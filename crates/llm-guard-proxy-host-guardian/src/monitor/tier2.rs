//! Generation-bound handoff to the proxy's existing local-recovery coordinator.
use super::{
    CgroupTarget, ConfigHandle, Duration, File, GuardianConfig, Instant, MemoryGuardian, OFlag,
    OpenOptions, PathBuf, RecoveryTarget, Uid, io, read_mem_available_compact, read_registration,
};
use crate::{EscalationConfig, EscalationEpisode};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::sync::{Arc, Mutex};
use tokio::sync::{Notify, mpsc, oneshot};

/// A content-free terminal acknowledgment, never a launcher acknowledgment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecoveryOutcome {
    /// Owned action exited successfully and readiness passed before the deadline.
    Succeeded,
    /// No action was admitted (busy, stale, receipt failure, or boot already claimed).
    NotAdmitted,
    /// Owned action failed after its process group was settled.
    Failed,
    /// Owner cancellation was observed and the process group settled.
    Cancelled,
    /// Shared action deadline expired and the process group settled.
    TimedOut,
    /// Cleanup or terminal durability is unconfirmed; never counts as success.
    Unconfirmed,
}

/// Exact retained generation plus an owner-controlled spawn/cancel fence.
/// Cancellation and synchronous action dispatch share one lock. No argv or
/// environment is retained here. External registration changes are revalidated
/// immediately before dispatch and continuously by the owning guardian.
#[derive(Debug)]
pub struct RecoveryAuthority {
    cancelled: Mutex<bool>,
    notify: Notify,
    target: CgroupTarget,
    config: ConfigHandle,
    revision: u64,
    policy: GuardianConfig,
    runtime_dir: PathBuf,
    meminfo: File,
    /// One absolute deadline for admission, action and readiness.
    pub deadline: Instant,
}

impl RecoveryAuthority {
    /// Constructs a fence from a successfully verified Tier-1 generation.
    /// The caller must retain the successful write/empty proof for this target.
    #[must_use]
    pub fn new(
        target: CgroupTarget,
        config: ConfigHandle,
        policy: GuardianConfig,
        runtime_dir: PathBuf,
        meminfo: File,
        deadline: Instant,
    ) -> Self {
        Self {
            cancelled: Mutex::new(false),
            notify: Notify::new(),
            target,
            revision: config.revision(),
            config,
            policy,
            runtime_dir,
            meminfo,
            deadline,
        }
    }

    fn valid(&self) -> bool {
        if Instant::now() >= self.deadline
            || self.config.guardian_snapshot().ok().as_ref() != Some(&self.policy)
            || !read_mem_available_compact(&self.meminfo).is_ok_and(|bytes| {
                bytes
                    <= self
                        .policy
                        .escalation_mem_threshold_gib
                        .saturating_mul(1024 * 1024 * 1024)
            })
            || self.target.is_empty() != Ok(true)
        {
            return false;
        }
        let registration = self
            .runtime_dir
            .join(self.policy.effective_registration_file());
        if read_registration(&registration, Uid::effective().as_raw())
            .ok()
            .as_ref()
            != Some(&self.target.registration)
        {
            return false;
        }
        let path = self.policy.cgroup_root.join(
            self.target
                .registration
                .control_group
                .trim_start_matches('/'),
        );
        let current = OpenOptions::new()
            .read(true)
            .custom_flags((OFlag::O_DIRECTORY | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC).bits())
            .open(path);
        match (
            current.and_then(|file| file.metadata()),
            self.target.directory.metadata(),
        ) {
            (Ok(current), Ok(held)) => current.dev() == held.dev() && current.ino() == held.ino(),
            _ => false,
        }
    }

    /// Runs a synchronous dispatch only while the exact owner remains valid.
    pub fn dispatch<T>(&self, action: impl FnOnce() -> T) -> Option<T> {
        let cancelled = self.cancelled.lock().ok()?;
        if *cancelled || !self.valid() {
            return None;
        }
        self.config.at_revision(self.revision, action)
    }

    /// Revokes future actions and wakes the executor to settle its actual child.
    pub fn cancel(&self) {
        if let Ok(mut cancelled) = self.cancelled.lock() {
            *cancelled = true;
        }
        self.notify.notify_waiters();
    }

    /// Waits for owner cancellation, generation invalidation, or the shared deadline.
    pub async fn cancelled(&self) {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            let _enabled = notified.as_mut().enable();
            if self.dispatch(|| ()).is_none() {
                return;
            }
            tokio::select! {
                () = notified => {},
                () = tokio::time::sleep(Duration::from_millis(50)) => {},
            }
        }
    }

    /// Installed configuration revision bound to this owner.
    #[must_use]
    pub const fn config_revision(&self) -> u64 {
        self.revision
    }

    /// Returns the retained target identity, independent of pathname replacement.
    ///
    /// # Errors
    /// Returns the descriptor metadata error if the retained target cannot be inspected.
    pub fn target_identity(&self) -> io::Result<(u64, u64, &str)> {
        let metadata = self.target.directory.metadata()?;
        Ok((
            metadata.dev(),
            metadata.ino(),
            &self.target.registration.container_id,
        ))
    }
}

/// One bounded handoff; the executor must settle ownership before acknowledging.
#[derive(Debug)]
pub struct RecoveryRequest {
    pub authority: Arc<RecoveryAuthority>,
    pub episode: u64,
    pub profile: String,
    pub available_bytes: u64,
    pub threshold_bytes: u64,
    pub grace_secs: u64,
    pub completion: oneshot::Sender<RecoveryOutcome>,
}

#[derive(Debug)]
struct Invocation {
    authority: Arc<RecoveryAuthority>,
    completion: oneshot::Receiver<RecoveryOutcome>,
}

#[derive(Debug, Default)]
pub(super) struct GuardianEscalation {
    sender: Option<mpsc::Sender<RecoveryRequest>>,
    pub(super) episode: EscalationEpisode,
    generation: Option<(u64, u64)>,
    next_episode: u64,
    invocation: Option<Invocation>,
    outcome: Option<RecoveryOutcome>,
}

impl GuardianEscalation {
    pub(super) fn revalidate(&mut self) {
        if let Some(invocation) = &mut self.invocation {
            if invocation.authority.dispatch(|| ()).is_none() {
                invocation.authority.cancel();
            }
            match invocation.completion.try_recv() {
                Ok(outcome) => {
                    self.outcome = Some(outcome);
                    self.invocation = None;
                }
                Err(oneshot::error::TryRecvError::Closed) => {
                    self.outcome = Some(RecoveryOutcome::Unconfirmed);
                    self.invocation = None;
                }
                Err(oneshot::error::TryRecvError::Empty) => {}
            }
        }
    }

    pub(super) async fn shutdown(&mut self) {
        if let Some(mut invocation) = self.invocation.take() {
            invocation.authority.cancel();
            self.outcome = Some(
                match tokio::time::timeout(Duration::from_secs(4), &mut invocation.completion).await
                {
                    Ok(Ok(outcome)) => outcome,
                    _ => RecoveryOutcome::Unconfirmed,
                },
            );
        }
    }
}

impl Drop for GuardianEscalation {
    fn drop(&mut self) {
        if let Some(invocation) = &self.invocation {
            invocation.authority.cancel();
        }
    }
}

impl CgroupTarget {
    fn try_clone(&self) -> io::Result<Self> {
        Ok(Self {
            directory: self.directory.try_clone()?,
            kill: self.kill.try_clone()?,
            events: self.events.try_clone()?,
            registration: self.registration.clone(),
        })
    }
}

impl MemoryGuardian {
    /// Installs the bounded in-process executor. Standalone mode cannot escalate.
    pub fn set_recovery_sender(&mut self, sender: mpsc::Sender<RecoveryRequest>) {
        self.escalation.sender = Some(sender);
    }

    /// Returns the most recent exact-episode completion, not a target-empty signal.
    #[must_use]
    pub const fn escalation_outcome(&self) -> Option<RecoveryOutcome> {
        self.escalation.outcome
    }

    pub(super) fn update_escalation(&mut self) {
        if !self.active_policy.escalation_enabled {
            return;
        }
        self.escalation.revalidate();
        let policy = &self.active_policy;
        let available = read_mem_available_compact(&self.proc_meminfo).ok();
        let target = match &self.target {
            Some(RecoveryTarget::Cgroup(target)) => Some(target),
            _ => None,
        };
        let generation = target
            .and_then(|target| target.directory.metadata().ok())
            .map(|m| (m.dev(), m.ino()));
        let eligible = policy.escalation_enabled
            && self.latched
            && self.controller.as_ref().is_some_and(|controller| {
                controller.target_is_verified() && controller.last_write_result() == Some(Ok(()))
            })
            && target.is_some_and(|target| target.is_empty() == Ok(true))
            && generation.is_some()
            && available.is_some_and(|bytes| {
                bytes
                    <= policy
                        .escalation_mem_threshold_gib
                        .saturating_mul(1024 * 1024 * 1024)
            });
        if !eligible || generation != self.escalation.generation {
            if let Some(invocation) = &self.escalation.invocation {
                invocation.authority.cancel();
            }
            self.escalation.episode.clear();
            self.escalation.generation = generation;
        }
        if !eligible || self.escalation.invocation.is_some() {
            return;
        }
        let now = Instant::now();
        let grace = Duration::from_secs(policy.escalation_grace_secs);
        self.escalation.episode.arm(
            &EscalationConfig {
                enabled: true,
                unit: None,
                grace_period: grace,
            },
            now,
        );
        if !self.escalation.episode.take_due(grace, now) {
            return;
        }
        let Some(sender) = &self.escalation.sender else {
            self.escalation.outcome = Some(RecoveryOutcome::NotAdmitted);
            return;
        };
        let Some(target) = target else {
            return;
        };
        let cloned = target
            .try_clone()
            .and_then(|target| Ok((target, self.proc_meminfo.try_clone()?)));
        let Ok((target, meminfo)) = cloned else {
            return;
        };
        let Some(episode) = self.escalation.next_episode.checked_add(1) else {
            return;
        };
        self.escalation.next_episode = episode;
        let authority = Arc::new(RecoveryAuthority::new(
            target,
            self.config_handle.clone(),
            policy.clone(),
            self.runtime_dir.clone(),
            meminfo,
            now + Duration::from_secs(policy.escalation_timeout_secs),
        ));
        let (completion, receiver) = oneshot::channel();
        let request = RecoveryRequest {
            authority: Arc::clone(&authority),
            episode,
            profile: policy.escalation_profile.clone(),
            available_bytes: available.unwrap_or_default(),
            threshold_bytes: policy
                .escalation_mem_threshold_gib
                .saturating_mul(1024 * 1024 * 1024),
            grace_secs: policy.escalation_grace_secs,
            completion,
        };
        if sender.try_send(request).is_ok() {
            self.escalation.invocation = Some(Invocation {
                authority,
                completion: receiver,
            });
        } else {
            authority.cancel();
            self.escalation.outcome = Some(RecoveryOutcome::NotAdmitted);
        }
    }
}
