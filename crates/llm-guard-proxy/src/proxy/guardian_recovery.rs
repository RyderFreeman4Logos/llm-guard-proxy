//! In-process Tier-2 adapter; admission is shared with request/watchdog recovery.
use super::{
    Arc, Instant, LocalRecoveryCause, LocalRecoveryPolicy, ProxyState, RecoveryProcessGuard, Stdio,
    UpstreamProfileConfig, configure_recovery_command, finish_local_recovery_episode,
    local_recovery_admission_failure, recovery_receipt, send_local_recovery_readiness_probe,
    terminate_timed_out_recovery_child, watchdog_upstream_profiles,
};
use llm_guard_proxy_host_guardian::monitor::tier2::{RecoveryOutcome, RecoveryRequest};
use llm_guard_proxy_state::GuardianRecoveryIdentity;
use std::os::unix::fs::MetadataExt;
use tokio::{process::Command, sync::mpsc};

impl ProxyState {
    /// One bounded worker, sharing the existing profile coordinator and persistence drain.
    pub(crate) fn spawn_guardian_recovery(
        &self,
    ) -> (mpsc::Sender<RecoveryRequest>, tokio::task::JoinHandle<()>) {
        let (sender, mut receiver) = mpsc::channel(1);
        let state = self.clone();
        let worker = tokio::spawn(async move {
            loop {
                let request = tokio::select! {
                    () = state.wait_for_shutdown() => break,
                    request = receiver.recv() => match request { Some(request) => request, None => break },
                };
                state.run_guardian_recovery(request).await;
            }
        });
        (sender, worker)
    }

    async fn run_guardian_recovery(&self, request: RecoveryRequest) {
        let outcome = self.guardian_recovery_outcome(&request).await;
        let _acknowledged = request.completion.send(outcome);
    }

    async fn guardian_recovery_outcome(&self, request: &RecoveryRequest) -> RecoveryOutcome {
        let Some(profile) = self.config.snapshot().ok().and_then(|config| {
            watchdog_upstream_profiles(&config)
                .into_iter()
                .find(|profile| profile.name == request.profile)
        }) else {
            return RecoveryOutcome::NotAdmitted;
        };
        let policy = LocalRecoveryPolicy::from_config(&profile.local_recovery);
        if !policy.is_configured()
            || !foreground_command(&policy.restart_command)
            || request.authority.dispatch(|| ()).is_none()
            || self.shutdown.is_shutting_down()
        {
            return RecoveryOutcome::NotAdmitted;
        }
        let coordinator = self.local_recovery.coordinator_for(&profile.name);
        let Ok(mut state) = coordinator.state.try_lock() else {
            return RecoveryOutcome::NotAdmitted;
        };
        // A Guardian must not join, cancel, or claim success from somebody else's recovery.
        if state.running
            || local_recovery_admission_failure(&policy, &mut state, Instant::now()).is_some()
        {
            return RecoveryOutcome::NotAdmitted;
        }
        let Some(episode) = state.ensure_active_recovery_episode() else {
            return RecoveryOutcome::NotAdmitted;
        };
        let Some(identity) = identity(request) else {
            return RecoveryOutcome::NotAdmitted;
        };
        state.running = true;
        state.active_guardian_recovery = true;
        state.recovery_started = Some(Instant::now());
        state.recovery_deadline = Some(request.authority.deadline.into());
        state.runs_in_window = state.runs_in_window.saturating_add(1);
        drop(state);
        let receipt = recovery_receipt::Context::new(
            self.store.clone(),
            &profile,
            Arc::clone(&self.persistence_tasks),
        )
        .episode(episode, LocalRecoveryCause::UpstreamStall, &policy)
        .guardian(identity)
        .deadline(request.authority.deadline.into())
        .shutdown(Arc::clone(&self.shutdown));
        let _tracked = receipt.track();
        let mut metadata = match receipt.pre_spawn(None).await {
            Ok(metadata) => metadata,
            Err(metadata) => {
                #[cfg(test)]
                eprintln!("guardian admission failure={metadata:?}");
                finish_local_recovery_episode(&coordinator, episode, metadata).await;
                return RecoveryOutcome::NotAdmitted;
            }
        };
        let outcome = self
            .guardian_owned_action(request, &profile, &policy, &receipt)
            .await;
        let label = match outcome {
            RecoveryOutcome::Succeeded => "succeeded",
            RecoveryOutcome::NotAdmitted => "not_admitted",
            RecoveryOutcome::Failed => "failed",
            RecoveryOutcome::Cancelled => "cancelled",
            RecoveryOutcome::TimedOut => "episode_timeout",
            RecoveryOutcome::Unconfirmed => "cleanup_unconfirmed",
        };
        metadata.insert(String::from("local_recovery_status"), String::from(label));
        let terminal = receipt.finish(&metadata).await;
        #[cfg(test)]
        eprintln!("guardian terminal outcome={outcome:?} durability={terminal:?}");
        let effective_outcome = match terminal {
            Ok(()) => outcome,
            Err(category) => {
                metadata.insert(
                    String::from("local_recovery_status"),
                    String::from("receipt_failed"),
                );
                metadata.insert(
                    String::from("local_recovery_receipt_error"),
                    category.to_owned(),
                );
                RecoveryOutcome::Unconfirmed
            }
        };
        // Only physically unconfirmed cleanup retains the admission fence; a settled
        // child with failed terminal durability publishes failure and wakes waiters.
        if outcome != RecoveryOutcome::Unconfirmed {
            finish_local_recovery_episode(&coordinator, episode, metadata).await;
        }
        effective_outcome
    }

    async fn guardian_owned_action(
        &self,
        request: &RecoveryRequest,
        profile: &UpstreamProfileConfig,
        policy: &LocalRecoveryPolicy,
        receipt: &recovery_receipt::Context,
    ) -> RecoveryOutcome {
        let mut command = Command::new(&policy.restart_command[0]);
        command
            .args(&policy.restart_command[1..])
            .kill_on_drop(true)
            .env("LLM_GUARD_RECOVERY_RECEIPT_ID", receipt.id())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        configure_recovery_command(&mut command);
        // Cancellation and the actual spawn are serialized by the owner fence.
        let child = request.authority.dispatch(|| command.spawn());
        let mut child = match child {
            Some(Ok(child)) => RecoveryProcessGuard::new(child),
            Some(Err(_)) => return RecoveryOutcome::Failed,
            None => return cancelled_outcome(request),
        };
        let deadline = Instant::from_std(request.authority.deadline);
        let action_deadline = deadline.min(Instant::now() + policy.restart_timeout);
        let status = tokio::select! {
            biased;
            () = self.wait_for_shutdown() => None,
            () = request.authority.cancelled() => None,
            result = tokio::time::timeout_at(action_deadline, child.wait()) => Some(result),
        };
        match status {
            Some(Ok(Ok(status))) if status.success() => {}
            Some(Ok(Ok(_))) => return RecoveryOutcome::Failed,
            _ => {
                let cleanup = terminate_timed_out_recovery_child(&mut child).await;
                #[cfg(test)]
                eprintln!("guardian cleanup={cleanup:?}");
                if cleanup
                    .get("upstream_stall_recovery_timeout_cleanup_status")
                    .map(String::as_str)
                    != Some("terminated_after_kill")
                {
                    return RecoveryOutcome::Unconfirmed;
                }
                return if Instant::now() >= action_deadline {
                    RecoveryOutcome::TimedOut
                } else {
                    cancelled_outcome(request)
                };
            }
        }
        let deadline = deadline.min(Instant::now() + policy.readiness_deadline);
        loop {
            if request.authority.dispatch(|| ()).is_none() || self.shutdown.is_shutting_down() {
                return cancelled_outcome(request);
            }
            let ready = tokio::select! {
                biased;
                () = self.wait_for_shutdown() => return RecoveryOutcome::Cancelled,
                () = request.authority.cancelled() => return cancelled_outcome(request),
                result = tokio::time::timeout_at(deadline,
                    send_local_recovery_readiness_probe(&self.client, &profile.base_url, policy)) => result,
            };
            if request.authority.dispatch(|| ()).is_none() {
                return cancelled_outcome(request);
            }
            if matches!(ready, Ok(Ok(true))) {
                return RecoveryOutcome::Succeeded;
            }
            if ready.is_err() {
                return RecoveryOutcome::TimedOut;
            }
            tokio::select! {
                () = self.wait_for_shutdown() => return RecoveryOutcome::Cancelled,
                () = request.authority.cancelled() => return cancelled_outcome(request),
                () = tokio::time::sleep_until(deadline.min(Instant::now() + policy.readiness_interval)) => {},
            }
        }
    }
}

fn cancelled_outcome(request: &RecoveryRequest) -> RecoveryOutcome {
    if std::time::Instant::now() >= request.authority.deadline {
        RecoveryOutcome::TimedOut
    } else {
        RecoveryOutcome::Cancelled
    }
}

fn identity(request: &RecoveryRequest) -> Option<GuardianRecoveryIdentity> {
    let (device, inode, container_id) = request.authority.target_identity().ok()?;
    let binary = std::fs::File::open("/proc/self/exe")
        .ok()?
        .metadata()
        .ok()?;
    Some(GuardianRecoveryIdentity {
        boot_id: std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
            .ok()?
            .trim()
            .to_owned(),
        guardian_episode: request.episode,
        config_revision: request.authority.config_revision(),
        target_device: device,
        target_inode: inode,
        container_id: container_id.to_owned(),
        available_bytes: request.available_bytes,
        threshold_bytes: request.threshold_bytes,
        grace_secs: request.grace_secs,
        binary_device: binary.dev(),
        binary_inode: binary.ino(),
    })
}

fn foreground_command(command: &[String]) -> bool {
    // These launchers transfer the work outside the owned process group. Arbitrary
    // configured helpers remain trusted: they must stay foreground and must not setsid/daemonize.
    command.first().is_some_and(|program| {
        std::path::Path::new(program).is_absolute()
            && !matches!(
                std::path::Path::new(program)
                    .file_name()
                    .and_then(|name| name.to_str()),
                Some("systemctl" | "systemd-run" | "docker" | "sh" | "bash" | "sudo")
            )
    })
}
