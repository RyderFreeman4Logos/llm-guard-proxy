use std::{
    ops::{Deref, DerefMut},
    process::{Child, Command},
    thread,
    time::{Duration, Instant},
};

use nix::sys::signal::Signal;

use super::{
    DeferredSignalState, DeferredWorkflowProcess, LinuxProcessIdentity, SharedDeferredReaper,
    SignalAuthority, SpawnedChildGuard, configure_process_group, signal_owned_workflow,
};

pub(super) struct TestDeferredProcess {
    process: Option<DeferredWorkflowProcess>,
    identity: LinuxProcessIdentity,
}

impl TestDeferredProcess {
    pub(super) const fn new(
        process: DeferredWorkflowProcess,
        identity: LinuxProcessIdentity,
    ) -> Self {
        Self {
            process: Some(process),
            identity,
        }
    }

    pub(super) fn spawn_true(signal_state: DeferredSignalState) -> Self {
        let mut command = Command::new("/bin/true");
        configure_process_group(&mut command);
        let child = command.spawn().expect("deferred true fixture should spawn");
        let cleanup_deadline = Instant::now() + super::PROCESS_REAP_GRACE;
        let mut spawned = SpawnedChildGuard::new(child, cleanup_deadline);
        let identity = LinuxProcessIdentity::capture(spawned.pid())
            .expect("deferred true fixture identity should be captured");
        spawned.set_signal_authority(SignalAuthority::new(identity));
        let child = spawned
            .disarm()
            .expect("spawn guard should transfer the deferred true fixture");
        Self::new(DeferredWorkflowProcess::new(child, signal_state), identity)
    }

    pub(super) fn into_process(mut self) -> DeferredWorkflowProcess {
        self.process
            .take()
            .expect("deferred fixture should transfer exactly once")
    }
}

impl Deref for TestDeferredProcess {
    type Target = DeferredWorkflowProcess;

    fn deref(&self) -> &Self::Target {
        self.process
            .as_ref()
            .expect("deferred fixture should remain armed")
    }
}

impl DerefMut for TestDeferredProcess {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.process
            .as_mut()
            .expect("deferred fixture should remain armed")
    }
}

impl Drop for TestDeferredProcess {
    fn drop(&mut self) {
        let Some(mut process) = self.process.take() else {
            return;
        };
        let authority = SignalAuthority::new(self.identity);
        let _signal = signal_owned_workflow(&authority, Signal::SIGKILL);
        let deadline = Instant::now() + super::PROCESS_REAP_GRACE;
        while Instant::now() < deadline {
            match process.child.try_wait() {
                Ok(Some(_)) | Err(_) => return,
                Ok(None) => thread::sleep(super::PROCESS_REAP_POLL),
            }
        }
    }
}

pub(super) struct TestLocalDeferredReaper {
    reaper: SharedDeferredReaper,
}

impl TestLocalDeferredReaper {
    pub(super) const fn new(reaper: SharedDeferredReaper) -> Self {
        Self { reaper }
    }
}

impl Deref for TestLocalDeferredReaper {
    type Target = SharedDeferredReaper;

    fn deref(&self) -> &Self::Target {
        &self.reaper
    }
}

impl Drop for TestLocalDeferredReaper {
    fn drop(&mut self) {
        let mut processes = self
            .reaper
            .processes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for mut process in processes.drain(..) {
            let deadline = Instant::now() + Duration::from_secs(1);
            while Instant::now() < deadline {
                match process.child.try_wait() {
                    Ok(Some(_)) | Err(_) => break,
                    Ok(None) => thread::sleep(Duration::from_millis(10)),
                }
            }
        }
        self.reaper
            .pending_cleanups
            .store(0, std::sync::atomic::Ordering::SeqCst);
    }
}

pub(super) struct TestProcessGroup {
    child: Option<Child>,
    identity: LinuxProcessIdentity,
}

impl TestProcessGroup {
    pub(super) fn spawn(command: &mut Command) -> Self {
        let child = command.spawn().expect("test process should spawn");
        let cleanup_deadline = Instant::now() + Duration::from_secs(1);
        let mut spawned = SpawnedChildGuard::new(child, cleanup_deadline);
        let identity = LinuxProcessIdentity::capture(spawned.pid())
            .expect("test process identity should be captured");
        spawned.set_signal_authority(SignalAuthority::new(identity));
        let child = spawned
            .disarm()
            .expect("spawn guard should transfer the process-group fixture");
        Self {
            child: Some(child),
            identity,
        }
    }

    pub(super) const fn identity(&self) -> LinuxProcessIdentity {
        self.identity
    }

    pub(super) fn child_mut(&mut self) -> &mut Child {
        self.child.as_mut().expect("test child should remain armed")
    }

    pub(super) fn finish_after_signal(mut self) {
        let deadline = Instant::now() + Duration::from_secs(1);
        while Instant::now() < deadline {
            match self.child_mut().try_wait() {
                Ok(Some(_)) => {
                    self.child.take();
                    return;
                }
                Ok(None) | Err(_) => thread::sleep(Duration::from_millis(10)),
            }
        }
        panic!("signalled test process should exit before cleanup deadline");
    }
}

impl Drop for TestProcessGroup {
    fn drop(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };
        let authority = SignalAuthority::new(self.identity);
        let _signal = signal_owned_workflow(&authority, Signal::SIGKILL);
        let deadline = Instant::now() + Duration::from_secs(1);
        while Instant::now() < deadline {
            match child.try_wait() {
                Ok(Some(_)) | Err(_) => return,
                Ok(None) => thread::sleep(Duration::from_millis(10)),
            }
        }
    }
}
