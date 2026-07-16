use std::{
    fs::File,
    io,
    os::unix::net::UnixStream,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::mpsc,
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use llm_guard_proxy_core::WorkflowConfig;

use super::pipe_io::read_bounded_deadline_or_cancel;

pub(super) struct PipeDrainHandle {
    cancel_endpoint: Option<UnixStream>,
    result_receiver: mpsc::Receiver<io::Result<Vec<u8>>>,
    thread: Option<JoinHandle<()>>,
}

impl PipeDrainHandle {
    pub(super) fn cancel(&mut self) {
        // Closing the socket endpoint is a non-blocking cancellation send. The
        // peer observes POLLHUP whether cancellation happens before or during poll.
        self.cancel_endpoint.take();
    }

    pub(super) fn finish(mut self, deadline: Instant, cancel: bool) -> io::Result<Vec<u8>> {
        if cancel {
            self.cancel();
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        let result = match self.result_receiver.recv_timeout(remaining) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.cancel();
                Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "workflow stderr drain exceeded cleanup deadline",
                ))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                self.cancel();
                Err(io::Error::other(
                    "workflow stderr drain completion channel disconnected",
                ))
            }
        };
        self.join_until(deadline);
        result
    }

    fn join_until(&mut self, deadline: Instant) {
        loop {
            let Some(handle) = self.thread.as_ref() else {
                return;
            };
            if handle.is_finished() {
                let handle = self.thread.take().expect("checked drain thread handle");
                let _join = handle.join();
                return;
            }
            let now = Instant::now();
            if now >= deadline {
                // Cancellation was already sent on timeout, or is sent by Drop.
                // Detaching here is the only bounded response to scheduler failure.
                return;
            }
            thread::sleep(Duration::from_millis(1).min(deadline.saturating_duration_since(now)));
        }
    }
}

impl Drop for PipeDrainHandle {
    fn drop(&mut self) {
        self.cancel();
        self.thread.take();
    }
}

#[cfg(test)]
std::thread_local! {
    static STDERR_DRAIN_COMPLETION_PROBE: std::cell::RefCell<Option<mpsc::Sender<()>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) struct StderrDrainCompletionProbeGuard;

#[cfg(test)]
pub(crate) fn install_stderr_drain_completion_probe(
    sender: mpsc::Sender<()>,
) -> StderrDrainCompletionProbeGuard {
    STDERR_DRAIN_COMPLETION_PROBE.with(|probe| {
        assert!(
            probe.borrow_mut().replace(sender).is_none(),
            "stderr drain completion probe was already installed on this test thread"
        );
    });
    StderrDrainCompletionProbeGuard
}

#[cfg(test)]
impl Drop for StderrDrainCompletionProbeGuard {
    fn drop(&mut self) {
        STDERR_DRAIN_COMPLETION_PROBE.with(|probe| {
            probe.borrow_mut().take();
        });
    }
}

pub(super) fn spawn_pipe_drain(
    pipe: File,
    execution_deadline: Instant,
) -> io::Result<PipeDrainHandle> {
    let (cancel_endpoint, cancel_receiver) = UnixStream::pair()?;
    let (result_sender, result_receiver) = mpsc::channel();
    #[cfg(test)]
    let completion_probe = STDERR_DRAIN_COMPLETION_PROBE.with(|probe| probe.borrow_mut().take());
    let thread = thread::Builder::new()
        .name(String::from("llm-guard-workflow-stderr"))
        .spawn(move || {
            let result = catch_unwind(AssertUnwindSafe(move || {
                read_bounded_deadline_or_cancel(
                    pipe,
                    cancel_receiver,
                    WorkflowConfig::default_max_stdout_bytes(),
                    execution_deadline,
                )
            }))
            .unwrap_or_else(|_| Err(io::Error::other("workflow stderr drain thread panicked")));
            #[cfg(test)]
            if let Some(completion_probe) = completion_probe {
                let _completed = completion_probe.send(());
            }
            let _result = result_sender.send(result);
        })?;
    Ok(PipeDrainHandle {
        cancel_endpoint: Some(cancel_endpoint),
        result_receiver,
        thread: Some(thread),
    })
}

pub(super) fn cleanup_and_finish_drain_with<T, Drain, Cleanup, Finish>(
    cleanup: Cleanup,
    drain: Option<Drain>,
    cleanup_deadline: Instant,
    finish: Finish,
) -> io::Result<(T, Option<io::Result<Vec<u8>>>)>
where
    Cleanup: FnOnce() -> io::Result<T>,
    Finish: FnOnce(Drain, Instant, bool) -> io::Result<Vec<u8>>,
{
    match cleanup() {
        Ok(value) => Ok((
            value,
            drain.map(|drain| finish(drain, cleanup_deadline, false)),
        )),
        Err(error) => {
            if let Some(drain) = drain {
                let _drain = finish(drain, cleanup_deadline, true);
            }
            Err(error)
        }
    }
}
