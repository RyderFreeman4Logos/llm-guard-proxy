//! Deadline-driven Linux pipe I/O for workflow stdin, stdout, and stderr.

use std::{
    io::{self, Read, Write},
    os::fd::AsFd,
    time::{Duration, Instant},
};

use nix::{
    errno::Errno,
    poll::{PollFd, PollFlags, PollTimeout, poll},
};

const READ_CHUNK_BYTES: usize = 8 * 1024;
const PIPE_ATOMIC_WRITE_BYTES: usize = 4 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PollAttempt {
    Ready,
    TimedOut,
}

pub(super) fn write_all_deadline<W>(
    mut writer: W,
    payload: &[u8],
    deadline: Instant,
) -> io::Result<()>
where
    W: Write + AsFd,
{
    write_all_deadline_with(
        &mut writer,
        payload,
        deadline,
        |writer| wait_until_ready(writer, PollFlags::POLLOUT, deadline, "workflow stdin write"),
        Instant::now,
    )
}

pub(super) fn read_bounded_deadline<R>(
    mut reader: R,
    max_bytes: usize,
    deadline: Instant,
) -> io::Result<Vec<u8>>
where
    R: Read + AsFd,
{
    read_bounded_deadline_with(
        &mut reader,
        max_bytes,
        deadline,
        |reader| wait_until_ready(reader, PollFlags::POLLIN, deadline, "workflow pipe read"),
        Instant::now,
    )
}

pub(super) fn read_bounded_deadline_or_cancel<R, Cancel>(
    mut reader: R,
    cancel: Cancel,
    max_bytes: usize,
    deadline: Instant,
) -> io::Result<Vec<u8>>
where
    R: Read + AsFd,
    Cancel: AsFd,
{
    read_bounded_deadline_with(
        &mut reader,
        max_bytes,
        deadline,
        |reader| wait_until_readable_or_cancel(reader, &cancel, deadline),
        Instant::now,
    )
}

fn wait_until_readable_or_cancel<R, Cancel>(
    reader: &R,
    cancel: &Cancel,
    deadline: Instant,
) -> io::Result<()>
where
    R: AsFd,
    Cancel: AsFd,
{
    loop {
        let current = Instant::now();
        ensure_before_deadline(current, deadline, "workflow stderr drain")?;
        let remaining = deadline.saturating_duration_since(current);
        let timeout = PollTimeout::try_from(remaining).unwrap_or(PollTimeout::MAX);
        let mut descriptors = [
            PollFd::new(
                reader.as_fd(),
                PollFlags::POLLIN | PollFlags::POLLHUP | PollFlags::POLLERR,
            ),
            PollFd::new(
                cancel.as_fd(),
                PollFlags::POLLIN | PollFlags::POLLHUP | PollFlags::POLLERR,
            ),
        ];
        match poll(&mut descriptors, timeout) {
            Ok(0) => return Err(deadline_error("workflow stderr drain")),
            Ok(_) => {
                if descriptors[1]
                    .revents()
                    .is_some_and(|events| !events.is_empty())
                {
                    return Err(io::Error::new(
                        io::ErrorKind::Interrupted,
                        "workflow stderr drain cancelled",
                    ));
                }
                if descriptors[0]
                    .revents()
                    .is_some_and(|events| !events.is_empty())
                {
                    return Ok(());
                }
            }
            Err(Errno::EINTR) => {}
            Err(error) => return Err(io::Error::from_raw_os_error(error as i32)),
        }
    }
}

pub(super) fn write_all_deadline_with<W, WaitReady, Now>(
    writer: &mut W,
    payload: &[u8],
    deadline: Instant,
    mut wait_ready: WaitReady,
    mut now: Now,
) -> io::Result<()>
where
    W: Write,
    WaitReady: FnMut(&W) -> io::Result<()>,
    Now: FnMut() -> Instant,
{
    let mut offset = 0;
    while offset < payload.len() {
        ensure_before_deadline(now(), deadline, "workflow stdin write")?;
        wait_ready(writer)?;
        ensure_before_deadline(now(), deadline, "workflow stdin write")?;
        let end = offset
            .saturating_add(PIPE_ATOMIC_WRITE_BYTES)
            .min(payload.len());
        match writer.write(&payload[offset..end]) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "workflow stdin write returned zero bytes",
                ));
            }
            Ok(written) => offset = offset.saturating_add(written),
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::Interrupted | io::ErrorKind::WouldBlock
                ) => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

pub(super) fn read_bounded_deadline_with<R, WaitReady, Now>(
    reader: &mut R,
    max_bytes: usize,
    deadline: Instant,
    mut wait_ready: WaitReady,
    mut now: Now,
) -> io::Result<Vec<u8>>
where
    R: Read,
    WaitReady: FnMut(&R) -> io::Result<()>,
    Now: FnMut() -> Instant,
{
    let mut output = Vec::new();
    let mut buffer = [0_u8; READ_CHUNK_BYTES];
    loop {
        ensure_before_deadline(now(), deadline, "workflow pipe read")?;
        wait_ready(reader)?;
        ensure_before_deadline(now(), deadline, "workflow pipe read")?;
        match reader.read(&mut buffer) {
            Ok(0) => return Ok(output),
            Ok(read) => {
                if output.len().saturating_add(read) > max_bytes {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("workflow output exceeded byte limit={max_bytes}"),
                    ));
                }
                output.extend_from_slice(&buffer[..read]);
            }
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::Interrupted | io::ErrorKind::WouldBlock
                ) => {}
            Err(error) => return Err(error),
        }
    }
}

fn wait_until_ready<Fd>(
    fd: &Fd,
    events: PollFlags,
    deadline: Instant,
    operation: &'static str,
) -> io::Result<()>
where
    Fd: AsFd,
{
    wait_until_ready_with(deadline, operation, Instant::now, |remaining| {
        poll_once(fd, events, remaining)
    })
}

pub(super) fn wait_until_ready_with<Now, Poll>(
    deadline: Instant,
    operation: &'static str,
    mut now: Now,
    mut poll_once: Poll,
) -> io::Result<()>
where
    Now: FnMut() -> Instant,
    Poll: FnMut(Duration) -> io::Result<PollAttempt>,
{
    loop {
        let current = now();
        ensure_before_deadline(current, deadline, operation)?;
        let remaining = deadline.saturating_duration_since(current);
        match poll_once(remaining) {
            Ok(PollAttempt::TimedOut) => return Err(deadline_error(operation)),
            Ok(PollAttempt::Ready) => {
                ensure_before_deadline(now(), deadline, operation)?;
                return Ok(());
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
}

fn poll_once<Fd>(fd: &Fd, events: PollFlags, remaining: Duration) -> io::Result<PollAttempt>
where
    Fd: AsFd,
{
    let timeout = PollTimeout::try_from(remaining).unwrap_or(PollTimeout::MAX);
    let mut poll_fds = [PollFd::new(fd.as_fd(), events)];
    match poll(&mut poll_fds, timeout) {
        Ok(0) => Ok(PollAttempt::TimedOut),
        Ok(_) => {
            let Some(revents) = poll_fds[0].revents() else {
                return Err(io::Error::other(
                    "poll returned unknown workflow pipe flags",
                ));
            };
            if revents.contains(PollFlags::POLLNVAL) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "poll reported an invalid workflow pipe",
                ));
            }
            if revents.intersects(events | PollFlags::POLLERR | PollFlags::POLLHUP) {
                Ok(PollAttempt::Ready)
            } else {
                Err(io::Error::other(
                    "poll returned without workflow pipe readiness",
                ))
            }
        }
        Err(Errno::EINTR) => Err(io::Error::from(io::ErrorKind::Interrupted)),
        Err(error) => Err(io::Error::from_raw_os_error(error as i32)),
    }
}

fn ensure_before_deadline(
    now: Instant,
    deadline: Instant,
    operation: &'static str,
) -> io::Result<()> {
    if now >= deadline {
        Err(deadline_error(operation))
    } else {
        Ok(())
    }
}

fn deadline_error(operation: &'static str) -> io::Error {
    io::Error::new(
        io::ErrorKind::TimedOut,
        format!("{operation} exceeded workflow execution deadline"),
    )
}

#[cfg(test)]
#[path = "pipe_io_tests.rs"]
mod tests;
