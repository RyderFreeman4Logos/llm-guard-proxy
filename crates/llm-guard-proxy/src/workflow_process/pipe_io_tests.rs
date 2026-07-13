use std::{
    collections::VecDeque,
    io::{self, Read, Write},
    os::unix::net::UnixStream,
    time::{Duration, Instant},
};

use super::{
    PollAttempt, read_bounded_deadline_or_cancel, read_bounded_deadline_with,
    wait_until_ready_with, write_all_deadline_with,
};

#[test]
fn stderr_drain_cancelled_before_poll_returns_without_waiting_for_pipe_eof() {
    let (reader, _writer) = UnixStream::pair().expect("pipe fixture should open");
    let (cancel_owner, cancel_receiver) = UnixStream::pair().expect("cancel fixture should open");
    drop(cancel_owner);

    let result = read_bounded_deadline_or_cancel(
        reader,
        cancel_receiver,
        1024,
        Instant::now() + Duration::from_secs(30),
    );

    assert_eq!(
        result
            .expect_err("pre-poll cancellation should interrupt the drain")
            .kind(),
        io::ErrorKind::Interrupted
    );
}

#[test]
fn partial_writes_retry_interrupted_and_would_block_without_resetting_deadline() {
    let started = Instant::now();
    let mut writer = ScriptedWriter::new([
        WriteStep::Bytes(2),
        WriteStep::Error(io::ErrorKind::Interrupted),
        WriteStep::Error(io::ErrorKind::WouldBlock),
        WriteStep::Bytes(4),
    ]);
    let mut ready_calls = 0_usize;

    write_all_deadline_with(
        &mut writer,
        b"abcdef",
        started + Duration::from_secs(1),
        |_writer| {
            ready_calls += 1;
            Ok(())
        },
        || started,
    )
    .expect("scripted partial write should complete");

    assert_eq!(writer.output, b"abcdef");
    assert_eq!(ready_calls, 4);
}

#[test]
fn partial_reads_retry_interrupted_and_would_block_without_resetting_deadline() {
    let started = Instant::now();
    let mut reader = ScriptedReader::new([
        ReadStep::Bytes(b"ab".to_vec()),
        ReadStep::Error(io::ErrorKind::Interrupted),
        ReadStep::Error(io::ErrorKind::WouldBlock),
        ReadStep::Bytes(b"cd".to_vec()),
        ReadStep::Eof,
    ]);
    let mut ready_calls = 0_usize;

    let output = read_bounded_deadline_with(
        &mut reader,
        16,
        started + Duration::from_secs(1),
        |_writer| {
            ready_calls += 1;
            Ok(())
        },
        || started,
    )
    .expect("scripted partial read should complete");

    assert_eq!(output, b"abcd");
    assert_eq!(ready_calls, 5);
}

#[test]
fn ready_poll_is_rejected_when_the_absolute_deadline_elapsed_during_poll() {
    let started = Instant::now();
    let deadline = started + Duration::from_millis(50);
    let mut now = VecDeque::from([started, deadline]);

    let error = wait_until_ready_with(
        deadline,
        "test pipe operation",
        || now.pop_front().expect("test clock should have a sample"),
        |_remaining| Ok(PollAttempt::Ready),
    )
    .expect_err("readiness after the deadline must time out before I/O");

    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
}

#[test]
fn interrupted_poll_recomputes_remaining_time_against_the_same_deadline() {
    let started = Instant::now();
    let deadline = started + Duration::from_millis(100);
    let mut now = VecDeque::from([
        started,
        started + Duration::from_millis(10),
        started + Duration::from_millis(20),
    ]);
    let mut remaining = Vec::new();
    let mut attempts = VecDeque::from([
        Err(io::Error::from(io::ErrorKind::Interrupted)),
        Ok(PollAttempt::Ready),
    ]);

    wait_until_ready_with(
        deadline,
        "test pipe operation",
        || now.pop_front().expect("test clock should have a sample"),
        |timeout| {
            remaining.push(timeout);
            attempts
                .pop_front()
                .expect("test poll should have a scripted result")
        },
    )
    .expect("poll should retry EINTR before the deadline");

    assert_eq!(
        remaining,
        vec![Duration::from_millis(100), Duration::from_millis(90)]
    );
}

enum WriteStep {
    Bytes(usize),
    Error(io::ErrorKind),
}

struct ScriptedWriter {
    steps: VecDeque<WriteStep>,
    output: Vec<u8>,
}

impl ScriptedWriter {
    fn new(steps: impl IntoIterator<Item = WriteStep>) -> Self {
        Self {
            steps: steps.into_iter().collect(),
            output: Vec::new(),
        }
    }
}

impl Write for ScriptedWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        match self.steps.pop_front().expect("write step should exist") {
            WriteStep::Bytes(count) => {
                let count = count.min(buffer.len());
                self.output.extend_from_slice(&buffer[..count]);
                Ok(count)
            }
            WriteStep::Error(kind) => Err(io::Error::from(kind)),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

enum ReadStep {
    Bytes(Vec<u8>),
    Error(io::ErrorKind),
    Eof,
}

struct ScriptedReader {
    steps: VecDeque<ReadStep>,
}

impl ScriptedReader {
    fn new(steps: impl IntoIterator<Item = ReadStep>) -> Self {
        Self {
            steps: steps.into_iter().collect(),
        }
    }
}

impl Read for ScriptedReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        match self.steps.pop_front().expect("read step should exist") {
            ReadStep::Bytes(bytes) => {
                let count = bytes.len().min(buffer.len());
                buffer[..count].copy_from_slice(&bytes[..count]);
                Ok(count)
            }
            ReadStep::Error(kind) => Err(io::Error::from(kind)),
            ReadStep::Eof => Ok(0),
        }
    }
}
