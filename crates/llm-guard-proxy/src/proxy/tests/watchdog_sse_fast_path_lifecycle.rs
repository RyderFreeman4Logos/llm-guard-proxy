use super::*;
use std::{sync::Arc, time::Duration};
use tokio::time::Instant;

const SSE_RESIDUAL_CAP: usize = 64 * 1024;

fn oversized_sse_chunk(trailing_event: &[u8]) -> Vec<u8> {
    let mut chunk = Vec::new();
    while chunk.len() <= SSE_RESIDUAL_CAP {
        chunk.extend_from_slice(b": ignored framing\n");
    }
    chunk.extend_from_slice(b"data: ");
    chunk.extend_from_slice(trailing_event);
    chunk.push(b'\n');
    chunk
}

fn chat_event_with_content_len(content_len: usize) -> Vec<u8> {
    let prefix = b"{\"choices\":[{\"delta\":{\"content\":\"";
    let suffix = b"\"}}]}";
    let mut event = Vec::with_capacity(prefix.len() + content_len + suffix.len());
    event.extend_from_slice(prefix);
    event.extend(std::iter::repeat_n(b'x', content_len));
    event.extend_from_slice(suffix);
    event
}

#[test]
fn watchdog_sse_fast_path_progress_suppresses_mature_overlap_recovery() {
    let detection_window = Duration::from_secs(2);
    let tracker = Arc::new(StuckWatchdogTokenTracker::default());
    let stalled = tracker.begin_request("default", WatchdogProgressUnit::Chat, detection_window);
    {
        let mut windows = tracker
            .windows
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let attempt = windows
            .get_mut("default")
            .and_then(|window| window.attempts.values_mut().next())
            .expect("the stalled request must have an active watchdog attempt");
        attempt.started_at = Instant::now()
            .checked_sub(Duration::from_secs(3))
            .expect("test Instant must support watchdog maturation");
    }
    assert!(
        tracker.has_too_few_output_progress_units("default", detection_window, 1),
        "the test must begin with a mature stalled attempt eligible for recovery"
    );

    let healthy = tracker.begin_request("default", WatchdogProgressUnit::Chat, detection_window);
    let chunk = oversized_sse_chunk(b"{\"choices\":[{\"delta\":{\"content\":\"healthy\"}}]}");
    assert!(
        !healthy.record_emitted_chunk(&chunk),
        "an unterminated healthy SSE event must not record progress before dispatch"
    );
    assert!(
        healthy.record_emitted_chunk(b"\n"),
        "the retained healthy event must record progress once its boundary arrives"
    );
    assert_eq!(
        tracker.sample_count("default"),
        1,
        "the split healthy event must publish exactly one progress sample"
    );
    assert!(
        !tracker.has_too_few_output_progress_units("default", detection_window, 1),
        "the healthy split event must suppress recovery for the mature overlap"
    );

    drop(healthy);
    drop(stalled);
}

#[test]
fn watchdog_sse_fast_path_sticky_unobservable_excludes_a_mature_attempt_from_recovery() {
    let detection_window = Duration::from_secs(2);
    let tracker = Arc::new(StuckWatchdogTokenTracker::default());
    let request =
        tracker.begin_request("unobservable", WatchdogProgressUnit::Chat, detection_window);
    {
        let mut windows = tracker
            .windows
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let attempt = windows
            .get_mut("unobservable")
            .and_then(|window| window.attempts.values_mut().next())
            .expect("the test request must have an active watchdog attempt");
        attempt.started_at = Instant::now()
            .checked_sub(Duration::from_secs(3))
            .expect("test Instant must support watchdog maturation");
    }

    let data_syntax_len = b"{\"choices\":[{\"delta\":{\"content\":\"".len() + b"\"}}]}".len();
    let cap_plus_one_event = chat_event_with_content_len(SSE_RESIDUAL_CAP + 1 - data_syntax_len);
    assert!(
        !request.record_emitted_chunk(&oversized_sse_chunk(&cap_plus_one_event)),
        "an unretainable fast-path event cannot publish observable progress"
    );
    {
        let windows = tracker
            .windows
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let attempt = windows
            .get("unobservable")
            .and_then(|window| window.attempts.values().next())
            .expect("the unobservable attempt must remain active");
        assert!(attempt.unobservable_progress && attempt.observation_suspended);
    }
    assert!(
        !tracker.has_too_few_output_progress_units("unobservable", detection_window, 1),
        "sticky unobservability must exclude a mature attempt from false watchdog recovery"
    );

    drop(request);
}
