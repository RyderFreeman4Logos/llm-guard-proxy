use super::*;

const SSE_RESIDUAL_CAP: usize = 64 * 1024;
const HEALTHY_EVENT: &[u8] = b"data: {\"choices\":[{\"delta\":{\"content\":\"healthy\"}}]}";

#[test]
fn watchdog_sse_fast_path_counts_completed_events_before_retaining_the_next_event() {
    let mut chunk = Vec::new();
    while chunk.len() <= SSE_RESIDUAL_CAP {
        chunk.extend_from_slice(b": ignored framing\n");
    }
    chunk.extend_from_slice(HEALTHY_EVENT);
    chunk.extend_from_slice(b"\n\n");
    chunk.extend_from_slice(HEALTHY_EVENT);
    chunk.push(b'\n');

    let mut parser = WatchdogProgressParser::default();
    assert_eq!(
        emitted_progress(WatchdogProgressUnit::Chat, &mut parser, &chunk),
        WatchdogProgressState::Progress(1),
        "a completed event before the trailing residual must count immediately"
    );
    assert_eq!(
        emitted_progress(WatchdogProgressUnit::Chat, &mut parser, b"\n"),
        WatchdogProgressState::Progress(1),
        "the retained trailing event must dispatch once after its later blank line"
    );
    assert_eq!(
        emitted_progress(WatchdogProgressUnit::Chat, &mut parser, b"\n"),
        WatchdogProgressState::Incomplete,
        "neither completed nor retained events may be counted twice"
    );
}
