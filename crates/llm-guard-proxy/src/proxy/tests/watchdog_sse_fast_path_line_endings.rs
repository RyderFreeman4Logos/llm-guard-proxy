use super::*;

const SSE_RESIDUAL_CAP: usize = 64 * 1024;

#[test]
fn watchdog_sse_fast_path_preserves_a_crlf_boundary_split_after_trailing_data() {
    let mut chunk = Vec::new();
    while chunk.len() <= SSE_RESIDUAL_CAP {
        chunk.extend_from_slice(b": ignored framing\r\n");
    }
    chunk.extend_from_slice(b"data: {\"choices\":[{\"delta\":{\"content\":\"healthy\"}}]}\r");
    assert!(
        chunk.len() > SSE_RESIDUAL_CAP,
        "the fixture must enter the oversized fast path"
    );

    let mut parser = WatchdogProgressParser::default();
    assert_eq!(
        emitted_progress(WatchdogProgressUnit::Chat, &mut parser, &chunk),
        WatchdogProgressState::Incomplete,
        "a CR-terminated data line remains pending until the following blank line"
    );
    assert_eq!(
        emitted_progress(WatchdogProgressUnit::Chat, &mut parser, b"\n\r\n"),
        WatchdogProgressState::Progress(1),
        "the LF paired with the split CR must not create a premature blank event"
    );
}
