use super::*;

fn parse_sse_chunks(
    progress_unit: WatchdogProgressUnit,
    chunks: &[&[u8]],
) -> WatchdogProgressState {
    let mut parser = WatchdogProgressParser::default();
    let mut state = WatchdogProgressState::Incomplete;
    let mut progress = 0_u64;
    for chunk in chunks {
        state = emitted_progress(progress_unit, &mut parser, chunk);
        if let WatchdogProgressState::Progress(emitted) = state {
            progress = progress.saturating_add(emitted);
        }
    }
    if progress > 0 {
        WatchdogProgressState::Progress(progress)
    } else {
        state
    }
}

#[test]
fn watchdog_sse_event_framing_joins_multiline_data_for_every_progress_protocol() {
    let cases = [
        (
            "chat",
            WatchdogProgressUnit::Chat,
            b"data: {\"choices\":[{\"delta\":\ndata: {\"content\":\"healthy\"}}]}\n\n".as_slice(),
        ),
        (
            "completion",
            WatchdogProgressUnit::Completion,
            b"data: {\"choices\":[\ndata: {\"text\":\"healthy\"}]}\n\n".as_slice(),
        ),
        (
            "embedding",
            WatchdogProgressUnit::Embedding,
            b"data: {\"data\":[\ndata: {\"embedding\":[0.1]}]}\n\n".as_slice(),
        ),
        (
            "reranker",
            WatchdogProgressUnit::Reranker,
            b"data: {\"results\":[\ndata: {\"relevance_score\":0.9}]}\n\n".as_slice(),
        ),
    ];

    for (name, progress_unit, event) in cases {
        assert_eq!(
            parse_sse_chunks(progress_unit, &[event]),
            WatchdogProgressState::Progress(1),
            "a multiline {name} SSE event must be joined before semantic parsing"
        );
    }
}

#[test]
fn watchdog_sse_event_framing_accepts_all_line_endings_at_every_chunk_boundary() {
    for (name, ending) in [("lf", "\n"), ("crlf", "\r\n"), ("cr", "\r")] {
        let event = format!(
            "data: {{\"choices\":[{{\"delta\":{ending}data: {{\"content\":\"healthy\"}}}}]}}{ending}{ending}"
        );
        let event_bytes = event.as_bytes();
        for split in 0..=event_bytes.len() {
            assert_eq!(
                parse_sse_chunks(
                    WatchdogProgressUnit::Chat,
                    &[&event_bytes[..split], &event_bytes[split..]],
                ),
                WatchdogProgressState::Progress(1),
                "{name} event must parse when split at byte boundary {split}"
            );
        }
    }
}

#[test]
fn watchdog_sse_event_framing_aggregates_bounded_multiline_data_in_a_large_complete_chunk() {
    let mut chunk = Vec::new();
    for _ in 0..7_000 {
        chunk.extend_from_slice(b": ignored framing\n");
    }
    chunk.extend_from_slice(b"data: {\"choices\":[{\"delta\":\n");
    chunk.extend_from_slice(b"data: {\"content\":\"healthy\"}}]}\n\n");

    assert_eq!(
        parse_sse_chunks(WatchdogProgressUnit::Chat, &[&chunk]),
        WatchdogProgressState::Progress(1),
        "large complete chunks must still join bounded per-event data without retaining ignored frames",
    );
}

#[test]
fn watchdog_sse_event_framing_ignores_nonsemantic_fields_and_resets_malformed_events() {
    let mut parser = WatchdogProgressParser::default();
    let ignored =
        b": heartbeat\r\nevent: chunk\r\nid: 42\r\nretry: 10\r\ndata:\r\n\r\ndata: [DONE]\r\n\r\n";
    assert_eq!(
        emitted_progress(WatchdogProgressUnit::Chat, &mut parser, ignored),
        WatchdogProgressState::Incomplete,
        "comments, non-data fields, empty data, and [DONE] are not semantic progress"
    );
    assert_eq!(
        emitted_progress(
            WatchdogProgressUnit::Chat,
            &mut parser,
            b"data: {\"choices\":[}\r\n\r\n",
        ),
        WatchdogProgressState::Incomplete,
        "malformed dispatched JSON is not progress"
    );
    assert_eq!(
        emitted_progress(
            WatchdogProgressUnit::Chat,
            &mut parser,
            b"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}",
        ),
        WatchdogProgressState::Incomplete,
        "an unterminated event is not progress"
    );
    assert_eq!(
        emitted_progress(WatchdogProgressUnit::Chat, &mut parser, b"\r\r"),
        WatchdogProgressState::Progress(1),
        "a following complete CR-delimited event boundary dispatches exactly once"
    );
}

#[test]
fn watchdog_sse_event_framing_makes_oversized_incomplete_events_sticky_unobservable() {
    let mut parser = WatchdogProgressParser::default();
    let mut oversized = b"data: ".to_vec();
    oversized.extend(std::iter::repeat_n(b'x', 64 * 1024));

    assert_eq!(
        emitted_progress(
            WatchdogProgressUnit::Chat,
            &mut parser,
            &oversized[..64 * 1024],
        ),
        WatchdogProgressState::Incomplete,
        "the event may fill, but not exceed, the explicit residual cap"
    );
    assert_eq!(
        emitted_progress(
            WatchdogProgressUnit::Chat,
            &mut parser,
            &oversized[64 * 1024..],
        ),
        WatchdogProgressState::UnobservableOversize,
        "an oversized incomplete event must suspend observation instead of parsing a tail"
    );
    assert_eq!(
        emitted_progress(
            WatchdogProgressUnit::Chat,
            &mut parser,
            b"data: {\"choices\":[{\"delta\":{\"content\":\"later\"}}]}\n\n",
        ),
        WatchdogProgressState::UnobservableOversize,
        "oversized event degradation must stay sticky for the physical attempt"
    );
}
