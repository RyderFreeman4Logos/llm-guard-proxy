use super::*;

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
fn watchdog_sse_fast_path_dispatches_bounded_trailing_events_for_every_route_and_entry() {
    let cases = [
        (
            "/v1/chat/completions",
            b"{\"choices\":[{\"delta\":{\"content\":\"healthy\"}}]}".as_slice(),
        ),
        (
            "/v1/completions",
            b"{\"choices\":[{\"text\":\"healthy\"}]}".as_slice(),
        ),
        (
            "/v1/embeddings",
            b"{\"data\":[{\"embedding\":[0.1]}]}".as_slice(),
        ),
        (
            "/v1/rerank",
            b"{\"results\":[{\"relevance_score\":0.9}]}".as_slice(),
        ),
    ];

    for (route, event) in cases {
        let uri = route.parse().expect("test route must parse as a URI");
        let unit = watchdog_progress_unit(&uri).expect("watched route must map to a progress unit");
        for entered_sse_mode in [false, true] {
            let mut parser = WatchdogProgressParser::default();
            if entered_sse_mode {
                assert_eq!(
                    emitted_progress(unit, &mut parser, b": frame mode\n"),
                    WatchdogProgressState::Incomplete,
                    "the setup comment must enter SSE framing without dispatching {route} progress"
                );
            }

            let chunk = oversized_sse_chunk(event);
            assert!(
                chunk.len() > SSE_RESIDUAL_CAP,
                "the {route} fixture must take the oversized SSE fast path"
            );
            assert_eq!(
                emitted_progress(unit, &mut parser, &chunk),
                WatchdogProgressState::Incomplete,
                "an unterminated {route} event must not dispatch before its blank line"
            );
            assert_eq!(
                emitted_progress(unit, &mut parser, b"\n"),
                WatchdogProgressState::Progress(1),
                "the following blank line must dispatch exactly one retained {route} event"
            );
            assert_eq!(
                emitted_progress(unit, &mut parser, b"\n"),
                WatchdogProgressState::Incomplete,
                "a second blank line must not double-dispatch the {route} event"
            );
        }
    }
}

#[test]
fn watchdog_sse_fast_path_retains_exactly_capped_data_and_makes_cap_plus_one_sticky() {
    let data_syntax_len = b"{\"choices\":[{\"delta\":{\"content\":\"".len() + b"\"}}]}".len();
    let exact_cap_event = chat_event_with_content_len(SSE_RESIDUAL_CAP - data_syntax_len);
    assert_eq!(exact_cap_event.len(), SSE_RESIDUAL_CAP);
    let mut exact_cap_parser = WatchdogProgressParser::default();
    assert_eq!(
        emitted_progress(
            WatchdogProgressUnit::Chat,
            &mut exact_cap_parser,
            &oversized_sse_chunk(&exact_cap_event),
        ),
        WatchdogProgressState::Incomplete,
        "a capped event remains pending until its blank event delimiter arrives"
    );
    assert_eq!(
        emitted_progress(WatchdogProgressUnit::Chat, &mut exact_cap_parser, b"\n",),
        WatchdogProgressState::Progress(1),
        "data exactly at the residual cap must remain observable across the oversized fast path"
    );

    let cap_plus_one_event = chat_event_with_content_len(SSE_RESIDUAL_CAP + 1 - data_syntax_len);
    assert_eq!(cap_plus_one_event.len(), SSE_RESIDUAL_CAP + 1);
    let mut cap_plus_one_parser = WatchdogProgressParser::default();
    assert_eq!(
        emitted_progress(
            WatchdogProgressUnit::Chat,
            &mut cap_plus_one_parser,
            &oversized_sse_chunk(&cap_plus_one_event),
        ),
        WatchdogProgressState::UnobservableOversize,
        "an unretainable cap-plus-one event must suspend observation instead of silently losing data"
    );
    assert_eq!(
        emitted_progress(WatchdogProgressUnit::Chat, &mut cap_plus_one_parser, b"\n",),
        WatchdogProgressState::UnobservableOversize,
        "unobservable oversized state must stay sticky after a later event boundary"
    );
}
