//! Thread-safe registry of in-flight requests for live per-request observability.
//!
//! Unlike [`crate::ObservabilityStore`] (which records only completed requests
//! and attempts in `SQLite`), the live registry tracks requests that are
//! currently active or queued. This lets operators inspect streaming progress,
//! retry rungs, queue wait times, and stage timelines directly from the request
//! port — without attaching to process logs.
//!
//! The registry is metadata-only: no raw prompts or response bodies are
//! captured here. Only counters and state transitions are recorded.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Thread-safe registry of in-flight requests keyed by request id.
///
/// Implemented with `std::sync::RwLock<HashMap<..>>` to avoid introducing new
/// dependencies (the workspace has no `dashmap`). All entry mutations
/// (counter increments, state updates, profile/metadata updates) update
/// `last_updated_at_ms`.
#[derive(Clone, Debug, Default)]
pub struct LiveRequestRegistry {
    entries: Arc<RwLock<HashMap<String, LiveRequestEntry>>>,
}

/// A single in-flight request observed by the live registry.
#[derive(Clone, Debug)]
pub struct LiveRequestEntry {
    /// Stable identifier matching [`crate::RequestId`].
    pub request_id: String,
    /// Downstream listener name, if known.
    pub listener: Option<String>,
    /// Selected upstream profile name, if known.
    pub profile: Option<String>,
    /// Requested model id, if known.
    pub model: Option<String>,
    /// Upstream target URL (redacted form), if known.
    pub upstream_target: Option<String>,
    /// `"streaming"` or `"non_stream_json"`.
    pub downstream_mode: String,
    /// Current lifecycle state of the request.
    pub state: LiveRequestState,
    /// Wall-clock milliseconds since `UNIX_EPOCH` when the entry was created.
    pub created_at_ms: u64,
    /// Wall-clock milliseconds since `UNIX_EPOCH` of the last mutation.
    pub last_updated_at_ms: u64,
    /// Time spent waiting in the admission queue, in milliseconds.
    pub queue_wait_ms: Option<u64>,
    /// Elapsed upstream wall-clock time, in milliseconds (set when known).
    pub upstream_elapsed_ms: Option<u64>,
    /// Latency from upstream connect to first token, in milliseconds.
    pub first_token_latency_ms: Option<u64>,
    /// Active retry-ladder rung name, if currently retrying.
    pub active_ladder_rung: Option<String>,
    /// Zero-based index of the active attempt, if known.
    pub active_attempt_index: Option<u32>,
    /// Number of SSE chunks streamed downstream so far.
    pub chunks_downstream: u64,
    /// Bytes streamed downstream so far.
    pub bytes_downstream: u64,
    /// Number of chunks received from upstream so far.
    pub chunks_upstream: u64,
    /// Bytes received from upstream so far.
    pub bytes_upstream: u64,
    /// Wall-clock milliseconds of the last progress event.
    pub last_progress_at_ms: Option<u64>,
    /// Ordered list of lifecycle events for the request timeline.
    pub timeline: Vec<TimelineEvent>,
}

/// Lifecycle state of a live request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LiveRequestState {
    /// Request received but not yet admitted.
    Accepted,
    /// Waiting in the admission queue for an in-flight slot.
    Queued,
    /// Admitted and holding an in-flight slot.
    Admitted,
    /// Building/sending the upstream request.
    UpstreamConnecting,
    /// Upstream response headers received.
    UpstreamHeadersReceived,
    /// First upstream token observed.
    FirstTokenSeen,
    /// Actively streaming chunks downstream.
    Streaming,
    /// Buffering for loop-guard analysis.
    LoopGuardBuffering,
    /// Retrying after a failed attempt.
    Retrying,
    /// A shadow attempt is running alongside the primary.
    ShadowAttemptRunning,
    /// A paired (shadow) comparison is running.
    PairedComparisonRunning,
    /// The downstream client disconnected.
    ClientDisconnected,
    /// The request completed successfully.
    Completed,
    /// The request failed.
    Failed,
}

impl LiveRequestState {
    /// Returns the stable string label for this state, used in JSON output.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Queued => "queued",
            Self::Admitted => "admitted",
            Self::UpstreamConnecting => "upstream_connecting",
            Self::UpstreamHeadersReceived => "upstream_headers_received",
            Self::FirstTokenSeen => "first_token_seen",
            Self::Streaming => "streaming",
            Self::LoopGuardBuffering => "loop_guard_buffering",
            Self::Retrying => "retrying",
            Self::ShadowAttemptRunning => "shadow_attempt_running",
            Self::PairedComparisonRunning => "paired_comparison_running",
            Self::ClientDisconnected => "client_disconnected",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

/// A single event on a request's lifecycle timeline.
#[derive(Clone, Debug)]
pub struct TimelineEvent {
    /// Wall-clock milliseconds since `UNIX_EPOCH`.
    pub at_ms: u64,
    /// Human-readable event label (typically a state name).
    pub event: String,
}

/// Compact summary of a live request, returned by the list endpoint.
#[derive(Clone, Debug)]
pub struct LiveRequestSummary {
    pub request_id: String,
    pub state: String,
    pub model: Option<String>,
    pub profile: Option<String>,
    pub downstream_mode: String,
    /// Elapsed milliseconds since entry creation.
    pub elapsed_ms: u64,
    pub queue_wait_ms: Option<u64>,
    pub first_token_latency_ms: Option<u64>,
    pub active_ladder_rung: Option<String>,
    pub active_attempt_index: Option<u32>,
    pub chunks_downstream: u64,
    pub bytes_downstream: u64,
    pub last_progress_at_ms: Option<u64>,
}

impl LiveRequestRegistry {
    /// Creates a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new in-flight request.
    ///
    /// If an entry already exists for `request_id` it is replaced.
    pub fn register(&self, request_id: &str, downstream_mode: &str) {
        let now = now_unix_millis();
        let entry = LiveRequestEntry {
            request_id: request_id.to_owned(),
            listener: None,
            profile: None,
            model: None,
            upstream_target: None,
            downstream_mode: downstream_mode.to_owned(),
            state: LiveRequestState::Accepted,
            created_at_ms: now,
            last_updated_at_ms: now,
            queue_wait_ms: None,
            upstream_elapsed_ms: None,
            first_token_latency_ms: None,
            active_ladder_rung: None,
            active_attempt_index: None,
            chunks_downstream: 0,
            bytes_downstream: 0,
            chunks_upstream: 0,
            bytes_upstream: 0,
            last_progress_at_ms: None,
            timeline: vec![TimelineEvent {
                at_ms: now,
                event: LiveRequestState::Accepted.as_str().to_owned(),
            }],
        };
        if let Ok(mut entries) = self.entries.write() {
            entries.insert(request_id.to_owned(), entry);
        }
    }

    /// Updates the lifecycle state of a request and appends a timeline event.
    ///
    /// No-op if the request is not registered.
    pub fn update_state(&self, request_id: &str, state: LiveRequestState) {
        if let Ok(mut entries) = self.entries.write()
            && let Some(entry) = entries.get_mut(request_id)
        {
            let now = now_unix_millis();
            entry.state = state;
            entry.last_updated_at_ms = now;
            entry.timeline.push(TimelineEvent {
                at_ms: now,
                event: state.as_str().to_owned(),
            });
            if matches!(
                state,
                LiveRequestState::FirstTokenSeen
                    | LiveRequestState::Streaming
                    | LiveRequestState::UpstreamHeadersReceived
            ) {
                entry.last_progress_at_ms = Some(now);
            }
        }
    }

    /// Records profile/model metadata for a request.
    pub fn update_profile(&self, request_id: &str, profile: Option<String>, model: Option<String>) {
        if let Ok(mut entries) = self.entries.write()
            && let Some(entry) = entries.get_mut(request_id)
        {
            let now = now_unix_millis();
            if let Some(profile) = profile {
                entry.profile = Some(profile);
            }
            if let Some(model) = model {
                entry.model = Some(model);
            }
            entry.last_updated_at_ms = now;
        }
    }

    /// Records the downstream listener name and redacted upstream target.
    pub fn update_target(
        &self,
        request_id: &str,
        listener: Option<String>,
        upstream_target: Option<String>,
    ) {
        if let Ok(mut entries) = self.entries.write()
            && let Some(entry) = entries.get_mut(request_id)
        {
            let now = now_unix_millis();
            if let Some(listener) = listener {
                entry.listener = Some(listener);
            }
            if let Some(upstream_target) = upstream_target {
                entry.upstream_target = Some(upstream_target);
            }
            entry.last_updated_at_ms = now;
        }
    }

    /// Records the queue wait time once a request is admitted.
    pub fn record_queue_wait(&self, request_id: &str, queue_wait_ms: u64) {
        if let Ok(mut entries) = self.entries.write()
            && let Some(entry) = entries.get_mut(request_id)
        {
            let now = now_unix_millis();
            entry.queue_wait_ms = Some(queue_wait_ms);
            entry.last_updated_at_ms = now;
        }
    }

    /// Records first-token latency and transitions to `FirstTokenSeen`.
    pub fn record_first_token(&self, request_id: &str, latency_ms: u64) {
        if let Ok(mut entries) = self.entries.write()
            && let Some(entry) = entries.get_mut(request_id)
        {
            let now = now_unix_millis();
            entry.first_token_latency_ms = Some(latency_ms);
            entry.state = LiveRequestState::FirstTokenSeen;
            entry.last_updated_at_ms = now;
            entry.last_progress_at_ms = Some(now);
            entry.timeline.push(TimelineEvent {
                at_ms: now,
                event: LiveRequestState::FirstTokenSeen.as_str().to_owned(),
            });
        }
    }

    /// Increments downstream chunk/byte counters.
    pub fn record_downstream_chunk(&self, request_id: &str, bytes: u64) {
        if let Ok(mut entries) = self.entries.write()
            && let Some(entry) = entries.get_mut(request_id)
        {
            let now = now_unix_millis();
            entry.chunks_downstream = entry.chunks_downstream.saturating_add(1);
            entry.bytes_downstream = entry.bytes_downstream.saturating_add(bytes);
            entry.last_progress_at_ms = Some(now);
            entry.last_updated_at_ms = now;
        }
    }

    /// Increments upstream chunk/byte counters.
    pub fn record_upstream_chunk(&self, request_id: &str, bytes: u64) {
        if let Ok(mut entries) = self.entries.write()
            && let Some(entry) = entries.get_mut(request_id)
        {
            let now = now_unix_millis();
            entry.chunks_upstream = entry.chunks_upstream.saturating_add(1);
            entry.bytes_upstream = entry.bytes_upstream.saturating_add(bytes);
            entry.last_progress_at_ms = Some(now);
            entry.last_updated_at_ms = now;
        }
    }

    /// Records a retry attempt with the active ladder rung and attempt index.
    pub fn record_retry(&self, request_id: &str, rung: &str, attempt_index: u32) {
        if let Ok(mut entries) = self.entries.write()
            && let Some(entry) = entries.get_mut(request_id)
        {
            let now = now_unix_millis();
            entry.active_ladder_rung = Some(rung.to_owned());
            entry.active_attempt_index = Some(attempt_index);
            entry.state = LiveRequestState::Retrying;
            entry.last_updated_at_ms = now;
            entry.timeline.push(TimelineEvent {
                at_ms: now,
                event: format!("retrying:{rung}:{attempt_index}"),
            });
        }
    }

    /// Marks the request completed and removes it from the registry.
    pub fn complete(&self, request_id: &str) {
        if let Ok(mut entries) = self.entries.write() {
            entries.remove(request_id);
        }
    }

    /// Marks the request failed and removes it from the registry.
    pub fn fail(&self, request_id: &str) {
        self.complete(request_id);
    }

    /// Returns a snapshot summary of every active entry.
    ///
    /// `elapsed_ms` is computed relative to the snapshot time.
    #[must_use]
    pub fn list_active(&self) -> Vec<LiveRequestSummary> {
        let now = now_unix_millis();
        let Ok(entries) = self.entries.read() else {
            return Vec::new();
        };
        entries
            .values()
            .map(|entry| LiveRequestSummary {
                request_id: entry.request_id.clone(),
                state: entry.state.as_str().to_owned(),
                model: entry.model.clone(),
                profile: entry.profile.clone(),
                downstream_mode: entry.downstream_mode.clone(),
                elapsed_ms: now.saturating_sub(entry.created_at_ms),
                queue_wait_ms: entry.queue_wait_ms,
                first_token_latency_ms: entry.first_token_latency_ms,
                active_ladder_rung: entry.active_ladder_rung.clone(),
                active_attempt_index: entry.active_attempt_index,
                chunks_downstream: entry.chunks_downstream,
                bytes_downstream: entry.bytes_downstream,
                last_progress_at_ms: entry.last_progress_at_ms,
            })
            .collect()
    }

    /// Returns a clone of a single entry, including its full timeline.
    #[must_use]
    pub fn get(&self, request_id: &str) -> Option<LiveRequestEntry> {
        let Ok(entries) = self.entries.read() else {
            return None;
        };
        entries.get(request_id).cloned()
    }

    /// Removes entries older than `max_age_ms` as a safety net for leaked
    /// registrations (e.g. when a `complete`/`fail` call is missed).
    pub fn prune_stale(&self, max_age_ms: u64) {
        let now = now_unix_millis();
        let cutoff = now.saturating_sub(max_age_ms);
        if let Ok(mut entries) = self.entries.write() {
            entries.retain(|_, entry| entry.created_at_ms >= cutoff);
        }
    }

    /// Returns the number of currently registered entries.
    #[must_use]
    pub fn len(&self) -> usize {
        let Ok(entries) = self.entries.read() else {
            return 0;
        };
        entries.len()
    }

    /// Returns `true` if no entries are currently registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn now_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_retrieve_entry() {
        let registry = LiveRequestRegistry::new();
        registry.register("req-1", "streaming");
        let entry = registry.get("req-1").expect("entry should be registered");
        assert_eq!(entry.request_id, "req-1");
        assert_eq!(entry.downstream_mode, "streaming");
        assert_eq!(entry.state, LiveRequestState::Accepted);
        assert_eq!(entry.chunks_downstream, 0);
        assert_eq!(entry.chunks_upstream, 0);
        // Timeline starts with the Accepted event.
        assert_eq!(entry.timeline.len(), 1);
        assert_eq!(entry.timeline[0].event, "accepted");
    }

    #[test]
    fn state_transitions_recorded_in_timeline() {
        let registry = LiveRequestRegistry::new();
        registry.register("req-1", "streaming");
        registry.update_state("req-1", LiveRequestState::Admitted);
        registry.update_state("req-1", LiveRequestState::UpstreamConnecting);
        registry.update_state("req-1", LiveRequestState::Streaming);
        let entry = registry.get("req-1").expect("entry should exist");
        assert_eq!(entry.state, LiveRequestState::Streaming);
        let events: Vec<&str> = entry.timeline.iter().map(|e| e.event.as_str()).collect();
        assert_eq!(
            events,
            vec!["accepted", "admitted", "upstream_connecting", "streaming",]
        );
        // Progress events update last_progress_at_ms.
        assert!(entry.last_progress_at_ms.is_some());
    }

    #[test]
    fn chunk_counters_increment_correctly() {
        let registry = LiveRequestRegistry::new();
        registry.register("req-1", "streaming");
        registry.record_downstream_chunk("req-1", 128);
        registry.record_downstream_chunk("req-1", 256);
        registry.record_upstream_chunk("req-1", 64);
        registry.record_upstream_chunk("req-1", 192);
        let entry = registry.get("req-1").expect("entry should exist");
        assert_eq!(entry.chunks_downstream, 2);
        assert_eq!(entry.bytes_downstream, 384);
        assert_eq!(entry.chunks_upstream, 2);
        assert_eq!(entry.bytes_upstream, 256);
        assert!(entry.last_progress_at_ms.is_some());
    }

    #[test]
    fn complete_removes_entry_from_registry() {
        let registry = LiveRequestRegistry::new();
        registry.register("req-1", "non_stream_json");
        assert!(registry.get("req-1").is_some());
        registry.complete("req-1");
        assert!(registry.get("req-1").is_none());
        assert!(registry.is_empty());
    }

    #[test]
    fn list_active_returns_summaries() {
        let registry = LiveRequestRegistry::new();
        registry.register("req-1", "streaming");
        registry.register("req-2", "non_stream_json");
        registry.update_state("req-1", LiveRequestState::Streaming);
        registry.record_downstream_chunk("req-1", 100);
        let summaries = registry.list_active();
        assert_eq!(summaries.len(), 2);
        let req1 = summaries
            .iter()
            .find(|s| s.request_id == "req-1")
            .expect("req-1 should be present");
        assert_eq!(req1.state, "streaming");
        assert_eq!(req1.downstream_mode, "streaming");
        assert_eq!(req1.chunks_downstream, 1);
        assert_eq!(req1.bytes_downstream, 100);
        assert!(req1.elapsed_ms < 5_000, "elapsed should be small");
    }

    #[test]
    fn prune_stale_removes_old_entries() {
        let registry = LiveRequestRegistry::new();
        registry.register("req-1", "streaming");
        // Manually backdate the entry to simulate an old leaked registration.
        {
            let Ok(mut entries) = registry.entries.write() else {
                panic!("lock poisoned");
            };
            let entry = entries.get_mut("req-1").expect("entry exists");
            // Set created_at_ms to 1 hour ago.
            entry.created_at_ms = entry.created_at_ms.saturating_sub(3_600_000);
        }
        registry.register("req-2", "streaming");
        // Prune anything older than 5 minutes (300_000 ms).
        registry.prune_stale(300_000);
        assert!(
            registry.get("req-1").is_none(),
            "old entry should be pruned"
        );
        assert!(
            registry.get("req-2").is_some(),
            "recent entry should remain"
        );
    }

    #[test]
    fn record_retry_sets_rung_and_attempt() {
        let registry = LiveRequestRegistry::new();
        registry.register("req-1", "streaming");
        registry.record_retry("req-1", "rung-1", 2);
        let entry = registry.get("req-1").expect("entry should exist");
        assert_eq!(entry.state, LiveRequestState::Retrying);
        assert_eq!(entry.active_ladder_rung.as_deref(), Some("rung-1"));
        assert_eq!(entry.active_attempt_index, Some(2));
        assert!(
            entry
                .timeline
                .iter()
                .any(|e| e.event == "retrying:rung-1:2")
        );
    }

    #[test]
    fn record_first_token_sets_latency_and_state() {
        let registry = LiveRequestRegistry::new();
        registry.register("req-1", "streaming");
        registry.record_first_token("req-1", 250);
        let entry = registry.get("req-1").expect("entry should exist");
        assert_eq!(entry.state, LiveRequestState::FirstTokenSeen);
        assert_eq!(entry.first_token_latency_ms, Some(250));
    }

    #[test]
    fn update_profile_sets_model_and_profile() {
        let registry = LiveRequestRegistry::new();
        registry.register("req-1", "streaming");
        registry.update_profile("req-1", Some("openai".to_owned()), Some("gpt-4".to_owned()));
        let entry = registry.get("req-1").expect("entry should exist");
        assert_eq!(entry.profile.as_deref(), Some("openai"));
        assert_eq!(entry.model.as_deref(), Some("gpt-4"));
    }

    #[test]
    fn update_state_for_unknown_request_is_noop() {
        let registry = LiveRequestRegistry::new();
        registry.update_state("unknown", LiveRequestState::Streaming);
        assert!(registry.is_empty());
    }
}
