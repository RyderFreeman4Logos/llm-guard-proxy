//! Shared replay-safe local recovery gate.
//!
//! Both shielded and generic forwarding enter through this gate before any
//! downstream byte. It owns the per-request recovery budget and the mandatory
//! post-await commit/drop rechecks.

use super::{
    Arc, AtomicU64, Client, DownstreamCommitSignal, DownstreamDropSignal, Duration,
    LocalRecoveryCause, LocalRecoveryGate, LocalRecoveryPolicy, LocalRecoveryRunOptions, Ordering,
    RequestDeadline, UpstreamStallRecoveryCoordinator, applied_local_recovery_gate,
    local_recovery_completed_ready, local_recovery_downstream_commit_observed,
    local_recovery_metadata, local_recovery_permits_retry,
    run_local_recovery_for_profile_observing, skipped_local_recovery_gate,
    unapplied_local_recovery_gate,
};

pub(super) struct Context<'request> {
    pub(super) policy: &'request LocalRecoveryPolicy,
    pub(super) coordinator: &'request Arc<UpstreamStallRecoveryCoordinator>,
    pub(super) client: Client,
    pub(super) base_url: &'request str,
    pub(super) profile_name: &'request str,
    pub(super) attempts: &'request AtomicU64,
    pub(super) downstream_commit_signal: Option<&'request DownstreamCommitSignal>,
    pub(super) downstream_drop_signal: Option<&'request DownstreamDropSignal>,
    pub(super) request_deadline: RequestDeadline,
    pub(super) episode_timeout: Option<Duration>,
}

pub(super) async fn gate(
    context: Context<'_>,
    can_retry: bool,
    cause: LocalRecoveryCause,
) -> LocalRecoveryGate {
    if cause == LocalRecoveryCause::RequestDeadline && !context.policy.trigger_on_request_deadline {
        return unapplied_local_recovery_gate();
    }
    // Transient upstream failures must still enter local recovery after the
    // retry ladder is exhausted. Otherwise a ready recovered upstream is
    // stranded behind a permanent client 502 (#233).
    let allow_without_ladder = matches!(
        cause,
        LocalRecoveryCause::RequestDeadline
            | LocalRecoveryCause::TransientTransport
            | LocalRecoveryCause::TransientStatus
            | LocalRecoveryCause::UpstreamStall
            | LocalRecoveryCause::StuckWatchdog
    );
    if (!can_retry && !allow_without_ladder)
        || (!context.policy.enabled && context.policy.restart_command.is_empty())
    {
        return unapplied_local_recovery_gate();
    }
    if !context.policy.enabled || context.policy.restart_command.is_empty() {
        return unapplied_local_recovery_gate();
    }

    let mut metadata = local_recovery_metadata(context.policy, context.profile_name, cause);
    if context
        .downstream_drop_signal
        .is_some_and(DownstreamDropSignal::is_dropped)
    {
        return skipped_local_recovery_gate(metadata, "skipped_downstream_dropped", false);
    }
    if local_recovery_downstream_commit_observed(context.downstream_commit_signal, &mut metadata) {
        return skipped_local_recovery_gate(metadata, "skipped_downstream_committed", false);
    }
    let Some(remaining_request_budget) = context
        .request_deadline
        .remaining()
        .filter(|remaining| !remaining.is_zero())
    else {
        return skipped_local_recovery_gate(metadata, "skipped_request_deadline_exhausted", false);
    };

    let previous_attempts = context.attempts.fetch_add(1, Ordering::SeqCst);
    if previous_attempts >= u64::from(context.policy.max_attempts_per_request) {
        metadata.insert(
            String::from("local_recovery_status"),
            String::from("skipped_request_budget_exhausted"),
        );
        metadata.insert(
            String::from("local_recovery_permits_retry"),
            String::from("false"),
        );
        metadata.insert(
            String::from("local_recovery_request_attempts_used"),
            previous_attempts.to_string(),
        );
        return applied_local_recovery_gate(metadata, false, false);
    }
    metadata.insert(
        String::from("local_recovery_request_attempts_used"),
        previous_attempts.saturating_add(1).to_string(),
    );

    let recovery_metadata = run_local_recovery_for_profile_observing(
        context.policy,
        context.coordinator,
        context.client,
        context.base_url.to_owned(),
        cause,
        LocalRecoveryRunOptions {
            episode_timeout: context.episode_timeout,
            caller_timeout: Some(remaining_request_budget),
            recovery_episode_observer: None,
            downstream_commit_signal: context.downstream_commit_signal.cloned(),
        },
    )
    .await;
    metadata.extend(recovery_metadata);

    if context
        .downstream_drop_signal
        .is_some_and(DownstreamDropSignal::is_dropped)
    {
        return skipped_local_recovery_gate(metadata, "skipped_downstream_dropped", false);
    }
    if local_recovery_downstream_commit_observed(context.downstream_commit_signal, &mut metadata) {
        return skipped_local_recovery_gate(metadata, "skipped_downstream_committed", false);
    }

    let permits_retry = local_recovery_permits_retry(&metadata);
    let permits_replay =
        local_recovery_completed_ready(&metadata) && !context.request_deadline.is_exhausted();
    metadata.insert(
        String::from("local_recovery_permits_retry"),
        permits_retry.to_string(),
    );
    applied_local_recovery_gate(metadata, permits_retry, permits_replay)
}
