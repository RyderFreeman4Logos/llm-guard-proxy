//! Pre-commit local recovery for generic OpenAI-compatible forwarding.
//!
//! Generic fallbacks use the same recovery coordinator as shielded requests,
//! but may replay only before any downstream response is returned.

use super::{
    AttemptId, AttemptRecord, AttemptStatus, BTreeMap, Duration, FailedAttemptRecordInput,
    GenericForwardContext, LocalRecoveryCause, LocalRecoveryPolicy, ProxyError, RawPayloads,
    ReqwestFailureKind, SentUpstreamResponse, StatusCode, TokenUsage, failed_attempt_record,
    precommit_recovery, prepare_generic_attempt_request, response_metadata, retry_after,
    send_generic_upstream_attempt, unix_time_millis, upstream_mode_from_headers,
};

pub(super) async fn complete(
    context: &GenericForwardContext<'_>,
    first: Result<SentUpstreamResponse, ProxyError>,
) -> Result<SentUpstreamResponse, ProxyError> {
    let mut current = recover_and_replay(context, first).await?;
    let rate_limit_retry_budget = retry_after::RetryBudget::default();
    loop {
        let response = &current.response;
        if response.status() != StatusCode::TOO_MANY_REQUESTS || !context.config.retry.enabled {
            if context.uri.path() != "/v1/chat/completions"
                || !response.status().is_success()
                || context.request_deadline.is_exhausted()
            {
                return Ok(current);
            }
            match current.response.hold_first_nonempty_body_chunk().await {
                Ok(()) => return Ok(current),
                Err(failure) => {
                    let error = precommit_body_failure(context, current, failure);
                    current = recover_and_replay(context, Err(error)).await?;
                    continue;
                }
            }
        }
        let Some(remaining) = context.request_deadline.remaining() else {
            return Ok(current);
        };
        let Some(delay) = rate_limit_retry_budget.claim_delay(
            response.headers(),
            Duration::from_secs(context.config.retry.max_retry_after_secs),
            current.attempt_number,
        ) else {
            return Ok(current);
        };
        if !retry_after::wait_before_retry(delay, remaining, context.state.shutdown.subscribe())
            .await
        {
            return Ok(current);
        }

        let (next_attempt_number, mut completed_attempts) =
            status_retry_attempt_history(context, &current, BTreeMap::new());
        let retried = match send_generic_upstream_attempt(context, next_attempt_number).await {
            Ok(mut retried) => {
                completed_attempts.append(&mut retried.completed_attempt_records);
                retried.completed_attempt_records = completed_attempts;
                Ok(retried)
            }
            Err(error) => Err(error.with_completed_attempt_records(completed_attempts)),
        };
        current = recover_and_replay(context, retried).await?;
    }
}

fn precommit_body_failure(
    context: &GenericForwardContext<'_>,
    mut sent: SentUpstreamResponse,
    failure: ReqwestFailureKind,
) -> ProxyError {
    if let Some(lease) = sent.stuck_watchdog_attempt.take() {
        lease.end();
    }
    let error = ProxyError::UpstreamTransport {
        failure,
        observability: None,
    };
    let error_reason = error.to_string();
    let mut attempt_record = failed_attempt_record(FailedAttemptRecordInput {
        attempt_id: sent.attempt_id,
        attempt_number: sent.attempt_number,
        request_id: context.request_id.clone(),
        started_at_unix_ms: sent.attempt_started_at_unix_ms,
        finished_at_unix_ms: unix_time_millis(),
        error_type: error.error_type(),
        error_reason: &error_reason,
        request_metadata: sent.attempt_request_metadata,
        extra_response_metadata: BTreeMap::from([
            (String::from("precommit_body_failure"), String::from("true")),
            (
                String::from("upstream_response_received"),
                String::from("true"),
            ),
        ]),
    });
    attempt_record.http_status = Some(sent.response.status().as_u16());
    attempt_record.upstream_mode = upstream_mode_from_headers(sent.response.headers());
    error
        .with_observability(context.request_metadata.clone(), attempt_record)
        .with_completed_attempt_records(sent.completed_attempt_records)
}

async fn recover_and_replay(
    context: &GenericForwardContext<'_>,
    first: Result<SentUpstreamResponse, ProxyError>,
) -> Result<SentUpstreamResponse, ProxyError> {
    let cause = match &first {
        Ok(sent) if matches!(sent.response.status().as_u16(), 502..=504) => {
            Some(LocalRecoveryCause::TransientStatus)
        }
        Err(ProxyError::UpstreamTransport { failure, .. }) if failure.is_transient() => {
            Some(LocalRecoveryCause::TransientTransport)
        }
        _ => None,
    };
    let Some(cause) = cause else {
        return first;
    };
    if context.request_deadline.is_exhausted() {
        return first;
    }

    let policy = LocalRecoveryPolicy::from_config(&context.upstream_profile.local_recovery);
    let coordinator = context
        .state
        .local_recovery
        .coordinator_for(&context.upstream_profile.name);
    let gate = precommit_recovery::gate(
        precommit_recovery::Context {
            policy: &policy,
            coordinator: &coordinator,
            client: context.state.client.clone(),
            base_url: context.upstream_profile.primary_base_url(),
            profile_name: &context.upstream_profile.name,
            attempts: &context.local_recovery_attempts,
            downstream_commit_signal: None,
            downstream_drop_signal: None,
            request_deadline: context.request_deadline,
            episode_timeout: context.upstream_profile.restart_queue.enabled.then(|| {
                Duration::from_secs(context.upstream_profile.restart_queue.restart_timeout_secs)
            }),
        },
        false,
        cause,
    )
    .await;
    if !gate.permits_replay || context.request_deadline.is_exhausted() {
        return match first {
            Ok(mut sent) => {
                sent.attempt_request_metadata.extend(gate.metadata);
                Ok(sent)
            }
            Err(error) => Err(error.with_request_metadata(gate.metadata)),
        };
    }

    let (next_attempt_number, mut completed_attempts) = match &first {
        Ok(sent) => status_retry_attempt_history(context, sent, gate.metadata),
        Err(error) => transport_retry_attempt_history(context, error, gate.metadata),
    };
    match send_generic_upstream_attempt(context, next_attempt_number).await {
        Ok(mut replayed) => {
            completed_attempts.append(&mut replayed.completed_attempt_records);
            replayed.completed_attempt_records = completed_attempts;
            Ok(replayed)
        }
        Err(error) => Err(error.with_completed_attempt_records(completed_attempts)),
    }
}

fn status_retry_attempt_history(
    context: &GenericForwardContext<'_>,
    sent: &SentUpstreamResponse,
    recovery_metadata: BTreeMap<String, String>,
) -> (u32, Vec<AttemptRecord>) {
    let local_recovery = !recovery_metadata.is_empty();
    let retry_reason = if local_recovery {
        "local_recovery"
    } else {
        "transient_upstream_status"
    };
    let mut attempts = sent.completed_attempt_records.clone();
    let finished_at = unix_time_millis();
    let mut response_metadata = response_metadata(
        sent.response.status(),
        sent.response.headers(),
        0,
        finished_at.saturating_sub(sent.attempt_started_at_unix_ms),
    );
    response_metadata.insert(String::from("attempt_outcome"), String::from("retried"));
    response_metadata.insert(
        String::from("endpoint_disposition"),
        String::from("retryable_failure"),
    );
    let mut record = AttemptRecord {
        attempt_id: sent.attempt_id.clone(),
        request_id: context.request_id.clone(),
        attempt_number: sent.attempt_number,
        started_at_unix_ms: sent.attempt_started_at_unix_ms,
        finished_at_unix_ms: Some(finished_at),
        upstream_mode: upstream_mode_from_headers(sent.response.headers()),
        status: AttemptStatus::Retried,
        http_status: Some(sent.response.status().as_u16()),
        error_reason: Some(if local_recovery {
            format!(
                "upstream HTTP {} selected for local recovery",
                sent.response.status()
            )
        } else {
            String::from("bounded upstream rate limit retry")
        }),
        retry_reason: Some(String::from(retry_reason)),
        abort_reason: None,
        token_usage: TokenUsage::default(),
        request_metadata: sent.attempt_request_metadata.clone(),
        response_metadata,
        raw_payloads: RawPayloads::default(),
    };
    record.response_metadata.extend(recovery_metadata);
    attempts.push(record);
    (sent.attempt_number.saturating_add(1), attempts)
}

fn transport_retry_attempt_history(
    context: &GenericForwardContext<'_>,
    error: &ProxyError,
    recovery_metadata: BTreeMap<String, String>,
) -> (u32, Vec<AttemptRecord>) {
    let mut attempts = error.attempt_records();
    let attempt_number = attempts
        .last()
        .map_or(1, |attempt| attempt.attempt_number.max(1));
    if let Some(terminal) = attempts.last_mut() {
        terminal.status = AttemptStatus::Retried;
        terminal.retry_reason = Some(String::from("local_recovery"));
        terminal.response_metadata.extend(recovery_metadata);
    } else {
        let (_, request_metadata) = prepare_generic_attempt_request(context, attempt_number);
        let mut record = failed_attempt_record(FailedAttemptRecordInput {
            attempt_id: AttemptId::for_request(context.request_id, attempt_number),
            attempt_number,
            request_id: context.request_id.clone(),
            started_at_unix_ms: context.started_at_unix_ms,
            finished_at_unix_ms: unix_time_millis(),
            error_type: error.error_type(),
            error_reason: &error.to_string(),
            request_metadata,
            extra_response_metadata: recovery_metadata,
        });
        record.status = AttemptStatus::Retried;
        record.retry_reason = Some(String::from("local_recovery"));
        attempts.push(record);
    }
    (attempt_number.saturating_add(1), attempts)
}
