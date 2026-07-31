//! Strict handling for rate-limit retry delays and downstream headers.

use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};

use super::{Duration, HeaderMap, HeaderValue, RETRY_AFTER, ShutdownSubscription};

#[derive(Clone, Debug, Default)]
pub(super) struct RetryBudget {
    repeated_physical_attempt: Arc<AtomicU32>,
}

impl RetryBudget {
    pub(super) fn claim_delay(
        &self,
        headers: &HeaderMap,
        maximum: Duration,
        physical_attempt_number: u32,
    ) -> Option<Duration> {
        let delay = bounded_delay(headers, maximum)?;
        let repeated_physical_attempt = physical_attempt_number.saturating_add(1);
        self.repeated_physical_attempt
            .compare_exchange(
                0,
                repeated_physical_attempt,
                Ordering::Relaxed,
                Ordering::Relaxed,
            )
            .is_ok()
            .then_some(delay)
    }

    /// Maps a physical telemetry attempt to the ordinary non-429 ladder rung.
    ///
    /// The single accepted rate-limit retry repeats the current ordinary rung
    /// and therefore cannot spend an ordinary retry slot.
    pub(super) fn ordinary_attempt_number(&self, physical_attempt_number: u32) -> u32 {
        let repeated_physical_attempt = self.repeated_physical_attempt.load(Ordering::Relaxed);
        let rate_limit_offset = u32::from(
            repeated_physical_attempt != 0 && physical_attempt_number >= repeated_physical_attempt,
        );
        physical_attempt_number
            .saturating_sub(rate_limit_offset)
            .max(1)
    }
}

pub(super) fn bounded_delay(headers: &HeaderMap, maximum: Duration) -> Option<Duration> {
    let seconds = headers
        .get(RETRY_AFTER)?
        .to_str()
        .ok()?
        .parse::<u64>()
        .ok()?;
    let delay = Duration::from_secs(seconds);
    (seconds > 0 && delay <= maximum).then_some(delay)
}

pub(super) fn sanitize(headers: &mut HeaderMap, maximum: Duration) {
    let sanitized = bounded_delay(headers, maximum)
        .and_then(|delay| HeaderValue::from_str(&delay.as_secs().to_string()).ok());
    headers.remove(RETRY_AFTER);
    if let Some(value) = sanitized {
        headers.insert(RETRY_AFTER, value);
    }
}

pub(super) async fn wait_before_retry(
    delay: Duration,
    remaining: Duration,
    mut shutdown: ShutdownSubscription,
) -> bool {
    if delay >= remaining {
        return false;
    }
    tokio::select! {
        biased;
        () = shutdown.cancelled() => false,
        () = tokio::time::sleep(delay) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_positive_bounded_delta_seconds_are_usable_and_forwarded() {
        for (value, expected) in [
            (None, None),
            (Some("0"), None),
            (Some("-1"), None),
            (Some("invalid"), None),
            (Some("2"), None),
            (Some("18446744073709551616"), None),
            (Some("1"), Some(Duration::from_secs(1))),
        ] {
            let mut headers = HeaderMap::new();
            if let Some(value) = value {
                headers.insert(
                    RETRY_AFTER,
                    HeaderValue::from_str(value).expect("fixture must be a valid header value"),
                );
            }

            assert_eq!(bounded_delay(&headers, Duration::from_secs(1)), expected);
            sanitize(&mut headers, Duration::from_secs(1));
            assert_eq!(
                headers
                    .get(RETRY_AFTER)
                    .and_then(|value| value.to_str().ok()),
                expected.map(|_| "1")
            );
        }
    }

    #[test]
    fn budget_is_consumed_only_by_the_first_usable_delay() {
        let budget = RetryBudget::default();
        let mut invalid = HeaderMap::new();
        invalid.insert(RETRY_AFTER, HeaderValue::from_static("invalid"));
        let mut valid = HeaderMap::new();
        valid.insert(RETRY_AFTER, HeaderValue::from_static("1"));

        assert_eq!(
            budget.claim_delay(&invalid, Duration::from_secs(1), 1),
            None
        );
        assert_eq!(
            budget.claim_delay(&valid, Duration::from_secs(1), 1),
            Some(Duration::from_secs(1))
        );
        assert_eq!(budget.claim_delay(&valid, Duration::from_secs(1), 2), None);
    }

    #[test]
    fn rate_limit_retry_repeats_the_current_ordinary_attempt() {
        let budget = RetryBudget::default();
        assert_eq!(budget.ordinary_attempt_number(1), 1);

        let mut valid = HeaderMap::new();
        valid.insert(RETRY_AFTER, HeaderValue::from_static("1"));
        assert_eq!(
            budget.claim_delay(&valid, Duration::from_secs(1), 1),
            Some(Duration::from_secs(1))
        );
        assert_eq!(budget.ordinary_attempt_number(2), 1);
        assert_eq!(budget.ordinary_attempt_number(3), 2);
        assert_eq!(budget.ordinary_attempt_number(5), 4);
    }

    #[test]
    fn delayed_rate_limit_retry_does_not_relabel_prior_ordinary_attempts() {
        let budget = RetryBudget::default();
        let mut valid = HeaderMap::new();
        valid.insert(RETRY_AFTER, HeaderValue::from_static("1"));

        assert_eq!(
            budget.claim_delay(&valid, Duration::from_secs(1), 2),
            Some(Duration::from_secs(1))
        );
        assert_eq!(budget.ordinary_attempt_number(1), 1);
        assert_eq!(budget.ordinary_attempt_number(2), 2);
        assert_eq!(budget.ordinary_attempt_number(3), 2);
        assert_eq!(budget.ordinary_attempt_number(4), 3);
    }
}
