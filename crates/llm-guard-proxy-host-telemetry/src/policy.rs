//! Stateless-threshold and bounded-repeat policy for swap alerts.

use crate::{PolicyDecision, PressureReason, SwapGuardConfig, TelemetryEvent, TelemetryState};

/// Observer-only policy state; it never owns a recovery action.
#[derive(Debug)]
pub struct SwapGuard {
    config: SwapGuardConfig,
    state: TelemetryState,
    last_alert_at_unix_ms: Option<u64>,
}

impl SwapGuard {
    /// Creates a swap guard with an initially healthy state.
    #[must_use]
    pub fn new(config: SwapGuardConfig) -> Self {
        Self {
            config,
            state: TelemetryState::Healthy,
            last_alert_at_unix_ms: None,
        }
    }

    /// Evaluates one sample and emits only state changes or bounded alert repeats.
    #[must_use]
    pub fn observe(
        &mut self,
        sampled_at_unix_ms: u64,
        mem_available_kib: u64,
        swap_used_kib: u64,
    ) -> PolicyDecision {
        let state = self.state_for(mem_available_kib, swap_used_kib);
        let event = match state {
            TelemetryState::Alert(reason) => {
                let repeat_due = self.last_alert_at_unix_ms.is_none_or(|previous| {
                    sampled_at_unix_ms.saturating_sub(previous)
                        >= self
                            .config
                            .alert_repeat()
                            .as_millis()
                            .try_into()
                            .unwrap_or(u64::MAX)
                });
                if self.state != state || repeat_due {
                    self.last_alert_at_unix_ms = Some(sampled_at_unix_ms);
                    Some(TelemetryEvent::Alert(reason))
                } else {
                    None
                }
            }
            TelemetryState::SwapWarning if self.state != state => Some(TelemetryEvent::SwapWarning),
            TelemetryState::Healthy if self.state != TelemetryState::Healthy => {
                self.last_alert_at_unix_ms = None;
                Some(TelemetryEvent::Cleared)
            }
            _ => None,
        };
        self.state = state;
        PolicyDecision { state, event }
    }

    fn state_for(&self, mem_available_kib: u64, swap_used_kib: u64) -> TelemetryState {
        let memory_low = mem_available_kib < self.config.alert_mem_available_kib();
        let swap_high = swap_used_kib > self.config.alert_swap_kib();
        if memory_low || swap_high {
            return TelemetryState::Alert(match (memory_low, swap_high) {
                (true, true) => PressureReason::MemoryAndSwap,
                (true, false) => PressureReason::MemoryAvailable,
                (false, true) => PressureReason::Swap,
                (false, false) => unreachable!("alert requires a pressure condition"),
            });
        }
        if swap_used_kib >= self.config.warn_swap_kib() {
            TelemetryState::SwapWarning
        } else {
            TelemetryState::Healthy
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SwapGuard;
    use crate::{PressureReason, SwapGuardConfig, TelemetryEvent, TelemetryState};
    use std::time::Duration;

    fn guard() -> SwapGuard {
        SwapGuard::new(
            SwapGuardConfig::new(2, 4, 1, Duration::from_secs(60))
                .expect("test configuration is valid"),
        )
    }

    #[test]
    fn emits_bounded_alert_evidence_and_a_clear_transition() {
        let mut guard = guard();
        let alert = guard.observe(0, 512, 0);
        assert_eq!(
            alert.event,
            Some(TelemetryEvent::Alert(PressureReason::MemoryAvailable))
        );
        assert!(alert.event.is_some_and(TelemetryEvent::collects_evidence));
        assert_eq!(guard.observe(1_000, 512, 0).event, None);
        assert_eq!(
            guard.observe(60_000, 512, 0).event,
            Some(TelemetryEvent::Alert(PressureReason::MemoryAvailable))
        );
        assert_eq!(
            guard.observe(61_000, 2_048, 0).event,
            Some(TelemetryEvent::Cleared)
        );
    }

    #[test]
    fn warning_does_not_collect_alert_evidence() {
        let mut guard = guard();
        let warning = guard.observe(0, 2_048, 2 * 1024);
        assert_eq!(warning.state, TelemetryState::SwapWarning);
        assert_eq!(warning.event, Some(TelemetryEvent::SwapWarning));
    }
}
