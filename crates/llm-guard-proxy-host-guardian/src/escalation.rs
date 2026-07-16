//! Optional Tier-2 systemd escalation, deliberately disabled by default.

use crate::config::EscalationConfig;
use std::{
    process::Command,
    thread,
    time::{Duration, Instant},
};
use thiserror::Error;

const SYSTEMCTL_TIMEOUT: Duration = Duration::from_secs(10);

/// Tracks one low-memory episode so optional escalation cannot loop.
#[derive(Clone, Debug, Default)]
pub struct EscalationEpisode {
    armed_at: Option<Instant>,
    attempted: bool,
}

impl EscalationEpisode {
    /// Arms a fresh episode only when Tier 1 has successfully acted.
    pub fn arm(&mut self, config: &EscalationConfig, now: Instant) {
        if config.enabled() && self.armed_at.is_none() {
            self.armed_at = Some(now);
            self.attempted = false;
        }
    }

    /// Clears the episode after memory has rearmed.
    pub fn clear(&mut self) {
        self.armed_at = None;
        self.attempted = false;
    }

    /// Starts the configured service once after its grace period elapsed.
    ///
    /// # Errors
    ///
    /// Returns an error only after an enabled episode reaches its grace period
    /// and its validated `systemctl` invocation fails.
    pub fn maybe_run(
        &mut self,
        config: &EscalationConfig,
        now: Instant,
    ) -> Result<bool, EscalationError> {
        if !config.enabled() || self.attempted {
            return Ok(false);
        }
        let Some(armed_at) = self.armed_at else {
            return Ok(false);
        };
        if now.duration_since(armed_at) < config.grace_period() {
            return Ok(false);
        }
        self.attempted = true;
        let unit = config.unit().ok_or(EscalationError::MissingUnit)?;
        start_unit_with_timeout(unit, SYSTEMCTL_TIMEOUT)?;
        Ok(true)
    }
}

/// Escalation failures never change the Tier-1 latch state.
#[derive(Debug, Error)]
pub enum EscalationError {
    /// The enabled escalation block omitted its unit.
    #[error("escalation unit is missing")]
    MissingUnit,
    /// The unit is not a constrained systemd user-service name.
    #[error("invalid systemd service unit {0:?}")]
    InvalidUnit(String),
    /// `systemctl` could not be started.
    #[error("start systemctl: {0}")]
    Spawn(std::io::Error),
    /// `systemctl` could not be observed or reaped.
    #[error("wait for systemctl: {0}")]
    Wait(std::io::Error),
    /// The command exceeded its bounded wait.
    #[error("systemctl timed out after {0:?}")]
    Timeout(Duration),
    /// systemd rejected the requested unit.
    #[error("systemctl failed with {0}")]
    Failed(std::process::ExitStatus),
}

/// Validates a service name before passing it as a `systemctl` argument.
///
/// # Errors
///
/// Returns [`EscalationError::InvalidUnit`] when the name is not a constrained
/// `.service` identifier.
pub fn validate_unit_name(unit: &str) -> Result<(), EscalationError> {
    if !unit.ends_with(".service")
        || unit.len() <= ".service".len()
        || !unit
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b'@'))
    {
        return Err(EscalationError::InvalidUnit(unit.to_owned()));
    }
    Ok(())
}

/// Starts one validated user service with a bounded wait and reaping.
///
/// # Errors
///
/// Returns an error when the unit is invalid, `systemctl` cannot be run, it
/// exits unsuccessfully, or the timeout expires.
pub fn start_unit_with_timeout(unit: &str, timeout: Duration) -> Result<(), EscalationError> {
    validate_unit_name(unit)?;
    let mut child = Command::new("systemctl")
        .args(["--user", "start", "--", unit])
        .spawn()
        .map_err(EscalationError::Spawn)?;
    let started = Instant::now();
    loop {
        match child.try_wait().map_err(EscalationError::Wait)? {
            Some(status) if status.success() => return Ok(()),
            Some(status) => return Err(EscalationError::Failed(status)),
            None if started.elapsed() >= timeout => {
                child.kill().map_err(EscalationError::Wait)?;
                child.wait().map_err(EscalationError::Wait)?;
                return Err(EscalationError::Timeout(timeout));
            }
            None => thread::sleep(Duration::from_millis(20)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EscalationEpisode, validate_unit_name};
    use crate::config::EscalationConfig;
    use std::time::{Duration, Instant};

    fn config(enabled: bool, unit: Option<&str>) -> EscalationConfig {
        EscalationConfig {
            enabled,
            unit: unit.map(str::to_owned),
            grace_period: Duration::from_secs(60),
        }
    }

    #[test]
    fn accepts_safe_service_name() {
        assert!(validate_unit_name("vllm-aeon-27b.service").is_ok());
    }

    #[test]
    fn accepts_template_service_name() {
        assert!(validate_unit_name("worker@blue.service").is_ok());
    }

    #[test]
    fn rejects_missing_service_suffix() {
        assert!(validate_unit_name("vllm-aeon").is_err());
    }

    #[test]
    fn rejects_shell_metacharacters() {
        assert!(validate_unit_name("x.service;reboot").is_err());
    }

    #[test]
    fn rejects_path_separator() {
        assert!(validate_unit_name("dir/x.service").is_err());
    }

    #[test]
    fn disabled_config_never_arms_an_episode() {
        let mut episode = EscalationEpisode::default();
        let now = Instant::now();
        episode.arm(&config(false, None), now);
        assert!(
            !episode
                .maybe_run(&config(false, None), now + Duration::from_secs(61))
                .expect("disabled")
        );
    }

    #[test]
    fn enabled_episode_waits_for_grace_period() {
        let mut episode = EscalationEpisode::default();
        let now = Instant::now();
        let config = config(true, Some("test.service"));
        episode.arm(&config, now);
        assert!(
            !episode
                .maybe_run(&config, now + Duration::from_secs(59))
                .expect("grace")
        );
    }

    #[test]
    fn replacement_tier_one_actions_preserve_the_original_episode_deadline() {
        let mut episode = EscalationEpisode::default();
        let started = Instant::now();
        let config = config(true, Some("test.service"));
        episode.arm(&config, started);
        episode.arm(&config, started + Duration::from_secs(30));

        assert_eq!(episode.armed_at, Some(started));
        assert!(!episode.attempted);
    }

    #[test]
    fn clear_removes_pending_episode() {
        let mut episode = EscalationEpisode::default();
        let now = Instant::now();
        let config = config(true, Some("test.service"));
        episode.arm(&config, now);
        episode.clear();
        assert!(
            !episode
                .maybe_run(&config, now + Duration::from_secs(61))
                .expect("cleared")
        );
    }
}
