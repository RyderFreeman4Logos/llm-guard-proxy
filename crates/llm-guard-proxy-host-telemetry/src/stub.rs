//! Non-Linux no-op implementation of host telemetry.

use crate::{TelemetryConfig, TelemetryError, TelemetryIteration};
use std::path::PathBuf;

/// No-op host telemetry runtime for targets without Linux procfs.
#[derive(Debug)]
pub struct HostTelemetry;

impl HostTelemetry {
    /// Validates the configuration and creates a no-op sampler.
    ///
    /// # Errors
    ///
    /// Returns an error when the configuration file cannot be read or validated.
    pub fn open(config_path: impl Into<PathBuf>) -> Result<Self, TelemetryError> {
        let _config = TelemetryConfig::load(&config_path.into())?;
        Ok(Self)
    }

    /// Returns an explicit no-op iteration.
    pub const fn tick(&mut self) -> Result<TelemetryIteration, TelemetryError> {
        Ok(TelemetryIteration::Unsupported)
    }

    /// Waits for shutdown without reading host state or creating files.
    ///
    /// # Errors
    ///
    /// This stub returns successfully after shutdown.
    pub async fn run_until<F>(&mut self, shutdown: F) -> Result<(), TelemetryError>
    where
        F: std::future::Future<Output = ()>,
    {
        shutdown.await;
        Ok(())
    }
}
