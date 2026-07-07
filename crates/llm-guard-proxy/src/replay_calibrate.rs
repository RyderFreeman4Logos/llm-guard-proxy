//! `calibrate` subcommand: offline replay calibration (issue #112).
//!
//! Reads a JSONL archive of recorded [`ReplayRecord`]s, replays each one
//! through the [`ReplayRunner`] detector pipeline, and aggregates the results
//! into a [`CalibrationResult`].
//!
//! All replay types live in `llm-guard-proxy-core`; this module owns the CLI
//! plumbing (archive I/O and result serialization).

#![allow(dead_code)]

use std::fmt;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use llm_guard_proxy_core::replay::{CalibrationResult, ReplayConfig, ReplayRecord, ReplayRunner};
use thiserror::Error;

/// Errors produced by [`CalibrateCommand::run`].
#[derive(Debug, Error)]
pub enum ReplayError {
    /// The archive file could not be opened.
    #[error("failed to open archive {path}: {source}")]
    OpenArchive {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// The archive file could not be read.
    #[error("failed to read archive {path}: {source}")]
    ReadArchive {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// A JSONL line could not be parsed as a [`ReplayRecord`].
    #[error("failed to parse JSONL line {line} in {path}: {source}")]
    ParseRecord {
        path: PathBuf,
        line: usize,
        #[source]
        source: serde_json::Error,
    },
    /// The archive contained no records.
    #[error("archive {path} contained no records")]
    EmptyArchive { path: PathBuf },
    /// Writing the output file failed.
    #[error("failed to write output {path}: {source}")]
    WriteOutput {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Serializing the result to JSON failed.
    #[error("failed to serialize calibration result: {0}")]
    SerializeResult(serde_json::Error),
}

/// The `calibrate` subcommand: replay a JSONL archive and collect metrics.
pub struct CalibrateCommand {
    /// Path to the JSONL archive (one [`ReplayRecord`] per line).
    pub archive_path: PathBuf,
    /// Optional path to write the JSON [`CalibrationResult`].
    pub output_path: Option<PathBuf>,
    /// Calibration targets and token budgets.
    pub config: ReplayConfig,
}

impl CalibrateCommand {
    /// Runs the calibration: reads the archive, replays each record, and
    /// aggregates results. When `output_path` is set, the result is written as
    /// pretty-printed JSON.
    pub fn run(&self) -> Result<CalibrationResult, ReplayError> {
        let records = self.read_archive()?;
        if records.is_empty() {
            return Err(ReplayError::EmptyArchive {
                path: self.archive_path.clone(),
            });
        }

        let runner = ReplayRunner::new(self.config.clone());
        let results: Vec<_> = records
            .iter()
            .map(|record| runner.run_record(record))
            .collect();
        let calibration = CalibrationResult::from_results(&results, &records);

        if let Some(output_path) = &self.output_path {
            let json =
                serde_json::to_string_pretty(&calibration).map_err(ReplayError::SerializeResult)?;
            fs::write(output_path, json).map_err(|source| ReplayError::WriteOutput {
                path: output_path.clone(),
                source,
            })?;
        }

        Ok(calibration)
    }

    /// Reads and parses the JSONL archive into a vector of [`ReplayRecord`].
    fn read_archive(&self) -> Result<Vec<ReplayRecord>, ReplayError> {
        let file =
            fs::File::open(&self.archive_path).map_err(|source| ReplayError::OpenArchive {
                path: self.archive_path.clone(),
                source,
            })?;
        let reader = BufReader::new(file);
        let mut records = Vec::new();
        for (index, line_result) in reader.lines().enumerate() {
            let line = line_result.map_err(|source| ReplayError::ReadArchive {
                path: self.archive_path.clone(),
                source,
            })?;
            if line.trim().is_empty() {
                continue;
            }
            let record: ReplayRecord =
                serde_json::from_str(&line).map_err(|source| ReplayError::ParseRecord {
                    path: self.archive_path.clone(),
                    line: index + 1,
                    source,
                })?;
            records.push(record);
        }
        Ok(records)
    }
}

impl fmt::Display for CalibrateCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "calibrate {}", self.archive_path.display())?;
        if let Some(output) = &self.output_path {
            write!(f, " --output {}", output.display())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibrate_command_can_be_constructed() {
        let command = CalibrateCommand {
            archive_path: PathBuf::from("archive.jsonl"),
            output_path: Some(PathBuf::from("out.json")),
            config: ReplayConfig::default(),
        };
        assert_eq!(command.archive_path, PathBuf::from("archive.jsonl"));
        assert_eq!(
            command.output_path.as_deref(),
            Some(std::path::Path::new("out.json"))
        );
        assert!((command.config.hard_loop_recall_target - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn replay_error_open_archive_displays() {
        let error = ReplayError::OpenArchive {
            path: PathBuf::from("/missing/archive.jsonl"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "no such file"),
        };
        let message = error.to_string();
        assert!(message.contains("failed to open archive"), "{message}");
        assert!(message.contains("/missing/archive.jsonl"), "{message}");
        assert!(message.contains("no such file"), "{message}");
    }

    #[test]
    fn replay_error_empty_archive_displays() {
        let error = ReplayError::EmptyArchive {
            path: PathBuf::from("empty.jsonl"),
        };
        let message = error.to_string();
        assert!(message.contains("no records"), "{message}");
        assert!(message.contains("empty.jsonl"), "{message}");
    }

    #[test]
    fn replay_error_parse_record_displays() {
        let error = ReplayError::ParseRecord {
            path: PathBuf::from("bad.jsonl"),
            line: 7,
            source: serde_json::from_str::<serde_json::Value>("not json").unwrap_err(),
        };
        let message = error.to_string();
        assert!(message.contains("line 7"), "{message}");
        assert!(message.contains("bad.jsonl"), "{message}");
    }

    #[test]
    fn calibrate_command_display() {
        let command = CalibrateCommand {
            archive_path: PathBuf::from("archive.jsonl"),
            output_path: None,
            config: ReplayConfig::default(),
        };
        assert_eq!(command.to_string(), "calibrate archive.jsonl");

        let command = CalibrateCommand {
            archive_path: PathBuf::from("archive.jsonl"),
            output_path: Some(PathBuf::from("out.json")),
            config: ReplayConfig::default(),
        };
        assert_eq!(
            command.to_string(),
            "calibrate archive.jsonl --output out.json"
        );
    }
}
