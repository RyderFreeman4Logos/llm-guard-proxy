use std::{
    error::Error,
    fmt::{self, Display, Formatter},
};

/// Validation failure with the field name and clear requirement.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationError {
    field: &'static str,
    message: String,
}

impl ValidationError {
    /// Creates a validation failure for a configuration field.
    ///
    /// Operational adapters use this constructor to report validation that
    /// depends on external state while preserving the common error contract.
    pub fn new(field: &'static str, message: impl Into<String>) -> Self {
        Self {
            field,
            message: message.into(),
        }
    }

    /// Returns the invalid config field.
    #[must_use]
    pub const fn field(&self) -> &'static str {
        self.field
    }

    /// Returns the validation requirement.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl Display for ValidationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.field, self.message)
    }
}

impl Error for ValidationError {}

/// TOML parsing failure for the supported config subset.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConfigParseError {
    line: usize,
    message: String,
}

impl ConfigParseError {
    pub(crate) fn new(line: usize, message: impl Into<String>) -> Self {
        Self {
            line,
            message: message.into(),
        }
    }

    /// Returns the one-based input line number.
    #[must_use]
    pub const fn line(&self) -> usize {
        self.line
    }

    /// Returns the parse failure reason.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl Display for ConfigParseError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "line {}: {}", self.line, self.message)
    }
}

impl Error for ConfigParseError {}
