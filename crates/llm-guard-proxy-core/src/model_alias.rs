use std::{
    collections::HashMap,
    error::Error,
    fmt::{self, Display, Formatter},
};

/// Default timeout for workflow aliases, in milliseconds.
pub const DEFAULT_WORKFLOW_TIMEOUT_MS: u64 = 120_000;

/// Maximum timeout for workflow aliases, in milliseconds.
pub const MAX_WORKFLOW_TIMEOUT_MS: u64 = 600_000;

/// Kind of model alias.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AliasKind {
    /// Alias routes to an upstream profile.
    #[default]
    Upstream,
    /// Alias invokes a Guard Workflow Protocol workflow.
    Workflow,
}

impl AliasKind {
    /// Returns the TOML-compatible alias kind label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Upstream => "upstream",
            Self::Workflow => "workflow",
        }
    }
}

/// A configured model alias entry.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelAliasConfig {
    /// Virtual model ID exposed to clients.
    pub id: String,
    /// Alias kind: upstream or workflow.
    pub kind: AliasKind,
    /// For `kind = "upstream"`: target upstream profile name.
    pub upstream_profile: Option<String>,
    /// For `kind = "workflow"`: target workflow ID.
    pub workflow_id: Option<String>,
    /// For `kind = "workflow"`: timeout in milliseconds.
    pub workflow_timeout_ms: Option<u64>,
}

impl ModelAliasConfig {
    /// Returns the effective workflow timeout after applying defaulting and cap.
    ///
    /// `None` or `Some(0)` both resolve to `DEFAULT_WORKFLOW_TIMEOUT_MS`. The
    /// value is capped at `MAX_WORKFLOW_TIMEOUT_MS`.
    #[must_use]
    pub fn effective_workflow_timeout_ms(&self) -> u64 {
        self.workflow_timeout_ms
            .filter(|timeout| *timeout > 0)
            .unwrap_or(DEFAULT_WORKFLOW_TIMEOUT_MS)
            .min(MAX_WORKFLOW_TIMEOUT_MS)
    }
}

/// Resolved target for a model alias.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AliasTarget {
    /// Routes to an upstream profile by name.
    Upstream {
        /// Configured upstream profile name.
        profile_name: String,
    },
    /// Invokes a GWP workflow.
    Workflow {
        /// Configured workflow ID.
        workflow_id: String,
        /// Effective workflow timeout in milliseconds.
        timeout_ms: u64,
    },
}

/// Error returned when alias resolution fails.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AliasResolutionError {
    /// No alias with this ID exists.
    UnknownAlias {
        /// Request model string.
        model: String,
    },
    /// Alias is misconfigured.
    MisconfiguredAlias {
        /// Configured alias ID.
        id: String,
        /// Human-readable reason.
        reason: String,
    },
}

impl Display for AliasResolutionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownAlias { model } => write!(formatter, "unknown model alias {model:?}"),
            Self::MisconfiguredAlias { id, reason } => {
                write!(formatter, "misconfigured model alias {id:?}: {reason}")
            }
        }
    }
}

impl Error for AliasResolutionError {}

/// Resolves model alias strings to typed targets.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelAliasResolver {
    aliases: HashMap<String, ModelAliasConfig>,
}

impl ModelAliasResolver {
    /// Builds a resolver from configured aliases.
    #[must_use]
    pub fn new(aliases: Vec<ModelAliasConfig>) -> Self {
        Self {
            aliases: aliases
                .into_iter()
                .map(|alias| (alias.id.clone(), alias))
                .collect(),
        }
    }

    /// Resolves a request model string to a typed alias target.
    ///
    /// # Errors
    ///
    /// Returns [`AliasResolutionError::UnknownAlias`] when no configured alias
    /// has this ID, or [`AliasResolutionError::MisconfiguredAlias`] when the
    /// alias lacks the required target field for its kind.
    pub fn resolve(&self, model: &str) -> Result<AliasTarget, AliasResolutionError> {
        let alias = self
            .aliases
            .get(model)
            .ok_or_else(|| AliasResolutionError::UnknownAlias {
                model: model.to_owned(),
            })?;

        match alias.kind {
            AliasKind::Upstream => {
                let profile_name = non_empty_target(
                    alias,
                    alias.upstream_profile.as_deref(),
                    "upstream alias requires upstream_profile",
                )?;
                Ok(AliasTarget::Upstream { profile_name })
            }
            AliasKind::Workflow => {
                let workflow_id = non_empty_target(
                    alias,
                    alias.workflow_id.as_deref(),
                    "workflow alias requires workflow_id",
                )?;
                Ok(AliasTarget::Workflow {
                    workflow_id,
                    timeout_ms: alias.effective_workflow_timeout_ms(),
                })
            }
        }
    }

    /// Returns configured aliases sorted by ID for stable `/v1/models` output.
    #[must_use]
    pub fn list_aliases(&self) -> Vec<&ModelAliasConfig> {
        let mut aliases = self.aliases.values().collect::<Vec<_>>();
        aliases.sort_by(|left, right| left.id.cmp(&right.id));
        aliases
    }

    /// Returns true when a request model is a configured alias.
    #[must_use]
    pub fn is_alias(&self, model: &str) -> bool {
        self.aliases.contains_key(model)
    }
}

fn non_empty_target(
    alias: &ModelAliasConfig,
    value: Option<&str>,
    reason: &'static str,
) -> Result<String, AliasResolutionError> {
    value
        .filter(|value| !value.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| AliasResolutionError::MisconfiguredAlias {
            id: alias.id.clone(),
            reason: reason.to_owned(),
        })
}

#[cfg(test)]
mod tests {
    use super::{
        AliasKind, AliasResolutionError, AliasTarget, DEFAULT_WORKFLOW_TIMEOUT_MS,
        MAX_WORKFLOW_TIMEOUT_MS, ModelAliasConfig, ModelAliasResolver,
    };

    #[test]
    fn resolves_upstream_alias() {
        let resolver = ModelAliasResolver::new(vec![ModelAliasConfig {
            id: String::from("gpt-default"),
            kind: AliasKind::Upstream,
            upstream_profile: Some(String::from("default")),
            workflow_id: None,
            workflow_timeout_ms: None,
        }]);

        assert_eq!(
            resolver.resolve("gpt-default"),
            Ok(AliasTarget::Upstream {
                profile_name: String::from("default"),
            })
        );
    }

    #[test]
    fn resolves_workflow_alias_with_configured_timeout() {
        let resolver = ModelAliasResolver::new(vec![ModelAliasConfig {
            id: String::from("family/child-safe-general-v1"),
            kind: AliasKind::Workflow,
            upstream_profile: None,
            workflow_id: Some(String::from("family.child_safe_general.v1")),
            workflow_timeout_ms: Some(120_000),
        }]);

        assert_eq!(
            resolver.resolve("family/child-safe-general-v1"),
            Ok(AliasTarget::Workflow {
                workflow_id: String::from("family.child_safe_general.v1"),
                timeout_ms: 120_000,
            })
        );
    }

    #[test]
    fn rejects_unknown_alias() {
        let resolver = ModelAliasResolver::default();

        assert_eq!(
            resolver.resolve("missing"),
            Err(AliasResolutionError::UnknownAlias {
                model: String::from("missing"),
            })
        );
    }

    #[test]
    fn rejects_misconfigured_upstream_alias() {
        let resolver = ModelAliasResolver::new(vec![ModelAliasConfig {
            id: String::from("gpt-default"),
            kind: AliasKind::Upstream,
            upstream_profile: None,
            workflow_id: None,
            workflow_timeout_ms: None,
        }]);

        assert!(matches!(
            resolver.resolve("gpt-default"),
            Err(AliasResolutionError::MisconfiguredAlias { id, .. }) if id == "gpt-default"
        ));
    }

    #[test]
    fn defaults_workflow_timeout() {
        let resolver = ModelAliasResolver::new(vec![ModelAliasConfig {
            id: String::from("workflow-default-timeout"),
            kind: AliasKind::Workflow,
            upstream_profile: None,
            workflow_id: Some(String::from("workflow.default_timeout")),
            workflow_timeout_ms: None,
        }]);

        assert_eq!(
            resolver.resolve("workflow-default-timeout"),
            Ok(AliasTarget::Workflow {
                workflow_id: String::from("workflow.default_timeout"),
                timeout_ms: DEFAULT_WORKFLOW_TIMEOUT_MS,
            })
        );
    }

    #[test]
    fn caps_workflow_timeout() {
        let resolver = ModelAliasResolver::new(vec![ModelAliasConfig {
            id: String::from("workflow-capped-timeout"),
            kind: AliasKind::Workflow,
            upstream_profile: None,
            workflow_id: Some(String::from("workflow.capped_timeout")),
            workflow_timeout_ms: Some(MAX_WORKFLOW_TIMEOUT_MS + 1),
        }]);

        assert_eq!(
            resolver.resolve("workflow-capped-timeout"),
            Ok(AliasTarget::Workflow {
                workflow_id: String::from("workflow.capped_timeout"),
                timeout_ms: MAX_WORKFLOW_TIMEOUT_MS,
            })
        );
    }

    #[test]
    fn zero_workflow_timeout_resolves_to_default() {
        let resolver = ModelAliasResolver::new(vec![ModelAliasConfig {
            id: String::from("workflow-zero-timeout"),
            kind: AliasKind::Workflow,
            upstream_profile: None,
            workflow_id: Some(String::from("workflow.zero_timeout")),
            workflow_timeout_ms: Some(0),
        }]);

        assert_eq!(
            resolver.resolve("workflow-zero-timeout"),
            Ok(AliasTarget::Workflow {
                workflow_id: String::from("workflow.zero_timeout"),
                timeout_ms: DEFAULT_WORKFLOW_TIMEOUT_MS,
            })
        );
    }

    #[test]
    fn lists_all_configured_aliases() {
        let resolver = ModelAliasResolver::new(vec![
            ModelAliasConfig {
                id: String::from("zeta"),
                kind: AliasKind::Upstream,
                upstream_profile: Some(String::from("default")),
                workflow_id: None,
                workflow_timeout_ms: None,
            },
            ModelAliasConfig {
                id: String::from("alpha"),
                kind: AliasKind::Workflow,
                upstream_profile: None,
                workflow_id: Some(String::from("workflow.alpha")),
                workflow_timeout_ms: None,
            },
        ]);

        let alias_ids = resolver
            .list_aliases()
            .into_iter()
            .map(|alias| alias.id.as_str())
            .collect::<Vec<_>>();

        assert_eq!(alias_ids, vec!["alpha", "zeta"]);
        assert!(resolver.is_alias("alpha"));
        assert!(!resolver.is_alias("missing"));
    }
}
