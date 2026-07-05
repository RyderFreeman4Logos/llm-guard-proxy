//! Caller profile policy types.

use crate::{GwpProfile, GwpProfileKind};

/// Default profile name used when no profile key is provided.
pub const DEFAULT_PROFILE_NAME: &str = "default";

/// Kind of caller profile.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileKind {
    /// Child caller profile.
    Child,
    /// Adult caller profile.
    Adult,
}

impl From<&ProfileKind> for GwpProfileKind {
    fn from(kind: &ProfileKind) -> Self {
        match kind {
            ProfileKind::Child => Self::Child,
            ProfileKind::Adult => Self::Adult,
        }
    }
}

/// Shielded buffering mode for a profile.
#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShieldedBuffering {
    /// No buffering: stream responses directly.
    #[default]
    Off,
    /// Buffer full SSE response before returning to client.
    BufferedSse,
    /// Buffer and sanitize text chunks before streaming.
    Sanitized,
}

/// Configuration for one caller profile.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProfileConfig {
    /// Profile kind (child/adult).
    pub kind: ProfileKind,
    /// Model aliases this profile is allowed to use.
    #[serde(default)]
    pub allowed_models: Vec<String>,
    /// Maximum requests per day (`None` means unlimited).
    #[serde(default)]
    pub daily_request_limit: Option<u32>,
    /// Shielded buffering mode.
    #[serde(default)]
    pub shielded_buffering: ShieldedBuffering,
    /// Guard pack name (references a guard workflow or policy pack).
    #[serde(default)]
    pub guard_pack: Option<String>,
}

impl ProfileConfig {
    /// Check if a model alias is allowed for this profile.
    #[must_use]
    pub fn is_model_allowed(&self, model: &str) -> bool {
        self.allowed_models.iter().any(|allowed| allowed == model)
    }

    /// Check a request against this profile.
    #[must_use]
    pub fn check_request(&self, model: &str, daily_count: u32) -> ProfileCheckResult {
        if !self.is_model_allowed(model) {
            return ProfileCheckResult::Block {
                reason: BlockReason::ModelNotAllowed {
                    model: model.to_owned(),
                },
            };
        }
        if let Some(limit) = self.daily_request_limit {
            if daily_count >= limit {
                return ProfileCheckResult::Block {
                    reason: BlockReason::DailyLimitExceeded { limit },
                };
            }
        }
        ProfileCheckResult::Allow
    }

    /// Convert to GWP profile metadata for GWP invocation.
    #[must_use]
    pub fn to_gwp_profile(&self, id: &str) -> GwpProfile {
        GwpProfile {
            id: id.to_owned(),
            kind: GwpProfileKind::from(&self.kind),
        }
    }
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            kind: ProfileKind::Adult,
            allowed_models: Vec::new(),
            daily_request_limit: None,
            shielded_buffering: ShieldedBuffering::Off,
            guard_pack: None,
        }
    }
}

/// Result of checking a request against a profile.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProfileCheckResult {
    /// Request is allowed.
    Allow,
    /// Request is blocked with a reason.
    Block {
        /// Reason the profile blocked the request.
        reason: BlockReason,
    },
}

/// Reason a profile check blocked a request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BlockReason {
    /// Model alias not in the profile's allowed list.
    ModelNotAllowed {
        /// Requested model alias.
        model: String,
    },
    /// Daily request limit exceeded.
    DailyLimitExceeded {
        /// Configured daily request limit.
        limit: u32,
    },
    /// Profile kind does not match the required kind for this model.
    KindMismatch,
}

#[cfg(test)]
mod tests {
    use super::{BlockReason, ProfileCheckResult, ProfileConfig, ProfileKind, ShieldedBuffering};
    use crate::GwpProfileKind;

    fn adult_profile() -> ProfileConfig {
        ProfileConfig {
            kind: ProfileKind::Adult,
            allowed_models: vec![
                String::from("gpt-default"),
                String::from("family/child-safe-general-v1"),
            ],
            daily_request_limit: None,
            shielded_buffering: ShieldedBuffering::Off,
            guard_pack: None,
        }
    }

    #[test]
    fn adult_profile_allows_models_in_allowlist() {
        let profile = adult_profile();

        assert!(profile.is_model_allowed("gpt-default"));
        assert_eq!(
            profile.check_request("family/child-safe-general-v1", 10),
            ProfileCheckResult::Allow
        );
    }

    #[test]
    fn child_profile_blocks_model_not_in_allowlist() {
        let profile = ProfileConfig {
            kind: ProfileKind::Child,
            allowed_models: vec![String::from("family/child-safe-general-v1")],
            daily_request_limit: None,
            shielded_buffering: ShieldedBuffering::BufferedSse,
            guard_pack: Some(String::from("family_basic")),
        };

        assert_eq!(
            profile.check_request("gpt-default", 0),
            ProfileCheckResult::Block {
                reason: BlockReason::ModelNotAllowed {
                    model: String::from("gpt-default"),
                },
            }
        );
    }

    #[test]
    fn daily_limit_exceeded_blocks_request() {
        let mut profile = adult_profile();
        profile.daily_request_limit = Some(50);

        assert_eq!(
            profile.check_request("gpt-default", 50),
            ProfileCheckResult::Block {
                reason: BlockReason::DailyLimitExceeded { limit: 50 },
            }
        );
    }

    #[test]
    fn missing_daily_limit_allows_allowed_model_for_any_count() {
        let profile = adult_profile();

        assert_eq!(
            profile.check_request("gpt-default", u32::MAX),
            ProfileCheckResult::Allow
        );
    }

    #[test]
    fn unknown_model_blocks_as_not_allowed() {
        let profile = adult_profile();

        assert_eq!(
            profile.check_request("unknown", 0),
            ProfileCheckResult::Block {
                reason: BlockReason::ModelNotAllowed {
                    model: String::from("unknown"),
                },
            }
        );
    }

    #[test]
    fn empty_allowed_models_allows_nothing() {
        let profile = ProfileConfig::default();

        assert!(!profile.is_model_allowed("gpt-default"));
        assert_eq!(
            profile.check_request("gpt-default", 0),
            ProfileCheckResult::Block {
                reason: BlockReason::ModelNotAllowed {
                    model: String::from("gpt-default"),
                },
            }
        );
    }

    #[test]
    fn to_gwp_profile_preserves_kind_and_id() {
        let profile = ProfileConfig {
            kind: ProfileKind::Child,
            ..ProfileConfig::default()
        };

        let gwp_profile = profile.to_gwp_profile("child_default");

        assert_eq!(gwp_profile.id, "child_default");
        assert_eq!(gwp_profile.kind, GwpProfileKind::Child);
    }
}
