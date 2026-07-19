use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use axum::http::{HeaderMap, Uri};
use llm_guard_proxy_core::{
    ConfigHandle, EndpointSelectionMode, UpstreamEndpointConfig, UpstreamEndpointProtocol,
    UpstreamPriority, UpstreamProfileConfig,
};
use reqwest::Client;
use tokio::{
    sync::Mutex as AsyncMutex,
    time::{Instant, sleep, timeout},
};

use super::{
    ShutdownGate, build_upstream_url,
    reranker_protocol::{self, CanonicalRerankerRequest},
};

const BACKGROUND_PROBE_INTERVAL: Duration = Duration::from_secs(30);
const HEALTH_PROBE_HEADER: &str = "x-llm-guard-proxy-probe";

#[derive(Debug, Default)]
pub(super) struct UpstreamHealthRegistry {
    endpoints: Mutex<HashMap<String, Arc<EndpointHealth>>>,
    round_robin_positions: Mutex<HashMap<String, RoundRobinState>>,
    background_started: std::sync::atomic::AtomicBool,
}

#[derive(Debug, Default)]
struct EndpointHealth {
    probe_lock: AsyncMutex<()>,
    snapshot: Mutex<ProbeSnapshot>,
}

#[derive(Clone, Copy, Debug, Default)]
struct ProbeSnapshot {
    checked_at: Option<Instant>,
    healthy: bool,
    recovery_trial_in_progress: bool,
}

#[derive(Clone, Debug, Default)]
struct RoundRobinState {
    endpoint_identities: Vec<String>,
    next: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SelectedUpstreamEndpoint {
    pub(super) base_url: String,
    pub(super) priority: UpstreamPriority,
    pub(super) endpoint: UpstreamEndpointConfig,
    pub(super) selection_order: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum EndpointSelectionError {
    Shutdown,
    Incompatible { profile: String },
    Unavailable { profile: String, waited_ms: u64 },
}

pub(super) struct EndpointSelectionConstraints<'request> {
    pub(super) request: Option<&'request CanonicalRerankerRequest>,
    pub(super) request_headers: Option<&'request HeaderMap>,
    pub(super) request_deadline: Option<Instant>,
    pub(super) preferred_base_urls: Option<&'request [String]>,
    pub(super) excluded_base_urls: &'request [String],
}

impl UpstreamHealthRegistry {
    pub(super) async fn select_endpoint(
        &self,
        client: &Client,
        profile: &UpstreamProfileConfig,
        shutdown: &ShutdownGate,
        request: Option<&CanonicalRerankerRequest>,
        request_headers: Option<&HeaderMap>,
        request_deadline: Option<Instant>,
    ) -> Result<SelectedUpstreamEndpoint, EndpointSelectionError> {
        self.select_endpoint_excluding(
            client,
            profile,
            shutdown,
            EndpointSelectionConstraints {
                request,
                request_headers,
                request_deadline,
                preferred_base_urls: None,
                excluded_base_urls: &[],
            },
        )
        .await
    }

    pub(super) async fn select_endpoint_excluding(
        &self,
        client: &Client,
        profile: &UpstreamProfileConfig,
        shutdown: &ShutdownGate,
        constraints: EndpointSelectionConstraints<'_>,
    ) -> Result<SelectedUpstreamEndpoint, EndpointSelectionError> {
        if !profile.has_endpoint_failover() {
            return Ok(legacy_selected_endpoint(profile));
        }

        let started_at = Instant::now();
        let profile_deadline = started_at + Duration::from_millis(profile.health_probe_max_wait_ms);
        let deadline = constraints
            .request_deadline
            .map_or(profile_deadline, |request_deadline| {
                profile_deadline.min(request_deadline)
            });
        let protocol_compatible = profile.endpoints.iter().any(|endpoint| {
            reranker_protocol::is_compatible_with_endpoint(
                endpoint,
                constraints.request,
                constraints.request_headers,
            )
        });
        if !protocol_compatible {
            return Err(EndpointSelectionError::Incompatible {
                profile: profile.name.clone(),
            });
        }
        let mut candidates =
            Self::selection_order(profile, constraints.request, constraints.request_headers);
        if let Some(preferred_base_urls) = constraints.preferred_base_urls {
            order_preferred_endpoints(&mut candidates, preferred_base_urls);
        }
        candidates.retain(|endpoint| {
            !constraints
                .excluded_base_urls
                .iter()
                .any(|base_url| base_url == &endpoint.base_url)
        });
        if candidates.is_empty() {
            return Err(EndpointSelectionError::Unavailable {
                profile: profile.name.clone(),
                waited_ms: 0,
            });
        }

        loop {
            let mut eligible = Vec::with_capacity(candidates.len());
            for endpoint in &candidates {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    break;
                }
                if self
                    .probe_endpoint(
                        client,
                        endpoint,
                        Duration::from_millis(profile.health_probe_interval_ms),
                        Duration::from_millis(profile.health_probe_timeout_ms).min(remaining),
                        shutdown,
                        true,
                    )
                    .await?
                {
                    eligible.push(endpoint.clone());
                }
            }

            let eligible = self.ordered_eligible_endpoints(
                profile,
                eligible,
                constraints.excluded_base_urls.is_empty(),
                constraints.preferred_base_urls.is_some(),
            );
            if let Some(endpoint) = eligible.first().cloned() {
                return Ok(SelectedUpstreamEndpoint {
                    base_url: endpoint.base_url.clone(),
                    priority: endpoint.priority,
                    endpoint,
                    selection_order: eligible
                        .into_iter()
                        .map(|endpoint| endpoint.base_url)
                        .collect(),
                });
            }

            let now = Instant::now();
            if now >= deadline {
                return Err(EndpointSelectionError::Unavailable {
                    profile: profile.name.clone(),
                    waited_ms: duration_millis_u64(now.saturating_duration_since(started_at)),
                });
            }
            let delay = Duration::from_millis(profile.health_probe_interval_ms)
                .min(deadline.saturating_duration_since(now));
            let mut shutdown_subscription = shutdown.subscribe();
            tokio::select! {
                () = sleep(delay) => {}
                () = shutdown_subscription.cancelled() => {
                    return Err(EndpointSelectionError::Shutdown);
                }
            }
        }
    }

    async fn probe_endpoint(
        &self,
        client: &Client,
        endpoint: &UpstreamEndpointConfig,
        probe_interval: Duration,
        probe_timeout: Duration,
        shutdown: &ShutdownGate,
        reserve_passive_trial: bool,
    ) -> Result<bool, EndpointSelectionError> {
        if endpoint.protocol == UpstreamEndpointProtocol::DeepInfraQwen3Rerank {
            if !super::reranker_protocol::has_runtime_credential(endpoint) {
                return Ok(false);
            }
            let health = self.endpoint_health(endpoint);
            // Cloud health is passive: a cooldown blocks selection, expiry grants one real
            // request as the recovery trial, and no paid inference health probe is issued.
            return Ok(passive_cloud_eligible(
                &health,
                probe_interval,
                reserve_passive_trial,
            ));
        }
        let health = self.endpoint_health(endpoint);
        if let Some(healthy) = recent_health(&health, probe_interval) {
            return Ok(healthy);
        }

        let observed_check = health_snapshot(&health).checked_at;
        let _probe_guard = health.probe_lock.lock().await;
        let current = health_snapshot(&health);
        if current.checked_at != observed_check
            && current
                .checked_at
                .is_some_and(|checked_at| checked_at.elapsed() < probe_interval)
        {
            return Ok(current.healthy);
        }
        if let Some(healthy) = recent_health(&health, probe_interval) {
            return Ok(healthy);
        }

        let healthy = probe_models(client, &endpoint.base_url, probe_timeout, shutdown).await?;
        let mut snapshot = health_snapshot_mut(&health);
        snapshot.checked_at = Some(Instant::now());
        snapshot.healthy = healthy;
        Ok(healthy)
    }

    fn endpoint_health(&self, endpoint: &UpstreamEndpointConfig) -> Arc<EndpointHealth> {
        let identity = endpoint_identity(endpoint);
        let mut endpoints = mutex_guard(&self.endpoints);
        Arc::clone(
            endpoints
                .entry(identity)
                .or_insert_with(|| Arc::new(EndpointHealth::default())),
        )
    }

    fn selection_order(
        profile: &UpstreamProfileConfig,
        request: Option<&CanonicalRerankerRequest>,
        request_headers: Option<&HeaderMap>,
    ) -> Vec<UpstreamEndpointConfig> {
        let mut endpoints = profile
            .endpoints
            .iter()
            .filter(|endpoint| {
                reranker_protocol::is_compatible_with_endpoint(endpoint, request, request_headers)
            })
            .filter(|endpoint| reranker_protocol::has_runtime_credential(endpoint))
            .cloned()
            .collect::<Vec<_>>();
        endpoints.sort_by_key(|endpoint| match endpoint.priority {
            UpstreamPriority::Primary => 0_u8,
            UpstreamPriority::Failover => 1_u8,
        });
        endpoints
    }

    pub(super) fn eligible_endpoint_count(
        profile: &UpstreamProfileConfig,
        request: Option<&CanonicalRerankerRequest>,
        request_headers: Option<&HeaderMap>,
    ) -> usize {
        Self::selection_order(profile, request, request_headers).len()
    }

    fn ordered_eligible_endpoints(
        &self,
        profile: &UpstreamProfileConfig,
        mut eligible: Vec<UpstreamEndpointConfig>,
        advance_cursor: bool,
        preserve_preferred_order: bool,
    ) -> Vec<UpstreamEndpointConfig> {
        if eligible.is_empty() {
            return eligible;
        }
        if profile.endpoint_selection != EndpointSelectionMode::RoundRobin || eligible.len() == 1 {
            return eligible;
        }
        if preserve_preferred_order {
            return eligible;
        }
        let endpoint_identities = eligible.iter().map(endpoint_identity).collect::<Vec<_>>();
        let offset = if advance_cursor {
            self.next_round_robin_offset(&profile.name, &endpoint_identities)
        } else {
            0
        };
        eligible.rotate_left(offset);
        eligible
    }

    fn next_round_robin_offset(&self, profile_name: &str, endpoint_identities: &[String]) -> usize {
        let mut positions = mutex_guard(&self.round_robin_positions);
        let state = positions.entry(profile_name.to_owned()).or_default();
        if state.endpoint_identities != endpoint_identities {
            state.endpoint_identities = endpoint_identities.to_vec();
            state.next = 0;
        }
        let offset = state.next % endpoint_identities.len();
        state.next = state.next.wrapping_add(1);
        offset
    }

    pub(super) fn mark_unhealthy(&self, endpoint: &UpstreamEndpointConfig) {
        let health = self.endpoint_health(endpoint);
        let mut snapshot = health_snapshot_mut(&health);
        snapshot.checked_at = Some(Instant::now());
        snapshot.healthy = false;
        snapshot.recovery_trial_in_progress = false;
    }

    pub(super) fn mark_healthy(&self, endpoint: &UpstreamEndpointConfig) {
        let health = self.endpoint_health(endpoint);
        let mut snapshot = health_snapshot_mut(&health);
        snapshot.checked_at = Some(Instant::now());
        snapshot.healthy = true;
        snapshot.recovery_trial_in_progress = false;
    }

    pub(super) fn release_recovery_trial(&self, endpoint: &UpstreamEndpointConfig) {
        let health = self.endpoint_health(endpoint);
        health_snapshot_mut(&health).recovery_trial_in_progress = false;
    }

    pub(super) fn start_background_polling(
        self: &Arc<Self>,
        config: ConfigHandle,
        client: Client,
        shutdown: Arc<ShutdownGate>,
    ) {
        if self
            .background_started
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            return;
        }
        let registry = Arc::clone(self);
        tokio::spawn(async move {
            registry
                .background_poll_loop(config, client, shutdown)
                .await;
        });
    }

    async fn background_poll_loop(
        &self,
        config: ConfigHandle,
        client: Client,
        shutdown: Arc<ShutdownGate>,
    ) {
        loop {
            if shutdown.is_shutting_down() {
                return;
            }
            if let Ok(config) = config.snapshot() {
                self.reconcile_generation(&config);
                for profile in &config.upstream_profiles {
                    if !profile.has_endpoint_failover() {
                        continue;
                    }
                    for endpoint in &profile.endpoints {
                        if self
                            .probe_endpoint(
                                &client,
                                endpoint,
                                Duration::from_millis(profile.health_probe_interval_ms),
                                Duration::from_millis(profile.health_probe_timeout_ms),
                                &shutdown,
                                false,
                            )
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                }
            }
            let mut shutdown_subscription = shutdown.subscribe();
            tokio::select! {
                () = sleep(BACKGROUND_PROBE_INTERVAL) => {}
                () = shutdown_subscription.cancelled() => return,
            }
        }
    }

    fn reconcile_generation(&self, config: &llm_guard_proxy_core::AppConfig) {
        let active = config
            .upstream_profiles
            .iter()
            .flat_map(|profile| profile.endpoints.iter())
            .map(endpoint_identity)
            .collect::<HashSet<_>>();
        mutex_guard(&self.endpoints).retain(|identity, _health| active.contains(identity));
        mutex_guard(&self.round_robin_positions).retain(|profile_name, state| {
            config
                .upstream_profile_by_name(profile_name)
                .is_some_and(|profile| {
                    profile
                        .endpoints
                        .iter()
                        .map(endpoint_identity)
                        .collect::<Vec<_>>()
                        == state.endpoint_identities
                })
        });
    }
}

fn legacy_selected_endpoint(profile: &UpstreamProfileConfig) -> SelectedUpstreamEndpoint {
    let endpoint = UpstreamEndpointConfig {
        base_url: profile.base_url.clone(),
        priority: UpstreamPriority::Primary,
        ..UpstreamEndpointConfig::default()
    };
    SelectedUpstreamEndpoint {
        base_url: profile.base_url.clone(),
        priority: UpstreamPriority::Primary,
        endpoint,
        selection_order: vec![profile.base_url.clone()],
    }
}

fn endpoint_identity(endpoint: &UpstreamEndpointConfig) -> String {
    format!(
        "{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}",
        endpoint.base_url,
        endpoint.protocol.as_str(),
        endpoint.model.as_deref().unwrap_or_default(),
        endpoint.model_revision.as_deref().unwrap_or_default(),
        endpoint.api_key_env.as_deref().unwrap_or_default(),
        endpoint.priority.as_str(),
    )
}

fn order_preferred_endpoints(
    endpoints: &mut [UpstreamEndpointConfig],
    preferred_base_urls: &[String],
) {
    endpoints.sort_by_key(|endpoint| {
        preferred_base_urls
            .iter()
            .position(|base_url| base_url == &endpoint.base_url)
            .unwrap_or(usize::MAX)
    });
}

async fn probe_models(
    client: &Client,
    base_url: &str,
    probe_timeout: Duration,
    shutdown: &ShutdownGate,
) -> Result<bool, EndpointSelectionError> {
    let uri = Uri::from_static("/v1/models");
    let Ok(url) = build_upstream_url(base_url, &uri) else {
        return Ok(false);
    };
    let request = client
        .get(url)
        .header(HEALTH_PROBE_HEADER, "same-model-health")
        .send();
    let mut shutdown_subscription = shutdown.subscribe();
    tokio::select! {
        result = timeout(probe_timeout, request) => {
            Ok(matches!(result, Ok(Ok(response)) if response.status().is_success()))
        }
        () = shutdown_subscription.cancelled() => Err(EndpointSelectionError::Shutdown),
    }
}

fn passive_cloud_eligible(
    health: &EndpointHealth,
    interval: Duration,
    reserve_trial: bool,
) -> bool {
    let mut snapshot = health_snapshot_mut(health);
    if snapshot.checked_at.is_none() || snapshot.healthy {
        return true;
    }
    if snapshot
        .checked_at
        .is_some_and(|checked_at| checked_at.elapsed() < interval)
        || snapshot.recovery_trial_in_progress
    {
        return false;
    }
    if reserve_trial {
        snapshot.recovery_trial_in_progress = true;
    }
    true
}

fn recent_health(health: &EndpointHealth, interval: Duration) -> Option<bool> {
    let snapshot = health_snapshot(health);
    snapshot
        .checked_at
        .filter(|checked_at| checked_at.elapsed() < interval)
        .map(|_checked_at| snapshot.healthy)
}

fn health_snapshot(health: &EndpointHealth) -> ProbeSnapshot {
    *health_snapshot_mut(health)
}

fn health_snapshot_mut(health: &EndpointHealth) -> MutexGuard<'_, ProbeSnapshot> {
    mutex_guard(&health.snapshot)
}

fn mutex_guard<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn duration_millis_u64(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Bytes;

    fn endpoint(base_url: &str) -> UpstreamEndpointConfig {
        UpstreamEndpointConfig {
            base_url: String::from(base_url),
            ..UpstreamEndpointConfig::default()
        }
    }

    fn round_robin_profile() -> UpstreamProfileConfig {
        UpstreamProfileConfig {
            name: String::from("reranker"),
            endpoint_selection: EndpointSelectionMode::RoundRobin,
            endpoints: vec![endpoint("http://first/v1"), endpoint("http://second/v1")],
            ..UpstreamProfileConfig::default()
        }
    }

    #[test]
    fn round_robin_advances_only_over_currently_eligible_endpoints() {
        let registry = UpstreamHealthRegistry::default();
        let profile = round_robin_profile();
        let eligible = vec![profile.endpoints[0].clone(), profile.endpoints[1].clone()];
        assert_eq!(
            registry
                .ordered_eligible_endpoints(&profile, eligible.clone(), true, false)
                .into_iter()
                .next()
                .expect("first endpoint")
                .base_url,
            "http://first/v1"
        );
        assert_eq!(
            registry
                .ordered_eligible_endpoints(&profile, eligible.clone(), true, false)
                .into_iter()
                .next()
                .expect("second endpoint")
                .base_url,
            "http://second/v1"
        );
        let only_second = vec![profile.endpoints[1].clone()];
        assert_eq!(
            registry
                .ordered_eligible_endpoints(&profile, only_second, true, false)
                .into_iter()
                .next()
                .expect("only healthy endpoint")
                .base_url,
            "http://second/v1"
        );
        assert_eq!(
            registry
                .ordered_eligible_endpoints(&profile, eligible, true, false)
                .into_iter()
                .next()
                .expect("membership change resets fairly")
                .base_url,
            "http://first/v1"
        );
    }

    #[test]
    fn retry_follows_the_initial_round_robin_remaining_order() {
        let registry = UpstreamHealthRegistry::default();
        let mut profile = round_robin_profile();
        profile.endpoints.push(endpoint("http://third/v1"));
        let first =
            registry.ordered_eligible_endpoints(&profile, profile.endpoints.clone(), true, false);
        let second =
            registry.ordered_eligible_endpoints(&profile, profile.endpoints.clone(), true, false);
        assert_eq!(first[0].base_url, "http://first/v1");
        assert_eq!(second[0].base_url, "http://second/v1");
        let preferred = second
            .iter()
            .map(|endpoint| endpoint.base_url.clone())
            .collect::<Vec<_>>();
        let mut retry_candidates = vec![profile.endpoints[0].clone(), profile.endpoints[2].clone()];
        order_preferred_endpoints(&mut retry_candidates, &preferred);
        let retry_order =
            registry.ordered_eligible_endpoints(&profile, retry_candidates, false, true);
        assert_eq!(retry_order[0].base_url, "http://third/v1");
    }

    #[test]
    fn incompatible_deepinfra_never_enters_a_generic_request_order() {
        let mut profile = round_robin_profile();
        profile.endpoints.push(UpstreamEndpointConfig {
            base_url: String::from("https://api.deepinfra.com"),
            priority: UpstreamPriority::Failover,
            protocol: UpstreamEndpointProtocol::DeepInfraQwen3Rerank,
            model: Some(String::from("Qwen/Qwen3-Reranker-8B")),
            model_revision: Some(String::from("5fa94080caafeaa45a15d11f969d7978e087a3db")),
            api_key_env: Some(String::from("UNSET_TEST_DEEPINFRA_KEY")),
        });
        let order = UpstreamHealthRegistry::selection_order(&profile, None, None);
        assert!(
            order
                .iter()
                .all(|endpoint| endpoint.protocol == UpstreamEndpointProtocol::OpenAi)
        );
        let opaque_score = CanonicalRerankerRequest::UnsupportedScore;
        assert!(reranker_protocol::is_compatible_with_endpoint(
            &profile.endpoints[0],
            Some(&opaque_score),
            None,
        ));
        assert!(!reranker_protocol::is_compatible_with_endpoint(
            &profile.endpoints[2],
            Some(&opaque_score),
            None,
        ));
    }

    #[test]
    fn endpoint_identity_includes_model_revision_and_credential_binding() {
        let mut endpoint = UpstreamEndpointConfig {
            base_url: String::from("https://api.deepinfra.com"),
            priority: UpstreamPriority::Failover,
            protocol: UpstreamEndpointProtocol::DeepInfraQwen3Rerank,
            model: Some(String::from("Qwen/Qwen3-Reranker-8B")),
            model_revision: Some(String::from("5fa94080caafeaa45a15d11f969d7978e087a3db")),
            api_key_env: Some(String::from("FIRST_KEY")),
        };
        let original = endpoint_identity(&endpoint);
        endpoint.model_revision = Some(String::from("6fa94080caafeaa45a15d11f969d7978e087a3db"));
        assert_ne!(original, endpoint_identity(&endpoint));
        endpoint.model_revision = Some(String::from("5fa94080caafeaa45a15d11f969d7978e087a3db"));
        endpoint.api_key_env = Some(String::from("SECOND_KEY"));
        assert_ne!(original, endpoint_identity(&endpoint));
    }

    #[test]
    fn compatible_local_order_preserves_opaque_score_without_parsing_it() {
        let profile = round_robin_profile();
        let request = CanonicalRerankerRequest::Score {
            forward_uri: Uri::from_static("/v1/score"),
            body: Bytes::from_static(br#"{"future":"opaque"}"#),
        };
        let order = UpstreamHealthRegistry::selection_order(&profile, Some(&request), None);
        assert_eq!(order.len(), 2);
    }
}
