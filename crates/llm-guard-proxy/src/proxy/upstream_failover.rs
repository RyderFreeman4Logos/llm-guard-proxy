use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use axum::http::Uri;
use llm_guard_proxy_core::{
    ConfigHandle, EndpointSelectionMode, UpstreamEndpointConfig, UpstreamEndpointProtocol,
    UpstreamPriority, UpstreamProfileConfig,
};
use reqwest::Client;
use tokio::{
    sync::Mutex as AsyncMutex,
    time::{Instant, sleep, timeout},
};

use super::{ShutdownGate, build_upstream_url};

const BACKGROUND_PROBE_INTERVAL: Duration = Duration::from_secs(30);
const HEALTH_PROBE_HEADER: &str = "x-llm-guard-proxy-probe";

#[derive(Debug, Default)]
pub(super) struct UpstreamHealthRegistry {
    endpoints: Mutex<HashMap<String, Arc<EndpointHealth>>>,
    round_robin_positions: Mutex<HashMap<String, usize>>,
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
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SelectedUpstreamEndpoint {
    pub(super) base_url: String,
    pub(super) priority: UpstreamPriority,
    pub(super) endpoint: UpstreamEndpointConfig,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum EndpointSelectionError {
    Shutdown,
    Unavailable { profile: String, waited_ms: u64 },
}

impl UpstreamHealthRegistry {
    pub(super) async fn select_endpoint(
        &self,
        client: &Client,
        profile: &UpstreamProfileConfig,
        shutdown: &ShutdownGate,
    ) -> Result<SelectedUpstreamEndpoint, EndpointSelectionError> {
        self.select_endpoint_excluding(client, profile, shutdown, &[])
            .await
    }

    pub(super) async fn select_endpoint_excluding(
        &self,
        client: &Client,
        profile: &UpstreamProfileConfig,
        shutdown: &ShutdownGate,
        excluded_base_urls: &[String],
    ) -> Result<SelectedUpstreamEndpoint, EndpointSelectionError> {
        if !profile.has_endpoint_failover() {
            let endpoint = UpstreamEndpointConfig {
                base_url: profile.base_url.clone(),
                priority: UpstreamPriority::Primary,
                ..UpstreamEndpointConfig::default()
            };
            return Ok(SelectedUpstreamEndpoint {
                base_url: profile.base_url.clone(),
                priority: UpstreamPriority::Primary,
                endpoint,
            });
        }

        let started_at = Instant::now();
        let deadline = started_at + Duration::from_millis(profile.health_probe_max_wait_ms);
        let endpoints = self.selection_order(profile);

        loop {
            for endpoint in &endpoints {
                if excluded_base_urls
                    .iter()
                    .any(|base_url| base_url == &endpoint.base_url)
                {
                    continue;
                }
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
                    )
                    .await?
                {
                    return Ok(SelectedUpstreamEndpoint {
                        base_url: endpoint.base_url.clone(),
                        priority: endpoint.priority,
                        endpoint: endpoint.clone(),
                    });
                }
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
    ) -> Result<bool, EndpointSelectionError> {
        if endpoint.protocol == UpstreamEndpointProtocol::DeepInfraQwen3Rerank {
            return Ok(super::reranker_protocol::has_runtime_credential(endpoint));
        }
        let health = self.endpoint_health(&endpoint.base_url);
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

    fn endpoint_health(&self, base_url: &str) -> Arc<EndpointHealth> {
        let mut endpoints = mutex_guard(&self.endpoints);
        Arc::clone(
            endpoints
                .entry(base_url.to_owned())
                .or_insert_with(|| Arc::new(EndpointHealth::default())),
        )
    }

    fn selection_order(&self, profile: &UpstreamProfileConfig) -> Vec<UpstreamEndpointConfig> {
        let mut endpoints = profile.endpoints.clone();
        endpoints.sort_by_key(|endpoint| match endpoint.priority {
            UpstreamPriority::Primary => 0_u8,
            UpstreamPriority::Failover => 1_u8,
        });
        if profile.endpoint_selection != EndpointSelectionMode::RoundRobin || endpoints.len() < 2 {
            return endpoints;
        }
        let offset = self.next_round_robin_offset(&profile.name, endpoints.len());
        endpoints.rotate_left(offset);
        endpoints
    }

    fn next_round_robin_offset(&self, profile_name: &str, endpoint_count: usize) -> usize {
        let mut positions = mutex_guard(&self.round_robin_positions);
        let next = positions.entry(profile_name.to_owned()).or_insert(0);
        let offset = *next % endpoint_count;
        *next = next.wrapping_add(1);
        offset
    }

    pub(super) fn mark_unhealthy(&self, base_url: &str) {
        let health = self.endpoint_health(base_url);
        let mut snapshot = health_snapshot_mut(&health);
        snapshot.checked_at = Some(Instant::now());
        snapshot.healthy = false;
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
