use super::{
    AppConfig, Body, Bytes, CONTENT_TYPE, ConfigHandle, EvidenceStore, ObservabilityStore,
    ProxyState, Request, Response, Router, State, StatusCode, TcpListener, build_http_client,
    router,
};

#[cfg(feature = "guard")]
use super::BudgetStore;

use std::{
    collections::BTreeMap,
    convert::Infallible,
    future::IntoFuture,
    net::{Ipv4Addr, SocketAddr},
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::{sync::oneshot, task::JoinHandle, time::timeout};

const SELF_TEST_DEADLINE: Duration = Duration::from_secs(5);
const CLIENT_CHUNK_DEADLINE: Duration = Duration::from_secs(2);
const FIRST_CHUNK_TIMEOUT_MS: u64 = 50;
const HEARTBEAT: &[u8] = b": llm-guard-proxy heartbeat\n\n";
const REQUEST_BODY: &str = r#"{"model":"self-test","messages":[{"role":"user","content":"transport check"}],"stream":true}"#;
const SUCCESS_SSE: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Arm {
    Control,
    Committed,
}

#[derive(Clone)]
pub(super) struct Context {
    inner: Arc<ContextInner>,
}

struct ContextInner {
    arm: Arm,
    request_claims: AtomicU64,
    rejected_request_claims: AtomicU64,
    guard_business_attempts: AtomicU64,
    guard_probe_dispatches: AtomicU64,
    started: Instant,
    last_stamp_ns: AtomicU64,
    pre_await_gate_ns: AtomicU64,
    recovery_await_entered_ns: AtomicU64,
    body_emitted_ns: AtomicU64,
    client_ack_ns: AtomicU64,
    recovery_await_completed_ns: AtomicU64,
    post_await_committed_ns: AtomicU64,
    emit_tx: Mutex<Option<oneshot::Sender<()>>>,
    emit_rx: Mutex<Option<oneshot::Receiver<()>>>,
    ack_tx: Mutex<Option<oneshot::Sender<()>>>,
    ack_rx: Mutex<Option<oneshot::Receiver<()>>>,
}

impl std::fmt::Debug for Context {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PostAwaitSelfTestContext")
            .field("arm", &self.inner.arm)
            .finish()
    }
}

impl Context {
    fn new(arm: Arm) -> Self {
        let (emit_tx, emit_rx) = oneshot::channel();
        let (ack_tx, ack_rx) = oneshot::channel();
        Self {
            inner: Arc::new(ContextInner {
                arm,
                request_claims: AtomicU64::new(0),
                rejected_request_claims: AtomicU64::new(0),
                guard_business_attempts: AtomicU64::new(0),
                guard_probe_dispatches: AtomicU64::new(0),
                started: Instant::now(),
                last_stamp_ns: AtomicU64::new(0),
                pre_await_gate_ns: AtomicU64::new(0),
                recovery_await_entered_ns: AtomicU64::new(0),
                body_emitted_ns: AtomicU64::new(0),
                client_ack_ns: AtomicU64::new(0),
                recovery_await_completed_ns: AtomicU64::new(0),
                post_await_committed_ns: AtomicU64::new(0),
                emit_tx: Mutex::new(Some(emit_tx)),
                emit_rx: Mutex::new(Some(emit_rx)),
                ack_tx: Mutex::new(Some(ack_tx)),
                ack_rx: Mutex::new(Some(ack_rx)),
            }),
        }
    }

    pub(super) fn claim_request(&self) -> Result<Self, String> {
        if self
            .inner
            .request_claims
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            return Ok(self.clone());
        }
        self.inner
            .rejected_request_claims
            .fetch_add(1, Ordering::AcqRel);
        Err(failure("request_claim_rejected"))
    }

    pub(super) fn request_claim_counts(&self) -> (u64, u64) {
        (
            self.inner.request_claims.load(Ordering::Acquire),
            self.inner.rejected_request_claims.load(Ordering::Acquire),
        )
    }

    pub(super) fn mark_guard_business_attempt(&self) {
        self.inner
            .guard_business_attempts
            .fetch_add(1, Ordering::AcqRel);
    }

    pub(super) fn mark_guard_probe_dispatch(&self) {
        self.inner
            .guard_probe_dispatches
            .fetch_add(1, Ordering::AcqRel);
    }

    pub(super) fn guard_attempt_counts(&self) -> (u64, u64) {
        (
            self.inner.guard_business_attempts.load(Ordering::Acquire),
            self.inner.guard_probe_dispatches.load(Ordering::Acquire),
        )
    }

    pub(super) fn mark_pre_await_gate(&self) {
        self.mark(&self.inner.pre_await_gate_ns);
    }

    pub(super) fn take_emit_receiver(&self) -> Option<oneshot::Receiver<()>> {
        if self.inner.arm != Arm::Committed {
            return None;
        }
        take_channel(&self.inner.emit_rx)
    }

    pub(super) fn mark_body_emitted(&self) {
        self.mark(&self.inner.body_emitted_ns);
    }

    fn acknowledge_client_read(&self) -> Result<(), String> {
        self.mark(&self.inner.client_ack_ns);
        take_channel(&self.inner.ack_tx)
            .ok_or_else(|| failure("client_ack_sender_missing"))?
            .send(())
            .map_err(|()| failure("client_ack_receiver_closed"))
    }

    pub(super) async fn restart_metadata(&self) -> BTreeMap<String, String> {
        self.mark(&self.inner.recovery_await_entered_ns);
        if self.inner.arm == Arm::Committed {
            let emitted = take_channel(&self.inner.emit_tx).and_then(|sender| sender.send(()).ok());
            let acknowledged = match take_channel(&self.inner.ack_rx) {
                Some(receiver) => receiver.await.is_ok(),
                None => false,
            };
            if emitted.is_none() || !acknowledged {
                return BTreeMap::from([
                    (
                        String::from("local_recovery_restart_status"),
                        String::from("offline_coordination_failed"),
                    ),
                    (
                        String::from("local_recovery_status"),
                        String::from("offline_coordination_failed"),
                    ),
                ]);
            }
        }
        self.mark(&self.inner.recovery_await_completed_ns);
        BTreeMap::from([(
            String::from("local_recovery_restart_status"),
            String::from("succeeded"),
        )])
    }

    pub(super) fn mark_post_await_committed(&self) {
        self.mark(&self.inner.post_await_committed_ns);
    }

    fn phases(&self) -> PhaseReceipt {
        PhaseReceipt {
            pre_await_gate: self.inner.pre_await_gate_ns.load(Ordering::Acquire),
            recovery_await_entered_ns: self.inner.recovery_await_entered_ns.load(Ordering::Acquire),
            body_emitted_ns: self.inner.body_emitted_ns.load(Ordering::Acquire),
            client_ack_ns: self.inner.client_ack_ns.load(Ordering::Acquire),
            recovery_await_completed_ns: self
                .inner
                .recovery_await_completed_ns
                .load(Ordering::Acquire),
            post_await_committed_ns: self.inner.post_await_committed_ns.load(Ordering::Acquire),
        }
    }

    fn mark(&self, target: &AtomicU64) {
        let elapsed = u64::try_from(self.inner.started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let mut previous = self.inner.last_stamp_ns.load(Ordering::Acquire);
        loop {
            let stamp = elapsed.max(previous.saturating_add(1)).max(1);
            match self.inner.last_stamp_ns.compare_exchange_weak(
                previous,
                stamp,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    target.store(stamp, Ordering::Release);
                    return;
                }
                Err(current) => previous = current,
            }
        }
    }
}

fn take_channel<T>(slot: &Mutex<Option<T>>) -> Option<T> {
    slot.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
}

#[derive(Debug)]
struct FakeUpstreamState {
    arm: Arm,
    business_count: AtomicU64,
    probe_count: AtomicU64,
    ordered_roles: Mutex<Vec<String>>,
    first_business_payload: Mutex<Option<BusinessPayload>>,
    same_payload: AtomicBool,
    validation_error: Mutex<Option<String>>,
}

impl FakeUpstreamState {
    fn new(arm: Arm) -> Self {
        Self {
            arm,
            business_count: AtomicU64::new(0),
            probe_count: AtomicU64::new(0),
            ordered_roles: Mutex::new(Vec::new()),
            first_business_payload: Mutex::new(None),
            same_payload: AtomicBool::new(true),
            validation_error: Mutex::new(None),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct BusinessPayload {
    model: String,
    messages: Vec<BusinessMessage>,
    stream: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct BusinessMessage {
    role: String,
    content: String,
}

#[derive(Debug)]
struct LoopbackServer {
    addr: SocketAddr,
    task: Option<JoinHandle<std::io::Result<()>>>,
}

impl LoopbackServer {
    async fn stop(mut self) -> bool {
        let Some(task) = self.task.take() else {
            return true;
        };
        task.abort();
        matches!(task.await, Err(error) if error.is_cancelled())
    }
}

impl Drop for LoopbackServer {
    fn drop(&mut self) {
        if let Some(task) = &self.task {
            task.abort();
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PhaseReceipt {
    #[serde(rename = "pre_await_gate_ns")]
    pre_await_gate: u64,
    recovery_await_entered_ns: u64,
    body_emitted_ns: u64,
    client_ack_ns: u64,
    recovery_await_completed_ns: u64,
    post_await_committed_ns: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ArmReceipt {
    ordered_roles: Vec<String>,
    request_claims: u64,
    rejected_request_claims: u64,
    guard_business_count: u64,
    guard_probe_count: u64,
    business_count: u64,
    probe_count: u64,
    #[serde(flatten)]
    fault: FaultReceipt,
    first_byte_wait_ms: u64,
    #[serde(flatten)]
    client: ClientEvidence,
    #[serde(flatten)]
    completion: CompletionEvidence,
    phases: PhaseReceipt,
    #[serde(flatten)]
    cleanup: CleanupEvidence,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FaultReceipt {
    same_payload: bool,
    first_chunk_stall: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ClientEvidence {
    client_observed_heartbeat: bool,
    done_observed: bool,
    terminal_error_observed: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CompletionEvidence {
    eof_observed: bool,
    post_await_committed: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CleanupEvidence {
    loopback_only: bool,
    cleanup_complete: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    self_test: String,
    status: String,
    control: ArmReceipt,
    committed: ArmReceipt,
    same_payload_across_arms: bool,
}

struct ArmRun {
    receipt: ArmReceipt,
    first_payload: BusinessPayload,
}

pub(super) async fn run() -> Result<serde_json::Value, String> {
    timeout(SELF_TEST_DEADLINE, async {
        let control = run_arm(Arm::Control).await?;
        let committed = run_arm(Arm::Committed).await?;
        let same_payload_across_arms = control.first_payload == committed.first_payload;
        let receipt = Receipt {
            self_test: String::from("post-await-no-replay"),
            status: String::from("passed"),
            control: control.receipt,
            committed: committed.receipt,
            same_payload_across_arms,
        };
        validate_receipt(&receipt)?;
        serde_json::to_value(receipt).map_err(|_| failure("serialize_receipt"))
    })
    .await
    .map_err(|_| failure("deadline_exceeded"))?
}

async fn run_arm(arm: Arm) -> Result<ArmRun, String> {
    let fake_state = Arc::new(FakeUpstreamState::new(arm));
    let fake_server = spawn_fake_upstream(Arc::clone(&fake_state)).await?;
    let loopback_only = fake_server.addr.ip().is_loopback();
    let context = Context::new(arm);
    let mut proxy_state = build_proxy_state(fake_server.addr, context.clone())?;
    proxy_state.post_await_self_test = Some(context.clone());
    let cleanup_state = proxy_state.clone();
    let proxy_server = spawn_proxy(proxy_state).await?;
    let loopback_only = loopback_only && proxy_server.addr.ip().is_loopback();

    let client_result = run_internal_client(proxy_server.addr, arm, &context).await;
    cleanup_state.begin_shutdown();
    cleanup_state.flush_persistence().await;
    let proxy_stopped = proxy_server.stop().await;
    let fake_stopped = fake_server.stop().await;
    let cleanup_complete = proxy_stopped && fake_stopped;
    let client = match client_result {
        Ok(client) => client,
        Err(error) => {
            return Err(fake_state
                .validation_error
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
                .unwrap_or(error));
        }
    };
    let ordered_roles = fake_state
        .ordered_roles
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    let first_payload = fake_state
        .first_business_payload
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .ok_or_else(|| failure("missing_first_business_payload"))?;

    let phases = context.phases();
    let (request_claims, rejected_request_claims) = context.request_claim_counts();
    let (guard_business_count, guard_probe_count) = context.guard_attempt_counts();
    Ok(ArmRun {
        receipt: ArmReceipt {
            ordered_roles,
            request_claims,
            rejected_request_claims,
            guard_business_count,
            guard_probe_count,
            business_count: fake_state.business_count.load(Ordering::Acquire),
            probe_count: fake_state.probe_count.load(Ordering::Acquire),
            fault: FaultReceipt {
                same_payload: fake_state.same_payload.load(Ordering::Acquire),
                first_chunk_stall: phases.pre_await_gate > 0,
            },
            first_byte_wait_ms: client.first_byte_wait_ms,
            client: ClientEvidence {
                client_observed_heartbeat: client.outcome == ClientOutcome::CommittedBlocked,
                done_observed: client.outcome == ClientOutcome::ControlSuccess,
                terminal_error_observed: client.outcome == ClientOutcome::CommittedBlocked,
            },
            completion: CompletionEvidence {
                eof_observed: true,
                post_await_committed: phases.post_await_committed_ns > 0,
            },
            phases,
            cleanup: CleanupEvidence {
                loopback_only,
                cleanup_complete,
            },
        },
        first_payload,
    })
}

fn build_proxy_state(upstream_addr: SocketAddr, context: Context) -> Result<ProxyState, String> {
    let mut config = AppConfig::default();
    config.upstream.base_url = format!("http://{upstream_addr}/v1");
    config.upstream.request_timeout_ms = 1_000;
    config.upstream.local_recovery.enabled = true;
    config.upstream.local_recovery.restart_timeout_ms = 500;
    config.upstream.local_recovery.readiness_request_timeout_ms = 500;
    config.upstream.local_recovery.readiness_deadline_ms = 500;
    config.upstream.local_recovery.readiness_interval_ms = 10;
    config.upstream.local_recovery.cooldown_ms = 1;
    config.upstream.local_recovery.budget_window_ms = 1_000;
    config.retry.enabled = true;
    config.retry.max_attempts = 2;
    config.retry.request_deadline_ms = 2_000;
    config.retry.shielded_streaming_enabled = true;
    config.thinking.enabled = false;
    config.upstream_stall.enabled = true;
    config.upstream_stall.first_chunk_timeout_ms = FIRST_CHUNK_TIMEOUT_MS;
    config.upstream_stall.idle_timeout_ms = 500;
    config.observability.enabled = false;
    config.observability.sqlite_path = PathBuf::from(":memory:");
    config.evidence.enabled = false;
    config.validate().map_err(|_| failure("validate_config"))?;

    let handle = ConfigHandle::new(config);
    let store = ObservabilityStore::open(handle.clone())
        .map_err(|_| failure("open_memory_observability"))?;
    let evidence_store = EvidenceStore::open(handle.clone());
    #[cfg(feature = "guard")]
    let budget_store =
        Arc::new(BudgetStore::open(":memory:").map_err(|_| failure("open_memory_budget"))?);
    let mut state = ProxyState::new(
        handle,
        PathBuf::from(":memory:"),
        store,
        evidence_store,
        #[cfg(feature = "guard")]
        budget_store,
        build_http_client().map_err(|_| failure("build_proxy_client"))?,
    );
    state.post_await_self_test = Some(context);
    Ok(state)
}

async fn spawn_fake_upstream(state: Arc<FakeUpstreamState>) -> Result<LoopbackServer, String> {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .map_err(|_| failure("bind_fake_upstream"))?;
    let addr = listener
        .local_addr()
        .map_err(|_| failure("fake_upstream_addr"))?;
    let app = Router::new()
        .fallback(fake_upstream_handler)
        .with_state(state);
    let task = tokio::spawn(axum::serve(listener, app).into_future());
    Ok(LoopbackServer {
        addr,
        task: Some(task),
    })
}

async fn spawn_proxy(state: ProxyState) -> Result<LoopbackServer, String> {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .map_err(|_| failure("bind_proxy"))?;
    let addr = listener.local_addr().map_err(|_| failure("proxy_addr"))?;
    let task = tokio::spawn(axum::serve(listener, router(state)).into_future());
    Ok(LoopbackServer {
        addr,
        task: Some(task),
    })
}

async fn fake_upstream_handler(
    State(state): State<Arc<FakeUpstreamState>>,
    request: Request<Body>,
) -> Response<Body> {
    let path = request.uri().path().to_owned();
    let is_probe = request
        .headers()
        .get("x-llm-guard-proxy-probe")
        .is_some_and(|value| value == "local-recovery");
    let Ok(body) = axum::body::to_bytes(request.into_body(), 64 * 1024).await else {
        return status_response(StatusCode::BAD_REQUEST);
    };
    if is_probe {
        state.probe_count.fetch_add(1, Ordering::AcqRel);
        push_role(&state, "recovery_probe");
        return json_body(
            r#"{"choices":[{"message":{"role":"assistant","content":"ready"},"finish_reason":"stop"}]}"#,
        );
    }

    let payload = match parse_business_payload(&path, &body) {
        Ok(payload) => payload,
        Err(error) => {
            *state
                .validation_error
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(error);
            return status_response(StatusCode::BAD_REQUEST);
        }
    };

    let business_number = state.business_count.fetch_add(1, Ordering::AcqRel) + 1;
    push_role(&state, "business");
    if business_number == 1 {
        *state
            .first_business_payload
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(payload);
        return Response::builder()
            .status(StatusCode::OK)
            .header(CONTENT_TYPE, "text/event-stream")
            .body(Body::from_stream(futures_util::stream::pending::<
                Result<Bytes, Infallible>,
            >()))
            .unwrap_or_else(|_| status_response(StatusCode::INTERNAL_SERVER_ERROR));
    }
    if state.arm == Arm::Committed {
        return status_response(StatusCode::INTERNAL_SERVER_ERROR);
    }

    let same_payload = state
        .first_business_payload
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .is_some_and(|first| first == &payload);
    state.same_payload.store(same_payload, Ordering::Release);
    sse_body(SUCCESS_SSE)
}

fn parse_business_payload(path: &str, body: &Bytes) -> Result<BusinessPayload, String> {
    require(path == "/v1/chat/completions", "business_endpoint")?;
    let payload: BusinessPayload =
        serde_json::from_slice(body).map_err(|_| failure("business_payload_schema"))?;
    require(payload.model == "self-test", "business_model")?;
    require(payload.stream, "business_stream")?;
    require(
        payload.messages
            == [BusinessMessage {
                role: String::from("user"),
                content: String::from("transport check"),
            }],
        "business_messages",
    )?;
    Ok(payload)
}

fn push_role(state: &FakeUpstreamState, role: &str) {
    state
        .ordered_roles
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(String::from(role));
}

fn status_response(status: StatusCode) -> Response<Body> {
    Response::builder()
        .status(status)
        .body(Body::empty())
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

fn json_body(body: &'static str) -> Response<Body> {
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .unwrap_or_else(|_| status_response(StatusCode::INTERNAL_SERVER_ERROR))
}

fn sse_body(body: &'static str) -> Response<Body> {
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/event-stream")
        .body(Body::from(body))
        .unwrap_or_else(|_| status_response(StatusCode::INTERNAL_SERVER_ERROR))
}

#[derive(Debug)]
struct ClientReceipt {
    first_byte_wait_ms: u64,
    outcome: ClientOutcome,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ClientOutcome {
    ControlSuccess,
    CommittedBlocked,
}

async fn run_internal_client(
    proxy_addr: SocketAddr,
    arm: Arm,
    context: &Context,
) -> Result<ClientReceipt, String> {
    let client = build_http_client().map_err(|_| failure("build_internal_client"))?;
    let response = client
        .post(format!("http://{proxy_addr}/v1/chat/completions"))
        .header(CONTENT_TYPE, "application/json")
        .body(REQUEST_BODY)
        .send()
        .await
        .map_err(|_| failure("client_request"))?;
    if response.status() != reqwest::StatusCode::OK {
        return Err(failure("unexpected_downstream_status"));
    }

    let started = Instant::now();
    let mut stream = response.bytes_stream();
    let mut first_byte_wait_ms = 0;
    let mut heartbeat = false;
    let mut done = false;
    let mut terminal_error = false;
    let mut saw_chunk = false;
    loop {
        let next = timeout(CLIENT_CHUNK_DEADLINE, stream.next())
            .await
            .map_err(|_| failure("client_chunk_timeout"))?;
        let Some(chunk) = next else {
            break;
        };
        let chunk = chunk.map_err(|_| failure("client_body"))?;
        if !saw_chunk && !chunk.is_empty() {
            saw_chunk = true;
            first_byte_wait_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        }
        if chunk.as_ref() == HEARTBEAT {
            heartbeat = true;
            if arm == Arm::Committed {
                context.acknowledge_client_read()?;
            }
        }
        done |= contains(&chunk, b"data: [DONE]");
        terminal_error |= contains(&chunk, b"event: error");
    }
    let outcome = match (arm, heartbeat, done, terminal_error) {
        (Arm::Control, false, true, false) => ClientOutcome::ControlSuccess,
        (Arm::Committed, true, false, true) => ClientOutcome::CommittedBlocked,
        _ => return Err(failure("client_evidence")),
    };
    Ok(ClientReceipt {
        first_byte_wait_ms,
        outcome,
    })
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

fn validate_receipt(receipt: &Receipt) -> Result<(), String> {
    require(
        receipt.self_test == "post-await-no-replay",
        "self_test_name",
    )?;
    require(receipt.status == "passed", "status")?;
    validate_control(receipt)?;
    validate_committed(receipt)?;
    require(
        receipt.same_payload_across_arms,
        "cross_arm_payload_changed",
    )
}

fn validate_control(receipt: &Receipt) -> Result<(), String> {
    require(
        receipt.control.request_claims == 1,
        "control_request_claims",
    )?;
    require(
        receipt.control.rejected_request_claims == 0,
        "control_rejected_request_claims",
    )?;
    require(
        receipt.control.guard_business_count == 2,
        "control_guard_business_count",
    )?;
    require(
        receipt.control.guard_probe_count == 1,
        "control_guard_probe_count",
    )?;
    require(
        receipt.control.ordered_roles == ["business", "recovery_probe", "business"],
        "control_role_order",
    )?;
    require(
        receipt.control.business_count == 2,
        "control_business_count",
    )?;
    require(receipt.control.probe_count == 1, "control_probe_count")?;
    require(
        receipt.control.guard_business_count == receipt.control.business_count,
        "control_business_count_mismatch",
    )?;
    require(
        receipt.control.guard_probe_count == receipt.control.probe_count,
        "control_probe_count_mismatch",
    )?;
    require(
        receipt.control.fault.same_payload,
        "control_payload_changed",
    )?;
    require(receipt.control.fault.first_chunk_stall, "control_no_stall")?;
    require(receipt.control.first_byte_wait_ms > 0, "control_no_wait")?;
    require(
        !receipt.control.client.client_observed_heartbeat,
        "control_early_emit",
    )?;
    require(
        receipt.control.client.done_observed,
        "control_client_no_success",
    )?;
    require(
        !receipt.control.client.terminal_error_observed,
        "control_terminal_error",
    )?;
    require(receipt.control.completion.eof_observed, "control_no_eof")?;
    require(
        receipt.control.cleanup.loopback_only,
        "control_non_loopback",
    )?;
    require(receipt.control.cleanup.cleanup_complete, "control_cleanup")
}

fn validate_committed(receipt: &Receipt) -> Result<(), String> {
    require(
        receipt.committed.request_claims == 1,
        "committed_request_claims",
    )?;
    require(
        receipt.committed.rejected_request_claims == 0,
        "committed_rejected_request_claims",
    )?;
    require(
        receipt.committed.guard_business_count == 1,
        "committed_guard_business_count",
    )?;
    require(
        receipt.committed.guard_probe_count == 0,
        "committed_guard_probe_count",
    )?;
    require(
        receipt.committed.ordered_roles == ["business"],
        "committed_role_order",
    )?;
    require(
        receipt.committed.business_count == 1,
        "committed_business_count",
    )?;
    require(receipt.committed.probe_count == 0, "committed_probe_count")?;
    require(
        receipt.committed.guard_business_count == receipt.committed.business_count,
        "committed_business_count_mismatch",
    )?;
    require(
        receipt.committed.guard_probe_count == receipt.committed.probe_count,
        "committed_probe_count_mismatch",
    )?;
    require(
        receipt.committed.fault.same_payload,
        "committed_payload_state",
    )?;
    require(
        receipt.committed.fault.first_chunk_stall,
        "committed_no_stall",
    )?;
    require(
        receipt.committed.first_byte_wait_ms > 0,
        "committed_no_wait",
    )?;
    require(
        receipt.committed.client.client_observed_heartbeat,
        "committed_client_no_heartbeat",
    )?;
    require(
        !receipt.committed.client.done_observed,
        "committed_unsafe_success",
    )?;
    require(
        receipt.committed.client.terminal_error_observed,
        "committed_no_terminal_error",
    )?;
    require(
        receipt.committed.completion.eof_observed,
        "committed_no_eof",
    )?;
    require(
        receipt.committed.completion.post_await_committed,
        "committed_gate_not_observed",
    )?;
    require(
        receipt.committed.cleanup.loopback_only,
        "committed_non_loopback",
    )?;
    require(
        receipt.committed.cleanup.cleanup_complete,
        "committed_cleanup",
    )?;
    let phases = &receipt.committed.phases;
    require(
        0 < phases.pre_await_gate
            && phases.pre_await_gate < phases.recovery_await_entered_ns
            && phases.recovery_await_entered_ns < phases.body_emitted_ns
            && phases.body_emitted_ns < phases.client_ack_ns
            && phases.client_ack_ns < phases.recovery_await_completed_ns
            && phases.recovery_await_completed_ns < phases.post_await_committed_ns,
        "committed_causal_order",
    )
}

#[cfg(test)]
fn validate_receipt_value(value: serde_json::Value) -> Result<(), String> {
    let receipt: Receipt = serde_json::from_value(value).map_err(|_| failure("receipt_schema"))?;
    validate_receipt(&receipt)
}

fn require(condition: bool, code: &'static str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(failure(code))
    }
}

fn failure(code: &str) -> String {
    format!("post_await_no_replay_self_test_failed:{code}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_claim_is_single_owner_and_second_claim_is_rejected() {
        let context = Context::new(Arm::Control);

        assert!(context.claim_request().is_ok());
        assert_eq!(context.request_claim_counts(), (1, 0));
        assert!(context.claim_request().is_err());
        assert_eq!(context.request_claim_counts(), (1, 1));
    }

    #[test]
    fn guard_attempt_accounting_is_independent_from_upstream_observation() {
        let context = Context::new(Arm::Control);

        context.mark_guard_business_attempt();
        context.mark_guard_business_attempt();
        context.mark_guard_probe_dispatch();

        assert_eq!(context.guard_attempt_counts(), (2, 1));
    }

    #[test]
    fn business_payload_is_structured_and_order_independent() {
        let expected = parse_business_payload(
            "/v1/chat/completions",
            &Bytes::from_static(REQUEST_BODY.as_bytes()),
        )
        .expect("canonical payload should pass");
        let reordered = Bytes::from_static(
            br#"{"stream":true,"messages":[{"content":"transport check","role":"user"}],"model":"self-test"}"#,
        );
        assert_eq!(
            parse_business_payload("/v1/chat/completions", &reordered)
                .expect("JSON key order must not matter"),
            expected
        );

        for (path, body) in [
            ("/v1/other", REQUEST_BODY.as_bytes()),
            ("/v1/chat/completions", br#"{"arbitrary":true}"#),
            (
                "/v1/chat/completions",
                br#"{"model":"self-test","messages":[{"role":"user","content":"wrong"}],"stream":true}"#,
            ),
        ] {
            assert!(parse_business_payload(path, &Bytes::copy_from_slice(body)).is_err());
        }
    }

    #[tokio::test]
    async fn committed_fake_upstream_rejects_second_business_payload() {
        let state = Arc::new(FakeUpstreamState::new(Arm::Committed));

        let first = fake_upstream_handler(State(Arc::clone(&state)), business_request()).await;
        let second = fake_upstream_handler(State(Arc::clone(&state)), business_request()).await;

        assert_eq!(first.status(), StatusCode::OK);
        assert_eq!(second.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(state.business_count.load(Ordering::Acquire), 2);
    }

    #[test]
    fn receipt_schema_rejects_sensitive_and_unmeasured_fields() {
        let valid = valid_receipt_value();
        for unsupported in [
            "external_network_used",
            "config_or_credentials_read",
            "persistence_used",
        ] {
            assert!(valid.get(unsupported).is_none());
        }

        for field in ["prompt", "body", "headers", "diagnostic", "hash"] {
            let top_level = mutate(&valid, |value| {
                value
                    .as_object_mut()
                    .expect("receipt object")
                    .insert(String::from(field), serde_json::json!("forbidden"));
            });
            assert!(validate_receipt_value(top_level).is_err());

            let nested = mutate(&valid, |value| {
                value["control"]
                    .as_object_mut()
                    .expect("control object")
                    .insert(String::from(field), serde_json::json!("forbidden"));
            });
            assert!(validate_receipt_value(nested).is_err());
        }
    }

    #[test]
    fn receipt_validation_rejects_missing_or_self_confirming_evidence() {
        let valid = valid_receipt_value();
        assert!(validate_receipt_value(valid.clone()).is_ok());

        let mut mutations = Vec::new();
        mutations.push(mutate(&valid, |value| {
            value
                .as_object_mut()
                .expect("receipt object")
                .remove("control");
        }));
        mutations.push(mutate(&valid, |value| {
            value["control"]["probe_count"] = 0.into();
        }));
        mutations.push(mutate(&valid, |value| {
            value["control"]["rejected_request_claims"] = 1.into();
        }));
        mutations.push(mutate(&valid, |value| {
            value["control"]["guard_business_count"] = 1.into();
        }));
        mutations.push(mutate(&valid, |value| {
            value["control"]["guard_probe_count"] = 0.into();
        }));
        mutations.push(mutate(&valid, |value| {
            value["control"]["business_count"] = 1.into();
        }));
        mutations.push(mutate(&valid, |value| {
            value["control"]["same_payload"] = false.into();
        }));
        mutations.push(mutate(&valid, |value| {
            value["control"]["done_observed"] = false.into();
        }));
        mutations.push(mutate(&valid, |value| {
            value["committed"]["phases"]["body_emitted_ns"] = 1.into();
        }));
        mutations.push(mutate(&valid, |value| {
            value["committed"]["probe_count"] = 1.into();
        }));
        mutations.push(mutate(&valid, |value| {
            value["committed"]["request_claims"] = 2.into();
        }));
        mutations.push(mutate(&valid, |value| {
            value["committed"]["guard_business_count"] = 2.into();
        }));
        mutations.push(mutate(&valid, |value| {
            value["committed"]["guard_probe_count"] = 1.into();
        }));
        mutations.push(mutate(&valid, |value| {
            value["committed"]["business_count"] = 2.into();
        }));
        mutations.push(mutate(&valid, |value| {
            value["committed"]["ordered_roles"] = serde_json::json!(["business", "business"]);
        }));
        mutations.push(mutate(&valid, |value| {
            value["committed"]["cleanup_complete"] = false.into();
        }));
        mutations.push(mutate(&valid, |value| {
            value
                .as_object_mut()
                .expect("receipt object")
                .insert(String::from("raw_prompt"), String::from("forbidden").into());
        }));

        for mutation in mutations {
            assert!(validate_receipt_value(mutation).is_err());
        }
    }

    fn mutate(
        value: &serde_json::Value,
        change: impl FnOnce(&mut serde_json::Value),
    ) -> serde_json::Value {
        let mut value = value.clone();
        change(&mut value);
        value
    }

    fn business_request() -> Request<Body> {
        Request::builder()
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(REQUEST_BODY))
            .expect("self-test request")
    }

    fn valid_receipt_value() -> serde_json::Value {
        serde_json::json!({
            "self_test": "post-await-no-replay",
            "status": "passed",
            "control": {
                "ordered_roles": ["business", "recovery_probe", "business"],
                "request_claims": 1,
                "rejected_request_claims": 0,
                "guard_business_count": 2,
                "guard_probe_count": 1,
                "business_count": 2,
                "probe_count": 1,
                "same_payload": true,
                "first_chunk_stall": true,
                "first_byte_wait_ms": 50,
                "client_observed_heartbeat": false,
                "done_observed": true,
                "terminal_error_observed": false,
                "eof_observed": true,
                "post_await_committed": false,
                "phases": {
                    "pre_await_gate_ns": 1,
                    "recovery_await_entered_ns": 2,
                    "body_emitted_ns": 0,
                    "client_ack_ns": 0,
                    "recovery_await_completed_ns": 3,
                    "post_await_committed_ns": 0
                },
                "loopback_only": true,
                "cleanup_complete": true
            },
            "committed": {
                "ordered_roles": ["business"],
                "request_claims": 1,
                "rejected_request_claims": 0,
                "guard_business_count": 1,
                "guard_probe_count": 0,
                "business_count": 1,
                "probe_count": 0,
                "same_payload": true,
                "first_chunk_stall": true,
                "first_byte_wait_ms": 50,
                "client_observed_heartbeat": true,
                "done_observed": false,
                "terminal_error_observed": true,
                "eof_observed": true,
                "post_await_committed": true,
                "phases": {
                    "pre_await_gate_ns": 1,
                    "recovery_await_entered_ns": 2,
                    "body_emitted_ns": 3,
                    "client_ack_ns": 4,
                    "recovery_await_completed_ns": 5,
                    "post_await_committed_ns": 6
                },
                "loopback_only": true,
                "cleanup_complete": true
            },
            "same_payload_across_arms": true
        })
    }
}
