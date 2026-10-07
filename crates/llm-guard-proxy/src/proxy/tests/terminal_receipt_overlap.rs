//! Real wire/control experiment at the shared durable writer mutex.
use super::*;

type TraceEvent = (String, String, u128);
type Arm = (StatusCode, Vec<AttemptChainRow>, Vec<TraceEvent>, usize);
type ArmResponse = Result<
    Result<Result<StatusCode, reqwest::Error>, tokio::task::JoinError>,
    tokio::time::error::Elapsed,
>;

struct OverlapUpstream {
    addr: std::net::SocketAddr,
    server: tokio::task::JoinHandle<std::io::Result<()>>,
    wire: mpsc::Receiver<ObservedRequest>,
    ready_tx: tokio::sync::watch::Sender<bool>,
}

struct OverlapFixture {
    root: std::path::PathBuf,
    proxy: ProxyFixture,
    fake_server: tokio::task::JoinHandle<std::io::Result<()>>,
    wire: mpsc::Receiver<ObservedRequest>,
    trace: Arc<Mutex<Vec<TraceEvent>>>,
    events_rx: mpsc::UnboundedReceiver<String>,
    release_tx: std::sync::mpsc::Sender<()>,
}

async fn spawn_overlap_upstream() -> OverlapUpstream {
    let (sender, wire) = mpsc::channel(16);
    let fake_state = FakeUpstreamState {
        sender,
        changing_model_len: Arc::new(AtomicU64::new(128_000)),
        attempt_counts: Arc::new(Mutex::new(HashMap::new())),
        models_body: None,
        models_status: StatusCode::OK,
        models_label: "models",
        models_delay: None,
        pre_response_delay: None,
        rerank_status: None,
        deepinfra_response: None,
    };
    let (ready_tx, ready_rx) = tokio::sync::watch::channel(false);
    let count = Arc::new(AtomicU64::new(0));
    let app =
        Router::new().fallback(move |request: Request<Body>| {
            let state = fake_state.clone();
            let count = Arc::clone(&count);
            let mut ready = ready_rx.clone();
            async move {
                if request.uri().query().is_some_and(|query| {
                    query.contains("test=shielded-429-then-two-503-then-success")
                }) && count.fetch_add(1, Ordering::SeqCst) == 2
                {
                    // The third wire request waits for the actual first terminal boundary.
                    let _ = timeout(Duration::from_secs(2), ready.wait_for(|value| *value)).await;
                }
                fake_upstream_handler(State(state), request).await
            }
        });
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("fake bind");
    let addr = listener.local_addr().expect("fake address");
    let server = tokio::spawn(async move { axum::serve(listener, app).await });
    OverlapUpstream {
        addr,
        server,
        wire,
        ready_tx,
    }
}

fn install_overlap_hook(
    proxy: &ProxyFixture,
    overlap: bool,
    ready_tx: tokio::sync::watch::Sender<bool>,
    start: std::time::Instant,
) -> (
    Arc<Mutex<Vec<TraceEvent>>>,
    mpsc::UnboundedReceiver<String>,
    std::sync::mpsc::Sender<()>,
) {
    let trace = Arc::new(Mutex::new(Vec::new()));
    let observed = Arc::clone(&trace);
    let (events_tx, events_rx) = mpsc::unbounded_channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let release_rx = Mutex::new(release_rx);
    let first_terminal = AtomicBool::new(true);
    proxy
        .store
        .set_recovery_receipt_test_hook(move |stage, id| {
            observed.lock().expect("trace lock").push((
                stage.to_owned(),
                id.to_owned(),
                start.elapsed().as_micros(),
            ));
            let _ = events_tx.send(stage.to_owned());
            if stage == "terminal_locked" && first_terminal.swap(false, Ordering::SeqCst) && overlap
            {
                ready_tx.send_replace(true);
                // Finite test-only stall, after acquiring the real store mutex.
                let released = release_rx
                    .lock()
                    .expect("release lock")
                    .recv_timeout(Duration::from_secs(2))
                    .is_ok();
                observed.lock().expect("trace lock").push((
                    if released {
                        "terminal_release"
                    } else {
                        "barrier_timeout"
                    }
                    .to_owned(),
                    id.to_owned(),
                    start.elapsed().as_micros(),
                ));
            } else if stage == "terminal_ack" && !overlap {
                ready_tx.send_replace(true);
            }
        });
    (trace, events_rx, release_tx)
}

async fn setup_overlap_arm(overlap: bool) -> OverlapFixture {
    let root = unique_test_dir("terminal-receipt-overlap");
    fs::create_dir_all(&root).expect("owned recovery root");
    let marker = root.join("restart-ran");
    let OverlapUpstream {
        addr,
        server: fake_server,
        wire,
        ready_tx,
    } = spawn_overlap_upstream().await;
    let proxy = ProxyFixture::spawn_with_options(
        &format!("http://{addr}/v1"),
        true,
        AppConfig::default().server.max_in_flight_requests,
        &residual_guard_polish::multi_recovery_max_attempts_two_config(&marker),
    )
    .await;
    let (trace, events_rx, release_tx) =
        install_overlap_hook(&proxy, overlap, ready_tx, std::time::Instant::now());
    OverlapFixture {
        root,
        proxy,
        fake_server,
        wire,
        trace,
        events_rx,
        release_tx,
    }
}

async fn run_overlap_request(fixture: &mut OverlapFixture, overlap: bool) -> ArmResponse {
    let client = fixture.proxy.client.clone();
    let url = format!(
        "{}/v1/chat/completions?test=shielded-429-then-two-503-then-success",
        fixture.proxy.base_url
    );
    let mut request = tokio::spawn(async move {
        let response = client
            .post(url)
            .json(&json!({"model":"test-chat","messages":[]}))
            .send()
            .await?;
        let status = response.status();
        response.bytes().await?;
        Ok::<_, reqwest::Error>(status)
    });
    if overlap {
        // Release at actual mutex contention, not worker dispatch or a fixed delay.
        // The finite guard only fails the fixture; it cannot authorize a recovery.
        let _ = timeout(Duration::from_secs(2), async {
            while let Some(stage) = fixture.events_rx.recv().await {
                if stage == "record_busy" {
                    break;
                }
            }
        })
        .await;
    }
    let _ = fixture.release_tx.send(());
    let response = timeout(Duration::from_secs(8), &mut request).await;
    if response.is_err() {
        request.abort();
        let _ = request.await;
    }
    response
}

async fn finish_overlap_arm(fixture: OverlapFixture, response: ArmResponse, overlap: bool) -> Arm {
    let OverlapFixture {
        root,
        mut proxy,
        fake_server,
        mut wire,
        trace,
        ..
    } = fixture;
    // Release, settle tracked writers and join both servers BEFORE any behavior assertion.
    let drained = timeout(Duration::from_secs(3), proxy.state.flush_persistence()).await;
    proxy.server.task.abort();
    let _ = (&mut proxy.server.task).await;
    fake_server.abort();
    let _ = fake_server.await;
    let attempts = read_attempt_chain_rows(&proxy.sqlite_path);
    let receipts: i64 = rusqlite::Connection::open(&proxy.sqlite_path)
        .expect("settled receipt DB")
        .query_row(
            "SELECT COUNT(*) FROM local_recovery_receipts WHERE outcome = 'succeeded'",
            [],
            |row| row.get(0),
        )
        .expect("real durable terminal rows");
    let mut wire_count = 0;
    while wire.try_recv().is_ok() {
        wire_count += 1;
    }
    let stages = trace.lock().expect("trace lock").clone();
    drop(proxy);
    remove_dir_all(&root);
    assert!(drained.is_ok(), "tracked persistence must settle");
    assert!(
        !stages.iter().any(|event| event.0 == "barrier_timeout"),
        "bounded barrier must be explicitly released"
    );
    if overlap {
        assert!(
            stages.iter().any(|event| event.0 == "record_busy"),
            "second writer must witness real mutex contention: {stages:?}"
        );
        let locked = stages
            .iter()
            .find(|event| event.0 == "terminal_locked")
            .expect("actual writer mutex");
        let second = stages
            .iter()
            .find(|event| event.0 == "record_busy")
            .expect("actual contended mutex");
        let release = stages
            .iter()
            .find(|event| event.0 == "terminal_release")
            .expect("barrier release");
        assert!(
            locked.2 < second.2 && second.2 < release.2,
            "required overlap witnessed: {stages:?}"
        );
    }
    let status = response
        .expect("bounded request")
        .expect("joined request")
        .expect("wire response");
    if status == StatusCode::OK {
        assert_eq!(receipts, 2, "both real recovery receipts durably settled");
    }
    (status, attempts, stages, wire_count)
}

async fn overlap_arm(overlap: bool) -> Arm {
    let mut fixture = setup_overlap_arm(overlap).await;
    let response = run_overlap_request(&mut fixture, overlap).await;
    finish_overlap_arm(fixture, response, overlap).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn second_required_recovery_survives_first_terminal_audit_overlap() {
    let control = overlap_arm(false).await;
    let overlap = overlap_arm(true).await;
    for (name, arm) in [("control", &control), ("overlap", &overlap)] {
        eprintln!(
            "E1 {name} status={} attempts={} wire={} stages={:?} recovery={:?}",
            arm.0,
            arm.1.len(),
            arm.3,
            arm.2,
            arm.1
                .iter()
                .map(|row| row
                    .response_metadata
                    .as_object()
                    .expect("metadata")
                    .iter()
                    .filter(|(key, _)| key.starts_with("local_recovery_"))
                    .collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );
    }
    assert_eq!(control.0, StatusCode::OK, "settled control");
    assert_eq!(control.1.len(), 4, "identical control wire policy");
    assert_eq!(
        overlap.0,
        StatusCode::OK,
        "second required recovery must not be vetoed by first terminal audit: {overlap:?}"
    );
    assert_eq!(overlap.1.len(), 4);
    assert_eq!(overlap.3, control.3);
    for arm in [&control, &overlap] {
        for index in [1, 2] {
            assert_eq!(
                arm.1[index].response_metadata["local_recovery_status"],
                "succeeded"
            );
            assert!(
                arm.1[index]
                    .response_metadata
                    .get("local_recovery_receipt_id")
                    .is_some()
            );
            let id = arm.1[index].response_metadata["local_recovery_receipt_id"]
                .as_str()
                .expect("receipt id");
            let at = |stage| {
                arm.2
                    .iter()
                    .find(|event| event.0 == stage && event.1 == id)
                    .expect("actual stage")
                    .2
            };
            assert!(at("record_sql_end") < at("record_ack"));
            assert!(
                at("record_ack") - at("record_submit") < 250_000,
                "unchanged durable acknowledgment budget"
            );
        }
    }
}
