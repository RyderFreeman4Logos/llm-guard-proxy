use super::*;

#[tokio::test]
async fn legacy_abort_cannot_readmit_before_owned_task_is_polled() {
    let coordinator = Arc::new(UpstreamStallRecoveryCoordinator::default());
    let config = UpstreamStallConfig::default();
    let mut legacy_policy = UpstreamStallPolicy::from_config(&config);
    legacy_policy.recovery_command = vec![String::from("/usr/bin/true")];
    let mut recovery = Box::pin(run_upstream_stall_recovery(&legacy_policy, &coordinator));
    assert!(poll_once(recovery.as_mut()).await.is_pending());
    let episode = coordinator
        .state
        .lock()
        .await
        .active_recovery_episode_id
        .expect("legacy task registered");
    assert!(
        abort_local_recovery_episode(
            &coordinator,
            episode,
            BTreeMap::from([(
                String::from("local_recovery_status"),
                String::from("cancelled")
            )])
        )
        .await
    );
    let policy = recovery_policy(vec![String::from("/usr/bin/true")]);
    let denied = local_recovery_admission_failure(
        &policy,
        &mut *coordinator.state.lock().await,
        Instant::now(),
    );
    let _result = recovery.await;
    assert!(
        denied.is_some(),
        "legacy task admission cannot precede physical ownership registration"
    );
}

#[tokio::test]
async fn ordinary_restart_and_episode_timeouts_settle_before_readmission() {
    for episode_timeout in [None, Some(Duration::from_millis(100))] {
        let root = unique_test_dir("recovery-timeout-settlement");
        fs::create_dir_all(&root).expect("fixture");
        let _cleanup = TestDirectoryCleanup::new(&root);
        let pid_file = root.join("pid");
        let mut policy = recovery_policy(vec![
            String::from("/usr/bin/python3"),
            String::from("-c"),
            String::from(
                "import os,pathlib,sys,time; p=pathlib.Path(sys.argv[1]); t=p.with_suffix('.new'); t.write_text(str(os.getpid())); t.replace(p); time.sleep(30)",
            ),
            pid_file.display().to_string(),
        ]);
        policy.restart_timeout = Duration::from_millis(500);
        let coordinator = Arc::new(UpstreamStallRecoveryCoordinator::default());
        let upstream = FakeUpstream::spawn().await;
        let context = test_recovery_receipt_context();
        let mut recovery = Box::pin(run_local_recovery_for_profile_observing(
            &policy,
            &coordinator,
            Client::new(),
            upstream.base_url.clone(),
            LocalRecoveryCause::UpstreamStall,
            LocalRecoveryRunOptions {
                receipt: context,
                episode_timeout: None,
                caller_timeout: None,
                recovery_episode_observer: None,
                downstream_commit_signal: None,
                post_await_self_test: None,
            },
        ));
        let owner = Arc::clone(&coordinator.state.lock().await.physical_recovery_owners);
        let receipt = test_recovery_receipt_context().physical_owner(owner);
        let ran = AtomicBool::new(false);
        let mut restart = Box::pin(run_local_recovery_restart_command(
            &policy, &ran, &receipt, None,
        ));
        let deadline = Instant::now() + Duration::from_secs(2);
        while !pid_file.exists() {
            let pending = if episode_timeout.is_some() {
                poll_once(restart.as_mut()).await.is_pending()
            } else {
                poll_once(recovery.as_mut()).await.is_pending()
            };
            assert!(pending, "fixture must reach a real held child");
            assert!(
                Instant::now() < deadline,
                "child entered before bounded timeout"
            );
            tokio::task::yield_now().await;
        }
        let pid: i32 = fs::read_to_string(&pid_file)
            .expect("pid")
            .parse()
            .expect("pid integer");
        let identity = LinuxProcessIdentity::capture(u32::try_from(pid).expect("positive pid"))
            .expect("live owned identity");
        if let Some(duration) = episode_timeout {
            // Establish the physical child before starting the narrow drop clock.
            // This is the same real command-future Drop used by the episode timer,
            // without accidentally testing pre-action SQL admission instead.
            assert!(timeout(duration, restart.as_mut()).await.is_err());
            drop(restart);
            drop(recovery);
        } else {
            let result = recovery.await;
            assert_eq!(result["local_recovery_status"], "timeout_killed");
            drop(restart);
        }
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            let mut state = coordinator.state.lock().await;
            if state.physical_recovery_owners.load(Ordering::Acquire) == 0 && !state.running {
                break;
            }
            assert!(
                local_recovery_admission_failure(&policy, &mut state, Instant::now()).is_some(),
                "timeout cannot admit another physical owner before settlement"
            );
            drop(state);
            assert!(
                Instant::now() < deadline,
                "owned task and group settlement bounded"
            );
            tokio::task::yield_now().await;
        }
        assert_process_reaped(identity).await;
        let ready_policy = recovery_policy(vec![String::from("/usr/bin/true")]);
        let next = run_local_recovery_for_profile(
            &ready_policy,
            &coordinator,
            Client::new(),
            upstream.base_url.clone(),
            LocalRecoveryCause::UpstreamStall,
            None,
        )
        .await;
        assert_eq!(
            next["local_recovery_status"], "succeeded",
            "settled timeout permits real ordinary recovery"
        );
    }
}

use std::{future::Future, task::Poll};

async fn poll_once<F: Future>(future: std::pin::Pin<&mut F>) -> Poll<F::Output> {
    let mut future = future;
    std::future::poll_fn(|cx| Poll::Ready(future.as_mut().poll(cx))).await
}

fn recovery_policy(command: Vec<String>) -> LocalRecoveryPolicy {
    let mut policy = LocalRecoveryPolicy::from_config(&LocalRecoveryConfig::default());
    policy.enabled = true;
    policy.restart_command = command;
    policy.cooldown = Duration::ZERO;
    policy.max_per_window = 10;
    policy.restart_timeout = Duration::from_secs(2);
    policy
}

#[tokio::test]
async fn ordinary_abort_cannot_readmit_before_owned_task_has_dropped() {
    let coordinator = Arc::new(UpstreamStallRecoveryCoordinator::default());
    let policy = recovery_policy(vec![String::from("/usr/bin/sleep"), String::from("30")]);
    let mut recovery = Box::pin(run_local_recovery_for_profile(
        &policy,
        &coordinator,
        Client::new(),
        String::from("http://127.0.0.1:1/v1"),
        LocalRecoveryCause::UpstreamStall,
        None,
    ));
    assert!(poll_once(recovery.as_mut()).await.is_pending());
    let episode = coordinator
        .state
        .lock()
        .await
        .active_recovery_episode_id
        .expect("owned task registered");
    assert!(
        abort_local_recovery_episode(
            &coordinator,
            episode,
            BTreeMap::from([(
                String::from("local_recovery_status"),
                String::from("cancelled")
            )])
        )
        .await
    );
    // Current-thread runtime: abort was requested but the owned task has not been polled again.
    let denied = local_recovery_admission_failure(
        &policy,
        &mut *coordinator.state.lock().await,
        Instant::now(),
    );
    let result = recovery.await;
    assert_eq!(result["local_recovery_status"], "cancelled");
    assert!(
        denied.is_some(),
        "abort publication must not release physical/task ownership"
    );
}

pub(super) async fn external_reap_recovery(
    coordinator: &Arc<UpstreamStallRecoveryCoordinator>,
) -> BTreeMap<String, String> {
    let root = unique_test_dir("recovery-external-reap");
    fs::create_dir_all(&root).expect("fixture");
    let _cleanup = TestDirectoryCleanup::new(&root);
    let pid_file = root.join("pid");
    let policy = recovery_policy(vec![
        String::from("/usr/bin/python3"),
        String::from("-c"),
        String::from(
            "import os,pathlib,sys,time; p=pathlib.Path(sys.argv[1]); t=p.with_suffix('.new'); t.write_text(str(os.getpid())); t.replace(p); time.sleep(30)",
        ),
        pid_file.display().to_string(),
    ]);
    let context = test_recovery_receipt_context();
    let mut recovery = Box::pin(run_local_recovery_for_profile_observing(
        &policy,
        coordinator,
        Client::new(),
        String::from("http://127.0.0.1:1/v1"),
        LocalRecoveryCause::UpstreamStall,
        LocalRecoveryRunOptions {
            receipt: context,
            episode_timeout: None,
            caller_timeout: None,
            recovery_episode_observer: None,
            downstream_commit_signal: None,
            post_await_self_test: None,
        },
    ));
    let deadline = Instant::now() + Duration::from_secs(2);
    while !pid_file.exists() {
        assert!(poll_once(recovery.as_mut()).await.is_pending());
        assert!(Instant::now() < deadline, "real child start bounded");
        tokio::task::yield_now().await;
    }
    let pid: i32 = fs::read_to_string(&pid_file)
        .expect("pid")
        .parse()
        .expect("pid integer");
    kill(Pid::from_raw(pid), Signal::SIGKILL).expect("kill owned test child");
    // Safe error witness: reap only this owned, killable child before the guard's
    // next observation, making waitid report ECHILD instead of fabricating D-state.
    nix::sys::wait::waitpid(Pid::from_raw(pid), None).expect("externally reap fixture child");
    let metadata = recovery.await;
    assert_eq!(
        metadata["local_recovery_restart_status"], "wait_failed",
        "must reach actual wait-error decision"
    );
    metadata
}

#[tokio::test]
async fn ordinary_wait_error_keeps_shared_admission_fail_closed() {
    let coordinator = Arc::new(UpstreamStallRecoveryCoordinator::default());
    let _metadata = external_reap_recovery(&coordinator).await;
    let policy = recovery_policy(vec![String::from("/usr/bin/true")]);
    let result = run_local_recovery_for_profile(
        &policy,
        &coordinator,
        Client::new(),
        String::from("http://127.0.0.1:1/v1"),
        LocalRecoveryCause::UpstreamStall,
        None,
    )
    .await;
    assert_eq!(
        result["local_recovery_status"], "cleanup_unconfirmed",
        "actual ordinary re-admission must fail closed"
    );
}
