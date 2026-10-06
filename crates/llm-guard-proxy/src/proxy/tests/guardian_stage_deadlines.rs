use super::*;
use std::{future::Future, task::Poll};

#[tokio::test]
async fn tier2_denies_ordinary_unconfirmed_physical_cleanup() {
    let (_upstream, proxy, runtime) = fixture(vec![String::from("/usr/bin/true")]).await;
    let profile = proxy
        .state
        .config
        .snapshot()
        .expect("config")
        .default_upstream_profile();
    let coordinator = proxy.state.local_recovery.coordinator_for(&profile.name);
    let _failed = super::super::recovery_physical_fence::external_reap_recovery(&coordinator).await;
    let (request, receiver) =
        make_request(proxy.state.config.clone(), &runtime, Duration::from_secs(5));
    assert_eq!(
        run(&proxy, request, receiver).await,
        RecoveryOutcome::NotAdmitted,
        "Guardian cannot bypass an ordinary unconfirmed physical owner"
    );
}

#[tokio::test]
async fn tier2_admits_after_ordinary_timeout_physically_settles() {
    let (_upstream, proxy, runtime) = fixture(vec![String::from("/usr/bin/true")]).await;
    let profile = proxy
        .state
        .config
        .snapshot()
        .expect("config")
        .default_upstream_profile();
    let coordinator = proxy.state.local_recovery.coordinator_for(&profile.name);
    let mut policy = super::super::LocalRecoveryPolicy::from_config(&profile.local_recovery);
    policy.restart_command = vec![String::from("/usr/bin/sleep"), String::from("30")];
    policy.restart_timeout = Duration::from_millis(100);
    let failed = super::super::run_local_recovery_for_profile(
        &policy,
        &coordinator,
        super::super::Client::new(),
        profile.base_url.clone(),
        super::super::LocalRecoveryCause::UpstreamStall,
        None,
    )
    .await;
    assert_eq!(
        failed["upstream_stall_recovery_timeout_cleanup_status"],
        "terminated_after_kill"
    );
    tokio::time::sleep(policy.cooldown).await;
    let (request, receiver) =
        make_request(proxy.state.config.clone(), &runtime, Duration::from_secs(5));
    assert_eq!(
        run(&proxy, request, receiver).await,
        RecoveryOutcome::Succeeded,
        "confirmed ordinary timeout must permit Guardian admission"
    );
}

async fn poll_once<F: Future>(future: std::pin::Pin<&mut F>) -> Poll<F::Output> {
    let mut future = future;
    std::future::poll_fn(|cx| Poll::Ready(future.as_mut().poll(cx))).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tier2_action_rejects_success_first_polled_after_stage_deadline() {
    let (_upstream, proxy, runtime) = fixture(vec![String::from("/usr/bin/true")]).await;
    let entered = runtime.join("entered");
    let release = runtime.join("release");
    let exited = runtime.join("exited");
    let profile = proxy
        .state
        .config
        .snapshot()
        .expect("config")
        .default_upstream_profile();
    let mut policy = super::super::LocalRecoveryPolicy::from_config(&profile.local_recovery);
    policy.restart_timeout = Duration::from_millis(100);
    policy.restart_command = vec![
        String::from("/usr/bin/python3"),
        String::from("-c"),
        String::from(
            "import pathlib,sys,time; a,b,c=map(pathlib.Path,sys.argv[1:]); a.touch();\nwhile not b.exists(): time.sleep(.001)\nc.touch()",
        ),
        entered.display().to_string(),
        release.display().to_string(),
        exited.display().to_string(),
    ];
    let (request, _receiver) =
        make_request(proxy.state.config.clone(), &runtime, Duration::from_secs(5));
    let receipt = super::super::test_recovery_receipt_context();
    let mut action = Box::pin(
        proxy
            .state
            .guardian_owned_action(&request, &profile, &policy, &receipt),
    );
    let watchdog = tokio::time::Instant::now() + Duration::from_secs(2);
    while !entered.exists() {
        assert!(
            poll_once(action.as_mut()).await.is_pending(),
            "must reach held real child"
        );
        assert!(
            tokio::time::Instant::now() < watchdog,
            "child entry bounded"
        );
        tokio::task::yield_now().await;
    }
    fs::write(&release, b"go").expect("release");
    // The child becomes ready while the owning future is deliberately not polled.
    tokio::time::sleep(Duration::from_millis(160)).await;
    assert!(exited.exists(), "real successful child reached exit");
    assert_eq!(
        action.await,
        RecoveryOutcome::TimedOut,
        "late-ready action success must not advance to readiness"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tier2_readiness_rejects_success_first_polled_after_stage_deadline() {
    let (_upstream, proxy, runtime, mut barrier) = terminal_failure_fixture().await;
    let profile = proxy
        .state
        .config
        .snapshot()
        .expect("config")
        .default_upstream_profile();
    let mut policy = super::super::LocalRecoveryPolicy::from_config(&profile.local_recovery);
    policy.readiness_deadline = Duration::from_millis(100);
    let (request, _receiver) =
        make_request(proxy.state.config.clone(), &runtime, Duration::from_secs(5));
    let receipt = super::super::test_recovery_receipt_context();
    let mut action = Box::pin(
        proxy
            .state
            .guardian_owned_action(&request, &profile, &policy, &receipt),
    );
    let watchdog = tokio::time::Instant::now() + Duration::from_secs(2);
    while !*barrier.entered.borrow_and_update() {
        assert!(
            poll_once(action.as_mut()).await.is_pending(),
            "must reach held real HTTP probe"
        );
        assert!(
            tokio::time::Instant::now() < watchdog,
            "probe entry bounded"
        );
        tokio::task::yield_now().await;
    }
    barrier.release.send_replace(true);
    tokio::time::sleep(Duration::from_millis(160)).await;
    assert_eq!(
        action.await,
        RecoveryOutcome::TimedOut,
        "late-ready HTTP success must not release recovery"
    );
    barrier.server.abort();
    let _ = barrier.server.await;
}
