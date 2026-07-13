use std::{
    process::{Child, Command},
    sync::{
        Arc, Barrier,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use super::{InFlightLimiter, WorkflowExecutionTaskError, run_workflow_execution};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_caller_keeps_workflow_capacity_until_blocking_execution_finishes() {
    let limiter = Arc::new(InFlightLimiter::default());
    let (started_tx, started_rx) = mpsc::sync_channel(1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);

    let first_limiter = Arc::clone(&limiter);
    let first = tokio::spawn(async move {
        run_workflow_execution(first_limiter, 1, move |lease| {
            let child = Command::new("/bin/sleep")
                .arg("30")
                .spawn()
                .expect("admitted workflow process should spawn");
            let _process = LeasedTestChild {
                child,
                _lease: lease,
            };
            started_tx.send(()).expect("start should be observed");
            release_rx.recv().expect("execution should be released");
        })
        .await
    });
    started_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("first blocking execution should start");

    first.abort();
    assert!(
        first
            .await
            .expect_err("aborted caller should be cancelled")
            .is_cancelled()
    );

    let second_entered = Arc::new(AtomicBool::new(false));
    let second_marker = Arc::clone(&second_entered);
    let second = run_workflow_execution(Arc::clone(&limiter), 1, move |_lease| {
        second_marker.store(true, Ordering::SeqCst);
    })
    .await;

    assert!(matches!(
        second,
        Err(WorkflowExecutionTaskError::AtCapacity {
            max_in_flight_executions: 1
        })
    ));
    assert!(
        !second_entered.load(Ordering::SeqCst),
        "saturated admission must not enter or queue a blocking closure"
    );

    release_tx.send(()).expect("first execution should release");
    let deadline = Instant::now() + Duration::from_secs(1);
    while limiter.snapshot_counts().active != 0 {
        assert!(
            Instant::now() < deadline,
            "workflow capacity did not recover"
        );
        tokio::task::yield_now().await;
    }

    run_workflow_execution(limiter, 1, |_lease| ())
        .await
        .expect("capacity should recover after execution cleanup");
}

struct LeasedTestChild {
    child: Child,
    _lease: crate::workflow_execution::WorkflowExecutionLease,
}

impl Drop for LeasedTestChild {
    fn drop(&mut self) {
        let _killed = self.child.kill();
        let _reaped = self.child.wait();
    }
}

#[test]
fn workflow_admission_failure_respects_guard_fail_policy() {
    let failure = WorkflowExecutionTaskError::AtCapacity {
        max_in_flight_executions: 1,
    };

    assert!(matches!(
        super::guard_outcome_after_workflow_task(Err(failure.clone()), true),
        llm_guard_proxy_core::GuardOutcome::Block { .. }
    ));
    assert!(matches!(
        super::guard_outcome_after_workflow_task(Err(failure), false),
        llm_guard_proxy_core::GuardOutcome::Allow
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 16)]
async fn sixteen_callers_overlap_without_exceeding_workflow_execution_bound() {
    const CALLERS: usize = 16;
    const LIMIT: usize = 4;
    let limiter = Arc::new(InFlightLimiter::default());
    let barrier = Arc::new(tokio::sync::Barrier::new(CALLERS + 1));
    let entered = Arc::new(AtomicUsize::new(0));
    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));
    let release = Arc::new(Barrier::new(LIMIT + 1));
    let mut callers = Vec::new();

    for _ in 0..CALLERS {
        let limiter = Arc::clone(&limiter);
        let barrier = Arc::clone(&barrier);
        let entered = Arc::clone(&entered);
        let active = Arc::clone(&active);
        let max_active = Arc::clone(&max_active);
        let release = Arc::clone(&release);
        callers.push(tokio::spawn(async move {
            barrier.wait().await;
            run_workflow_execution(limiter, LIMIT, move |_lease| {
                entered.fetch_add(1, Ordering::SeqCst);
                let now_active = active.fetch_add(1, Ordering::SeqCst) + 1;
                max_active.fetch_max(now_active, Ordering::SeqCst);
                release.wait();
                active.fetch_sub(1, Ordering::SeqCst);
            })
            .await
        }));
    }

    barrier.wait().await;
    let deadline = Instant::now() + Duration::from_secs(1);
    while entered.load(Ordering::SeqCst) < LIMIT {
        assert!(
            Instant::now() < deadline,
            "all admitted closures should enter"
        );
        tokio::task::yield_now().await;
    }
    release.wait();

    let mut rejected = 0;
    for caller in callers {
        let result = caller.await.expect("overlap caller should join");
        rejected += usize::from(matches!(
            result,
            Err(WorkflowExecutionTaskError::AtCapacity { .. })
        ));
    }
    assert!(rejected >= CALLERS - LIMIT);
    assert_eq!(entered.load(Ordering::SeqCst), LIMIT);
    assert!(max_active.load(Ordering::SeqCst) <= LIMIT);
}
