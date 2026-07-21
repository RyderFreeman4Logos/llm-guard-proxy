use std::{sync::Arc, time::Duration};

use tokio::time::timeout;

use super::*;

const FLUSH_TIMEOUT: Duration = Duration::from_millis(25);
const FLUSH_COMPLETION_TIMEOUT: Duration = Duration::from_millis(500);
const TEST_COMPLETION_TIMEOUT: Duration = Duration::from_secs(2);

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn persistence_flush_returns_when_spawn_blocking_task_exceeds_shutdown_deadline() {
    let tasks = Arc::new(PersistenceTasks::default());
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();

    tasks.spawn_blocking(move || {
        started_tx
            .send(())
            .expect("stalled persistence task startup receiver should remain open");
        release_rx
            .recv()
            .expect("stalled persistence task should be released after flush returns");
    });
    started_rx
        .recv_timeout(TEST_COMPLETION_TIMEOUT)
        .expect("stalled persistence task should start before flushing");

    timeout(FLUSH_COMPLETION_TIMEOUT, tasks.flush(FLUSH_TIMEOUT))
        .await
        .expect("flush must return after its configured deadline");
    assert_eq!(
        tasks.in_flight.load(Ordering::SeqCst),
        1,
        "flush must return while the stalled persistence task remains in flight"
    );

    release_tx
        .send(())
        .expect("stalled persistence task should still be waiting after flush returns");
    timeout(TEST_COMPLETION_TIMEOUT, tasks.flush(FLUSH_TIMEOUT))
        .await
        .expect("released persistence task should drain");
}
