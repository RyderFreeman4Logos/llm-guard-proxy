use super::*;

async fn unit_fixture(wait_for_cancel: bool) -> (FakeUpstream, ProxyFixture, PathBuf, PathBuf) {
    let (upstream, proxy, runtime) = fixture(vec![String::from("/usr/bin/true")]).await;
    fs::set_permissions(&runtime, fs::Permissions::from_mode(0o700)).expect("private runtime");
    let mut config = proxy.state.config.snapshot().expect("config");
    config.guardian.escalation_unit = Some(String::from("gb10-stack-recovery@.service"));
    proxy
        .state
        .config
        .apply_reloadable(&config)
        .expect("unit policy");
    let control = runtime.join("systemctl-fixture");
    fs::write(&control, format!(r"#!/usr/bin/python3
import hashlib,json,os,pathlib,sys,time
root=pathlib.Path({runtime:?})
unit=sys.argv[-1]
run=root/'recovery'/unit.split('@')[1].removesuffix('.service')
inv='a'*32
if 'show' in sys.argv:
    started=(run/'ready.json').exists() and not (run/'stop-job-finished').exists()
    print('Id='+unit)
    print('InvocationID='+(inv if started else ''))
    print('ActiveState='+('activating' if started else 'inactive'))
    print('SubState='+('start' if started else 'dead'))
    print('Job=0')
    print('Result=success')
    sys.exit(0)
def record(name,data):
    tmp=run/(name+'.tmp')
    with open(tmp,'x') as f:
        os.chmod(tmp,0o600)
        json.dump(data,f)
    os.rename(tmp,run/name)
if 'stop' in sys.argv:
    (run/'stop').touch()
    time.sleep(.2) # wait-start already returned; ExecStopPost is still running
    (run/'stop-job-finished').touch()
    sys.exit(0)
assert '--wait' in sys.argv
manifest=(run/'manifest.json').read_bytes()
m=json.loads(manifest)
gate=dict(version=1,receipt_id=m['receipt_id'],manifest_sha256=hashlib.sha256(manifest).hexdigest(),invocation_id=inv)
record('ready.json',gate)
end=time.monotonic()+8
while not (run/'permit.json').exists():
    if (run/'stop').exists(): sys.exit(1)
    if time.monotonic()>end: sys.exit(1)
    time.sleep(.01)
assert json.loads((run/'permit.json').read_bytes())==gate
(root/'target.v1').unlink()
(root/'meminfo').write_text('MemAvailable: 2097152 kB\n')
(root/'owned').touch()
if {wait_for_cancel}:
    while not (run/'stop').exists():
        if time.monotonic()>end: sys.exit(1)
        time.sleep(.01)
    record('result.json',dict(gate,status='stopped'))
    sys.exit(1)
record('result.json',dict(gate,status='succeeded'))
", runtime=runtime.to_string_lossy(), wait_for_cancel=if wait_for_cancel { "True" } else { "False" })).expect("control fixture");
    fs::set_permissions(&control, fs::Permissions::from_mode(0o700)).expect("executable");
    (upstream, proxy, runtime, control)
}

#[tokio::test]
async fn systemd_gate_accepts_owned_memory_recovery_and_registration_turnover() {
    let (_upstream, proxy, runtime, control) = unit_fixture(false).await;
    let (request, _receiver) =
        make_request(proxy.state.config.clone(), &runtime, Duration::from_secs(5));
    let profile = proxy
        .state
        .config
        .snapshot()
        .expect("config")
        .default_upstream_profile();
    let receipt = crate::proxy::recovery_receipt::Context::new(
        proxy.state.store.clone(),
        &profile,
        Arc::clone(&proxy.state.persistence_tasks),
    );
    let outcome = proxy
        .state
        .guardian_unit_action_using(&request, &receipt, control.to_str().expect("path"))
        .await;
    assert_eq!(outcome, RecoveryOutcome::Succeeded);
    assert!(runtime.join("owned").exists());
    assert!(!runtime.join("target.v1").exists());
}

#[tokio::test]
async fn systemd_gate_cancellation_waits_for_owned_unit_cleanup() {
    let (_upstream, proxy, runtime, control) = unit_fixture(true).await;
    let (request, _receiver) =
        make_request(proxy.state.config.clone(), &runtime, Duration::from_secs(5));
    let authority = Arc::clone(&request.authority);
    let profile = proxy
        .state
        .config
        .snapshot()
        .expect("config")
        .default_upstream_profile();
    let receipt = crate::proxy::recovery_receipt::Context::new(
        proxy.state.store.clone(),
        &profile,
        Arc::clone(&proxy.state.persistence_tasks),
    );
    let action =
        proxy
            .state
            .guardian_unit_action_using(&request, &receipt, control.to_str().expect("path"));
    let cancel = async {
        tokio::time::timeout(Duration::from_secs(3), async {
            while !runtime.join("owned").exists() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("owned action reached");
        authority.cancel();
    };
    let (outcome, ()) = tokio::join!(action, cancel);
    assert_eq!(outcome, RecoveryOutcome::Cancelled);
    let runs = fs::read_dir(runtime.join("recovery")).expect("runs");
    assert!(runs.into_iter().all(|entry| {
        entry
            .expect("run")
            .path()
            .join("stop-job-finished")
            .exists()
    }));
    assert_eq!(
        receipt
            .physical_owner
            .load(std::sync::atomic::Ordering::Acquire),
        0
    );
}

#[tokio::test]
async fn systemd_gate_missing_cleanup_receipt_is_unconfirmed() {
    let (_upstream, proxy, runtime, control) = unit_fixture(true).await;
    let script = fs::read_to_string(&control).expect("fixture").replace(
        "record('result.json',dict(gate,status='stopped'))",
        "pass # no confirmed cleanup",
    );
    fs::write(&control, script).expect("failed cleanup fixture");
    let (request, _receiver) =
        make_request(proxy.state.config.clone(), &runtime, Duration::from_secs(5));
    let authority = Arc::clone(&request.authority);
    let profile = proxy
        .state
        .config
        .snapshot()
        .expect("config")
        .default_upstream_profile();
    let receipt = crate::proxy::recovery_receipt::Context::new(
        proxy.state.store.clone(),
        &profile,
        Arc::clone(&proxy.state.persistence_tasks),
    );
    let action =
        proxy
            .state
            .guardian_unit_action_using(&request, &receipt, control.to_str().expect("path"));
    let cancel = async {
        tokio::time::timeout(Duration::from_secs(3), async {
            while !runtime.join("owned").exists() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("owned action");
        authority.cancel();
    };
    let (outcome, ()) = tokio::join!(action, cancel);
    assert_eq!(outcome, RecoveryOutcome::Unconfirmed);
}

#[tokio::test]
async fn systemd_gate_rejects_wrong_result_and_waits_for_post_stop() {
    for (replacement, expected) in [
        (
            "dict(gate,status='succeeded',invocation_id='b'*32)",
            RecoveryOutcome::Unconfirmed,
        ),
        ("dict(gate,status='succeeded')", RecoveryOutcome::Succeeded),
    ] {
        let (_upstream, proxy, runtime, control) = unit_fixture(false).await;
        let script = fs::read_to_string(&control)
            .expect("fixture")
            .replace("dict(gate,status='succeeded')", replacement)
            + "time.sleep(.15)\n(root/'post-stop-finished').touch()\n";
        fs::write(&control, script).expect("terminal fixture");
        let (request, _receiver) =
            make_request(proxy.state.config.clone(), &runtime, Duration::from_secs(5));
        let profile = proxy
            .state
            .config
            .snapshot()
            .expect("config")
            .default_upstream_profile();
        let receipt = crate::proxy::recovery_receipt::Context::new(
            proxy.state.store.clone(),
            &profile,
            Arc::clone(&proxy.state.persistence_tasks),
        );
        let outcome = proxy
            .state
            .guardian_unit_action_using(&request, &receipt, control.to_str().expect("path"))
            .await;
        assert_eq!(outcome, expected);
        assert!(
            runtime.join("post-stop-finished").exists(),
            "terminal ACK must follow the retained waiter's completion"
        );
    }
}

#[tokio::test]
async fn systemd_gate_rejects_symlink_ready_without_publishing_permit() {
    let (_upstream, proxy, runtime, control) = unit_fixture(false).await;
    let script = fs::read_to_string(&control).expect("fixture").replace(
        "record('ready.json',gate)",
        "record('foreign.json',gate)\nos.symlink(run/'foreign.json',run/'ready.json')",
    );
    fs::write(&control, script).expect("hostile gate");
    let (request, _receiver) =
        make_request(proxy.state.config.clone(), &runtime, Duration::from_secs(5));
    let profile = proxy
        .state
        .config
        .snapshot()
        .expect("config")
        .default_upstream_profile();
    let receipt = crate::proxy::recovery_receipt::Context::new(
        proxy.state.store.clone(),
        &profile,
        Arc::clone(&proxy.state.persistence_tasks),
    );
    let outcome = proxy
        .state
        .guardian_unit_action_using(&request, &receipt, control.to_str().expect("path"))
        .await;
    assert_eq!(outcome, RecoveryOutcome::Unconfirmed);
    assert!(!runtime.join("owned").exists());
    assert!(
        fs::read_dir(runtime.join("recovery"))
            .expect("runs")
            .all(|entry| !entry.expect("run").path().join("permit.json").exists())
    );
}
