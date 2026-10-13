//! Systemd transactions stay behind an immutable startup gate until admission is final.
use super::{
    ProxyState, RecoveryProcessGuard, configure_recovery_command,
    guardian_recovery::cancelled_outcome, recovery_receipt, terminate_timed_out_recovery_child,
};
use llm_guard_proxy_host_guardian::monitor::tier2::{RecoveryOutcome, RecoveryRequest};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    os::{
        fd::AsRawFd,
        unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt},
    },
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::{process::Command, time::Instant};

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Gate {
    version: u8,
    receipt_id: String,
    manifest_sha256: String,
    invocation_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultRecord {
    version: u8,
    receipt_id: String,
    manifest_sha256: String,
    invocation_id: String,
    status: String,
}

struct Run {
    directory: File,
    unit: String,
    receipt: String,
    digest: String,
    control: String,
}

fn invalid(message: &str) -> io::Error {
    io::Error::other(message)
}

fn hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn private_directory(path: &Path) -> io::Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    let meta = file.metadata()?;
    if meta.uid() != nix::unistd::Uid::effective().as_raw() || meta.mode() & 0o077 != 0 {
        return Err(invalid("unsafe recovery directory"));
    }
    Ok(file)
}

fn fd_path(directory: &File, name: &str) -> PathBuf {
    PathBuf::from(format!("/proc/self/fd/{}/{name}", directory.as_raw_fd()))
}

fn write_once(directory: &File, name: &str, bytes: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(fd_path(directory, name))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    directory.sync_all()
}

fn publish_permit(directory: &File, bytes: &[u8]) -> io::Result<()> {
    write_once(directory, "permit.pending", bytes)?;
    // No fallible operation may follow visibility: the caller transfers ownership on Ok.
    nix::fcntl::renameat2(
        Some(directory.as_raw_fd()),
        "permit.pending",
        Some(directory.as_raw_fd()),
        "permit.json",
        nix::fcntl::RenameFlags::RENAME_NOREPLACE,
    )
    .map_err(io::Error::from)
}

fn read_record(directory: &File, name: &str) -> io::Result<Vec<u8>> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(fd_path(directory, name))?;
    let meta = file.metadata()?;
    if !meta.is_file()
        || meta.uid() != nix::unistd::Uid::effective().as_raw()
        || meta.mode() & 0o777 != 0o600
        || meta.nlink() != 1
        || meta.len() > 4096
    {
        return Err(invalid("unsafe recovery record"));
    }
    let mut bytes = Vec::new();
    file.take(4097).read_to_end(&mut bytes)?;
    if bytes.len() > 4096 {
        return Err(invalid("oversized recovery record"));
    }
    Ok(bytes)
}

fn boottime_ms() -> io::Result<u64> {
    let value = nix::time::clock_gettime(nix::time::ClockId::CLOCK_BOOTTIME)?;
    let secs = u64::try_from(value.tv_sec()).map_err(|_| invalid("negative boottime"))?;
    let nanos = u64::try_from(value.tv_nsec()).map_err(|_| invalid("negative boottime"))?;
    secs.checked_mul(1000)
        .and_then(|v| v.checked_add(nanos / 1_000_000))
        .ok_or_else(|| invalid("boottime overflow"))
}

async fn output(program: &str, arguments: &[&str], deadline: Instant) -> io::Result<Vec<u8>> {
    output_with_budget(program, arguments, deadline, Duration::from_secs(5)).await
}

async fn output_with_budget(
    program: &str,
    arguments: &[&str],
    deadline: Instant,
    budget: Duration,
) -> io::Result<Vec<u8>> {
    let mut command = Command::new(program);
    command
        .args(arguments)
        .kill_on_drop(true)
        .stdin(std::process::Stdio::null());
    let result = tokio::time::timeout_at(deadline.min(Instant::now() + budget), command.output())
        .await
        .map_err(|_| invalid("control command timeout"))??;
    if !result.status.success() || result.stdout.len() > 8192 {
        return Err(invalid("control command failed"));
    }
    Ok(result.stdout)
}

async fn show(
    control: &str,
    unit: &str,
    deadline: Instant,
) -> io::Result<BTreeMap<String, String>> {
    let bytes = output(
        control,
        &[
            "--user",
            "show",
            "--property=Id,InvocationID,ActiveState,SubState,Job,Result",
            "--",
            unit,
        ],
        deadline,
    )
    .await?;
    let text = std::str::from_utf8(&bytes).map_err(|_| invalid("invalid unit properties"))?;
    let mut fields = BTreeMap::new();
    for line in text.lines() {
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| invalid("malformed unit property"))?;
        if fields.insert(key.to_owned(), value.to_owned()).is_some() {
            return Err(invalid("duplicate unit property"));
        }
    }
    if fields.len() != 6 || fields.get("Id").map(String::as_str) != Some(unit) {
        return Err(invalid("wrong unit identity"));
    }
    Ok(fields)
}

fn settled(fields: &BTreeMap<String, String>) -> bool {
    matches!(
        fields.get("ActiveState").map(String::as_str),
        Some("inactive" | "failed")
    ) && matches!(fields.get("Job").map(String::as_str), Some("0" | ""))
}

impl Run {
    async fn prepare(request: &RecoveryRequest, receipt: &str, control: &str) -> io::Result<Self> {
        if receipt.len() > 120
            || receipt.is_empty()
            || !receipt
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
        {
            return Err(invalid("invalid receipt identifier"));
        }
        let template = request
            .authority
            .recovery_unit()
            .ok_or_else(|| invalid("missing template"))?;
        let unit = template.replace("@.service", &format!("@{receipt}.service"));
        let runtime = private_directory(request.authority.runtime_dir())?;
        let recovery = fd_path(&runtime, "recovery");
        match fs::DirBuilder::new().mode(0o700).create(&recovery) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
        let root = private_directory(&recovery)?;
        let path = fd_path(&root, receipt);
        fs::DirBuilder::new().mode(0o700).create(&path)?;
        let directory = private_directory(&path)?;
        let (device, inode, cid) = request.authority.target_identity()?;
        let remaining = u64::try_from(
            request
                .authority
                .deadline
                .saturating_duration_since(std::time::Instant::now())
                .as_millis(),
        )
        .map_err(|_| invalid("deadline overflow"))?;
        let manifest = serde_json::to_vec(&serde_json::json!({
            "version": 1, "receipt_id": receipt, "unit_name": unit,
            "boot_id": fs::read_to_string("/proc/sys/kernel/random/boot_id")?.trim(),
            "config_revision": request.authority.config_revision(),
            "target_device": device, "target_inode": inode, "container_id": cid,
            "control_group": request.authority.control_group(), "threshold_bytes": request.threshold_bytes,
            "deadline_boottime_ms": boottime_ms()?.checked_add(remaining).ok_or_else(|| invalid("deadline overflow"))?,
        })).map_err(io::Error::other)?;
        write_once(&directory, "manifest.json", &manifest)?;
        // Hash the held directory's immutable manifest, not a re-resolved run pathname.
        let manifest_path = fd_path(&directory, "manifest.json");
        // The child cannot use CLOEXEC descriptors; /proc/<parent>/fd retains the held object.
        let path = manifest_path
            .to_string_lossy()
            .replace("/proc/self/", &format!("/proc/{}/", std::process::id()));
        let bytes = output(
            "/usr/bin/sha256sum",
            &[&path],
            request.authority.deadline.into(),
        )
        .await?;
        let digest = std::str::from_utf8(&bytes)
            .ok()
            .and_then(|v| v.split_whitespace().next())
            .filter(|v| hex(v, 64))
            .ok_or_else(|| invalid("invalid manifest digest"))?
            .to_owned();
        Ok(Self {
            directory,
            unit,
            receipt: receipt.to_owned(),
            digest,
            control: control.to_owned(),
        })
    }

    fn ready(&self) -> io::Result<(Gate, Vec<u8>)> {
        let bytes = read_record(&self.directory, "ready.json")?;
        let gate: Gate = serde_json::from_slice(&bytes).map_err(io::Error::other)?;
        if gate.version != 1
            || gate.receipt_id != self.receipt
            || gate.manifest_sha256 != self.digest
            || !hex(&gate.invocation_id, 32)
        {
            return Err(invalid("wrong startup gate identity"));
        }
        Ok((gate, bytes))
    }

    fn result(&self, gate: &Gate, status: &str) -> bool {
        read_record(&self.directory, "result.json")
            .ok()
            .and_then(|bytes| serde_json::from_slice::<ResultRecord>(&bytes).ok())
            .is_some_and(|record| {
                record.version == gate.version
                    && record.receipt_id == gate.receipt_id
                    && record.manifest_sha256 == gate.manifest_sha256
                    && record.invocation_id == gate.invocation_id
                    && record.status == status
            })
    }

    async fn settle(
        &self,
        gate: Option<&Gate>,
        owned: bool,
        waiter: &mut RecoveryProcessGuard,
        waiter_done: bool,
    ) -> bool {
        let deadline = Instant::now() + Duration::from_secs(240);
        let ready = self.ready().ok();
        let gate = gate.or_else(|| ready.as_ref().map(|(gate, _)| gate));
        let Some(gate) = gate else {
            let _cleanup = terminate_timed_out_recovery_child(waiter).await;
            return false;
        };
        let Ok(before) = show(&self.control, &self.unit, deadline).await else {
            return false;
        };
        let invocation = before.get("InvocationID").map(String::as_str);
        if invocation != Some(gate.invocation_id.as_str())
            && !(invocation == Some("") && settled(&before))
        {
            return false;
        }
        // Start-job cancellation can release --wait before ExecStopPost. Own a blocking stop job too.
        if output_with_budget(
            &self.control,
            &["--user", "stop", "--", &self.unit],
            deadline,
            Duration::from_secs(240),
        )
        .await
        .is_err()
        {
            return false;
        }
        if !waiter_done
            && !matches!(
                tokio::time::timeout_at(deadline, waiter.wait()).await,
                Ok(Ok(_))
            )
        {
            let _cleanup = terminate_timed_out_recovery_child(waiter).await;
            return false;
        }
        let Ok(after) = show(&self.control, &self.unit, deadline).await else {
            return false;
        };
        let invocation = after.get("InvocationID").map(String::as_str);
        settled(&after)
            && matches!(invocation, Some(value) if value.is_empty() || value == gate.invocation_id)
            && (!owned || self.result(gate, "stopped") || self.result(gate, "succeeded"))
    }
}

impl ProxyState {
    pub(super) async fn guardian_unit_action(
        &self,
        request: &RecoveryRequest,
        receipt: &recovery_receipt::Context,
    ) -> RecoveryOutcome {
        self.guardian_unit_action_using(request, receipt, "/usr/bin/systemctl")
            .await
    }

    pub(super) async fn guardian_unit_action_using(
        &self,
        request: &RecoveryRequest,
        receipt: &recovery_receipt::Context,
        control: &str,
    ) -> RecoveryOutcome {
        let Ok(run) = Run::prepare(request, receipt.id(), control).await else {
            return RecoveryOutcome::NotAdmitted;
        };
        let deadline: Instant = request.authority.deadline.into();
        let Ok(before) = show(control, &run.unit, deadline).await else {
            return RecoveryOutcome::NotAdmitted;
        };
        if !settled(&before)
            || before
                .get("InvocationID")
                .is_none_or(|value| !value.is_empty())
            || request.authority.dispatch(|| ()).is_none()
        {
            return RecoveryOutcome::NotAdmitted;
        }
        // --wait holds systemd's RefUnit and waits through ExecStopPost, not merely the start job.
        let mut command = Command::new(control);
        command
            .args(["--user", "--wait", "start", "--", &run.unit])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        configure_recovery_command(&mut command);
        let Some(Ok(child)) = request.authority.dispatch(|| command.spawn()) else {
            return RecoveryOutcome::NotAdmitted;
        };
        let mut waiter =
            RecoveryProcessGuard::new_owned(child, std::sync::Arc::clone(&receipt.physical_owner));
        let mut gate = None;
        let mut waiter_done = false;
        let outcome = loop {
            if self.shutdown.is_shutting_down() || request.authority.dispatch(|| ()).is_none() {
                break cancelled_outcome(request);
            }
            if gate.is_none() {
                match run.ready() {
                    Ok((ready, bytes)) => {
                        let Ok(fields) = show(control, &run.unit, deadline).await else {
                            break RecoveryOutcome::Failed;
                        };
                        if fields.get("InvocationID") != Some(&ready.invocation_id)
                            || fields.get("ActiveState").map(String::as_str) != Some("activating")
                        {
                            break RecoveryOutcome::Failed;
                        }
                        gate = Some(ready);
                        if !matches!(
                            request
                                .authority
                                .grant_unit(|| publish_permit(&run.directory, &bytes)),
                            Some(Ok(()))
                        ) {
                            break cancelled_outcome(request);
                        }
                    }
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                    Err(_) => break RecoveryOutcome::Failed,
                }
            }
            tokio::select! {
                biased;
                status = waiter.wait() => {
                    waiter_done = status.is_ok();
                    if status.is_ok_and(|status| status.success())
                        && request.authority.dispatch(|| ()).is_some()
                        && !self.shutdown.is_shutting_down()
                        && gate.as_ref().is_some_and(|gate| run.result(gate, "succeeded")) {
                        return RecoveryOutcome::Succeeded;
                    }
                    break RecoveryOutcome::Failed;
                },
                () = tokio::time::sleep(Duration::from_millis(50)) => {},
            }
        };
        if run
            .settle(
                gate.as_ref(),
                request.authority.owns_unit(),
                &mut waiter,
                waiter_done,
            )
            .await
        {
            outcome
        } else {
            RecoveryOutcome::Unconfirmed
        }
    }
}
