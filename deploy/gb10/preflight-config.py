#!/usr/bin/env python3
"""Fail-closed semantic preflight for the reviewed GB10 configuration."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import TypeAlias

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)

AEON_PROFILE = "aeon-chat"
AEON_RESTART_COMMAND = [
    "systemctl",
    "--user",
    "restart",
    "vllm-aeon-27b-dflash-n12.service",
]
RECOVERY_COMPLETION_GUARD_MS = 1_000


def load_config(path: Path) -> dict[str, JsonValue]:
    """Parse the project TOML dialect while preserving JSON readiness bodies."""
    source = path.read_text(encoding="utf-8")
    normalized_lines: list[str] = []
    for line in source.splitlines():
        if line.lstrip().startswith("readiness_body ="):
            key, separator, raw_body = line.partition("=")
            if not separator:
                raise ValueError("readiness_body assignment is malformed")
            json.loads(raw_body.strip())
            line = f"{key}= {json.dumps(raw_body.strip())}"
        normalized_lines.append(line)
    config = tomllib.loads("\n".join(normalized_lines))
    routes = [_table(config, "upstream")]
    profiles = config.get("upstreams")
    if isinstance(profiles, list):
        routes.extend(profile for profile in profiles if isinstance(profile, dict))
    for route in routes:
        recovery = _table(route, "local_recovery")
        readiness_body = recovery.get("readiness_body")
        if isinstance(readiness_body, str):
            recovery["readiness_body"] = json.loads(readiness_body)
    return config


def _table(parent: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    value = parent.get(key)
    return value if isinstance(value, dict) else {}


def _positive_int(table: dict[str, JsonValue], key: str) -> int | None:
    value = table.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _recovery_errors(label: str, recovery: dict[str, JsonValue]) -> list[str]:
    errors: list[str] = []
    if recovery.get("enabled") is not True:
        errors.append(f"{label}.local_recovery.enabled must be true")
    if recovery.get("trigger_on_request_deadline") is not False:
        errors.append(f"{label}.local_recovery.trigger_on_request_deadline must be false")
    if recovery.get("restart_command") != AEON_RESTART_COMMAND:
        errors.append(f"{label}.local_recovery.restart_command is not the reviewed AEON unit")
    if recovery.get("max_attempts_per_request") != 1:
        errors.append(f"{label}.local_recovery.max_attempts_per_request must equal 1")
    for field in (
        "restart_timeout_ms",
        "readiness_request_timeout_ms",
        "readiness_deadline_ms",
        "readiness_interval_ms",
        "cooldown_ms",
        "budget_window_ms",
        "max_per_window",
    ):
        if _positive_int(recovery, field) is None:
            errors.append(f"{label}.local_recovery.{field} must be a positive integer")
    readiness_body = recovery.get("readiness_body")
    if not isinstance(readiness_body, dict) or readiness_body.get("model") != "aeon-ultimate":
        errors.append(f"{label}.local_recovery.readiness_body must probe aeon-ultimate")
    return errors


def minimum_downstream_idle_timeout_ms(config: dict[str, JsonValue]) -> int | None:
    """Return the strict byte-silent bound including recovery handoff and replay."""
    retry = _table(config, "retry")
    request_deadline = _positive_int(retry, "request_deadline_ms")
    default_upstream = _table(config, "upstream")
    default_recovery = _table(default_upstream, "local_recovery")
    default_request_timeout = _positive_int(default_upstream, "request_timeout_ms")
    profiles = config.get("upstreams")
    if request_deadline is None or default_request_timeout is None or not isinstance(profiles, list):
        return None
    aeon = next(
        (
            profile
            for profile in profiles
            if isinstance(profile, dict) and profile.get("name") == AEON_PROFILE
        ),
        None,
    )
    if not isinstance(aeon, dict):
        return None
    aeon_recovery = _table(aeon, "local_recovery")
    aeon_request_timeout = _positive_int(aeon, "request_timeout_ms")
    route_bounds: list[int] = []
    for request_timeout, recovery in (
        (default_request_timeout, default_recovery),
        (aeon_request_timeout, aeon_recovery),
    ):
        restart_timeout = _positive_int(recovery, "restart_timeout_ms")
        readiness_deadline = _positive_int(recovery, "readiness_deadline_ms")
        if request_timeout is None or restart_timeout is None or readiness_deadline is None:
            return None
        # A final physical replay may consume request_timeout after restart,
        # readiness, and the runtime's completion-publication handoff guard.
        route_bounds.append(
            request_timeout
            + restart_timeout
            + readiness_deadline
            + RECOVERY_COMPLETION_GUARD_MS
        )
    return request_deadline + max(route_bounds)


def validate_snapshot(
    config: dict[str, JsonValue], downstream_idle_timeout_ms: int
) -> tuple[list[str], int | None]:
    errors: list[str] = []
    if "guard_workflows" in config:
        errors.append("guard_workflows must remain inactive in the reviewed snapshot")

    default_upstream = _table(config, "upstream")
    errors.extend(
        _recovery_errors("upstream", _table(default_upstream, "local_recovery"))
    )
    profiles = config.get("upstreams")
    aeon = (
        next(
            (
                profile
                for profile in profiles
                if isinstance(profile, dict) and profile.get("name") == AEON_PROFILE
            ),
            None,
        )
        if isinstance(profiles, list)
        else None
    )
    if not isinstance(aeon, dict):
        errors.append(f"required profile {AEON_PROFILE!r} is missing")
    else:
        errors.extend(
            _recovery_errors(
                f"upstreams[{AEON_PROFILE!r}]", _table(aeon, "local_recovery")
            )
        )

    retry = _table(config, "retry")
    maximum_retry_after = _positive_int(retry, "max_retry_after_secs")
    if maximum_retry_after is None or maximum_retry_after > 300:
        errors.append("retry.max_retry_after_secs must be in 1..=300")

    minimum_idle_timeout = minimum_downstream_idle_timeout_ms(config)
    if minimum_idle_timeout is None:
        errors.append("cannot derive a bounded downstream idle-timeout requirement")
    elif downstream_idle_timeout_ms <= minimum_idle_timeout:
        errors.append(
            "downstream idle timeout must be strictly greater than "
            f"{minimum_idle_timeout} ms"
        )
    return errors, minimum_idle_timeout


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--downstream-idle-timeout-ms", type=int, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report validation without installing or changing any files",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        config = load_config(args.config)
    except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
        print(f"result=error reason=config-unreadable detail={error}", file=sys.stderr)
        return 2
    if args.downstream_idle_timeout_ms <= 0:
        print("result=error reason=idle-timeout-not-positive", file=sys.stderr)
        return 2

    errors, minimum_idle_timeout = validate_snapshot(
        config, args.downstream_idle_timeout_ms
    )
    if errors:
        for error in errors:
            print(f"result=error reason={error}", file=sys.stderr)
        return 1
    mode = "dry-run" if args.dry_run else "preflight"
    print(
        f"result=ok mode={mode} profile={AEON_PROFILE} "
        f"minimum_downstream_idle_timeout_ms={minimum_idle_timeout}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
