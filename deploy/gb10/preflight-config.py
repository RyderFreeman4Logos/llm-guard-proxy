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
AEON_DEFAULT_NO_THINK = "aeon-default-no-think"
AEON_RESTART_COMMAND = [
    "systemctl",
    "--user",
    "restart",
    "vllm-aeon-27b-dflash-n12.service",
]
RECOVERY_COMPLETION_GUARD_MS = 1_000
RESERVED_INGRESS_MODEL_IDS = [
    "abliterated-qwen-latest-27b-nvfp4",
    "aeon",
    "aeon-ultimate",
]
FORCED_ALIAS_PROFILES: dict[str, dict[str, JsonValue]] = {
    "abliterated-qwen-latest-27b-nvfp4-none": {
        "upstream_model": "abliterated-qwen-latest-27b-nvfp4",
        "thinking_mode": "force_disable",
        "output_cap": 16_384,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "abliterated-qwen-latest-27b-nvfp4-low": {
        "upstream_model": "abliterated-qwen-latest-27b-nvfp4",
        "thinking_mode": "force_thinking",
        "thinking_budget": 65_536,
        "output_cap": 16_384,
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0,
        "presence_penalty": 0,
        "repetition_penalty": 1,
    },
    "abliterated-qwen-latest-27b-nvfp4-medium": {
        "upstream_model": "abliterated-qwen-latest-27b-nvfp4",
        "thinking_mode": "force_thinking",
        "thinking_budget": 65_536,
        "output_cap": 16_384,
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0,
        "presence_penalty": 0,
        "repetition_penalty": 1,
    },
}


def load_config(path: Path) -> dict[str, JsonValue]:
    """Parse the project TOML dialect while preserving JSON readiness bodies."""
    source = path.read_text(encoding="utf-8")
    normalized_lines: list[str] = []
    for line in source.splitlines():
        if line.lstrip().startswith("readiness_body ="):
            key, separator, raw_body = line.partition("=")
            if not separator:
                raise ValueError("readiness_body assignment is malformed")
            if not raw_body.lstrip().startswith('"'):
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
    attempts = _positive_int(recovery, "max_attempts_per_request")
    if attempts != 1:
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


def _forced_alias_profile_errors(config: dict[str, JsonValue]) -> list[str]:
    profiles = config.get("forced_model_alias_profiles")
    if isinstance(profiles, list):
        numeric_fields = (
            "thinking_budget",
            "output_cap",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "repetition_penalty",
        )
        if any(
            isinstance(profile, dict)
            and any(isinstance(profile.get(field), bool) for field in numeric_fields)
            for profile in profiles
        ):
            return ["forced_model_alias_profiles numeric fields must not be boolean"]
    expected = [
        {"alias": alias, **settings}
        for alias, settings in FORCED_ALIAS_PROFILES.items()
    ]
    return (
        []
        if profiles == expected
        else ["forced_model_alias_profiles must exactly match the reviewed aliases"]
    )


def _named_upstream_membership_errors(config: dict[str, JsonValue]) -> list[str]:
    expected = list(FORCED_ALIAS_PROFILES)
    profiles = config.get("upstreams")
    if not isinstance(profiles, list):
        return [f"{AEON_DEFAULT_NO_THINK} routing profile is missing"]
    names: list[str] = []
    typed_profiles: list[dict[str, JsonValue]] = []
    for profile in profiles:
        if not isinstance(profile, dict):
            return ["upstreams entries must be tables"]
        name = profile.get("name")
        match_models = profile.get("match_models")
        if not isinstance(name, str) or not isinstance(match_models, list):
            return ["upstreams entries must have string names and match_models lists"]
        if not all(isinstance(model, str) for model in match_models):
            return ["upstreams match_models entries must be strings"]
        names.append(name)
        typed_profiles.append(profile)
    if len(set(names)) != len(names) or "default" in names:
        return ["upstreams names must be unique and must not duplicate the implicit default profile"]
    default_chat = next(
        (
            profile
            for profile in typed_profiles
            if profile.get("name") == AEON_DEFAULT_NO_THINK
        ),
        None,
    )
    if not isinstance(default_chat, dict):
        return [f"{AEON_DEFAULT_NO_THINK} routing profile is missing"]
    if default_chat.get("match_models") != expected:
        return [
            f"{AEON_DEFAULT_NO_THINK} match_models must exactly match the public NVFP4 aliases"
        ]
    canonical_targets = {
        model
        for settings in FORCED_ALIAS_PROFILES.values()
        if isinstance(model := settings.get("upstream_model"), str)
    }
    membership = {alias: [] for alias in expected}
    for profile in typed_profiles:
        name = profile.get("name")
        match_models = profile.get("match_models")
        if not isinstance(name, str) or not isinstance(match_models, list):
            return ["upstreams entries must have string names and match_models lists"]
        for alias in expected:
            if alias in match_models:
                membership[alias].append(name)
        if name != AEON_DEFAULT_NO_THINK and any(
            isinstance(model, str) and model in canonical_targets
            for model in match_models
        ):
            return ["forced canonical targets must not match competing upstream profiles"]
    return (
        []
        if all(names == [AEON_DEFAULT_NO_THINK] for names in membership.values())
        else [
            f"public NVFP4 aliases must belong exclusively to {AEON_DEFAULT_NO_THINK}"
        ]
    )


def _reserved_ingress_errors(config: dict[str, JsonValue]) -> list[str]:
    reserved = _table(config, "upstream").get("reserved_ingress_model_ids")
    return (
        []
        if reserved == RESERVED_INGRESS_MODEL_IDS
        else ["upstream.reserved_ingress_model_ids must exactly match the reserved identities"]
    )


def minimum_downstream_idle_timeout_ms(config: dict[str, JsonValue]) -> int | None:
    """Return the strict byte-silent bound including recovery handoff and replay."""
    retry = _table(config, "retry")
    request_deadline = _positive_int(retry, "request_deadline_ms")
    default_upstream = _table(config, "upstream")
    profiles = config.get("upstreams")
    if request_deadline is None or not isinstance(profiles, list):
        return None
    routes = [default_upstream, *(profile for profile in profiles if isinstance(profile, dict))]
    route_bounds: list[int] = []
    for route in routes:
        recovery = _table(route, "local_recovery")
        if recovery.get("enabled") is not True:
            continue
        request_timeout = _positive_int(route, "request_timeout_ms")
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
    return request_deadline + max(route_bounds) if route_bounds else None


def validate_snapshot(
    config: dict[str, JsonValue], downstream_idle_timeout_ms: int
) -> tuple[list[str], int | None]:
    errors: list[str] = []
    errors.extend(_forced_alias_profile_errors(config))
    errors.extend(_reserved_ingress_errors(config))
    errors.extend(_named_upstream_membership_errors(config))
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
    if isinstance(profiles, list):
        for profile in profiles:
            if not isinstance(profile, dict) or profile is aeon:
                continue
            recovery = _table(profile, "local_recovery")
            if recovery.get("enabled") is True:
                errors.extend(
                    _recovery_errors(
                        f"upstreams[{profile.get('name')!r}]",
                        recovery,
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
