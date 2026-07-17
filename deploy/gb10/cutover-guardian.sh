#!/usr/bin/env bash
set -euo pipefail

readonly legacy_unit="gb10-memory-guardian.service"
readonly integrated_unit="llm-guard-proxy.service"
readonly registration_max_bytes=1024

die() {
    printf 'guardian cutover: %s\n' "$1" >&2
    exit 1
}

if (( $# > 1 )); then
    die "usage: $0 [registration-file]"
fi

uid="$(id -u)"
runtime_dir="${XDG_RUNTIME_DIR:-/run/user/${uid}}/gb10-memory-guardian"
registration="${1:-${runtime_dir}/text-cgroup.v1}"

[[ -f "${registration}" && ! -L "${registration}" ]] \
    || die "registration must be a regular, non-symlink file"
[[ "$(stat -c '%a:%u:%h' -- "${registration}")" == "600:${uid}:1" ]] \
    || die "registration must have mode, owner, and link count 600:${uid}:1"
registration_size="$(stat -c '%s' -- "${registration}")"
(( registration_size < registration_max_bytes )) \
    || die "registration must be smaller than ${registration_max_bytes} bytes"
if IFS= read -r -d '' _ <"${registration}"; then
    die "registration contains a NUL byte"
fi

version=""
container_id=""
scope=""
control_group=""
while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -n "${line}" && "${line}" != *$'\r'* && "${line}" == *=* ]] \
        || die "registration contains a malformed line"
    key="${line%%=*}"
    value="${line#*=}"
    [[ -n "${value}" && "${value}" != *=* ]] \
        || die "registration contains a malformed value"
    case "${key}" in
        version)
            [[ -z "${version}" ]] || die "registration contains a duplicate version"
            version="${value}"
            ;;
        container_id)
            [[ -z "${container_id}" ]] \
                || die "registration contains a duplicate container_id"
            container_id="${value}"
            ;;
        scope)
            [[ -z "${scope}" ]] || die "registration contains a duplicate scope"
            scope="${value}"
            ;;
        control_group)
            [[ -z "${control_group}" ]] \
                || die "registration contains a duplicate control_group"
            control_group="${value}"
            ;;
        *) die "registration contains an unknown field" ;;
    esac
done <"${registration}"

[[ "${version}" == "1" ]] || die "registration version must be 1"
[[ "${container_id}" =~ ^[0-9a-f]{64}$ ]] \
    || die "container_id must be exactly 64 lowercase hexadecimal characters"
expected_scope="docker-${container_id}.scope"
[[ "${scope}" == "${expected_scope}" ]] \
    || die "scope must match the registered container_id"
expected_group="/user.slice/user-${uid}.slice/user@${uid}.service/app.slice/${scope}"
[[ "${control_group}" == "${expected_group}" ]] \
    || die "control_group must match the current user and scope"

# All registration validation is complete before the first guardian systemctl
# call. A missing legacy unit is the expected fresh-install state.
legacy_load_state="$(
    systemctl --user show --property=LoadState --value "${legacy_unit}"
)"
if [[ "${legacy_load_state}" != "not-found" ]]; then
    [[ -n "${legacy_load_state}" ]] || die "legacy guardian load state is empty"
    systemctl --user disable --now "${legacy_unit}"
fi
systemctl --user enable --now "${integrated_unit}"
systemctl --user is-active "${integrated_unit}"
