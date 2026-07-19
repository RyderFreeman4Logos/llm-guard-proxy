#!/usr/bin/env bash
set -euo pipefail

readonly legacy_unit="gb10-memory-guardian.service"
readonly integrated_unit="llm-guard-proxy.service"
readonly registration_max_bytes=1024
readonly cgroup_root="${LLM_GUARD_CGROUP_ROOT:-/sys/fs/cgroup}"
readonly proc_root="${LLM_GUARD_PROC_ROOT:-/proc}"
readonly systemctl_timeout="${LLM_GUARD_SYSTEMCTL_TIMEOUT:-15s}"

die() {
    printf 'guardian cutover: %s\n' "$1" >&2
    exit 1
}

systemctl_bounded() {
    timeout --foreground --kill-after=1s "${systemctl_timeout}" systemctl "$@"
}

validate_unit_state() {
    local state_name="$1"
    local state="$2"
    shift 2
    local allowed
    for allowed in "$@"; do
        [[ "${state}" != "${allowed}" ]] || return 0
    done
    die "${state_name} has unexpected state ${state}"
}

restore_unit_state() {
    local unit="$1"
    local unit_file_state="$2"
    local active_state="$3"
    local failed=0

    case "${unit_file_state}" in
        enabled)
            systemctl_bounded --user enable "${unit}" || failed=1
            ;;
        enabled-runtime)
            systemctl_bounded --user disable "${unit}" || failed=1
            systemctl_bounded --user enable --runtime "${unit}" || failed=1
            ;;
        disabled)
            systemctl_bounded --user disable "${unit}" || failed=1
            ;;
        static | indirect | generated | transient) ;;
    esac
    case "${active_state}" in
        active) systemctl_bounded --user start "${unit}" || failed=1 ;;
        inactive) systemctl_bounded --user stop "${unit}" || failed=1 ;;
        *) failed=1 ;;
    esac
    return "${failed}"
}

attest_integrated_guardian() {
    local pid
    pid="$(
        systemctl_bounded --user show --property=MainPID --value "${integrated_unit}"
    )"
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || die "integrated guardian has no running main process"

    local process_dir="${proc_root}/${pid}"
    [[ -d "${process_dir}/fd" && -d "${process_dir}/fdinfo" ]] \
        || die "integrated guardian process descriptors are unavailable"
    local expected_events expected_kill
    expected_events="$(readlink -f -- "${cgroup_events}")"
    expected_kill="$(readlink -f -- "${cgroup_kill}")"
    local events_open=false
    local writable_kill_open=false
    local fd_link fd_target fd_number flags key value extra
    for fd_link in "${process_dir}"/fd/[0-9]*; do
        [[ -e "${fd_link}" || -L "${fd_link}" ]] || continue
        fd_target="$(readlink -f -- "${fd_link}" 2>/dev/null || true)"
        if [[ "${fd_target}" == "${expected_events}" ]]; then
            events_open=true
        fi
        [[ "${fd_target}" == "${expected_kill}" ]] || continue
        fd_number="${fd_link##*/}"
        flags=""
        while read -r key value extra; do
            if [[ "${key}" == "flags:" && -z "${extra:-}" ]]; then
                flags="${value}"
                break
            fi
        done <"${process_dir}/fdinfo/${fd_number}"
        if [[ "${flags}" =~ ^0?[0-7]*[123567]$ ]]; then
            writable_kill_open=true
        fi
    done
    [[ "${events_open}" == "true" ]] \
        || die "integrated guardian has not opened the registered cgroup.events"
    [[ "${writable_kill_open}" == "true" ]] \
        || die "integrated guardian has not opened registered cgroup.kill for writing"
}

if (( $# != 0 )); then
    die "usage: $0"
fi

uid="$(id -u)"
runtime_dir="${XDG_RUNTIME_DIR:-/run/user/${uid}}/gb10-memory-guardian"
registration="${runtime_dir}/text-cgroup.v1"

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

[[ "${cgroup_root}" == /* ]] || die "cgroup root must be absolute"
[[ "${proc_root}" == /* ]] || die "proc root must be absolute"
cgroup_path="${cgroup_root}${control_group}"
[[ -d "${cgroup_path}" && ! -L "${cgroup_path}" ]] \
    || die "registered cgroup does not exist"
cgroup_events="${cgroup_path}/cgroup.events"
[[ -f "${cgroup_events}" && ! -L "${cgroup_events}" ]] \
    || die "registered cgroup.events is unavailable"
cgroup_kill="${cgroup_path}/cgroup.kill"
[[ -f "${cgroup_kill}" && ! -L "${cgroup_kill}" ]] \
    || die "registered cgroup.kill is unavailable"
populated=""
while read -r event_key event_value event_extra; do
    if [[ "${event_key}" == "populated" ]]; then
        [[ -z "${populated}" && -z "${event_extra:-}" && "${event_value}" =~ ^[01]$ ]] \
            || die "registered cgroup.events has malformed populated state"
        populated="${event_value}"
    fi
done <"${cgroup_events}"
[[ "${populated}" == "1" ]] || die "registered cgroup is not populated"

# All registration and live-target validation is complete before the first guardian
# systemctl call. A missing legacy unit is the expected fresh-install state.
legacy_load_state="$(
    systemctl_bounded --user show --property=LoadState --value "${legacy_unit}"
)"
legacy_unit_file_state="disabled"
legacy_active_state="inactive"
if [[ "${legacy_load_state}" != "not-found" ]]; then
    [[ -n "${legacy_load_state}" ]] || die "legacy guardian load state is empty"
    legacy_unit_file_state="$(
        systemctl_bounded --user show --property=UnitFileState --value "${legacy_unit}"
    )"
    legacy_active_state="$(
        systemctl_bounded --user show --property=ActiveState --value "${legacy_unit}"
    )"
    validate_unit_state "legacy guardian unit file" "${legacy_unit_file_state}" \
        enabled enabled-runtime disabled static indirect generated transient
    validate_unit_state "legacy guardian activity" "${legacy_active_state}" \
        active inactive
fi
integrated_unit_file_state="$(
    systemctl_bounded --user show --property=UnitFileState --value "${integrated_unit}"
)"
integrated_active_state="$(
    systemctl_bounded --user show --property=ActiveState --value "${integrated_unit}"
)"
validate_unit_state "integrated guardian unit file" "${integrated_unit_file_state}" \
    enabled enabled-runtime disabled static indirect generated transient
validate_unit_state "integrated guardian activity" "${integrated_active_state}" \
    active inactive

cutover_started=false
cutover_complete=false
rollback_cutover() {
    local status=$?
    trap - EXIT INT TERM
    if [[ "${cutover_started}" == "true" && "${cutover_complete}" != "true" ]]; then
        set +e
        printf 'guardian cutover: restoring prior guardian unit states\n' >&2
        local rollback_failed=0
        restore_unit_state \
            "${integrated_unit}" "${integrated_unit_file_state}" "${integrated_active_state}" \
            || rollback_failed=1
        if [[ "${legacy_load_state}" != "not-found" ]]; then
            restore_unit_state \
                "${legacy_unit}" "${legacy_unit_file_state}" "${legacy_active_state}" \
                || rollback_failed=1
        fi
        if (( rollback_failed != 0 )); then
            printf 'guardian cutover: automatic rollback was incomplete\n' >&2
        fi
    fi
    exit "${status}"
}
trap rollback_cutover EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cutover_started=true
if [[ "${legacy_load_state}" != "not-found" ]]; then
    systemctl_bounded --user disable --now "${legacy_unit}"
fi
systemctl_bounded --user enable --now "${integrated_unit}"
systemctl_bounded --user is-active "${integrated_unit}"
attest_integrated_guardian
cutover_complete=true
trap - EXIT INT TERM
