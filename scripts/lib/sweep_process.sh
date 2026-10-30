_sweep_active_pid=""

_sweep_cleanup() {
    local child_pid="${_sweep_active_pid}"
    _sweep_active_pid=""
    if [[ -n "${child_pid}" ]]; then
        kill -TERM "${child_pid}" 2>/dev/null || true
        wait "${child_pid}" 2>/dev/null || true
    fi
}

_sweep_stop() {
    local exit_code="$1"
    trap '' INT TERM HUP
    _sweep_cleanup
    exit "${exit_code}"
}

trap '_sweep_stop 130' INT
trap '_sweep_stop 143' TERM
trap '_sweep_stop 129' HUP
trap '_sweep_cleanup' EXIT

run_training() {
    local status=0
    "$@" &
    _sweep_active_pid=$!
    wait "${_sweep_active_pid}" || status=$?
    _sweep_active_pid=""
    if (( status >= 128 )); then
        exit "${status}"
    fi
    return "${status}"
}
