#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${PROJECT_ROOT}/scripts/sweep_sam.sh" \
    --config "${PROJECT_ROOT}/configs/no_sam_sweep.yaml" "$@"
