#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
SWEEP_CONFIG="${SWEEP_CONFIG:-${PROJECT_ROOT}/configs/sam_sweep.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-wise-ft}"
exec "${PYTHON:-python}" "${PROJECT_ROOT}/utils/run_sam_sweep.py" \
    --config "${SWEEP_CONFIG}" --project "${WANDB_PROJECT}" "$@"
