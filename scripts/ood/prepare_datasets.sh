#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASETS=(cifar10_c cifar100_c svhn)
DATA_ROOT="${DATA_ROOT:-${SAM_DATASET_ROOT:-${ROOT}/datasets}}"
DATA_ROOT="${DATA_ROOT}" bash "${ROOT}/scripts/download_datasets.sh" "${DATASETS[@]}"
