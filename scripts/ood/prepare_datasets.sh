#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASETS=(cifar10_c cifar100_c svhn)
DATA_ROOT="${ROOT}/datasets"
"${ROOT}/env/bin/python" "${ROOT}/scripts/prepare_ood_datasets.py" \
    --datasets "${DATASETS[@]}" --data-root "${DATA_ROOT}"
