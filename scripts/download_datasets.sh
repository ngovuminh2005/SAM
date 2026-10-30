#!/usr/bin/env bash
set -euo pipefail

DATASETS=(imagenet)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${SAM_DATASET_ROOT:-${PROJECT_ROOT}/datasets}}"
PYTHON="${PYTHON:-${CONDA_PREFIX:-${PROJECT_ROOT}/env}/bin/python}"
if (( $# )); then DATASETS=("$@"); fi
for dataset in "${DATASETS[@]}"; do
    case "${dataset}" in
        imagenet|cifar10|cifar100|tiny-imagenet|cifar10_c|cifar100_c|svhn) ;;
        *) echo "Unknown dataset: ${dataset}" >&2; exit 2 ;;
    esac
done
for dataset in "${DATASETS[@]}"; do
    case "${dataset}" in
        imagenet)
            DATA_ROOT="${DATA_ROOT}" bash "${PROJECT_ROOT}/scripts/get_imagenet.sh"
            ;;
        cifar10|cifar100|tiny-imagenet)
            "${PYTHON}" "${PROJECT_ROOT}/scripts/prepare_datasets.py" --dataset "${dataset}" --data-root "${DATA_ROOT}"
            ;;
        cifar10_c|cifar100_c|svhn)
            "${PYTHON}" "${PROJECT_ROOT}/scripts/prepare_ood_datasets.py" --datasets "${dataset}" --data-root "${DATA_ROOT}"
            ;;
    esac
done
