#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${SAM_DATASET_ROOT:-${PROJECT_ROOT}/datasets}}"
IMAGENET_ROOT="${DATA_ROOT}/ImageNet"
IMAGENET_BASE_URL="${IMAGENET_BASE_URL:-https://image-net.org/data/ILSVRC/2012}"
IMAGENET_COOKIES="${IMAGENET_COOKIES:-}"
train=false
val=false
if (( $# == 0 )); then train=true; val=true; fi
for option in "$@"; do
    case "${option}" in
        --train) train=true ;;
        --val) val=true ;;
        *) echo "Usage: bash scripts/get_imagenet.sh [--train] [--val]" >&2; exit 2 ;;
    esac
done

for command in wget tar md5sum git find flock; do
    command -v "${command}" >/dev/null || { echo "Missing command: ${command}" >&2; exit 2; }
done
mkdir -p "${IMAGENET_ROOT}"
exec {dataset_lock}>"${IMAGENET_ROOT}/.download.lock"
flock -n "${dataset_lock}" || { echo "Another ImageNet download is running" >&2; exit 2; }

split_ready() {
    local directory="$1" expected="$2"
    [[ -d "${directory}" ]] || return 1
    [[ $(find "${directory}" -mindepth 1 -maxdepth 1 -type d -name 'n[0-9]*' | wc -l) -eq 1000 ]] || return 1
    [[ $(find "${directory}" -mindepth 2 -maxdepth 2 -type f -iname '*.jpeg' | wc -l) -eq "${expected}" ]]
}

if [[ "${train}" == true ]] && split_ready "${IMAGENET_ROOT}/train" 1281167; then
    echo "[SKIP] ImageNet train is already prepared"
    train=false
fi
if [[ "${val}" == true ]] && split_ready "${IMAGENET_ROOT}/val" 50000; then
    echo "[SKIP] ImageNet validation is already prepared"
    val=false
fi
if [[ "${train}" == false && "${val}" == false ]]; then exit 0; fi

minimum_gb=15
if [[ "${train}" == true ]]; then minimum_gb=300; fi
free_kb=$(df -Pk "${IMAGENET_ROOT}" | awk 'END {print $4}')
existing_kb=0
for item in "${IMAGENET_ROOT}/.train-extract" "${IMAGENET_ROOT}/.val-extract" "${IMAGENET_ROOT}"/*.tar*; do
    if [[ -e "${item}" ]]; then
        size_kb=$(du -sk "${item}" | awk '{print $1}')
        existing_kb=$((existing_kb + size_kb))
    fi
done
if (( free_kb + existing_kb < minimum_gb * 1024 * 1024 )); then
    echo "Need at least ${minimum_gb} GiB free at ${IMAGENET_ROOT}; choose a larger DATA_ROOT" >&2
    exit 2
fi

wget_args=(--continue --tries=3 --timeout=60)
if [[ -n "${IMAGENET_COOKIES}" ]]; then wget_args+=(--load-cookies "${IMAGENET_COOKIES}"); fi
fetch_archive() {
    local filename="$1" checksum="$2" archive="${IMAGENET_ROOT}/$1"
    if [[ ! -f "${archive}" ]]; then
        wget "${wget_args[@]}" -O "${archive}.part" "${IMAGENET_BASE_URL}/${filename}"
        printf '%s  %s\n' "${checksum}" "${archive}.part" | md5sum --check --status || {
            echo "Invalid ${filename}; check ImageNet access/cookies and the download URL" >&2; exit 1;
        }
        mv "${archive}.part" "${archive}"
    else
        printf '%s  %s\n' "${checksum}" "${archive}" | md5sum --check --status || {
            echo "Checksum mismatch: ${archive}" >&2; exit 1;
        }
    fi
}

if [[ "${train}" == true ]]; then
    [[ ! -e "${IMAGENET_ROOT}/train" ]] || { echo "Incomplete train directory exists; move it aside before retrying" >&2; exit 2; }
    staging="${IMAGENET_ROOT}/.train-extract"
    mkdir -p "${staging}"
    if [[ ! -f "${staging}/.archive-extracted" ]]; then
        fetch_archive ILSVRC2012_img_train.tar 1d675b47d978889d74fa0da5fadfb00e
        tar --extract --file "${IMAGENET_ROOT}/ILSVRC2012_img_train.tar" --directory "${staging}" --no-same-owner
        touch "${staging}/.archive-extracted"
    fi
    rm -f "${IMAGENET_ROOT}/ILSVRC2012_img_train.tar"
    shopt -s nullglob
    for archive in "${staging}"/*.tar; do
        class_name="$(basename "${archive}" .tar)"
        [[ "${class_name}" =~ ^n[0-9]{8}$ ]] || { echo "Unexpected class archive: ${class_name}" >&2; exit 1; }
        mkdir -p "${staging}/${class_name}"
        tar --extract --file "${archive}" --directory "${staging}/${class_name}" --no-same-owner
        rm "${archive}"
    done
    split_ready "${staging}" 1281167 || { echo "ImageNet train counts are incorrect; rerun to resume" >&2; exit 1; }
    rm "${staging}/.archive-extracted"
    mv "${staging}" "${IMAGENET_ROOT}/train"
fi

if [[ "${val}" == true ]]; then
    [[ ! -e "${IMAGENET_ROOT}/val" ]] || { echo "Incomplete val directory exists; move it aside before retrying" >&2; exit 2; }
    fetch_archive ILSVRC2012_img_val.tar 29b22e2961454d5413ddabcf34fc5622
    staging="${IMAGENET_ROOT}/.val-extract"
    mkdir -p "${staging}"
    tar --extract --file "${IMAGENET_ROOT}/ILSVRC2012_img_val.tar" --directory "${staging}" --no-same-owner
    mapping="${IMAGENET_ROOT}/valprep.sh"
    wget --tries=3 --timeout=60 -O "${mapping}.part" \
        https://raw.githubusercontent.com/soumith/imagenetloader.torch/b81dbb17ebe5a09d91517b979a29e4d8183b7b9d/valprep.sh
    [[ $(git hash-object "${mapping}.part") == a05b97191cba09e0b65ffc4fde811fc6bd9898a4 ]] || {
        echo "Validation mapping checksum mismatch" >&2; exit 1;
    }
    mv "${mapping}.part" "${mapping}"
    (cd "${staging}" && bash -e "${mapping}")
    split_ready "${staging}" 50000 || { echo "ImageNet validation counts are incorrect" >&2; exit 1; }
    mv "${staging}" "${IMAGENET_ROOT}/val"
    rm "${IMAGENET_ROOT}/ILSVRC2012_img_val.tar" "${mapping}"
fi

if [[ -d "${IMAGENET_ROOT}/train" && -d "${IMAGENET_ROOT}/val" ]]; then
    diff <(find "${IMAGENET_ROOT}/train" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort) \
         <(find "${IMAGENET_ROOT}/val" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
fi
echo "[READY] ${IMAGENET_ROOT}"
