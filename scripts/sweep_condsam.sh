#!/usr/bin/env bash
# Sweep rho for the logits-CondSAM baseline (CondSAM direction on lm_head), Mistral-7B /
# UltraFeedback, same setting as scripts/run_mistral_paper.sh (batch 128, lr 5e-7, beta 0.01,
# 1 epoch). Runs are sequential, each on all GPUs.
#
# Default grid: the paper's logits-SAM grid {1e-5, 1e-4, 1e-3} plus {1e-2, 1e-1}: at the same rho,
# the CondSAM direction raised the loss ~500x less than logits-SAM in a 1-step check
# (rho=1e-1: +0.003 vs +1.55), so the paper grid alone may barely perturb it.
#
# Usage:
#   bash scripts/sweep_condsam.sh
#   RHOS="1e-2 1e-1 1" SEEDS="42 43" bash scripts/sweep_condsam.sh
#   CONDSAM_ARGS="--condsam_beta 0.9" bash scripts/sweep_condsam.sh
#
# Env vars:
#   RHOS          rho grid (default "1e-5 1e-4 1e-3 1e-2 1e-1")
#   SEEDS         seeds (default "42")
#   CONDSAM_ARGS  extra CondSAM flags for every run (beta, eps, power_iters, update_interval)
#   DRY_RUN       1 = print the runs without launching them
# A run whose output directory already has train_results.json is skipped. Logs:
# logs/sweep_condsam/<run>.log; checkpoints: data/<run>/.
set -uo pipefail

RHOS="${RHOS:-1e-5 1e-4 1e-3 1e-2 1e-1}"
SEEDS="${SEEDS:-42}"
DRY_RUN="${DRY_RUN:-0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO/logs/sweep_condsam"
mkdir -p "$LOG_DIR"

runs=()
for seed in $SEEDS; do
  for rho in $RHOS; do
    runs+=("$rho|$seed|mistral7b_dpo_condsam_rho${rho}_seed${seed}")
  done
done

echo "CondSAM sweep: ${#runs[@]} runs (rho: $RHOS; seeds: $SEEDS)"
failed=()
for entry in "${runs[@]}"; do
  IFS='|' read -r rho seed run <<< "$entry"
  if [[ -f "$REPO/data/$run/train_results.json" ]]; then
    echo "[skip] $run (already finished)"
    continue
  fi
  echo "[run ] $run  ($(date '+%F %T'))  log: $LOG_DIR/$run.log"
  [[ "$DRY_RUN" == "1" ]] && continue
  SEED="$seed" RHO="$rho" RUN_NAME="$run" bash "$REPO/scripts/launch_dpo.sh" condsam > "$LOG_DIR/$run.log" 2>&1
  status=$?
  if [[ $status -eq 2 ]]; then
    echo "[stop] $run: not enough GPUs on $(hostname), see $LOG_DIR/$run.log"
    exit 2
  fi
  if [[ $status -ne 0 ]]; then
    echo "[fail] $run (exit $status)"
    failed+=("$run")
  else
    echo "[done] $run  ($(date '+%F %T'))"
  fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed runs: ${failed[*]}"
  exit 1
fi
echo "Sweep finished."
