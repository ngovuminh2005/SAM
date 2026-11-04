#!/usr/bin/env bash
# Table 2 of the logits-SAM paper (Mistral-7B / UltraFeedback), DPO rows, in one go on 2 GPUs:
#   1. DPO            (AdamW)
#   2. DPO+logits-SAM (rho = 1e-5, Table 7)
# Same setting for both: global batch 128, lr 5e-7, beta 0.01, 1 epoch, seed 42.
#
# Usage:
#   bash scripts/run_mistral_paper.sh
#   SEED=43 bash scripts/run_mistral_paper.sh
#
# Env vars: SEED (default 42), RHO (default 1e-5); anything launch_dpo.sh reads is passed through.
# A run whose output directory already has train_results.json is skipped, so the script can be
# relaunched after an interruption. Logs: logs/paper/<run>.log; checkpoints: data/<run>/.
set -uo pipefail

SEED="${SEED:-42}"
RHO="${RHO:-1e-5}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO/logs/paper"
mkdir -p "$LOG_DIR"

runs=(
  "baseline|mistral7b_dpo_baseline_seed${SEED}"
  "sam|mistral7b_dpo_logitssam_rho${RHO}_seed${SEED}"
)

failed=()
for entry in "${runs[@]}"; do
  IFS='|' read -r mode run <<< "$entry"
  if [[ -f "$REPO/data/$run/train_results.json" ]]; then
    echo "[skip] $run (already finished)"
    continue
  fi
  echo "[run ] $run  ($(date '+%F %T'))  log: $LOG_DIR/$run.log"
  SEED="$SEED" RHO="$RHO" RUN_NAME="$run" bash "$REPO/scripts/launch_dpo.sh" "$mode" > "$LOG_DIR/$run.log" 2>&1
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
echo "All runs finished."
