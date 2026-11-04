#!/usr/bin/env bash
# One DPO run of the Mistral-7B / UltraFeedback experiment (logits-SAM paper setting, Table 2).
# Used by scripts/run_mistral_paper.sh and scripts/sweep_condsam.sh; can also be run alone.
#
# Usage:
#   bash scripts/launch_dpo.sh baseline # plain DPO (AdamW)
#   bash scripts/launch_dpo.sh sam      # DPO + logits-SAM (rho=1e-5, Table 7)
#   bash scripts/launch_dpo.sh condsam  # DPO + logits-CondSAM baseline (CondSAM direction on lm_head)
#
# Optional env vars: SEED (default 42), RHO (default 1e-5), NUM_GPUS (default 2), PER_DEVICE (default 4),
#   CONDSAM_ARGS (extra CondSAM flags, e.g. "--condsam_beta 0.95 --condsam_power_iters 1"),
#   RUN_NAME (override output dir / wandb run name), WANDB_PROJECT (default sam-dpo), WANDB_MODE (online|offline|disabled), ENV_DIR (default <repo>/env), EXTRA_ARGS (appended to the dpo.py command, e.g. "--max_steps 2").
set -euo pipefail

MODE="${1:-sam}"
SEED="${SEED:-42}"
RHO="${RHO:-1e-5}"
NUM_GPUS="${NUM_GPUS:-2}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ENV_DIR:-$REPO/env}"
CONFIG="$REPO/recipes/zephyr-7b-beta/dpo/config_full_paper_2gpu.yaml"
RUN_NAME="${RUN_NAME:-}"

# Keep the global batch at 128: PER_DEVICE x NUM_GPUS x grad_accum.
PER_DEVICE="${PER_DEVICE:-4}"
GRAD_ACCUM=$((128 / (PER_DEVICE * NUM_GPUS)))

case "$MODE" in
  sam)      SAM_ARGS=(--use_sam_trainer --sam_method sam --sam_rho "$RHO"); RUN="mistral7b_dpo_logitssam_rho${RHO}_seed${SEED}" ;;
  condsam)  SAM_ARGS=(--use_sam_trainer --sam_method condsam --sam_rho "$RHO"); RUN="mistral7b_dpo_condsam_rho${RHO}_seed${SEED}" ;;
  baseline) SAM_ARGS=();                                   RUN="mistral7b_dpo_baseline_seed${SEED}" ;;
  *) echo "MODE must be 'sam', 'condsam' or 'baseline'" >&2; exit 1 ;;
esac

RUN="${RUN_NAME:-$RUN}"

# Fail fast on a node without enough GPUs (e.g. the login node), instead of being killed later.
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU" || true)
if [[ "$GPU_COUNT" -lt "$NUM_GPUS" ]]; then
  echo "[launch_dpo] $(hostname): found ${GPU_COUNT} GPU(s), need ${NUM_GPUS}. Run this on the GPU node (tmux window 0 on worker-0)." >&2
  exit 2
fi
export PATH="$ENV_DIR/bin:$PATH"
export ACCELERATE_LOG_LEVEL=info
export TOKENIZERS_PARALLELISM=false
# Less fragmentation near the 80 GB limit (allocator setting only, numerics unchanged).
# Weights & Biases (project sam-dpo). Without a login (no WANDB_API_KEY, no api.wandb.ai entry
# in ~/.netrc) wandb would stop and prompt for a key inside tmux, so fall back to offline mode;
# upload later with:  env/bin/wandb sync wandb/offline-run-*
export WANDB_PROJECT="${WANDB_PROJECT:-sam-dpo}"
if [[ -z "${WANDB_MODE:-}" && -z "${WANDB_API_KEY:-}" ]] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
  echo "[wandb] not logged in: logging offline (run 'env/bin/wandb login' once to log online)" >&2
  export WANDB_MODE=offline
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$REPO"
accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml \
  --num_processes="$NUM_GPUS" \
  scripts/dpo.py \
  --config "$CONFIG" \
  --seed "$SEED" \
  --per_device_train_batch_size "$PER_DEVICE" \
  --per_device_eval_batch_size "$PER_DEVICE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --output_dir "data/$RUN" \
  --run_name "$RUN" \
  "${SAM_ARGS[@]}" \
  ${CONDSAM_ARGS:-} \
  ${EXTRA_ARGS:-}
