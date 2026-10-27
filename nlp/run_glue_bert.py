#!/usr/bin/env python3
"""
BERT-base GLUE fine-tuning: baseline SGD vs original SAM variants (SGD+momentum inner update).

This entrypoint is for comparing SGD with paper-style SAM family optimizers defined in this repo
(SAM, ASAM, ESAM, lookSAM, FriendlySAM, FisherSAM, MongeSAM, GSAM). It does **not**
use AdamW-backed sharpness methods (AdamW_SAM / AdamW_FriendlySAM).

Training: linear LR (HF schedule), global grad clip (Args.grad_clip_norm), batch/epochs/lr from ini + CLI.
Per-task sharpness ρ (overrides ini [SAM] rhos); see ``_glue_sam_rho_for_task`` + env ``SAM_GLUE_RHO_OVERRIDE``.

Run:
  torchrun --standalone --nproc_per_node=1 nlp/run_glue_bert.py --ifile configs/NLP_GLUE_BERT.ini --glue_task sst2 --optimizer SAM

ρ sweep (same run, overrides per-task table): ``SAM_GLUE_RHO_OVERRIDE=0.006 torchrun ...``

STS-B is regression (MSE): use --optimizer SGD, or **SAM** (two-step loss uses MSE).
Other SAM-family optimizers on this path still use cross-entropy inside ``optimizer.step`` and stay disabled for STS-B.
Excluded here: BayesianSAM, AdamW_MSAM (API), AdamW_SAM, AdamW_FriendlySAM (AdamW hybrid).
MSAM uses the same epoch-wise ascent API as main.py (not SAM.step(model,...)).

Validation metrics logged to metrics_history.json are **official GLUE dev columns only**
(CoLA Mcc., SST-2/MNLI/... Acc., MRPC/QQP Acc./F1., STS-B Pear./Spea.; see nlp/glue_metrics.py).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

# Allow `python nlp/run_glue_bert.py` from repo root (package `nlp` + `utility`).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from nlp.glue_metrics import (
    add_mnli_mean_if_complete,
    extract_official_glue_metrics,
    format_official_metrics_line,
)
from nlp.glue_model import BertGlueWrapper
from nlp.glue_tasks import GLUE_TASK_KEYS, glue_num_labels, is_glue_regression, mnli_eval_split_names
from optimizer import getOptimizer
from utility.args import Args
from utility import tracking
from utility.evaluation import should_evaluate
from utility.nlp_glue_args import register_nlp_glue_args
from utility.utils import initialize

register_nlp_glue_args()
import main as _vision_main  # noqa: F401, E402 — registers logDir, epochs, dataset Args, etc.

try:
    import evaluate
except ImportError as exc:
    raise RuntimeError(
        "Install NLP deps: pip install transformers datasets evaluate scikit-learn"
    ) from exc

# Not supported on this GLUE path (API / design).
EXCLUDED_OPTIMIZERS = frozenset(
    {
        "BayesianSAM",
        "AdamW_MSAM",
        "AdamW_SAM",
        "AdamW_FriendlySAM",
        "AdamW",
        "Adam",
    }
)

# MSAM: momentum-ascent each epoch (see main.py); forward path like SGD, not optimizer.step(model,...).
MSAM_VARIANT = frozenset({"MSAM"})

# Two-step sharpness optimizers (expect model, batch_dict, labels); inner update is SGD-style-specific.
SAM_SGD_VARIANTS = frozenset(
    {
        "SAM",
        "CondSAM",
        "CondSam2",
        "ASAM",
        "ESAM",
        "lookSAM",
        "FriendlySAM",
        "FisherSAM",
        "MongeSAM",
        "GSAM",
    }
)

# GLUE regression (STS-B): only these implement ``step(..., regression=True)`` with MSE.
SAM_REGRESSION_STEP_VARIANTS = frozenset({"SAM", "CondSAM", "CondSam2"})


def _unwrap(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if isinstance(m, DDP) else m


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _tokenize_builder(task: str, tokenizer):
    k1, k2 = GLUE_TASK_KEYS[task.lower()]

    def _tok(examples):
        if k2 is None:
            enc = tokenizer(examples[k1], truncation=True, max_length=128)
        else:
            enc = tokenizer(examples[k1], examples[k2], truncation=True, max_length=128)
        enc["labels"] = examples["label"]
        return enc

    return _tok


def _build_metric(task: str):
    return evaluate.load("glue", task.lower())


def _sanitize_run_segment(s: str) -> str:
    s = str(s).strip().replace(" ", "")
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s)


def _run_directory(task: str) -> Path:
    """Unique folder under glue_output_root: task + optimizer + lr + bs + epochs + method rho."""
    repo = Path(__file__).resolve().parents[1]
    root = Path(Args.glue_output_root).expanduser()
    if not root.is_absolute():
        root = (repo / root).resolve()
    lr_tag = _sanitize_run_segment(f"{float(Args.learningRate):.6g}")
    if Args.optimizer == "SGD":
        rho_tag = "rho_na"
    else:
        rho_tag = _sanitize_run_segment(f"rho{_vision_main._resolve_optimizer_rho()}")
    name = "__".join(
        [
            _sanitize_run_segment(task),
            _sanitize_run_segment(Args.optimizer),
            f"lr{lr_tag}",
            f"bs{int(Args.batchSize)}",
            f"ep{int(Args.epochs)}",
            rho_tag,
        ]
    )
    return root / name


def _format_seconds(total_seconds: float) -> str:
    total_seconds = max(0.0, float(total_seconds))
    hh = int(total_seconds // 3600)
    mm = int((total_seconds % 3600) // 60)
    ss = total_seconds - (hh * 3600 + mm * 60)
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"


# GLUE benchmark: single ρ per run for all method-specific keys (getOptimizer + _resolve_optimizer_rho).
_GLUE_SAM_RHO_ATTRS = (
    "rho",
    "samRho",
    "condsamRho",
    "asamRho",
    "esamRho",
    "looksamRho",
    "friendlysamRho",
    "fishersamRho",
    "mongesamRho",
    "gsamRho",
    "bayesiansamRho",
    "msamRho",
    "adamwSamRho",
    "adamSamRho",
    "adamwMsamRho",
    "adamwFriendlysamRho",
)


def _glue_sam_rho_for_task(task: str) -> float:
    """
    One ρ for all SAM-family optimizers on GLUE (written to every *Rho on ``Args.data``).

    **Mechanics (why ρ must be small here)**

    - **SAM** (``optimizer/sam.py``): ascent ``ε = (ρ / ‖g‖₂) · g`` (concatenated grads), so **‖ε‖₂ = ρ**
      exactly. Descent uses the *second* gradient at ``w + ε``, after ``move_back``, with
      ``clip_grad_norm`` then SGD. Too large ρ makes the inner problem far from the outer one;
      with only 3 epochs and lr ``2e-5``, SST-2 collapsed (~51% vs SGD ~54%) at **ρ=0.05** in
      ``logs_nlp``.

    - **CoLA**: at **ρ=0.01**, SAM reported MCC 0 vs SGD ~1.8 (same logs); grammar +
      Matthews correlation is brittle — use the **smallest** bucket.

    - **Tiny dev tasks** (RTE, WNLI, STS-B): keep ρ in the same “small-data” band as before.

    **Tuning**: install ``evaluate`` and run ``scripts/sweep_glue_sam_rho.sh`` (or set
    ``SAM_GLUE_RHO_OVERRIDE``) on SST-2 / CoLA to fine-tune. This table is a conservative
    compromise aimed at **beating or matching SGD** without the catastrophic failure mode at 0.05.
    """
    t = task.lower().replace("-", "")
    if t == "sts-b":
        t = "stsb"
    if t == "cola":
        return 0.002
    if t in ("rte", "stsb", "wnli"):
        return 0.004
    return 0.005


def _apply_glue_sam_rho_overrides(task: str) -> float:
    """
    Set Args rho + every *Rho used by SAM family so all methods share the task-specific ρ.

    Sweep / ablation: set env ``SAM_GLUE_RHO_OVERRIDE`` to a positive float to use one ρ for this
    run (ignores per-task table). Unset or empty → use ``_glue_sam_rho_for_task``.
    """
    env = os.environ.get("SAM_GLUE_RHO_OVERRIDE", "").strip()
    if env:
        rho = float(env)
        if rho <= 0:
            raise RuntimeError(f"SAM_GLUE_RHO_OVERRIDE must be > 0, got {rho!r}")
    else:
        rho = float(_glue_sam_rho_for_task(task))
    for name in _GLUE_SAM_RHO_ATTRS:
        setattr(Args.data, name, rho)
    return rho


def main() -> None:
    Args.parse_args()
    if Args.evalEvery < 1:
        raise ValueError("evalEvery must be positive")

    task = Args.glue_task.lower().replace("-", "")
    if task == "sts-b":
        task = "stsb"
    if task not in GLUE_TASK_KEYS:
        raise RuntimeError(f"Unknown glue_task={Args.glue_task!r}. Choose from {sorted(GLUE_TASK_KEYS)}")

    sam_rho_applied = _apply_glue_sam_rho_overrides(task)

    if Args.optimizer in EXCLUDED_OPTIMIZERS:
        raise RuntimeError(
            f"Optimizer {Args.optimizer} is excluded on this benchmark. "
            f"Use SGD / MSAM / SAM / ASAM / ESAM / lookSAM / FriendlySAM / FisherSAM / MongeSAM / GSAM / CondSAM / CondSam2."
        )

    is_reg = is_glue_regression(task)
    if is_reg and Args.optimizer in SAM_SGD_VARIANTS and Args.optimizer not in SAM_REGRESSION_STEP_VARIANTS:
        raise RuntimeError(
            f"STS-B is regression (MSE). Optimizer {Args.optimizer} still uses cross-entropy in optimizer.step. "
            f"Use SGD, MSAM, SAM, CondSAM, CondSam2, for glue_task=stsb."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("nlp/run_glue_bert.py expects CUDA + NCCL (same as main.py).")

    rank = int(os.environ["SLURM_PROCID"]) if "SLURM_PROCID" in os.environ else int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank % torch.cuda.device_count())))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    run_seed = initialize()

    device = torch.device(f"cuda:{local_rank}")
    num_labels = glue_num_labels(task)

    raw = load_dataset("glue", task)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tok_fn = _tokenize_builder(task, tokenizer)

    train_ds = raw["train"].map(tok_fn, batched=True, remove_columns=raw["train"].column_names)
    collator = DataCollatorWithPadding(tokenizer)

    per_device_bs = max(1, Args.batchSize // world_size)
    if Args.batchSize % world_size != 0 and rank == 0:
        print(f"[WARN] batchSize {Args.batchSize} not divisible by world_size {world_size}; using per_device {per_device_bs}")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=per_device_bs,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=Args.dataThreads,
        collate_fn=collator,
        pin_memory=True,
    )

    hf = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    wrapper = BertGlueWrapper(hf).to(device)
    # SAM-family steps call model.module.setBatchNormTracking (vision DDP convention); always use DDP.
    ddp = DDP(wrapper, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    n_train = len(train_ds)
    optimizer = getOptimizer(_unwrap(ddp).parameters(), n_data=n_train)

    total_steps = max(1, len(train_loader)) * Args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    run_dir = _run_directory(task)
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[GLUE] task={task} SAM-family ρ (all rhos)={sam_rho_applied}")
        print(f"[RUN_DIR] {run_dir}")

    tracking.start({**vars(Args.data), "actual_seed": run_seed}, run_dir, job_type="glue")
    history: list[dict] = []
    epoch_train_times_sec: list[float] = []
    total_train_time_sec = 0.0

    for epoch in range(1, Args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch - 1)
        ddp.train()
        running_loss = 0.0
        n_batches = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        epoch_train_start = time.perf_counter()

        if Args.optimizer in MSAM_VARIANT:
            optimizer.move_up_to_momentumAscent()

        for batch in train_loader:
            batch = _to_device(batch, device)
            labels = batch["labels"]

            optimizer.zero_grad(set_to_none=True)

            if Args.optimizer in SAM_SGD_VARIANTS:
                use_sam_regression = is_reg and Args.optimizer in SAM_REGRESSION_STEP_VARIANTS
                loss, _ = optimizer.step(ddp, batch, labels, regression=use_sam_regression)
            else:
                # SGD, MSAM, etc.: one forward + backward + step (MSAM adds epoch-wise ascent in main.py style).
                logits = ddp(batch)
                if is_reg:
                    loss = F.mse_loss(logits.squeeze(-1), labels.float())
                else:
                    loss = F.cross_entropy(logits, labels)
                loss.backward()
                if Args.grad_clip_norm and Args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(ddp.parameters(), Args.grad_clip_norm)
                optimizer.step()

            scheduler.step()
            running_loss += float(loss.detach().mean() if loss.dim() > 0 else loss)
            n_batches += 1

        if Args.optimizer in MSAM_VARIANT:
            optimizer.move_back_from_momentumAscent()

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        epoch_train_elapsed = time.perf_counter() - epoch_train_start
        total_train_time_sec += epoch_train_elapsed
        epoch_train_times_sec.append(epoch_train_elapsed)

        if rank == 0 and Args.verbose and (
            os.environ.get("SAM_CONSOLE_EVAL_ONLY", "0") != "1"
            or should_evaluate(epoch, Args.epochs, Args.evalEvery)
        ):
            print(
                f"[train] epoch {epoch}/{Args.epochs} loss={running_loss / max(1, n_batches):.4f} "
                f"| train_time={epoch_train_elapsed:.3f}s"
            )

        loss_totals = torch.tensor([running_loss, n_batches], device=device, dtype=torch.float64)
        dist.all_reduce(loss_totals)
        global_train_loss = (loss_totals[0] / loss_totals[1].clamp_min(1)).item()
        if rank == 0:
            tracking.log({"epoch": epoch, "train/loss": global_train_loss,
                          "optimizer/lr": optimizer.param_groups[0]["lr"],
                          "time/train_seconds": epoch_train_elapsed})
            if should_evaluate(epoch, Args.epochs, Args.evalEvery):
                _unwrap(ddp).eval()
                metrics_out: dict[str, float] = {}
                for split in mnli_eval_split_names(task):
                    val_raw = raw[split]
                    val_tok = val_raw.map(tok_fn, batched=True, remove_columns=val_raw.column_names)
                    val_loader = DataLoader(
                        val_tok,
                        batch_size=per_device_bs,
                        shuffle=False,
                        num_workers=Args.dataThreads,
                        collate_fn=collator,
                        pin_memory=True,
                    )
                    metric = _build_metric(task)
                    preds_all: list = []
                    refs_all: list = []

                    with torch.no_grad():
                        for vb in val_loader:
                            vb = _to_device(vb, device)
                            lab = vb["labels"]
                            logits = _unwrap(ddp)(vb)
                            if is_reg:
                                preds_all.extend(logits.squeeze(-1).cpu().numpy().tolist())
                                refs_all.extend(lab.float().cpu().numpy().tolist())
                            else:
                                preds_all.extend(torch.argmax(logits, -1).cpu().numpy().tolist())
                                refs_all.extend(lab.cpu().numpy().tolist())

                    m = metric.compute(predictions=preds_all, references=refs_all)
                    sub = extract_official_glue_metrics(task, split, m)
                    metrics_out.update(sub)

                if task == "mnli":
                    add_mnli_mean_if_complete(metrics_out)

                tracking.log({"epoch": epoch, **{f"validation/{key}": value for key, value in metrics_out.items()}})
                history.append({"epoch": epoch, **metrics_out})
                print(
                    f"[val] epoch {epoch} official: {format_official_metrics_line(task, metrics_out)} "
                    f"| json_keys={list(metrics_out.keys())}"
                )
                with open(run_dir / "metrics_history.json", "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)
            with open(run_dir / "params.json", "w", encoding="utf-8") as f:
                json.dump(vars(Args.data), f, indent=2)

            ckpt = {
                "epoch": epoch,
                "model_state_dict": _unwrap(ddp).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "glue_task": task,
                "optimizer_name": Args.optimizer,
            }
            torch.save(ckpt, run_dir / f"checkpoint_epoch{epoch}.pt")

        dist.barrier()

    if rank == 0:
        avg_train_time_sec = total_train_time_sec / max(1, len(epoch_train_times_sec))
        with open(run_dir / "training_time.txt", "w", encoding="utf-8") as f:
            f.write(f"optimizer: {Args.optimizer}\n")
            f.write(f"glue_task: {task}\n")
            f.write(f"epochs: {Args.epochs}\n")
            f.write(f"total_train_time_seconds: {total_train_time_sec:.6f}\n")
            f.write(f"total_train_time_hhmmss: {_format_seconds(total_train_time_sec)}\n")
            f.write(f"avg_train_time_per_epoch_seconds: {avg_train_time_sec:.6f}\n")
            f.write(f"avg_train_time_per_epoch_hhmmss: {_format_seconds(avg_train_time_sec)}\n")
            f.write("epoch_train_times_seconds: " + ", ".join(f"{x:.6f}" for x in epoch_train_times_sec) + "\n")

        torch.save(
            {
                "model_state_dict": _unwrap(ddp).state_dict(),
                "glue_task": task,
                "optimizer_name": Args.optimizer,
            },
            run_dir / "checkpoint_final.pt",
        )
        print(f"[done] task={task} optimizer={Args.optimizer} artifacts -> {run_dir}")

    dist.destroy_process_group()


if __name__ == "__main__":
    tracking.execute(main)
