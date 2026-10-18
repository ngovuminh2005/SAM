#!/usr/bin/env python3
"""
Approximate top-k Hessian eigenvalues with Lanczos from a saved training checkpoint.

Accurate CIFAR-100 defaults (WRN comparison):
  - split=train
  - disable_aug=True
  - batch_size=128
  - num_batches=20
  - lanczos_iters=100

Typical use (run directory contains params.json + final.model):
  python scripts/compute_hessian_topk.py \
    --run_dir logs/CIFAR100_SAM_WRN_rho0.1_lr0.5
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import getModel
from utility.args import Args
from utility import tracking
from utility.data import DataLoader


def _sanitize_tag(value: object) -> str:
    return str(value).replace(" ", "").replace("/", "_")


def _pick_free_local_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _load_checkpoint_state_dict(checkpoint_path: Path) -> dict:
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict):
        if "modelState" in state:
            return state["modelState"]
        if "model_state_dict" in state:
            return state["model_state_dict"]
        return state
    raise RuntimeError(f"Unsupported checkpoint format at {checkpoint_path}")


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    # main.py may save compiled model weights with "_orig_mod." prefix.
    if not state_dict:
        return state_dict
    if all(str(k).startswith("_orig_mod.") for k in state_dict.keys()):
        return {str(k).replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _flatten_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def _hvp(model: torch.nn.Module, params: list[torch.nn.Parameter], batch, vec: torch.Tensor, device: torch.device) -> torch.Tensor:
    inputs, targets = (x.to(device, non_blocking=True) for x in batch)
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grads = _flatten_tensors([g for g in grads])
    gv = torch.dot(flat_grads, vec)
    hv = torch.autograd.grad(gv, params, retain_graph=False)
    return _flatten_tensors([h.detach() for h in hv])


def _lanczos_topk(hvp_fn, n_params: int, k: int, iters: int, device: torch.device) -> torch.Tensor:
    q = torch.randn(n_params, device=device)
    q = q / (torch.norm(q) + 1e-12)
    basis: list[torch.Tensor] = []
    alphas: list[torch.Tensor] = []
    betas: list[torch.Tensor] = []
    beta_prev = torch.tensor(0.0, device=device)
    q_prev = torch.zeros_like(q)

    for _ in range(iters):
        basis.append(q)
        z = hvp_fn(q)
        if len(basis) > 1:
            z = z - beta_prev * q_prev
        alpha = torch.dot(q, z)
        alphas.append(alpha)
        z = z - alpha * q

        # Full re-orthogonalization for numerical stability.
        for b in basis:
            z = z - torch.dot(z, b) * b

        beta = torch.norm(z)
        if beta < 1e-10:
            break
        betas.append(beta)
        q_prev = q
        q = z / beta
        beta_prev = beta

    m = len(alphas)
    t = torch.zeros((m, m), device=device)
    for i, a in enumerate(alphas):
        t[i, i] = a
    for i, b in enumerate(betas):
        if i + 1 < m:
            t[i, i + 1] = b
            t[i + 1, i] = b

    eigvals = torch.linalg.eigvalsh(t)
    return torch.topk(eigvals, k=min(k, eigvals.numel()), largest=True).values


def _resolve_paths(args) -> tuple[Path, Path]:
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        params_path = run_dir / "params.json"
        checkpoint_path = run_dir / args.checkpoint_name
    else:
        params_path = Path(args.params_json).expanduser().resolve()
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    return params_path, checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute top-k Hessian eigenvalues with Lanczos.")
    parser.add_argument("--run_dir", type=str, default="", help="Run directory containing params.json and checkpoint file.")
    parser.add_argument("--params_json", type=str, default="", help="Path to params.json (if not using --run_dir).")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint model file (if not using --run_dir).")
    parser.add_argument("--checkpoint_name", type=str, default="final.model", help="Checkpoint filename under --run_dir.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split for Hessian estimate.")
    parser.add_argument("--num_batches", type=int, default=20, help="Number of batches to average for Hessian-vector products.")
    parser.add_argument("--k", type=int, default=5, help="Top-k eigenvalues to report.")
    parser.add_argument("--lanczos_iters", type=int, default=100, help="Lanczos iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the Lanczos initial vector.")
    parser.add_argument("--disable_aug", action="store_true", default=True, help="Disable data augmentation flags from params.json for analysis (default: True).")
    parser.add_argument("--enable_aug", action="store_true", help="Use original augmentations from params.json (overrides --disable_aug).")
    parser.add_argument("--batch_size", type=int, default=128, help="Override batch size for Hessian evaluation (default: 128).")
    parser.add_argument("--output_dir", type=str, default="logs_hessian", help="Directory for saving Hessian result text/json files.")
    args = parser.parse_args()

    params_path, checkpoint_path = _resolve_paths(args)
    if not params_path.exists():
        raise RuntimeError(f"params.json not found: {params_path}")
    if not checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")

    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    if args.enable_aug:
        args.disable_aug = False

    if args.disable_aug:
        params["flip"] = False
        params["crop"] = False
        params["cut"] = False
        params["randAugment"] = False
        params["mixup"] = False
    if args.batch_size > 0:
        params["batchSize"] = int(args.batch_size)

    Args.data = SimpleNamespace(**params)
    if not torch.cuda.is_available():
        raise RuntimeError("This script expects CUDA (utility.data.DataLoader uses distributed CUDA samplers).")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    try:
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            # Avoid EADDRINUSE when multiple Hessian scripts run concurrently.
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = _pick_free_local_port()
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            dist.init_process_group(backend="nccl", init_method="env://", rank=0, world_size=1)

        data = DataLoader()
        model = getModel()(num_classes=data.numClasses).to(device)
        model_state = _normalize_state_dict_keys(_load_checkpoint_state_dict(checkpoint_path))
        model.load_state_dict(model_state, strict=True)
        model.eval()

        loader = data.train if args.split == "train" else data.test
        batches = []
        for idx, batch in enumerate(loader):
            batches.append(batch)
            if idx + 1 >= args.num_batches:
                break
        if not batches:
            raise RuntimeError("No batches were loaded for Hessian computation.")

        params_req = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in params_req)

        def hvp_avg(vec: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(vec)
            for batch in batches:
                out = out + _hvp(model, params_req, batch, vec, device)
            return out / float(len(batches))

        probe_vec = torch.randn(n_params, device=device)
        probe_vec = probe_vec / (torch.norm(probe_vec) + 1e-12)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = hvp_avg(probe_vec)
        torch.cuda.synchronize(device)
        hvp_avg_sec = time.perf_counter() - t0
        est_total_sec = hvp_avg_sec * float(args.lanczos_iters)

        topk = _lanczos_topk(
            hvp_fn=hvp_avg,
            n_params=n_params,
            k=args.k,
            iters=args.lanczos_iters,
            device=device,
        )

        print(f"[INFO] params_json: {params_path}")
        print(f"[INFO] checkpoint: {checkpoint_path}")
        print(
            f"[INFO] split={args.split} num_batches={len(batches)} k={args.k} "
            f"lanczos_iters={args.lanczos_iters} batch_size={params.get('batchSize')}"
        )
        print(
            "[ESTIMATE] "
            f"one_hvp_avg={hvp_avg_sec:.3f}s | expected_total~{est_total_sec/60.0:.2f} min "
            f"({est_total_sec:.1f}s)"
        )
        print("[RESULT] top-k Hessian eigenvalues (descending):")
        for i, value in enumerate(topk.tolist(), start=1):
            print(f"  lambda_{i}: {value:.6f}")

        dataset_tag = _sanitize_tag(params.get("dataset", "unknown_dataset"))
        model_tag = _sanitize_tag(params.get("model", "unknown_model"))
        if model_tag == "WRN":
            model_tag = f"{model_tag}{_sanitize_tag(params.get('depth', 'na'))}_{_sanitize_tag(params.get('widthFactor', 'na'))}"
        method_tag = _sanitize_tag(params.get("optimizer", "unknown_optimizer"))
        split_tag = _sanitize_tag(args.split)
        out_dir = Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            f"{dataset_tag}__{model_tag}__{method_tag}__split_{split_tag}"
            f"__k{args.k}__iters{args.lanczos_iters}__b{len(batches)}__seed{args.seed}"
        )
        txt_path = out_dir / f"{stem}.txt"
        json_path = out_dir / f"{stem}.json"

        txt_lines = [
            f"dataset={params.get('dataset')}",
            f"model={params.get('model')}",
            f"depth={params.get('depth')}",
            f"widthFactor={params.get('widthFactor')}",
            f"optimizer={params.get('optimizer')}",
            f"checkpoint={checkpoint_path}",
            f"params_json={params_path}",
            f"split={args.split}",
            f"num_batches={len(batches)}",
            f"batch_size={params.get('batchSize')}",
            f"k={args.k}",
            f"lanczos_iters={args.lanczos_iters}",
            f"seed={args.seed}",
            f"disable_aug={args.disable_aug}",
            f"one_hvp_avg_sec={hvp_avg_sec:.6f}",
            f"expected_total_sec={est_total_sec:.6f}",
            "topk_eigenvalues=" + ", ".join(f"{v:.8f}" for v in topk.tolist()),
        ]
        txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

        payload = {
            "dataset": params.get("dataset"),
            "model": params.get("model"),
            "depth": params.get("depth"),
            "widthFactor": params.get("widthFactor"),
            "optimizer": params.get("optimizer"),
            "checkpoint": str(checkpoint_path),
            "params_json": str(params_path),
            "split": args.split,
            "num_batches": len(batches),
            "batch_size": params.get("batchSize"),
            "k": args.k,
            "lanczos_iters": args.lanczos_iters,
            "seed": args.seed,
            "disable_aug": bool(args.disable_aug),
            "one_hvp_avg_sec": float(hvp_avg_sec),
            "expected_total_sec": float(est_total_sec),
            "topk_eigenvalues": [float(v) for v in topk.tolist()],
        }
        tracking.start({**params, **vars(args)}, out_dir, name=checkpoint_path.parent.name + "_hessian", job_type="hessian")
        tracking.log({f"hessian/lambda_{i+1}": float(value) for i, value in enumerate(topk.tolist())})
        tracking.summary(payload)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[SAVE] txt={txt_path}")
        print(f"[SAVE] json={json_path}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    tracking.execute(main)
