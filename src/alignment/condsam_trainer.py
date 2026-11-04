"""Logits-CondSAM: the CondSAM perturbation direction applied to the output head only.

Baseline for logits-SAM. Everything is shared with `LogitsSAMTrainer` (same forward to
the hidden states, same pre/post loss, same ZeRO-3 delta-logits path); only the
perturbation direction of `lm_head.weight` changes:

    logits-SAM :  e = rho * g / ||g||
    CondSAM    :  e = rho * d / ||d||,
                  d = ((x'Ax)(y'By))^2 / (2 s) * y x'  -  ||A||^2 ||B||^2 / (2 n (||g||^2 + eps)) * g,
                  s = y' g x (+/- eps),  n = numel(g)

as in CondSAM (OrthoSAM/optimizer/condsam.py, condsam1.py), with the empirical Kronecker
factors of the head W (vocab x hidden):

    A (hidden x hidden) = EMA of h h'           over the rows of the first pass,
    B (vocab  x vocab ) = EMA of (N g_z)(N g_z)' over the same rows,

where h are the hidden states entering lm_head, g_z the gradients of the pre-perturbation
loss w.r.t. the logits, and N the number of preference pairs of the micro-batch (the
loss is a mean over pairs, as CondSAM scales by the batch size). x and y follow the top
eigenvectors of A and B by power iteration (one step per update by default). Until the
first update the direction is the plain gradient, as in CondSAM.

Differences with the image-classification CondSAM, all forced by the DPO setting:
- Statistics use only the rows that enter the loss (completion tokens, `loss_mask`),
  not prompt or padding positions.
- The factors are updated on every SAM step, i.e. every micro-batch (`compute_loss`
  call), or every `condsam_update_interval`-th one.
- Scratch buffers are released after every compute_loss; only the factors persist.
- Factor statistics are all-reduced across ranks (as CondSAM does under DDP) unless
  `condsam_sync_stats=False`; the head gradient itself stays per rank, as in logits-SAM.
- The factors are not saved in checkpoints: resuming restarts them from the identity.

Memory: B is vocab x vocab in fp32 (32000^2 * 4 B = 4.1 GB for Mistral), built in
blocks of `condsam_vocab_block` rows so only one block is extra.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Union

import torch
import torch.distributed as dist

from .sam_trainer import LogitsSAMTrainer


@contextmanager
def _tf32(enabled: bool):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


class LogitsCondSAMTrainer(LogitsSAMTrainer):
    def __init__(
        self,
        *args,
        condsam_beta: float = 0.95,
        condsam_eps: float = 1e-8,
        condsam_power_iters: int = 1,
        condsam_update_interval: int = 1,
        condsam_sync_stats: bool = True,
        condsam_vocab_block: int = 4096,
        condsam_row_chunk: int = 4096,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.sam_adaptive:
            raise ValueError("LogitsCondSAMTrainer does not support sam_adaptive.")
        if not 0 <= condsam_beta < 1:
            raise ValueError("condsam_beta must be in [0, 1).")
        if condsam_eps <= 0:
            raise ValueError("condsam_eps must be positive.")
        for name, value in [
            ("condsam_power_iters", condsam_power_iters),
            ("condsam_update_interval", condsam_update_interval),
            ("condsam_vocab_block", condsam_vocab_block),
            ("condsam_row_chunk", condsam_row_chunk),
        ]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.condsam_beta = condsam_beta
        self.condsam_eps = condsam_eps
        self.condsam_power_iters = condsam_power_iters
        self.condsam_update_interval = condsam_update_interval
        self.condsam_sync_stats = condsam_sync_stats
        self.condsam_vocab_block = condsam_vocab_block
        self.condsam_row_chunk = condsam_row_chunk
        self._condsam_state: dict[str, torch.Tensor] = {}
        self._condsam_calls = 0
        self._condsam_prepared = None
        self._condsam_buffers: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ plumbing
    def _prepare_decoder_batch(self, batch):
        prepared = super()._prepare_decoder_batch(batch)
        # compute_loss prepares the batch once, before the perturbation is built.
        self._condsam_prepared = prepared
        return prepared

    def _condsam_distributed(self) -> bool:
        return self.condsam_sync_stats and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    def _condsam_rows(self, hidden_states: torch.Tensor, grad_logits: torch.Tensor):
        """(all hidden rows, all logit-gradient rows, rows entering the loss, pairs)."""
        h = hidden_states.detach().reshape(-1, hidden_states.shape[-1])
        g = grad_logits.detach().reshape(-1, grad_logits.shape[-1])
        prepared = self._condsam_prepared or {}
        mask = prepared.get("loss_mask")
        if mask is not None and tuple(mask.shape) == tuple(grad_logits.shape[:2]):
            index = mask.reshape(-1).nonzero(as_tuple=True)[0]
        else:
            index = torch.arange(h.shape[0], device=h.device)
        num_examples = int(prepared.get("num_examples") or grad_logits.shape[0])
        return h, g, index, num_examples

    def _rowwise_matmul_t(self, left: torch.Tensor, right: torch.Tensor, out: torch.Tensor, alpha: float = 1.0):
        """out += alpha * left' right in fp32, over chunks of rows (bf16 inputs are exact in TF32)."""
        low = left.dtype in (torch.bfloat16, torch.float16) and right.dtype in (torch.bfloat16, torch.float16)
        with _tf32(low and left.is_cuda):
            for start in range(0, left.shape[0], self.condsam_row_chunk):
                end = start + self.condsam_row_chunk
                out.addmm_(left[start:end].float().t(), right[start:end].float(), alpha=alpha)
        return out

    def _condsam_all_reduce(self, tensor: torch.Tensor):
        if self._condsam_distributed():
            dist.all_reduce(tensor)
        return tensor

    @staticmethod
    def _check_finite(checks: list[tuple[str, torch.Tensor]], distributed: bool):
        flags = torch.stack([~torch.isfinite(value).all() for _, value in checks]).to(torch.int32)
        if distributed:
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        for (name, _), bad in zip(checks, flags.tolist()):
            if bad:
                raise FloatingPointError(f"CondSAM non-finite {name}")

    # ------------------------------------------------------------------ factors
    @torch.no_grad()
    def _condsam_init(self, hidden_size: int, vocab_size: int, device):
        state = self._condsam_state
        state["A"] = torch.eye(hidden_size, device=device, dtype=torch.float32)
        state["B"] = torch.eye(vocab_size, device=device, dtype=torch.float32)
        for key, size in (("x", hidden_size), ("y", vocab_size)):
            v = torch.randn(size, device=device, dtype=torch.float32)
            v /= v.norm().clamp_min(torch.finfo(v.dtype).tiny)
            if self._condsam_distributed():
                dist.broadcast(v, src=0)
            state[key] = v

    def _condsam_buffer(self, name: str, shape, device) -> torch.Tensor:
        """Zeroed fp32 buffer cached across steps (condsam1: buffers are reused)."""
        buffer = self._condsam_buffers.get(name)
        if buffer is None or tuple(buffer.shape) != tuple(shape) or buffer.device != torch.device(device):
            buffer = torch.empty(shape, device=device, dtype=torch.float32)
            self._condsam_buffers[name] = buffer
        return buffer.zero_()

    @torch.no_grad()
    def _condsam_update(self, h_rows: torch.Tensor, g_rows: torch.Tensor, num_examples: int) -> list:
        """EMA of the factors and power iteration, in place. Returns the finite checks;
        the caller synchronizes them with the direction norm (one host sync per step)."""
        hidden_size, vocab_size = h_rows.shape[1], g_rows.shape[1]
        device = g_rows.device
        count = torch.tensor(float(h_rows.shape[0]), device=device, dtype=torch.float64)
        self._condsam_all_reduce(count)
        # A micro-batch with no completion tokens on any rank keeps the factors (weight 0).
        weight = (count > 0).float() * (1.0 - self.condsam_beta)
        denominator = count.clamp_min(1).float()
        state = self._condsam_state
        if "A" not in state or state["A"].shape[0] != hidden_size or state["B"].shape[0] != vocab_size:
            self._condsam_init(hidden_size, vocab_size, device)

        a_sum = self._condsam_buffer("a_sum", (hidden_size, hidden_size), device)
        self._rowwise_matmul_t(h_rows, h_rows, a_sum)
        self._condsam_all_reduce(a_sum)
        state["A"].lerp_(a_sum.div_(denominator), weight)

        # B = EMA of (N g)'(N g) / rows. B is symmetric: only the upper block triangle is
        # computed and reduced, then mirrored (condsam1's upper-triangular Gram), and the
        # N^2 factor goes through alpha instead of materializing N * g.
        scale = float(num_examples) ** 2
        g32 = g_rows.float()
        B, step = state["B"], self.condsam_vocab_block
        with _tf32(g_rows.dtype in (torch.bfloat16, torch.float16) and g_rows.is_cuda):
            # One flat buffer for the largest block; each block is a contiguous view of it
            # (NCCL needs contiguous tensors).
            storage = self._condsam_buffer("b_block", (min(step, vocab_size) * vocab_size,), device)
            for start in range(0, vocab_size, step):
                end = min(start + step, vocab_size)
                rows, cols = end - start, vocab_size - start
                block = storage[: rows * cols].view(rows, cols).zero_()
                block.addmm_(g32[:, start:end].t(), g32[:, start:], alpha=scale)
                self._condsam_all_reduce(block)
                upper = B[start:end, start:]
                upper.lerp_(block.div_(denominator), weight)
                if end < vocab_size:
                    B[end:, start:end].copy_(upper[:, end - start:].t())
        del g32

        checks = []
        tiny = torch.finfo(torch.float64).tiny
        with _tf32(False):
            for matrix, vector in (("A", "x"), ("B", "y")):
                for _ in range(self.condsam_power_iters):
                    v = state[matrix] @ state[vector]
                    norm = torch.linalg.vector_norm(v.double())
                    checks.append(norm)
                    candidate = (v.double() / norm.clamp_min(tiny)).float()
                    state[vector] = torch.where(norm > 0, candidate, state[vector])
            # Factor-only terms of the direction, refreshed with the factors.
            state["quad_A"] = state["x"] @ state["A"] @ state["x"]
            state["quad_B"] = state["y"] @ state["B"] @ state["y"]
            state["sq_A"] = torch.linalg.vector_norm(state["A"]).double().square()
            state["sq_B"] = torch.linalg.vector_norm(state["B"]).double().square()
        # A non-finite statistic reaches the factor through the EMA, hence its norm.
        return [state["sq_A"], state["sq_B"]] + checks

    # ------------------------------------------------------------------ direction
    @torch.no_grad()
    def _condsam_perturbation(self, hidden_states: torch.Tensor, grad_logits: torch.Tensor) -> torch.Tensor:
        """rho * d / ||d|| for the head weight (vocab x hidden, fp32)."""
        h, g, index, num_examples = self._condsam_rows(hidden_states, grad_logits)
        vocab_size, hidden_size = g.shape[1], h.shape[1]

        # Head gradient of the pre-perturbation loss (per rank, as in logits-SAM).
        # Rows outside the loss have zero logit gradient, so the selected rows suffice.
        h_rows, g_rows = h.index_select(0, index), g.index_select(0, index)
        grad_w = self._condsam_buffer("grad_w", (vocab_size, hidden_size), g.device)
        self._rowwise_matmul_t(g_rows, h_rows, grad_w)

        factor_checks = []
        if self._condsam_calls % self.condsam_update_interval == 0:
            factor_checks = self._condsam_update(h_rows, g_rows, num_examples)
        self._condsam_calls += 1
        del h_rows, g_rows

        state, eps = self._condsam_state, self.condsam_eps
        if "A" not in state:
            direction = grad_w.clone()
        else:
            x, y = state["x"], state["y"]
            with _tf32(False):
                s = y @ grad_w @ x
            denominator = s + torch.where(s >= 0, eps, -eps)
            spectral = state["quad_A"] * state["quad_B"]
            direction = (spectral.square() / (2 * denominator)) * torch.outer(y, x)
            ab = (state["sq_A"] * state["sq_B"]).float()
            direction -= (ab / (2 * grad_w.numel() * (grad_w.square().sum() + eps))) * grad_w
            direction = torch.where(torch.count_nonzero(direction) == 0, grad_w, direction)

        norm = torch.linalg.vector_norm(direction).double()
        # One host synchronization for every finite check and the norm (condsam1).
        invalid = torch.stack(
            [(~torch.isfinite(t)).any() for t in factor_checks] + [(~torch.isfinite(norm)).any()]
        ).to(torch.int32)
        if self._condsam_distributed():
            dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
        values = torch.cat([invalid.double(), norm.view(1)]).tolist()
        if factor_checks and any(values[: len(factor_checks)]):
            raise FloatingPointError("CondSAM non-finite factor statistics or power iteration")
        if values[len(factor_checks)]:
            raise FloatingPointError("CondSAM non-finite perturbation direction")
        if values[-1] == 0:
            return direction.zero_()
        return direction.mul_((self.sam_rho / norm).float())

    # ------------------------------------------------------------------ overrides
    def _compute_sam_perturbation(
        self,
        loss_pre: torch.Tensor,
        clean_logits_pre: torch.Tensor,
        hidden_states_pre: torch.Tensor,
        head_weight: torch.Tensor,
    ) -> torch.Tensor:
        (grad_logits,) = torch.autograd.grad(loss_pre, clean_logits_pre, retain_graph=False, create_graph=False)
        perturbation = self._condsam_perturbation(hidden_states_pre, grad_logits)
        return perturbation.to(head_weight.dtype)

    def _condsam_delta_logits(self, hidden_states, grad_logits, out_dtype):
        perturbation = self._condsam_perturbation(hidden_states, grad_logits)
        work_dtype = grad_logits.dtype
        h_post = hidden_states.reshape(-1, hidden_states.shape[-1]).to(work_dtype, copy=False)
        # Gradient flows to the hidden states through h_post, as in the logits-SAM path.
        delta_logits = h_post.matmul(perturbation.to(work_dtype).t())
        return delta_logits.to(out_dtype).view_as(grad_logits)

    def _compute_nonadaptive_delta_logits(self, hidden_states, grad_logits, out_dtype):
        return self._condsam_delta_logits(hidden_states, grad_logits, out_dtype)

    def _compute_nonadaptive_delta_logits_chunked(self, hidden_states, grad_logits, out_dtype, chunk_size=None):
        return self._condsam_delta_logits(hidden_states, grad_logits, out_dtype)

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        try:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        finally:
            self._condsam_prepared = None
            # Hand the scratch buffers (~1 GB for Mistral) back to the caching allocator, so
            # the optimizer step can reuse them; only the factors persist across steps.
            self._condsam_buffers.clear()
