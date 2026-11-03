"""Fused kernels for CondSAM: out += alpha * rows.T @ rows for its Kronecker factors,
and the power-iteration normalization.

On CUDA with bf16/fp16 rows, a Triton kernel reads the half-precision values
directly (their products are exact in fp32, which accumulates), computes only
the upper-triangular tiles of the symmetric result and splits the long row
dimension across programs. Partial tiles are summed in a fixed order, so the
result is deterministic. Elsewhere it falls back to chunked fp32 addmm.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

ROW_CHUNK = 262144
WORKSPACE_BYTES = 512 * 1024**2


def _addmm(rows, out, alpha):
    # bf16/fp16 values are exact in TF32, so TF32 products equal fp32 ones
    exact_in_tf32 = rows.dtype in (torch.bfloat16, torch.float16)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = previous or exact_in_tf32
    try:
        for chunk in rows.split(ROW_CHUNK):
            chunk = chunk.float()
            out.addmm_(chunk.T, chunk, alpha=alpha)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return out


if triton is not None:
    @triton.jit
    def _gram_kernel(x_ptr, ws_ptr, tile_i_ptr, tile_j_ptr, n_rows, dim, rows_per_split,
                     stride_n, stride_d, stride_ws,
                     BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr, INNER: tl.constexpr):
        pair = tl.program_id(0)
        split = tl.program_id(1)
        ti = tl.load(tile_i_ptr + pair)
        tj = tl.load(tile_j_ptr + pair)
        offs_i = ti * BLOCK_D + tl.arange(0, BLOCK_D)
        offs_j = tj * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_i = offs_i < dim
        mask_j = offs_j < dim
        start = split * rows_per_split
        stop = tl.minimum(start + rows_per_split, n_rows)
        # Two-level sum: tensor cores accumulate INNER blocks of rows, which
        # are then added into a separate fp32 total, bounding rounding growth.
        acc = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
        for outer in range(start, stop, BLOCK_N * INNER):
            part = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
            for k in range(outer, tl.minimum(outer + BLOCK_N * INNER, stop), BLOCK_N):
                offs_n = k + tl.arange(0, BLOCK_N)
                mask_n = offs_n < stop
                a = tl.load(x_ptr + offs_n[:, None] * stride_n + offs_i[None, :] * stride_d,
                            mask=mask_n[:, None] & mask_i[None, :], other=0.)
                b = tl.load(x_ptr + offs_n[:, None] * stride_n + offs_j[None, :] * stride_d,
                            mask=mask_n[:, None] & mask_j[None, :], other=0.)
                part = tl.dot(tl.trans(a), b, part)
            acc += part
        base = ws_ptr + split * stride_ws
        mask = mask_i[:, None] & mask_j[None, :]
        tl.store(base + offs_i[:, None] * dim + offs_j[None, :], acc, mask=mask)
        if ti != tj:
            # The mirrored tile of the symmetric result
            tl.store(base + offs_j[None, :] * dim + offs_i[:, None], acc, mask=mask)


_TILES = {}


def _tile_pairs(n_tiles, device):
    key = (n_tiles, device)
    if key not in _TILES:
        i, j = torch.triu_indices(n_tiles, n_tiles)
        _TILES[key] = (i.to(device=device, dtype=torch.int32), j.to(device=device, dtype=torch.int32))
    return _TILES[key]


def _gram_triton(rows, out, alpha):
    n_rows, dim = rows.shape
    # 128-wide tiles are faster per FLOP; 64 avoids heavy padding (e.g. d=147)
    padded = lambda b: triton.cdiv(dim, b) * b
    block_d = 128 if dim > 64 and padded(128) ** 2 <= 1.3 * padded(64) ** 2 else 64
    n_tiles = triton.cdiv(dim, block_d)
    tile_i, tile_j = _tile_pairs(n_tiles, rows.device)
    pairs = tile_i.numel()
    # Enough programs to fill the GPU, bounded by the workspace size
    sms = torch.cuda.get_device_properties(rows.device).multi_processor_count
    splits = max(1, min(triton.cdiv(4 * sms, pairs), triton.cdiv(n_rows, 4096),
                        WORKSPACE_BYTES // (4 * dim * dim)))
    block_n = 64 if block_d == 128 else 128
    rows_per_split = triton.cdiv(triton.cdiv(n_rows, splits), block_n) * block_n
    splits = triton.cdiv(n_rows, rows_per_split)
    ws = torch.empty((splits, dim, dim), device=rows.device, dtype=torch.float32)
    _gram_kernel[(pairs, splits)](
        rows, ws, tile_i, tile_j, n_rows, dim, rows_per_split,
        rows.stride(0), rows.stride(1), dim * dim,
        BLOCK_D=block_d, BLOCK_N=block_n, INNER=8, num_warps=8, num_stages=3)
    out.add_(ws.sum(0) if splits > 1 else ws[0], alpha=alpha)
    return out


def gram_accumulate(rows, out, alpha=1., use_triton=True):
    """out += alpha * rows.T @ rows, with rows of shape (n, d) and out (d, d) in fp32."""
    # Small products are launch-bound; cuBLAS handles them better
    # Rows must be 16-byte aligned for vectorized loads (e.g. not d=147)
    aligned = (rows.stride(0) * rows.element_size()) % 16 == 0 and rows.data_ptr() % 16 == 0
    if (use_triton and triton is not None and rows.is_cuda and out.is_cuda and aligned
            and rows.stride(1) == 1 and rows.dtype in (torch.bfloat16, torch.float16)
            and rows.shape[0] >= 16384):
        return _gram_triton(rows, out, alpha)
    return _addmm(rows, out, alpha)


if triton is not None:
    @triton.jit
    def _normalize_kernel(x_ptr, v_ptr, norm_ptr, active_ptr, n, dim, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        row = offs // dim
        norm = tl.load(norm_ptr + row, mask=mask, other=0.)
        take = (tl.load(active_ptr + row, mask=mask, other=0) != 0) & (norm > 0)
        v = tl.load(v_ptr + offs, mask=mask)
        x = tl.load(x_ptr + offs, mask=mask)
        # Division in float64, rounded once to float32, as in the torch version
        candidate = (v.to(tl.float64) / norm).to(tl.float32)   # IEEE div.rn.f64
        tl.store(x_ptr + offs, tl.where(take, candidate, x), mask=mask)


def normalize_into(x, v, norm, active):
    """x = (v / norm) where active and norm > 0, else x; in place, one kernel on CUDA.

    x and v are (d,) or a batch (k, d) of float32 vectors, with norm and active
    of shape () or (k,). norm is the float64 norm of v, so a positive norm is
    never below float64's tiny and the torch version's clamp_min never applies.
    """
    if triton is not None and x.is_cuda and x.is_contiguous() and v.is_contiguous():
        n = x.numel()
        _normalize_kernel[(triton.cdiv(n, 1024),)](x, v, norm, active, n, x.shape[-1], BLOCK=1024)
        return x
    norm, active = norm.unsqueeze(-1), active.unsqueeze(-1)
    candidate = (v.double() / norm.clamp_min(torch.finfo(torch.float64).tiny)).float()
    return x.copy_(torch.where(active & (norm > 0), candidate, x))
