"""Torch/NPU re-implementations of the tilelang kernels used by DeepSeek-V4-Flash.

Route-B (single-card expert-offload) replaces the CUDA/tilelang kernels in
``kernel.py`` with portable torch ops so the model can run on Ascend NPU.

Reference math (block sizes, scale formats, packed layouts) is taken verbatim
from ``/home/models/DeepSeek-V4-Flash/inference/kernel.py``.
"""

import math
import functools

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# FP4 tables
# ---------------------------------------------------------------------------
# E2M1 nibble -> float. Index = nibble value (0..15).
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
# Non-negative magnitudes of E2M1, sorted (used for round-to-nearest).
FP4_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

FP8_MAX = 448.0
FP4_MAX = 6.0


# ---------------------------------------------------------------------------
# scale helpers
# ---------------------------------------------------------------------------
def e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """UE8M0 scale -> float32. value = 2 ** (byte - 127)."""
    if scale.dtype == torch.float8_e8m0fnu:
        u = scale.view(torch.uint8)
    elif scale.dtype == torch.uint8:
        u = scale
    else:
        # already an integer exponent tensor
        return torch.pow(2.0, scale.float() - 127.0)
    return torch.pow(2.0, u.to(torch.int32).float() - 127.0)


def _fast_pow2_scale(amax: torch.Tensor, max_inv: float) -> torch.Tensor:
    """round_scale = 2 ** ceil(log2(amax * max_inv)); matches fast_round_scale()."""
    v = torch.clamp(amax * max_inv, min=2.0 ** -126)
    return torch.pow(2.0, torch.ceil(torch.log2(v)))


def _round_fp4(v: torch.Tensor) -> torch.Tensor:
    """Round values in [-6, 6] to the nearest E2M1 representable value (ties -> even nibble)."""
    tbl = FP4_POS.to(v.device)
    a = v.abs()
    idx = torch.searchsorted(tbl, a.contiguous())
    idx = idx.clamp(1, 7)
    lo = tbl[idx - 1]
    hi = tbl[idx]
    d_lo = a - lo
    d_hi = hi - a
    # candidate index (idx-1) or idx; on ties pick the even one
    take_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (((idx - 1) % 2) == 1))
    val = torch.where(take_hi, hi, lo)
    sign = torch.where(v < 0, -1.0, 1.0)
    return val * sign


# ---------------------------------------------------------------------------
# block-wise activation quantization (in-place fp8 / fp4 simulation)
# ---------------------------------------------------------------------------
def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt: str = "ue8m0",
              scale_dtype=None, inplace: bool = False):
    """Block-wise FP8 quantization.

    Accepts the reference call signature ``(x, block_size, scale_fmt, scale_dtype,
    inplace)``; ``scale_dtype`` is accepted for compatibility and ignored.

    inplace=True reproduces the reference ``act_quant(..., inplace=True)``:
    quantize to FP8, dequantize back and write into ``x`` (returns x).
    Otherwise returns (y_fp8, scale).
    """
    orig_dtype = x.dtype
    N = x.shape[-1]
    assert N % block_size == 0, (N, block_size)
    xf = x.float()
    xb = xf.reshape(-1, N // block_size, block_size)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    if scale_fmt is not None:
        s = _fast_pow2_scale(amax, 1.0 / FP8_MAX)
    else:
        s = amax / FP8_MAX
    y = torch.clamp(xb / s, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        out = (y.float() * s).reshape_as(x).to(orig_dtype)
        x.copy_(out)
        return x
    y = y.reshape_as(x)
    # NOTE: torch_npu cannot cast fp32 -> float8_e8m0fnu; route B never consumes
    # the fp8 activations/scales (activation quant runs inplace), so return the
    # (exact, power-of-two) scales as fp32.
    s_out = s.squeeze(-1).reshape(*x.shape[:-1], N // block_size)
    return y, s_out


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False):
    """Block-wise FP4 quantization (power-of-2 scale). inplace reproduces reference."""
    orig_dtype = x.dtype
    N = x.shape[-1]
    assert N % block_size == 0, (N, block_size)
    xf = x.float()
    xb = xf.reshape(-1, N // block_size, block_size)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=FP4_MAX * (2.0 ** -126))
    s = _fast_pow2_scale(amax, 1.0 / FP4_MAX)
    q = _round_fp4(torch.clamp(xb / s, -FP4_MAX, FP4_MAX))
    if inplace:
        out = (q * s).reshape_as(x).to(orig_dtype)
        x.copy_(out)
        return x
    raise NotImplementedError("non-inplace fp4 pack not needed on this path")


# ---------------------------------------------------------------------------
# weight dequantization (offline, to bf16)
# ---------------------------------------------------------------------------
# Packed byte -> (low-nibble value, high-nibble value); one 256-entry LUT lets a
# whole byte be decoded with a single gather (indices stay int32, no int64).
_TBL256_CACHE = {}


def _tbl256(device):
    t = _TBL256_CACHE.get(device)
    if t is None:
        idx = torch.arange(256)
        t = torch.stack([FP4_TABLE[idx & 0xF], FP4_TABLE[idx >> 4]],
                        dim=-1).contiguous().to(device)
        _TBL256_CACHE[device] = t
    return t


def dequant_fp4_weight(w_packed: torch.Tensor, scale: torch.Tensor,
                       out_dtype=torch.bfloat16) -> torch.Tensor:
    """Dequantize an FP4 expert weight.

    w_packed: [N, K//2] int8, nibbles packed along K (low nibble = k=2j, high = 2j+1)
    scale:    [N, K//32] UE8M0
    returns:  [N, K] in out_dtype

    One int32-indexed 256-entry LUT decodes both nibbles of each byte at once;
    the per-32 UE8M0 scale is applied as a broadcast multiply (no expand copy).
    """
    u = w_packed.view(torch.uint8)
    vals = _tbl256(u.device)[u.to(torch.int32)]      # [N, K//2, 2]
    N, K2, _ = vals.shape
    K = K2 * 2
    s = e8m0_to_fp32(scale).unsqueeze(-1)            # [N, K//32, 1]
    return (vals.reshape(N, K // 32, 32) * s).reshape(N, K).to(out_dtype)


def dequant_fp8_weight(w: torch.Tensor, scale: torch.Tensor, block: int = 128,
                       out_dtype=torch.bfloat16) -> torch.Tensor:
    """Dequantize a 128x128-block FP8 weight: [N, K] fp8 * scale[ceil(N/128), ceil(K/128)]."""
    N, K = w.shape
    s = e8m0_to_fp32(scale)  # [nb_n, nb_k]
    s = s.repeat_interleave(block, dim=0)[:N].repeat_interleave(block, dim=1)[:, :K]
    return (w.float() * s).to(out_dtype)


# ---------------------------------------------------------------------------
# rotary embedding (no complex dtype: precomputed cos/sin)
# ---------------------------------------------------------------------------
def precompute_rope(dim: int, seqlen: int, original_seq_len: int, base: float,
                    factor: float, beta_fast: float, beta_slow: float):
    """Returns (cos, sin) each [seqlen, dim//2] float32 (YaRN optional)."""

    def find_correction_dim(num_rot, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rot * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(mn, mx, dim):
        if mn == mx:
            mx += 0.001
        lin = (torch.arange(dim, dtype=torch.float32) - mn) / (mx - mn)
        return torch.clamp(lin, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen, dtype=torch.float32)
    angles = torch.outer(t, freqs)  # [seqlen, dim//2]
    return torch.cos(angles), torch.sin(angles)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                 inverse: bool = False) -> torch.Tensor:
    """In-place rotary. x last dim = dim (even). cos/sin: [seqlen, dim//2]."""
    if inverse:
        sin = -sin
    if x.ndim == 3:  # [b, s, d]
        cos = cos.view(1, x.size(1), x.size(-1) // 2)
        sin = sin.view(1, x.size(1), x.size(-1) // 2)
    else:  # [b, s, h, d]
        cos = cos.view(1, x.size(1), 1, x.size(-1) // 2)
        sin = sin.view(1, x.size(1), 1, x.size(-1) // 2)
    x2 = x.float().unflatten(-1, (-1, 2))
    xr, xi = x2[..., 0], x2[..., 1]
    out_r = xr * cos - xi * sin
    out_i = xr * sin + xi * cos
    x.copy_(torch.stack([out_r, out_i], dim=-1).flatten(-2).to(x.dtype))
    return x


# ---------------------------------------------------------------------------
# randomized Hadamard rotation (replaces fast_hadamard_transform)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=16)
def _hadamard(n: int):
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / math.sqrt(n)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Hadamard rotation on the last dim (power of two)."""
    n = x.size(-1)
    h = _hadamard(n).to(device=x.device, dtype=torch.float32)
    return (x.float() @ h).to(x.dtype)


# ---------------------------------------------------------------------------
# hyper-connection sinkhorn split
# ---------------------------------------------------------------------------
def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
                      hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    """mixes: [..., (2+hc)*hc] -> pre, post, comb. Mirrors hc_split_sinkhorn_kernel."""
    shape = mixes.shape[:-1]
    mixes = mixes.float()
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = mixes[..., 2 * hc:].reshape(*shape, hc, hc) * hc_scale[2] + \
        hc_base[2 * hc:].reshape(hc, hc)
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


# ---------------------------------------------------------------------------
# sparse attention (index gather + online softmax + attn sink)
# ---------------------------------------------------------------------------
def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor,
                topk_idxs: torch.Tensor, softmax_scale: float,
                query_chunk: int = 256) -> torch.Tensor:
    """q: [b, s, h, d], kv: [b, n, d], attn_sink: [h], topk_idxs: [b, s, topk] (int, -1 = pad).

    Returns [b, s, h, d]. Chunked over the query axis to bound memory.
    """
    b, s, h, d = q.shape
    out = torch.empty_like(q)
    sink = attn_sink.reshape(1, 1, h, 1)
    for lo in range(0, s, query_chunk):
        hi = min(lo + query_chunk, s)
        qc = q[:, lo:hi]                       # [b, sc, h, d]
        idx = topk_idxs[:, lo:hi].long()       # [b, sc, topk]
        valid = idx >= 0
        idx_c = idx.clamp(min=0)
        b_ar = torch.arange(b, device=kv.device).view(b, 1, 1).expand_as(idx_c)
        kv_g = kv[b_ar, idx_c]                 # [b, sc, topk, d]
        kv_g = kv_g * valid.unsqueeze(-1).to(kv_g.dtype)
        scores = torch.matmul(qc, kv_g.transpose(-1, -2)) * softmax_scale  # [b, sc, h, topk]
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
        row_max = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
        p = torch.exp(scores - row_max)
        denom = p.sum(dim=-1, keepdim=True) + torch.exp(sink - row_max)
        o = torch.matmul(p.to(kv_g.dtype), kv_g) / denom.to(kv_g.dtype)
        out[:, lo:hi] = o
    return out
