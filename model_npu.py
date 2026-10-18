"""Route-B: DeepSeek-V4-Flash inference on a single Ascend NPU with expert offload.

Ported from ``/home/models/DeepSeek-V4-Flash/inference/model.py``:

* the tilelang kernels (``kernel.py``) are replaced by torch / NPU ops
  (``npu_ops.py``);
* quantized (FP8 / FP4) weights are dequantized to bf16 at load time instead of
  being fed to the tilelang GEMMs;
* routed experts are streamed from a host-RAM bank (``offload.ExpertBank``)
  instead of being resident, which is what makes a single card viable.

Module attribute names mirror the checkpoint keys exactly so a streaming loader
can populate them by name.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from npu_ops import (act_quant, fp4_act_quant, rotate_activation,
                     apply_rotary as _apply_rotary, precompute_rope,
                     hc_split_sinkhorn, sparse_attn)

block_size = 128
fp4_block_size = 32
scale_fmt = "ue8m0"
scale_dtype = torch.float8_e8m0fnu  # informational; npu_ops ignores it
DEVICE = torch.device("npu:0")


def _t(*shape, dtype=torch.bfloat16):
    return torch.empty(*shape, dtype=dtype, device=DEVICE)


def _quant_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Non-destructive FP8 round-trip of ``x`` (matches the act_quant applied to
    activations by ``linear()`` for quantized weights, without touching ``x``)."""
    y = x.clone()
    return act_quant(y, block_size, scale_fmt, scale_dtype, True)


@dataclass
class ModelArgs:
    """Model hyperparameters. Field names match the config JSON keys."""
    max_batch_size: int = 1
    max_seq_len: int = 4096
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 2048
    n_layers: int = 43
    n_hash_layers: int = 3
    n_mtp_layers: int = 1
    n_heads: int = 64
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: str = "sqrtsoftplus"
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: Tuple[int, ...] = ()
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    @classmethod
    def from_json(cls, path: str) -> "ModelArgs":
        with open(path) as f:
            cfg = json.load(f)
        known = {k: v for k, v in cfg.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class Linear(nn.Module):
    """Weight-only linear.

    ``mode`` describes how the *checkpoint* weight is consumed:

    * ``"fp8"``  -- checkpoint FP8 (+scale); dequantized to bf16 at load, and the
      input goes through an FP8 round-trip to match QAT (reference ``linear``).
    * ``"bf16"`` -- kept/used as bf16, no activation quant (e.g. ``wo_a``).
    * ``"fp32"`` -- kept/used as fp32 (compressor / gate path).
    """

    def __init__(self, in_features: int, out_features: int, mode: str = "bf16"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        dt = torch.float32 if mode == "fp32" else torch.bfloat16
        self.weight = nn.Parameter(_t(out_features, in_features, dtype=dt),
                                   requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "fp8":
            x = _quant_roundtrip(x)
        return F.linear(x, self.weight)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32, device=DEVICE),
                                   requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


@lru_cache(8)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat([torch.arange(start_pos + 1, window_size),
                            torch.arange(0, start_pos + 1)], dim=0)
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(8)
def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


class Compressor(nn.Module):
    """Gated pooling over ``compress_ratio`` consecutive tokens."""

    def __init__(self, args: ModelArgs, compress_ratio: int = 4,
                 head_dim: int = 512, rotate: bool = False):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = head_dim - args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap

        self.ape = nn.Parameter(_t(compress_ratio, coff * self.head_dim, dtype=torch.float32),
                                requires_grad=False)
        self.wkv = Linear(self.dim, coff * self.head_dim, mode="fp32")
        self.wgate = Linear(self.dim, coff * self.head_dim, mode="fp32")
        self.norm = RMSNorm(self.head_dim, args.norm_eps)
        self.kv_cache: torch.Tensor = None
        self.register_buffer("kv_state", torch.zeros(
            args.max_batch_size, coff * compress_ratio, coff * self.head_dim,
            dtype=torch.float32, device=DEVICE), persistent=False)
        self.register_buffer("score_state", torch.full(
            (args.max_batch_size, coff * compress_ratio, coff * self.head_dim),
            float("-inf"), dtype=torch.float32, device=DEVICE), persistent=False)
        self.freqs_cos: torch.Tensor = None
        self.freqs_sin: torch.Tensor = None

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int):
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim
        dtype = x.dtype
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio: cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff - ratio: cutoff] + self.ape
            if remainder > 0:
                kv, self.kv_state[:bsz, offset: offset + remainder] = kv.split([cutoff, remainder], dim=1)
                self.score_state[:bsz, offset: offset + remainder] = score[:, cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score += self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d],
                                          self.kv_state[:bsz, ratio:, d:]], dim=1)
                    score_state = torch.cat([self.score_state[:bsz, :ratio, :d],
                                             self.score_state[:bsz, ratio:, d:]], dim=1)
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return
        kv = self.norm(kv.to(dtype))
        if start_pos == 0:
            cos, sin = self.freqs_cos[:cutoff:ratio], self.freqs_sin[:cutoff:ratio]
        else:
            cos = self.freqs_cos[start_pos + 1 - self.compress_ratio].unsqueeze(0)
            sin = self.freqs_sin[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        _apply_rotary(kv[..., -rd:], cos, sin)
        if self.rotate:
            kv = rotate_activation(kv)
            fp4_act_quant(kv, fp4_block_size, True)
        else:
            act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        if start_pos == 0:
            self.kv_cache[:bsz, :seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv


class Indexer(nn.Module):
    """Learned top-k selection over compressed KV positions."""

    def __init__(self, args: ModelArgs, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim, mode="fp8")
        self.weights_proj = Linear(self.dim, self.n_heads, mode="bf16")
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio

        self.compressor = Compressor(args, compress_ratio, self.head_dim, True)
        self.register_buffer("kv_cache", torch.zeros(
            args.max_batch_size, args.max_seq_len // compress_ratio, self.head_dim,
            device=DEVICE), persistent=False)
        self.freqs_cos: torch.Tensor = None
        self.freqs_sin: torch.Tensor = None

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        bsz, seqlen, _ = x.size()
        ratio, rd = self.compress_ratio, self.rope_head_dim
        end_pos = start_pos + seqlen
        cos = self.freqs_cos[start_pos:start_pos + seqlen]
        sin = self.freqs_sin[start_pos:start_pos + seqlen]
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cos = self.freqs_cos
            self.compressor.freqs_sin = self.freqs_sin
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        _apply_rotary(q[..., -rd:], cos, sin)
        q = rotate_activation(q)
        fp4_act_quant(q, fp4_block_size, True)
        self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        index_score = torch.matmul(q, self.kv_cache[:bsz, :end_pos // ratio].transpose(-1, -2))
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if start_pos == 0:
            mask = torch.arange(seqlen // ratio, device=x.device).repeat(seqlen, 1) >= \
                torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            index_score = index_score + torch.where(mask, float("-inf"), 0)
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs


class Attention(nn.Module):
    """MLA with sliding window + optional KV compression."""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = self.n_groups
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps

        self.attn_sink = nn.Parameter(_t(self.n_local_heads, dtype=torch.float32),
                                      requires_grad=False)
        self.wq_a = Linear(self.dim, self.q_lora_rank, mode="fp8")
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim, mode="fp8")
        self.wkv = Linear(self.dim, self.head_dim, mode="fp8")
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = Linear(self.n_heads * self.head_dim // self.n_groups,
                           self.n_groups * args.o_lora_rank, mode="bf16")
        self.wo_b = Linear(self.n_groups * args.o_lora_rank, self.dim, mode="fp8")
        self.softmax_scale = self.head_dim ** -0.5

        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            self.indexer = Indexer(args, self.compress_ratio) if self.compress_ratio == 4 else None

        kv_cache_size = args.window_size + (args.max_seq_len // self.compress_ratio
                                            if self.compress_ratio else 0)
        self.register_buffer("kv_cache", torch.zeros(
            args.max_batch_size, kv_cache_size, self.head_dim, device=DEVICE), persistent=False)
        if self.compress_ratio:
            original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
        else:
            original_seq_len, rope_theta = 0, args.rope_theta
        cos, sin = precompute_rope(self.rope_head_dim, args.max_seq_len, original_seq_len,
                                   rope_theta, args.rope_factor, args.beta_fast, args.beta_slow)
        self.register_buffer("freqs_cos", cos.to(DEVICE), persistent=False)
        self.register_buffer("freqs_sin", sin.to(DEVICE), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        cos = self.freqs_cos[start_pos:start_pos + seqlen]
        sin = self.freqs_sin[start_pos:start_pos + seqlen]
        win, ratio, rd = self.window_size, self.compress_ratio, self.rope_head_dim
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cos = self.freqs_cos
            self.compressor.freqs_sin = self.freqs_sin
            if self.indexer is not None:
                self.indexer.freqs_cos = self.freqs_cos
                self.indexer.freqs_sin = self.freqs_sin
        # q
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        _apply_rotary(q[..., -rd:], cos, sin)

        # win kv & topk_idxs
        kv = self.kv_norm(self.wkv(x))
        _apply_rotary(kv[..., -rd:], cos, sin)
        act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos).to(x.device)
        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = get_compress_topk_idxs(
                    ratio, bsz, seqlen, start_pos, offset).to(x.device)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff: win], self.kv_cache[:bsz, :cutoff] = \
                    kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                if (kv_compress := self.compressor(x, start_pos)) is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos)
            o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
        _apply_rotary(o[..., -rd:], cos, sin, inverse=True)

        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.matmul(o.unsqueeze(-2), wo_a.transpose(-1, -2)).squeeze(-2)
        return self.wo_b(o.flatten(2))


class Gate(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(_t(args.n_routed_experts, args.dim, dtype=torch.float32),
                                   requires_grad=False)
        if self.hash:
            self.tid2eid = nn.Parameter(torch.empty(
                args.vocab_size, args.n_activated_experts, dtype=torch.int64, device=DEVICE),
                requires_grad=False)
            self.bias = None
        else:
            self.bias = nn.Parameter(_t(args.n_routed_experts, dtype=torch.float32),
                                     requires_grad=False)

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None):
        scores = F.linear(x.float(), self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices


class Expert(nn.Module):
    """Dense SwiGLU expert (used for the shared experts)."""

    def __init__(self, dim: int, inter_dim: int, mode: str = "fp8", swiglu_limit: float = 0):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, mode=mode)
        self.w2 = Linear(inter_dim, dim, mode=mode)
        self.w3 = Linear(dim, inter_dim, mode=mode)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        h = F.silu(gate) * up
        if weights is not None:
            h = weights * h
        return self.w2(h.to(dtype))


class MoE(nn.Module):
    """Routed MoE whose expert weights live in a host-RAM ``ExpertBank``."""

    def __init__(self, layer_id: int, args: ModelArgs, bank=None):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.swiglu_limit = args.swiglu_limit
        self.gate = Gate(layer_id, args)
        self.bank = bank
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, mode="fp8",
                                     swiglu_limit=args.swiglu_limit)

    def _expert(self, wid: dict, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xq = _quant_roundtrip(x)
        gate = F.linear(xq, wid["w1"]).float()
        up = F.linear(xq, wid["w3"]).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        h = F.silu(gate) * up * weights
        return F.linear(_quant_roundtrip(h.to(dtype)), wid["w2"])

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten().cpu(),
                                minlength=self.n_routed_experts).tolist()
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            val = self._expert(self.bank.get(self.layer_id, i),
                               x[idx], weights[idx, top, None])
            y.index_add_(0, idx, val.float())
        y = y + self.shared_experts(x)
        return y.type_as(x).view(shape)


class Block(nn.Module):
    """Transformer block with Hyper-Connections."""

    def __init__(self, layer_id: int, args: ModelArgs, bank=None):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(layer_id, args, bank=bank)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_attn_fn = nn.Parameter(_t(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False)
        self.hc_ffn_fn = nn.Parameter(_t(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False)
        self.hc_attn_base = nn.Parameter(_t(mix_hc, dtype=torch.float32), requires_grad=False)
        self.hc_ffn_base = nn.Parameter(_t(mix_hc, dtype=torch.float32), requires_grad=False)
        self.hc_attn_scale = nn.Parameter(_t(3, dtype=torch.float32), requires_grad=False)
        self.hc_ffn_scale = nn.Parameter(_t(3, dtype=torch.float32), requires_grad=False)

    def hc_pre(self, x, hc_fn, hc_scale, hc_base):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult,
                                            self.hc_sinkhorn_iters, self.hc_eps)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(self, x, residual, post, comb):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + \
            torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(self, x, start_pos, input_ids=None):
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = nn.Parameter(_t(vocab_size, dim), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)


class ParallelHead(nn.Module):
    def __init__(self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.float32, device=DEVICE),
                                   requires_grad=False)

    def hc_head(self, x, hc_fn, hc_scale, hc_base):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)

    def forward(self, x, hc_fn, hc_scale, hc_base, norm):
        x = self.hc_head(x, hc_fn, hc_scale, hc_base)
        return F.linear(norm(x)[:, -1].float(), self.weight)


class MTPBlock(Block):
    def __init__(self, layer_id: int, args: ModelArgs, bank=None):
        super().__init__(layer_id, args, bank=bank)
        self.e_proj = Linear(args.dim, args.dim, mode="fp8")
        self.h_proj = Linear(args.dim, args.dim, mode="fp8")
        self.enorm = RMSNorm(args.dim, args.norm_eps)
        self.hnorm = RMSNorm(args.dim, args.norm_eps)
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.hc_head_fn = nn.Parameter(_t(args.hc_mult, args.hc_mult * args.dim,
                                          dtype=torch.float32), requires_grad=False)
        self.hc_head_base = nn.Parameter(_t(args.hc_mult, dtype=torch.float32), requires_grad=False)
        self.hc_head_scale = nn.Parameter(_t(1, dtype=torch.float32), requires_grad=False)
        self.embed: ParallelEmbedding = None
        self.head: ParallelHead = None

    @torch.inference_mode()
    def forward(self, x, start_pos, input_ids):
        assert self.embed is not None and self.head is not None
        e = self.enorm(self.embed(input_ids))
        x = self.hnorm(x)
        x = self.e_proj(e).unsqueeze(2) + self.h_proj(x)
        x = super().forward(x, start_pos, input_ids)
        return self.head(x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, bank=None, device=None):
        global DEVICE
        if device is not None:
            DEVICE = torch.device(device)
        super().__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.bank = bank
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([Block(i, args, bank=bank) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, self.norm_eps)
        self.head = ParallelHead(args.vocab_size, args.dim, self.norm_eps, self.hc_eps)
        self.mtp = nn.ModuleList()
        for i in range(args.n_mtp_layers):
            blk = MTPBlock(args.n_layers + i, args, bank=bank)
            blk.embed = self.embed
            blk.head = self.head
            self.mtp.append(blk)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_head_fn = nn.Parameter(_t(hc_mult, hc_mult * args.dim,
                                          dtype=torch.float32), requires_grad=False)
        self.hc_head_base = nn.Parameter(_t(hc_mult, dtype=torch.float32), requires_grad=False)
        self.hc_head_scale = nn.Parameter(_t(1, dtype=torch.float32), requires_grad=False)

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0):
        h = self.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        for layer in self.layers:
            h = layer(h, start_pos, input_ids)
        return self.head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
