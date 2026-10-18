"""Correctness self-test for npu_ops.py against hand-written torch references.

Run (from the repo root, inside an Ascend env with vLLM / torch-npu available):
    python test_npu_ops.py
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import npu_ops as O

DEV = "cpu"
try:
    import torch_npu  # noqa: F401
    if torch.npu.is_available():
        DEV = "npu:0"
except Exception:  # pragma: no cover
    pass


def dev():
    return torch.device(DEV)


FAIL = []


def check(name, a, b, atol=1e-3, rtol=1e-3):
    a = a.detach().to("cpu").float()
    b = b.detach().to("cpu").float()
    err = (a - b).abs().max().item()
    rel = ((a - b).abs() / (b.abs() + 1e-6)).max().item()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    flag = "OK " if ok else "FAIL"
    if not ok:
        FAIL.append(name)
    print(f"[{flag}] {name:34s} max_abs={err:.3e} max_rel={rel:.3e}")


# ---------------------------------------------------------------------------
# reference implementations (CPU, plain torch)
# ---------------------------------------------------------------------------
FP4_TABLE = O.FP4_TABLE


def ref_act_quant_inplace(x, block_size, fp8_max=448.0, round_scale=True):
    orig = x.dtype
    xf = x.float()
    N = x.shape[-1]
    xb = xf.reshape(-1, N // block_size, block_size)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    if round_scale:
        s = torch.pow(2.0, torch.ceil(torch.log2(amax / fp8_max)))
    else:
        s = amax / fp8_max
    y8 = torch.clamp(xb / s, -fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return (y8.float() * s).reshape_as(x).to(orig)


def ref_fp4_quant_inplace(x, block_size):
    orig = x.dtype
    xf = x.float()
    N = x.shape[-1]
    xb = xf.reshape(-1, N // block_size, block_size)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=6 * 2.0 ** -126)
    s = torch.pow(2.0, torch.ceil(torch.log2(amax / 6.0)))
    v = torch.clamp(xb / s, -6.0, 6.0)
    # brute-force nearest fp4 (ties -> even index)
    flat = v.reshape(-1)
    d = (flat.unsqueeze(-1) - FP4_TABLE.unsqueeze(0)).abs()
    mn = d.min(dim=-1, keepdim=True).values
    cand = (d <= mn + 1e-9)
    # on ties choose the even table index (round-half-to-even)
    idxs = torch.arange(16)
    big = torch.where(cand, (idxs % 2 == 0).float(), torch.full_like(d, -1e9))
    pick = big.argmax(dim=-1)
    q = FP4_TABLE[pick]
    s = s.expand(-1, -1, block_size).reshape(-1)
    return (q * s).reshape_as(x).to(orig)


def ref_dequant_fp4(w_packed, scale):
    u = w_packed.view(torch.uint8)
    N, Kh = u.shape
    K = Kh * 2
    vals = torch.empty(N, K, dtype=torch.float32)
    for j in range(Kh):
        vals[:, 2 * j] = FP4_TABLE[(u[:, j] & 0x0F).long()]
        vals[:, 2 * j + 1] = FP4_TABLE[((u[:, j] >> 4) & 0x0F).long()]
    s = torch.pow(2.0, scale.view(torch.uint8).to(torch.int32).float() - 127.0)
    s = s.unsqueeze(-1).expand(N, K // 32, 32).reshape(N, K)
    return vals * s


def ref_dequant_fp8(w, scale, block=128):
    N, K = w.shape
    s = torch.pow(2.0, scale.view(torch.uint8).to(torch.int32).float() - 127.0)
    s = s.repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return w.float() * s


def ref_rope_complex(x, dim, base=10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(x.size(1), dtype=torch.float32)
    ang = torch.outer(t, freqs)
    fc = torch.polar(torch.ones_like(ang), ang)
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    fc = fc.view(1, x.size(1), *([1] if x.dim() == 4 else []), -1)
    return torch.view_as_real(xc * fc).flatten(-2)


def ref_sinkhorn(mixes, scale, base, hc=4, iters=20, eps=1e-6):
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = mixes[..., 2 * hc:].reshape(*mixes.shape[:-1], hc, hc) * scale[2] + \
        base[2 * hc:].reshape(hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def ref_sparse_attn(q, kv, sink, idxs, scale):
    b, s, h, d = q.shape
    out = torch.empty_like(q)
    for bi in range(b):
        for si in range(s):
            ii = idxs[bi, si].long()
            valid = ii >= 0
            kvg = kv[bi, ii.clamp(min=0)] * valid.unsqueeze(-1)
            sc = (q[bi, si].float() @ kvg.float().T) * scale   # [h, topk]
            sc = sc.masked_fill(~valid.unsqueeze(0), float("-inf"))
            m = torch.maximum(sc.amax(-1, keepdim=True), sink.reshape(-1, 1).float())
            p = torch.exp(sc - m)
            denom = p.sum(-1, keepdim=True) + torch.exp(sink.reshape(-1, 1).float() - m)
            out[bi, si] = ((p @ kvg.float()) / denom).to(q.dtype)
    return out


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    d = dev()
    print("device:", d, "\n")

    # 1. _round_fp4 table check (nearest, ties -> even table index)
    vals = torch.arange(-6.0, 6.001, 0.01, dtype=torch.float32)
    got = O._round_fp4(vals)
    ref = []
    for v in vals:
        dists = (FP4_TABLE - v).abs()
        mn = dists.min()
        ties = torch.nonzero(dists <= mn + 1e-9).flatten()
        idx = ties[0] if len(ties) == 1 else next(
            (int(i) for i in ties if int(i) % 2 == 0), int(ties[0]))
        ref.append(float(FP4_TABLE[idx]))
    check("_round_fp4 (nearest-even)", got, torch.tensor(ref), atol=1e-6)

    # 2. act_quant inplace (power-of-2 scale)
    x = (torch.randn(64, 512) * 3).to(torch.bfloat16).to(d)
    x1 = x.clone().to(d)
    out = O.act_quant(x1, 128, "ue8m0", True)
    check("act_quant inplace ue8m0", out.cpu(), ref_act_quant_inplace(x.cpu(), 128, 448.0, True))

    # act_quant non-inplace returns fp8 + fp32 scale
    y, s = O.act_quant(x.to(d), 128, "ue8m0", False)
    xb = x.cpu().float().reshape(64, 4, 128)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    sref = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    check("act_quant scale (ue8m0)", s.cpu().reshape(64, 4), sref.reshape(64, 4), atol=1e-30)

    # 3. fp4_act_quant inplace
    x2 = (torch.randn(64, 256) * 2).to(torch.bfloat16).to(d)
    out2 = O.fp4_act_quant(x2.clone(), 32, True)
    check("fp4_act_quant inplace", out2.cpu(), ref_fp4_quant_inplace(x2.cpu(), 32))

    # 4. dequant_fp4_weight
    N, Kh = 8, 128
    import numpy as np
    rng = np.random.default_rng(0)
    packed = torch.from_numpy(rng.integers(0, 256, (N, Kh), dtype=np.uint8))
    packed_signed = packed.view(torch.int8)
    scale_bytes = rng.integers(120, 134, (N, (Kh * 2) // 32), dtype=np.uint8)
    sc = torch.from_numpy(scale_bytes).view(torch.uint8)
    sc_e8 = sc.view(torch.float8_e8m0fnu).to(d)
    wd = O.dequant_fp4_weight(packed_signed.to(d), sc_e8).cpu()
    check("dequant_fp4_weight", wd, ref_dequant_fp4(packed_signed, sc), atol=1e-6)

    # 5. dequant_fp8_weight
    w8 = (torch.randn(256, 256) * 2).to(torch.float8_e4m3fn)
    sl = torch.from_numpy(rng.integers(120, 134, (2, 2), dtype=np.uint8)).view(torch.uint8)
    wd8 = O.dequant_fp8_weight(w8.to(d), sl.view(torch.float8_e8m0fnu).to(d)).cpu()
    check("dequant_fp8_weight", wd8, ref_dequant_fp8(w8, sl), atol=1e-6)

    # 6. apply_rotary
    dim = 64
    xx = torch.randn(2, 16, 8, dim, dtype=torch.bfloat16).to(d)
    cos, sin = O.precompute_rope(dim, 16, 0, 10000.0, 40, 32, 1)
    cos, sin = cos.to(d), sin.to(d)
    rr = O.apply_rotary(xx.clone(), cos, sin).cpu()
    check("apply_rotary", rr, ref_rope_complex(xx.cpu(), dim), atol=2e-2, rtol=2e-2)
    # inverse round-trip
    back = O.apply_rotary(rr.to(d), cos, sin, inverse=True).cpu()
    check("apply_rotary inverse", back, xx.cpu().float(), atol=2e-2, rtol=2e-2)

    # 7. hc_split_sinkhorn
    mixes = torch.randn(3, 5, 24).to(d)
    scale = torch.tensor([0.5, 0.7, 0.9]).to(d)
    base = torch.randn(24).to(d)
    p1, q1, c1 = O.hc_split_sinkhorn(mixes, scale, base, 4, 20)
    p2, q2, c2 = ref_sinkhorn(mixes.cpu(), scale.cpu(), base.cpu(), 4, 20)
    check("sinkhorn pre", p1.cpu(), p2)
    check("sinkhorn post", q1.cpu(), q2)
    check("sinkhorn comb", c1.cpu(), c2, atol=1e-5)

    # 8. sparse_attn
    b, s, h, dd, topk = 2, 6, 64, 512, 16
    q = torch.randn(b, s, h, dd, dtype=torch.bfloat16).to(d)
    kv = torch.randn(b, 40, dd, dtype=torch.bfloat16).to(d)
    sink = torch.randn(h).to(d)
    idxs = torch.randint(0, 40, (b, s, topk), dtype=torch.int32)
    idxs[0, 0, :3] = -1
    idxs = idxs.to(d)
    o = O.sparse_attn(q, kv, sink, idxs, 1.0 / math.sqrt(dd))
    oref = ref_sparse_attn(q.cpu(), kv.cpu(), sink.cpu(), idxs.cpu(), 1.0 / math.sqrt(dd))
    check("sparse_attn", o.cpu(), oref, atol=3e-2, rtol=3e-2)

    # 9. rotate_activation orthonormal
    z = torch.randn(4, 128, dtype=torch.bfloat16).to(d)
    r = O.rotate_activation(z)
    check("rotate_activation norm", r.float().norm(dim=-1), z.float().norm(dim=-1), atol=2e-2)

    print("\n" + ("ALL PASS" if not FAIL else f"FAILED: {FAIL}"))


if __name__ == "__main__":
    main()
