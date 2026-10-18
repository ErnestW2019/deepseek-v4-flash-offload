"""Micro-benchmark: FP4 weight dequant variants on NPU.

Goal of p4: cut per-expert dequant from ~28 ms to ms-level.
Candidates:
  cur    : current impl (int64 .long() + 2 gathers + expand)
  tbl256 : single uint8-indexed [256,2] LUT (byte = low|high nibble pair)
  tbl256_bit : + e8m0 via (u<<23).view(f32) instead of pow
  arith  : integer/float arithmetic decode, no gather
Also measures: H2D packed vs H2D bf16, and pure transfer cost.
"""

import os
import sys
import time

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from npu_ops import FP4_TABLE, e8m0_to_fp32, dequant_fp4_weight  # noqa: E402

DEV = torch.device("npu:0")

# 256-entry LUT: byte -> (low nibble value, high nibble value)
_t = FP4_TABLE
TBL256 = torch.stack([_t[torch.arange(256) & 0xF], _t[torch.arange(256) >> 4]], dim=-1).contiguous()  # [256,2] f32


def e8m0_bit(scale):
    """2**(byte-127) via float32 bit trick: bits = byte<<23 (exact for byte>=1)."""
    if scale.dtype == torch.float8_e8m0fnu:
        u = scale.view(torch.uint8)
    elif scale.dtype == torch.uint8:
        u = scale
    else:
        return torch.pow(2.0, scale.float() - 127.0)
    return (u.to(torch.int32) << 23).view(torch.float32)


def dq_tbl256(w_packed, scale, use_bit=True, out_dtype=torch.bfloat16):
    u = w_packed.view(torch.uint8)
    tbl = TBL256.to(u.device)
    vals = tbl[u.to(torch.int32)]      # [N, K//2, 2] f32 (int32 index, no .long())
    N = vals.shape[0]
    K = vals.shape[1] * 2
    vals = vals.reshape(N, K // 32, 32)
    s = (e8m0_bit(scale) if use_bit else e8m0_to_fp32(scale)).unsqueeze(-1)
    return (vals * s).reshape(N, K).to(out_dtype)


def dq_arith(w_packed, scale, out_dtype=torch.bfloat16):
    u = w_packed.view(torch.uint8)

    def nib(n):
        ni = n.to(torch.int32)
        sign = (ni >> 3) & 1
        e = (ni >> 1) & 3
        m = ni & 1
        sh = (e - 1).clamp(min=0)
        mag = torch.where(e > 0, (2 + m) << sh, m).float() * 0.5
        return torch.where(sign.to(torch.bool), -mag, mag)

    lo = nib(u & 0x0F)
    hi = nib((u >> 4) & 0x0F)
    vals = torch.stack([lo, hi], dim=-1).reshape(*u.shape[:-1], u.shape[-1] * 2)
    N, K = vals.shape
    s = e8m0_bit(scale).unsqueeze(-1).expand(N, K // 32, 32).reshape(N, K)
    return (vals * s).to(out_dtype)


def timeit(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def main():
    torch.manual_seed(0)
    shapes = [(2048, 2048), (4096, 1024)]  # w1/w3, w2
    data = []
    for (N, K) in shapes:
        wp = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
        sc = torch.randint(120, 135, (N, K // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu)
        data.append((N, K, wp, sc))

    # correctness: tbl256 vs arith vs current vs bit-scale
    print("== correctness (vs current impl, on NPU) ==")
    for (N, K, wp, sc) in data:
        wd, sd = wp.to(DEV), sc.to(DEV)
        ref = dequant_fp4_weight(wd, sd)
        for name, fn in (("tbl256", lambda: dq_tbl256(wd, sd, True)),
                         ("tbl256_pow", lambda: dq_tbl256(wd, sd, False)),
                         ("arith", lambda: dq_arith(wd, sd))):
            got = fn()
            d = (got.float() - ref.float()).abs().max().item()
            print(f"  [{N}x{K}] {name:10s} max_abs_diff={d:.3e}")

    print("\n== timing (per call, on-device data) ==")
    for (N, K, wp, sc) in data:
        wd, sd = wp.to(DEV), sc.to(DEV)
        ms_cur = timeit(lambda: dequant_fp4_weight(wd, sd))
        ms_tbl = timeit(lambda: dq_tbl256(wd, sd, True))
        ms_tblp = timeit(lambda: dq_tbl256(wd, sd, False))
        ms_ar = timeit(lambda: dq_arith(wd, sd))
        print(f"  [{N}x{K}] cur={ms_cur:.2f}  tbl256_bit={ms_tbl:.2f}  "
              f"tbl256_pow={ms_tblp:.2f}  arith={ms_ar:.2f}  (ms)")

    print("\n== e8m0 scale alone ==")
    for (N, K, wp, sc) in data:
        sd = sc.to(DEV)
        a = timeit(lambda: e8m0_to_fp32(sd))
        b = timeit(lambda: e8m0_bit(sd))
        print(f"  [{N}x{K//32}] pow={a:.3f}  bit={b:.3f} (ms)")

    print("\n== H2D transfer (packed vs bf16) ==")
    for (N, K, wp, sc) in data:
        bf = torch.randn(N, K, dtype=torch.bfloat16)

        def h2d_packed():
            return wp.to(DEV, non_blocking=False)

        def h2d_bf16():
            return bf.to(DEV, non_blocking=False)

        print(f"  [{N}x{K}] packed={timeit(h2d_packed):.3f}ms  "
              f"bf16={timeit(h2d_bf16):.3f}ms")


if __name__ == "__main__":
    main()
