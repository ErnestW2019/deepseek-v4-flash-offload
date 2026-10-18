"""Loader test for route-B.

Part 1 (tiny synthetic checkpoint): writes a *complete* tiny checkpoint using
        the same packing conventions as the real one (packed FP4 experts +
        E8M0 scales, FP8 non-expert weights + block scales), loads it back
        through ``loader.load_model`` and runs a forward pass.
Part 2 (real checkpoint, no I/O): verifies every real checkpoint key maps onto
        a ``model_npu`` parameter/buffer (or is an expert / expert scale).
Part 3 (real shard spot check): reads a real FP8 weight + scale straight from a
        shard and dequantizes it, to confirm dtype / shape handling.
"""

import json
import os
import shutil
import tempfile

import torch

try:
    import torch_npu  # noqa: F401  (registers the npu backend)
except Exception:  # pragma: no cover
    pass

from safetensors.torch import save_file

from npu_ops import dequant_fp8_weight
from offload import ExpertBank
from model_npu import ModelArgs, Transformer, Linear, DEVICE
import loader as loader_mod
from loader import load_model, _EXPERT_RE

CKPT = os.environ.get("DSV4_MODEL", "/home/models/DeepSeek-V4-Flash")


def tiny_args():
    return ModelArgs(
        max_batch_size=1, max_seq_len=32, vocab_size=64, dim=256, moe_inter_dim=256,
        n_layers=2, n_hash_layers=1, n_mtp_layers=1, n_heads=4,
        n_routed_experts=8, n_shared_experts=1, n_activated_experts=2,
        q_lora_rank=128, head_dim=128, rope_head_dim=64, o_groups=2, o_lora_rank=64,
        window_size=8, compress_ratios=(0, 4, 0), rope_factor=16, original_seq_len=0,
        index_n_heads=4, index_head_dim=64, index_topk=4, hc_mult=2,
    )


def _fp8_weight_names(model):
    """Names of the Linear weights whose *checkpoint* format is FP8."""
    return {f"{n}.weight" for n, m in model.named_modules()
            if isinstance(m, Linear) and m.mode == "fp8"}


def build_tiny_ckpt(model, args, out_dir):
    """Write a complete tiny checkpoint mirroring the real packing layout."""
    g = torch.Generator().manual_seed(1234)
    fp8_names = _fp8_weight_names(model)
    tensors = {}
    for key, t in model.state_dict().items():
        shape = tuple(t.shape)
        base = torch.randn(shape, generator=g) * 0.02
        if key in fp8_names:
            h, w = shape
            tensors[key] = base.to(torch.float8_e4m3fn).contiguous()
            sh, sw = (h + 127) // 128, (w + 127) // 128
            tensors[key[:-len("weight")] + "scale"] = torch.randint(
                120, 134, (sh, sw), dtype=torch.uint8, generator=g).contiguous()
        else:
            if t.dtype == torch.int64:          # tid2eid
                base = torch.randint(0, args.n_routed_experts, shape,
                                     dtype=torch.int64, generator=g)
            elif "norm" in key:
                base = 1.0 + base
            tensors[key] = base.to(t.dtype).contiguous()

    # routed experts: packed FP4 (int8 nibbles) + per-32 E8M0 scale
    inter, dim = args.moe_inter_dim, args.dim
    for li in range(args.n_layers + args.n_mtp_layers):
        prefix = f"layers.{li}" if li < args.n_layers else f"mtp.{li - args.n_layers}"
        for eid in range(args.n_routed_experts):
            for wname, (N, K) in (("w1", (inter, dim)), ("w2", (dim, inter)),
                                  ("w3", (inter, dim))):
                tensors[f"{prefix}.ffn.experts.{eid}.{wname}.weight"] = torch.randint(
                    0, 256, (N, K // 2), dtype=torch.uint8, generator=g).contiguous()
                tensors[f"{prefix}.ffn.experts.{eid}.{wname}.scale"] = torch.randint(
                    100, 150, (N, K // 32), dtype=torch.uint8, generator=g).contiguous()

    shard = "model.safetensors"
    save_file(tensors, os.path.join(out_dir, shard))
    weight_map = {k: shard for k in tensors}
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    return weight_map


def test_tiny():
    args = tiny_args()
    bank = ExpertBank(capacity=32, device=DEVICE)
    model = Transformer(args, bank=bank)
    d = tempfile.mkdtemp(prefix="dsv4_synth_")
    try:
        build_tiny_ckpt(model, args, d)
        stats = load_model(model, d, bank, DEVICE, include_experts=True,
                           max_experts_per_layer=None)
        n_expect = (args.n_layers + args.n_mtp_layers) * args.n_routed_experts
        ok = not stats["unmapped"] and stats["experts_registered"] == n_expect
        print(f"[{'OK ' if ok else 'BAD'}] tiny load: experts={stats['experts_registered']}"
              f"/{n_expect} fp8={stats['fp8_dequant']} cast={stats['cast']} "
              f"unmapped={len(stats['unmapped'])}")
        if stats["unmapped"]:
            print("   unmapped:", stats["unmapped"][:10])
        assert ok, stats

        w = model.layers[0].attn.wq_a.weight
        assert w.device.type == DEVICE.type and bool(torch.isfinite(w).all()), "fp8 dequant on device"
        assert bank.has(0, 0) and bank.has(args.n_layers, 0), "experts registered"
        print(f"[OK ] sampled wq_a on {w.device} dtype={w.dtype} "
              f"bank_experts={bank.num_experts()}")

        model.eval()
        ids = torch.randint(0, args.vocab_size, (1, 8), device=DEVICE)
        out = model(ids, 0)
        fok = out.shape == (1, args.vocab_size) and bool(torch.isfinite(out).all())
        print(f"[{'OK ' if fok else 'BAD'}] tiny forward {tuple(out.shape)} "
              f"finite={bool(torch.isfinite(out).all())}")
        assert fok
        for pos in range(8, 11):
            o = model(ids[:, pos - 8:pos - 7], pos)
            assert o.shape == (1, args.vocab_size) and bool(torch.isfinite(o).all())
        print("[OK ] tiny decode 3 steps")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return True


def test_real_mapping():
    args = ModelArgs.from_json(os.path.join(CKPT, "inference/config.json"))
    model = Transformer(args, bank=None)
    with open(os.path.join(CKPT, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    n_expert = n_scale = n_fp8 = n_cast = 0
    unmapped = []
    for key in weight_map:
        if _EXPERT_RE.match(key):
            n_expert += 1
            continue
        if key.endswith(".scale"):
            n_scale += 1
            continue
        try:
            loader_mod._target(model, key)
        except AttributeError:
            unmapped.append(key)
            continue
        companion = (key[:-len("weight")] + "scale") if key.endswith(".weight") else None
        if companion is not None and companion in weight_map:
            n_fp8 += 1
        else:
            n_cast += 1
    ok = not unmapped
    print(f"[{'OK ' if ok else 'BAD'}] real mapping: expert_keys={n_expert} "
          f"fp8={n_fp8} cast={n_cast} orphan_scales={n_scale} unmapped={len(unmapped)}")
    if unmapped:
        print("   unmapped:", unmapped[:20])
    assert ok, unmapped
    del model
    if DEVICE.type == "npu":
        torch.npu.empty_cache()
    return True


def test_real_shard_spot():
    with open(os.path.join(CKPT, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    keys = [k for k in weight_map
            if k.startswith("layers.") and k.endswith("attn.wq_b.weight")][:2]
    assert keys, "no wq_b keys found"
    reader = loader_mod._Reader(CKPT, weight_map)
    try:
        for k in keys:
            sk = k[:-len("weight")] + "scale"
            w, s = reader.get(k), reader.get(sk)
            dq = dequant_fp8_weight(w, s, 128, torch.bfloat16)
            ok = dq.shape == w.shape and bool(torch.isfinite(dq).all())
            print(f"[{'OK ' if ok else 'BAD'}] real dequant {k} w={tuple(w.shape)}"
                  f"{w.dtype} s={tuple(s.shape)}{s.dtype} -> {dq.dtype}")
            assert ok
    finally:
        reader.close()
    return True


if __name__ == "__main__":
    all_ok = True
    all_ok &= test_tiny()
    all_ok &= test_real_shard_spot()
    all_ok &= test_real_mapping()
    print("ALL PASS" if all_ok else "FAILED")
