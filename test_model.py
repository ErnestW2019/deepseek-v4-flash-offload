"""Smoke test for model_npu.Transformer on npu:0.

Part 1: tiny config with an ExpertBank of random packed FP4 experts; runs a
        prefill forward (exercises hash gate + compressor/indexer + MoE offload
        + hyper-connections + head) and a few decode steps (start_pos > 0).
Part 2: builds the *real* config (weights are torch.empty) purely to validate
        construction and measure the resident (non-expert) memory budget.
"""

import torch

try:
    import torch_npu  # noqa: F401  (registers the npu backend)
except Exception:  # pragma: no cover
    pass

from offload import ExpertBank
from model_npu import ModelArgs, Transformer, RMSNorm, DEVICE


def rand_expert(seed, dim, inter):
    """Random packed FP4 expert as (w1p,w1s, w2p,w2s, w3p,w3s) on CPU."""
    g = torch.Generator().manual_seed(seed)

    def packed(N, K):
        return torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, generator=g)

    def scale(N, K):
        return torch.randint(100, 150, (N, K // 32), dtype=torch.uint8, generator=g)

    return (packed(inter, dim), scale(inter, dim),   # w1: [inter, dim]
            packed(dim, inter), scale(dim, inter),   # w2: [dim, inter]
            packed(inter, dim), scale(inter, dim))   # w3: [inter, dim]


def init_weights(model, n_experts):
    g = torch.Generator().manual_seed(0)
    for _, p in model.named_parameters():
        if p.dtype == torch.int64:  # tid2eid
            vals = torch.randint(0, n_experts, p.shape, dtype=torch.int64, generator=g)
            p.data.copy_(vals.to(p.device))
        else:
            p.data.copy_(torch.randn(p.shape, dtype=p.dtype, generator=g).to(p.device)
                         * 0.02)
    for m in model.modules():
        if isinstance(m, RMSNorm):
            m.weight.data.fill_(1.0)


def tiny_args():
    return ModelArgs(
        max_batch_size=1, max_seq_len=32, vocab_size=64, dim=256, moe_inter_dim=256,
        n_layers=2, n_hash_layers=1, n_mtp_layers=0, n_heads=4,
        n_routed_experts=8, n_shared_experts=1, n_activated_experts=2,
        q_lora_rank=128, head_dim=128, rope_head_dim=64, o_groups=2, o_lora_rank=64,
        window_size=8, compress_ratios=(0, 4), rope_factor=16, original_seq_len=0,
        index_n_heads=4, index_head_dim=64, index_topk=4, hc_mult=2,
    )


def test_tiny():
    args = tiny_args()
    bank = ExpertBank(capacity=4, device=DEVICE)
    for layer in range(args.n_layers):
        for eid in range(args.n_routed_experts):
            bank.register(layer, eid, *rand_expert(layer * 100 + eid, args.dim, args.moe_inter_dim))
    model = Transformer(args, bank=bank)
    init_weights(model, args.n_routed_experts)
    model.eval()

    torch.manual_seed(1)
    ids = torch.randint(0, args.vocab_size, (1, 8), device=DEVICE)
    out = model(ids, 0)
    ok = out.shape == (1, args.vocab_size)
    print(f"[{'OK ' if ok else 'BAD'}] prefill logits {tuple(out.shape)} finite={bool(torch.isfinite(out).all())}")
    assert ok and bool(torch.isfinite(out).all())

    for pos in range(8, 16):
        o = model(ids[:, pos - 8:pos - 7], pos)
        if o.shape != (1, args.vocab_size) or not bool(torch.isfinite(o).all()):
            print(f"[BAD] decode pos={pos} shape={tuple(o.shape)}")
            raise SystemExit(1)
    print("[OK ] decode 8 steps")
    print("[OK ] bank:", bank.stats())
    return True


def test_real_config():
    import json
    import os
    cfg_path = os.path.join(
        os.environ.get("DSV4_MODEL", "/home/models/DeepSeek-V4-Flash"),
        "inference", "config.json",
    )
    cfg = json.load(open(cfg_path))
    args = ModelArgs.from_json(cfg_path)
    assert args.n_layers == cfg["n_layers"] and args.moe_inter_dim == cfg["moe_inter_dim"]
    model = Transformer(args, bank=None)
    para = sum(p.numel() * p.element_size() for p in model.parameters())
    buf = sum(b.numel() * b.element_size() for b in model.buffers())
    print(f"[OK ] real config built: params={para/2**30:.2f} GiB buffers={buf/2**30:.2f} GiB")
    del model
    if DEVICE.type == "npu":
        torch.npu.empty_cache()
    return True


if __name__ == "__main__":
    all_ok = True
    all_ok &= test_tiny()
    all_ok &= test_real_config()
    print("ALL PASS" if all_ok else "FAILED")
