"""Self-test for ExpertBank (host packed bank + NPU LRU slot cache)."""

import torch

import npu_ops
from offload import ExpertBank

DEV = "npu:0"


def make_expert(seed, N1=64, K1=64, N2=64, K2=64):
    """Random packed FP4 expert weights in the real checkpoint layout."""
    g = torch.Generator().manual_seed(seed)
    def packed(N, K):
        return torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, generator=g)
    def scale(N, K):
        return torch.randint(100, 150, (N, K // 32), dtype=torch.uint8, generator=g)
    return (packed(N1, K1), scale(N1, K1),
            packed(N2, K2), scale(N2, K2),
            packed(N1, K1), scale(N1, K1))


def ref_dequant(wp, ws):
    return npu_ops.dequant_fp4_weight(wp.to(DEV), ws.to(DEV), torch.bfloat16)


def check(name, a, b, atol=0.0, rtol=0.0):
    a = a.detach().to("cpu").float()
    b = b.detach().to("cpu").float()
    diff = (a - b).abs().max().item()
    ok = diff <= atol + rtol * b.abs().max().item()
    print(f"[{'OK ' if ok else 'FAIL'}] {name:32s} max_abs={diff:.3e}")
    return ok


def main():
    all_ok = True
    torch.manual_seed(0)

    # register 5 experts for layer 0 and 3 for layer 1
    bank = ExpertBank(capacity=4, device=DEV, dtype=torch.bfloat16)
    refs = {}
    for eid in range(5):
        t = make_expert(100 + eid)
        bank.register(0, eid, *t)
        refs[(0, eid)] = t
    for eid in range(3):
        t = make_expert(200 + eid)
        bank.register(1, eid, *t)
        refs[(1, eid)] = t

    all_ok &= (bank.num_experts() == 8)
    all_ok &= bank.has(0, 4) and bank.has(1, 2) and not bank.has(0, 5)

    # 1) get() matches a direct dequant
    d = bank.get(0, 0)
    wp, ws = refs[(0, 0)][0], refs[(0, 0)][1]
    all_ok &= check("get w1 == direct dequant", d["w1"], ref_dequant(wp, ws))
    all_ok &= check("get w2 shape", torch.tensor(list(d["w2"].shape)),
                    torch.tensor([64, 64]))

    # 2) hit on the same key
    h0 = bank.hits
    bank.get(0, 0)
    all_ok &= (bank.hits == h0 + 1)

    # 3) LRU: fill cache to capacity, touch an early key, then overflow.
    bank.invalidate()
    bank.hits = bank.misses = bank.evictions = 0
    for eid in range(4):
        bank.get(0, eid)                     # fills 4 slots
    all_ok &= (len(bank._cache) == 4)
    bank.get(0, 0)                            # touch 0 -> MRU
    bank.get(1, 0)                            # miss -> evicts LRU (0,1)
    all_ok &= ((0, 1) not in bank._cache)
    all_ok &= ((0, 0) in bank._cache and (1, 0) in bank._cache)
    all_ok &= (bank.evictions == 1)
    s = bank.stats()
    all_ok &= (s["misses"] == 5 and s["hits"] == 1 and s["hit_rate"] == 1 / 6)

    # 4) capacity=0 -> never cached
    bank0 = ExpertBank(capacity=0, device=DEV)
    bank0.register(0, 0, *make_expert(7))
    bank0.get(0, 0)
    bank0.get(0, 0)
    all_ok &= (len(bank0._cache) == 0 and bank0.stats()["misses"] == 2)

    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
