"""组件级前向对齐：插件 ``OffloadDeepseekV4MoE`` vs 基线 ``model_npu.MoE``。

口径（用户拍板）：只对齐被改动的 MoE 组件。用真实 checkpoint 的 gate / 专家
权重，在同一输入上比对：

* 路由：top-k 专家索引（集合相等）+ 路由权重（数值一致）；
* 专家输出：逐 token 的 routed 输出（数值一致）。

覆盖两条路径：
* 非 hash 层（layer 3：带 ``e_score_correction_bias`` 的 ``noaux_tc`` topk）；
* hash 层（layer 0：``tid2eid`` 查表，不走 topk）。

另对每条路径追加 **a4w4**（``DSV4_MXFP4_GMM=1``，激活+权重均 MXFP4）相对 bf16
路径的容差对齐——该分支数值口径偏离 fp8 参考，故用 ``MXFP4_*`` 双阈值而非
``max_abs_diff=0``。

基线 ``model_npu.MoE`` 与插件共享同一个 ``ExpertBank``，故专家 bf16 权重逐位
一致，差异只可能来自路由 / 组合逻辑。
"""

import json
import os
import sys
import types

import torch
import torch.nn as nn

try:
    import torch_npu  # noqa: F401  (registers the npu backend)
except Exception:  # pragma: no cover
    pass

from safetensors import safe_open

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from offload import ExpertBank  # noqa: E402
from model_npu import ModelArgs, MoE, DEVICE  # noqa: E402
import dsv4_plugin.models.deepseek_v4 as dv4  # noqa: E402
from dsv4_plugin.models.deepseek_v4 import OffloadDeepseekV4MoE  # noqa: E402

CKPT = os.environ.get("DSV4_MODEL", "/home/models/DeepSeek-V4-Flash")
ATOL = 1e-3
RTOL = 1e-3
# a4w4（激活也转 MXFP4）相对参考 fp8/bf16 路径的容差——不再是 max_abs_diff=0。
# 两条路径共用同一批 FP4 专家权重（fp8 路径把 FP4 反量化到 bf16，E2M1 值在 bf16
# 中可精确表示），差异只来自激活量化：fp8-per128（E4M3，逐元素 rel~4%）vs
# MXFP4-per32（E2M1，1 位尾数 → 逐元素 rel 上界 ~0.25）。
# 实测（layer3，T=8）：相对 no-act-quant 参考 fp8 rel_max≈0.042、a4w4 rel_max≈0.226
# （≈5×，与位宽一致）；a4w4-vs-fp8 max_abs≈0.28 mean≈0.05。故用「worst-case +
# bulk」双阈值作回归基线（非逐元素 allclose）。
MXFP4_MAX_ABS = 0.35
MXFP4_MEAN_ABS = 0.08


class Reader:
    """按权重映射惰性读取 checkpoint 分片。"""

    def __init__(self, ckpt):
        self.dir = ckpt
        with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
            self.map = json.load(f)["weight_map"]
        self._h = {}

    def get(self, key):
        shard = self.map[key]
        h = self._h.get(shard)
        if h is None:
            h = safe_open(os.path.join(self.dir, shard), framework="pt")
            self._h[shard] = h
        return h.get_tensor(key)

    def close(self):
        for h in self._h.values():
            h.__exit__(None, None, None)
        self._h.clear()


class _ZeroShared(nn.Module):
    """零输出，用于把基线 MoE 的 shared 分支屏蔽，隔离 routed 路径。"""

    def forward(self, x, weights=None):
        return torch.zeros_like(x)


def _load_experts(reader, bank, layer, eids):
    for e in eids:
        s = {}
        for w in ("w1", "w2", "w3"):
            s[w + "p"] = reader.get(f"layers.{layer}.ffn.experts.{e}.{w}.weight")
            s[w + "s"] = reader.get(f"layers.{layer}.ffn.experts.{e}.{w}.scale")
        bank.register(layer, e, s["w1p"], s["w1s"], s["w2p"], s["w2s"], s["w3p"], s["w3s"])


def _make_plugin(layer_id, is_hash, baseline, bank):
    """用 ``__new__`` 绕过 __init__（其需要 vllm config），手工装配所需字段。"""
    m = OffloadDeepseekV4MoE.__new__(OffloadDeepseekV4MoE)
    nn.Module.__init__(m)
    m.hash = is_hash
    m.topk = 6
    m.renormalize = True               # config.norm_topk_prob
    m.scoring_func = "sqrtsoftplus"
    m.routed_scaling_factor = 1.5      # config.routed_scaling_factor
    m.swiglu_limit = 10.0              # config.swiglu_limit
    m.n_routed_experts = 256
    m.is_sequence_parallel = False
    m.shared_experts = None
    m.layer_idx = layer_id
    m.engine = types.SimpleNamespace(get=bank.get, get_packed=bank.get_packed)
    # 复用基线的 gate 张量，保证 gate 权重/偏置/查表逐位一致。
    m.gate = types.SimpleNamespace(
        weight=baseline.gate.weight,
        e_score_correction_bias=None if is_hash else baseline.gate.bias,
        tid2eid=None if not is_hash else baseline.gate.tid2eid,
    )
    return m


def _aligned_weights(w, idx):
    """按专家索引升序重排每行权重，消除 topk 排序差异（插件用 sorted=False）。"""
    order = torch.argsort(idx.long(), dim=-1)
    return torch.gather(w, 1, order)


def check_layer(reader, args, layer_id, is_hash):
    tag = f"layer{layer_id}{'(hash)' if is_hash else '(noaux_tc)'}"
    bank = ExpertBank(capacity=args.n_routed_experts, device=DEVICE)

    baseline = MoE(layer_id, args, bank=bank).to(DEVICE)
    baseline.eval()
    baseline.shared_experts = _ZeroShared()

    gw = reader.get(f"layers.{layer_id}.ffn.gate.weight").float().to(DEVICE)
    baseline.gate.weight.data.copy_(gw)
    if is_hash:
        tid = reader.get(f"layers.{layer_id}.ffn.gate.tid2eid").to(torch.long).to(DEVICE)
        baseline.gate.tid2eid.data.copy_(tid)
    else:
        bias = reader.get(f"layers.{layer_id}.ffn.gate.bias").float().to(DEVICE)
        baseline.gate.bias.data.copy_(bias)

    plugin = _make_plugin(layer_id, is_hash, baseline, bank)

    torch.manual_seed(1234)
    T = 8
    x = (torch.randn(T, args.dim, device=DEVICE, dtype=torch.bfloat16) * 0.5)
    ids = torch.randint(0, args.vocab_size, (T,), device=DEVICE, dtype=torch.int64)

    # -- 1) 路由对齐 ---------------------------------------------------
    wp, ip = plugin._route(x, ids)
    wb, ib = baseline.gate(x, ids.flatten())

    sp = torch.sort(ip.long(), dim=-1).values.cpu()
    sb = torch.sort(ib.long(), dim=-1).values.cpu()
    idx_ok = torch.equal(sp, sb)
    w_ok = torch.allclose(_aligned_weights(wp, ip).float().cpu(),
                          _aligned_weights(wb, ib).float().cpu(),
                          atol=1e-4, rtol=1e-4)
    print(f"[{'OK ' if idx_ok else 'BAD'}] {tag} route top-k: "
          f"indices_equal={idx_ok}  weights_close={w_ok}")
    print(f"      plugin idx[0]={ip[0].tolist()}  base idx[0]={ib[0].tolist()}")

    # -- 按需装载被选中的专家 -----------------------------------------
    need = sorted(set(ip.flatten().tolist()) | set(ib.flatten().tolist()))
    _load_experts(reader, bank, layer_id, need)
    print(f"      loaded {len(need)} experts into bank (capacity={bank.capacity})")

    # -- 2) 专家输出对齐 ------------------------------------------------
    out_p = plugin._routed(x, ids)
    out_b = baseline(x, ids)
    diff = (out_p.float() - out_b.float()).abs()
    out_ok = torch.allclose(out_p.float(), out_b.float(), atol=ATOL, rtol=RTOL)
    print(f"[{'OK ' if out_ok else 'BAD'}] {tag} routed output: "
          f"max_abs_diff={diff.max().item():.3e} mean={diff.mean().item():.3e}")
    print(f"      bank stats: {bank.stats()}")

    # -- 3) a4w4 路径（激活+权重均 MXFP4）相对 bf16 路径的容差对齐 --------
    dv4._MXFP4_GMM = True
    try:
        out4_p = plugin._routed(x, ids)
    finally:
        dv4._MXFP4_GMM = False
    diff4 = (out4_p.float() - out_p.float()).abs()
    ref = out_p.float().abs().max().item()
    out4_ok = (diff4.max().item() < MXFP4_MAX_ABS) and (diff4.mean().item() < MXFP4_MEAN_ABS)
    print(f"[{'OK ' if out4_ok else 'BAD'}] {tag} a4w4 vs bf16: "
          f"max_abs_diff={diff4.max().item():.3e} mean={diff4.mean().item():.3e} "
          f"ref_absmax={ref:.3e}")
    ps = bank.stats()
    print(f"      packed stats: hit_rate={ps['packed_hit_rate']:.3f} "
          f"cached={ps['packed_cached']}/{ps['packed_capacity']}")

    return idx_ok and w_ok and out_ok and out4_ok


def main():
    args = ModelArgs()  # 默认值即 DeepSeek-V4-Flash 真实配置
    assert args.dim == 4096 and args.n_routed_experts == 256 and args.n_hash_layers == 3
    reader = Reader(CKPT)
    try:
        ok = True
        ok &= check_layer(reader, args, layer_id=3, is_hash=False)  # noaux_tc + bias
        ok &= check_layer(reader, args, layer_id=0, is_hash=True)   # hash 查表
    finally:
        reader.close()
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
