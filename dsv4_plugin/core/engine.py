"""专家卸载引擎：包住既有 ``offload.ExpertBank``，供 vLLM 侧 MoE 使用。

设计要点（见 ../../DESIGN.md）：
* host 侧保存**打包态 FP4** 专家（int8 nibble + e8m0 per-32 scale），不解压；
* NPU 侧维护 bf16 专家槽位 LRU，miss 时才 H2D + 反量化；
* 全模型共享一个引擎实例（43 层共用一个专家池）。
"""

import os
import re
import sys

import torch

# 让既有顶层模块（offload.py / npu_ops.py）可被 import。
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from offload import ExpertBank  # noqa: E402

# 原始 checkpoint 键：layers.N.ffn.experts.M.w{1,2,3}.{weight,scale}（可带 model. 前缀）
_EXPERT_RE = re.compile(
    r"(?:^|\.)(?:layers\.(\d+)|mtp\.(\d+))\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$"
)
# 主干（非 MTP）专家：layers.N.ffn.experts.M.w{1,2,3}.{weight,scale}
_MAIN_EXPERT_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)" r"\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$"
)

_SLOT = {"weight": "p", "scale": "s"}  # pack / scale 后缀，对齐 ExpertBank.register


def parse_expert_name(name: str):
    """解析专家权重名 -> (layer, eid, w, kind)；非专家返回 None。"""
    m = _EXPERT_RE.search(name)
    if not m:
        return None
    layer = int(m.group(1)) if m.group(1) is not None else -1 - int(m.group(2))
    return layer, int(m.group(3)), m.group(4), m.group(5)


def is_expert_weight(name: str, include_mtp: bool = False) -> bool:
    """是否为专家权重。

    ``mtp.*`` 是 speculative-decoding 的 draft 层，其 ``load_weights`` 走的是
    ``DeepSeekV4MTP``（本插件未挂 tee），且主干 ``load_weights`` 本身会跳过它们
    （``deepseek_v4.py`` 的 ``get_spec_layer_idx_from_weight_name``）。故默认只认
    主干 ``layers.*`` 专家，避免把用不到的 MTP 专家误收进 host 池。
    """
    pattern = _EXPERT_RE if include_mtp else _MAIN_EXPERT_RE
    return pattern.search(name) is not None


class ExpertOffloadEngine:
    """host 打包专家池 + NPU 槽位缓存；按 (layer, eid) 粒度取用。"""

    def __init__(self, slot_capacity: int, device="npu:0", dtype=torch.bfloat16):
        self.bank = ExpertBank(capacity=slot_capacity, device=device, dtype=dtype)
        # (layer, eid) -> {"w1p":..,"w1s":..,...}，集齐 6 块后登记进 bank
        self._pending = {}
        self.captured = 0

    # -- 权重捕获（load_weights 拦截点调用）-----------------------------
    def capture(self, name: str, tensor: torch.Tensor) -> bool:
        """把专家张量登记进 host 池；返回是否被本引擎消费。"""
        parsed = parse_expert_name(name)
        if parsed is None:
            return False
        layer, eid, w, kind = parsed
        slots = self._pending.setdefault((layer, eid), {})
        slots[w + _SLOT[kind]] = tensor.detach().to("cpu")
        if len(slots) >= 6:
            self.bank.register(layer, eid,
                               slots["w1p"], slots["w1s"], slots["w2p"],
                               slots["w2s"], slots["w3p"], slots["w3s"])
            del self._pending[(layer, eid)]
            self.captured += 1
        return True

    # -- 取用 -------------------------------------------------------------
    def get(self, layer: int, eid: int):
        """返回 NPU 上 bf16 的 {"w1","w2","w3"}（必要时 H2D + 反量化）。"""
        return self.bank.get(layer, eid)

    def get_packed(self, layer: int, eid: int):
        """返回 NPU 上**打包态** FP4 专家（a4w4 grouped-GEMM 直用）。

        dict 键：``w{1,2,3}p``（[N, K//2] int8）、``w{1,2,3}s``（[N, K//32] E8M0）。
        """
        return self.bank.get_packed(layer, eid)

    def stats(self):
        s = self.bank.stats()
        s["captured"] = self.captured
        s["parked"] = len(self._pending)
        return s

    def __repr__(self):
        return f"ExpertOffloadEngine({self.bank!r}, captured={self.captured})"


# -- 全模型单例 --------------------------------------------------------
_ENGINE: ExpertOffloadEngine | None = None


def get_engine(slot_capacity: int | None = None, device="npu:0") -> ExpertOffloadEngine:
    """获取（或惰性创建）全局引擎。``slot_capacity`` 仅首次生效。"""
    global _ENGINE
    if _ENGINE is None:
        if slot_capacity is None:
            # 端到端实测（43 层 / 24 token）：64 槽命中 0.67、256 槽 0.67、768 槽 0.82、
            # 1024 槽 0.84，之后收益递减；512 为显存/命中率折中（约 24 GiB bf16 专家）。
            slot_capacity = int(os.environ.get("DSV4_EXPERT_SLOTS", "512"))
        _ENGINE = ExpertOffloadEngine(slot_capacity=slot_capacity, device=device)
    return _ENGINE
