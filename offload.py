"""Route-B expert offload: host-RAM expert bank + NPU bf16 slot cache with LRU.

The checkpoint keeps routed experts in FP4 (packed as int8 nibbles) plus an
E8M0 per-32 scale. Dequantizing all 256*43 experts to bf16 would need ~560 GB
of RAM, so the host side keeps the *packed* bytes verbatim and dequantizes on
demand. The NPU side holds a configurable number of dequantized bf16 experts
("slots"); a miss copies the packed bytes H2D (1/4 the traffic of bf16) and
dequantizes on device.

Reference:
  FreeToken / KTransformers style expert offloading.
"""

import collections
import os

import torch

from npu_ops import dequant_fp4_weight


class ExpertBank:
    """Host-resident packed expert weights + NPU-side LRU cache of bf16 experts.

    Parameters
    ----------
    capacity : int
        Number of experts kept dequantized on the NPU at once.
    device : str | torch.device
        NPU device for the slot cache.
    dtype : torch.dtype
        Compute dtype for the dequantized experts (bf16 by default).
    pin : bool | None
        Pin host packed experts in page-locked memory. 实测 H2D 20→51 GB/s
        （miss 帧 43→23ms）。``None`` 时取环境变量 ``DSV4_PIN_HOST``（默认开）。
    """

    def __init__(self, capacity: int, device, dtype=torch.bfloat16, verbose=False, pin=None):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.dtype = dtype
        self.verbose = verbose
        if pin is None:
            pin = os.environ.get("DSV4_PIN_HOST", "1") == "1"
        self.pin = pin
        self._host = {}       # (layer, eid) -> (w1p, w1s, w2p, w2s, w3p, w3s) on CPU
        self._cache = collections.OrderedDict()  # (layer, eid) -> dict(layer bf16)
        self._packed_cache = collections.OrderedDict()  # (layer, eid) -> dict(packed on device)
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        # 打包态专家只占 bf16 的 1/4，默认给 4× 槽位（与 bf16 槽位等显存），
        # 实测命中率 0.42→0.81、e2e 6.55→7.10 tok/s（vs bf16 4.23 tok/s = 1.68×）。
        self.packed_capacity = int(os.environ.get("DSV4_PACKED_SLOTS", str(4 * self.capacity)))
        self.packed_hits = 0
        self.packed_misses = 0

    # -- host bank --------------------------------------------------------
    def register(self, layer, eid, w1p, w1s, w2p, w2s, w3p, w3s):
        """Store the raw packed expert weights (CPU tensors, as loaded)."""
        tup = (w1p, w1s, w2p, w2s, w3p, w3s)
        if self.pin:
            try:
                tup = tuple(x if x.is_pinned() else x.pin_memory() for x in tup)
            except RuntimeError as exc:  # 锁页内存不可用则一次性降级
                print(f"[ExpertBank] pin_memory 失败，降级为非锁页: {exc}", flush=True)
                self.pin = False
        self._host[(int(layer), int(eid))] = tup

    def has(self, layer, eid):
        return (int(layer), int(eid)) in self._host

    def num_experts(self):
        return len(self._host)

    # -- device slot cache ------------------------------------------------
    def _dequant_on_device(self, layer, eid):
        w1p, w1s, w2p, w2s, w3p, w3s = self._host[(layer, eid)]
        d = self.device
        dt = self.dtype
        w1 = dequant_fp4_weight(w1p.to(d, non_blocking=True), w1s.to(d, non_blocking=True), dt)
        w2 = dequant_fp4_weight(w2p.to(d, non_blocking=True), w2s.to(d, non_blocking=True), dt)
        w3 = dequant_fp4_weight(w3p.to(d, non_blocking=True), w3s.to(d, non_blocking=True), dt)
        return {"w1": w1, "w2": w2, "w3": w3}

    def get(self, layer, eid):
        key = (int(layer), int(eid))
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            self._cache.move_to_end(key)
            return cached
        self.misses += 1
        vals = self._dequant_on_device(*key)
        if self.capacity > 0:
            self._cache[key] = vals
            while len(self._cache) > self.capacity:
                self._cache.popitem(last=False)
                self.evictions += 1
        return vals

    # -- device packed (MXFP4 a4w4) cache --------------------------------
    def get_packed(self, layer, eid):
        """设备侧**打包态**专家缓存：直接可喂 a4w4 grouped-GEMM，免反量化。

        返回 ``{"w{1,2,3}p", "w{1,2,3}s"}``：
        * ``wp`` — 打包 FP4 权重 ``[N, K//2]``（int8，原样）；
        * ``ws`` — E8M0 per-32 scale ``[N, K//32]``（2D）。

        调用方需自行拼成 ``[g, N, K//2]`` 再 ``transpose(1, 2)``（保留 transpose
        stride），见 ``_routed_mxfp4``。
        """
        key = (int(layer), int(eid))
        cached = self._packed_cache.get(key)
        if cached is not None:
            self.packed_hits += 1
            self._packed_cache.move_to_end(key)
            return cached
        self.packed_misses += 1
        p1, s1, p2, s2, p3, s3 = self._host[key]
        d = self.device
        move = lambda t: t.to(d, non_blocking=True).contiguous()  # noqa: E731
        vals = {"w1p": move(p1), "w1s": move(s1), "w2p": move(p2),
                "w2s": move(s2), "w3p": move(p3), "w3s": move(s3)}
        if self.packed_capacity > 0:
            self._packed_cache[key] = vals
            while len(self._packed_cache) > self.packed_capacity:
                self._packed_cache.popitem(last=False)
        return vals

    def invalidate(self):
        self._cache.clear()
        self._packed_cache.clear()

    def stats(self):
        tot = self.hits + self.misses
        rate = (self.hits / tot) if tot else 0.0
        ptot = self.packed_hits + self.packed_misses
        prate = (self.packed_hits / ptot) if ptot else 0.0
        return {"hits": self.hits, "misses": self.misses,
                "evictions": self.evictions, "hit_rate": rate,
                "cached": len(self._cache), "capacity": self.capacity,
                "packed_hits": self.packed_hits, "packed_misses": self.packed_misses,
                "packed_hit_rate": prate, "packed_cached": len(self._packed_cache),
                "packed_capacity": self.packed_capacity,
                "pinned": self.pin}

    def __repr__(self):
        s = self.stats()
        return (f"ExpertBank(experts={self.num_experts()}, cached={s['cached']}/"
                f"{s['capacity']}, hit_rate={s['hit_rate']:.3f})")
