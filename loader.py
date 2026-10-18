"""Streaming weight loader for route-B (single-card expert offload).

Reads the packed DeepSeek-V4-Flash checkpoint shards directly (no convert step)
by selecting per-tensor from the safetensors headers:

* routed experts (``layers.N.ffn.experts.M.w{1,2,3}.{weight,scale}`` and the MTP
  block) are kept **packed on host RAM** and registered into the ``ExpertBank``;
* FP8 non-expert weights are dequantized with their E8M0 block scale and moved
  to the NPU;
* everything else (bf16/fp32/int64) is cast to the target dtype and moved.

Checkpoint key names map 1:1 onto ``model_npu`` module attribute paths, so the
target tensor is resolved with ``model.get_parameter`` / ``model.get_buffer``.
"""

import json
import os
import re

import torch
from safetensors import safe_open

from npu_ops import dequant_fp8_weight

# layers.N.ffn.experts.M.wX.weight  |  mtp.K.ffn.experts.M.wX.weight  (and .scale)
_EXPERT_RE = re.compile(
    r"^(?:layers\.(\d+)|mtp\.(\d+))\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")


class _Reader:
    """Lazy per-shard safetensors reader (keeps headers open, reads on demand)."""

    def __init__(self, ckpt_dir, weight_map):
        self.dir = ckpt_dir
        self.map = weight_map
        self._handles = {}

    def get(self, key):
        shard = self.map[key]
        h = self._handles.get(shard)
        if h is None:
            h = safe_open(os.path.join(self.dir, shard), framework="pt")
            self._handles[shard] = h
        return h.get_tensor(key)

    def close(self):
        for h in self._handles.values():
            h.__exit__(None, None, None)
        self._handles.clear()


def _target(model, key):
    try:
        return model.get_parameter(key)
    except AttributeError:
        return model.get_buffer(key)


def load_model(model, ckpt_dir, bank, device, include_experts=True,
               max_experts_per_layer=None, verbose=True):
    """Populate ``model`` (and ``bank``) from the checkpoint at ``ckpt_dir``.

    Returns a stats dict. ``include_experts=False`` loads only the non-expert
    weights (useful for tests / dry runs). ``max_experts_per_layer`` caps how
    many experts per layer are pulled into the bank (for loader tests).
    """
    with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    reader = _Reader(ckpt_dir, weight_map)
    n_layers = model.args.n_layers
    device = torch.device(device)

    experts = {}          # (layer, eid) -> {wXp / wXs}
    registered = 0
    n_fp8 = n_cast = n_skip = 0
    unmapped = []

    try:
        for key in weight_map:
            m = _EXPERT_RE.match(key)
            if m:
                if not include_experts:
                    continue
                layer = int(m.group(1)) if m.group(1) is not None else n_layers + int(m.group(2))
                eid, w, kind = int(m.group(3)), m.group(4), m.group(5)
                if max_experts_per_layer is not None and eid >= max_experts_per_layer:
                    continue  # do not even read capped experts off disk
                slots = experts.setdefault((layer, eid), {})
                slots[w + ("p" if kind == "weight" else "s")] = reader.get(key)
                if len(slots) == 6:
                    bank.register(layer, eid, slots["w1p"], slots["w1s"], slots["w2p"],
                                  slots["w2s"], slots["w3p"], slots["w3s"])
                    registered += 1
                    del experts[(layer, eid)]
                continue

            if key.endswith(".scale"):
                continue  # companion of the preceding fp8 weight

            try:
                target = _target(model, key)
            except AttributeError:
                unmapped.append(key)
                n_skip += 1
                continue

            t = reader.get(key)
            scale_key = key[:-len("weight")] + "scale" if key.endswith(".weight") else None
            if scale_key is not None and scale_key in weight_map:
                val = dequant_fp8_weight(t, reader.get(scale_key), 128, target.dtype)
                n_fp8 += 1
            else:
                val = t.to(target.dtype)
                n_cast += 1
            target.data.copy_(val.to(device, non_blocking=True))
    finally:
        reader.close()

    # flush any partially-collected experts (no cap) left unregistered
    if include_experts and max_experts_per_layer is None:
        for (layer, eid), slots in experts.items():
            if len(slots) == 6:
                bank.register(layer, eid, slots["w1p"], slots["w1s"], slots["w2p"],
                              slots["w2s"], slots["w3p"], slots["w3s"])
                registered += 1
        experts.clear()

    stats = {"experts_registered": registered, "fp8_dequant": n_fp8,
             "cast": n_cast, "unmapped": unmapped}
    if verbose:
        print(f"[loader] experts={registered} fp8={n_fp8} cast={n_cast} "
              f"unmapped={len(unmapped)}")
        if unmapped:
            print("[loader] unmapped keys:", unmapped[:10])
    return stats
