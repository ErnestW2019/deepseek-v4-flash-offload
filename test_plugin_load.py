"""Integration test: plugin ``load_weights`` tee + expert capture.

Part 1  Real index.json name coverage: the plugin's ``is_expert_weight`` consumes
        exactly the main-model routed-expert keys (``layers.*.ffn.experts.*``)
        and lets the MTP spec-decode keys (``mtp.*``) fall through.
Part 2  Tee split: drive ``_offload_load_weights`` with a recording stub as the
        wrapped original, over a synthetic iterator carrying every real
        checkpoint key -> the stub must see exactly the non-main-expert keys,
        and the engine must capture exactly n_layers*n_experts experts.
Part 3  Real shard spot check: read two real experts' six packed tensors,
        capture them, and fetch via the NPU slot cache.
"""

import json
import os

import torch

try:
    import torch_npu  # noqa: F401  (registers the npu backend)
except Exception:  # pragma: no cover
    pass

import dsv4_plugin.core.engine as eng
import dsv4_plugin.models.deepseek_v4 as mod
from dsv4_plugin.core.engine import ExpertOffloadEngine, is_expert_weight, parse_expert_name

CKPT = os.environ.get("DSV4_MODEL", "/home/models/DeepSeek-V4-Flash")
INDEX = os.path.join(CKPT, "model.safetensors.index.json")
N_LAYERS = 43
N_EXPERTS = 256
N_PER_EXPERT = 6  # w1/w2/w3 x weight/scale


def load_weight_map():
    with open(INDEX) as f:
        return json.load(f)["weight_map"]


def test_name_coverage(wm):
    main = [k for k in wm if is_expert_weight(k)]
    mtp = [k for k in wm if ".ffn.experts." in k and not is_expert_weight(k)]
    n_expect = N_LAYERS * N_EXPERTS * N_PER_EXPERT
    ok = (
        len(main) == n_expect
        and all(".ffn.experts." in k and k.startswith("layers.") for k in main)
        and all(k.startswith("mtp.") for k in mtp)
        and len(mtp) == N_EXPERTS * N_PER_EXPERT  # 1 MTP draft layer
    )
    print(f"[{'OK ' if ok else 'BAD'}] name coverage: main={len(main)}/{n_expect} "
          f"mtp_fallthrough={len(mtp)}")
    assert ok

    # every main expert contributes exactly 6 blocks; parse maps layer/eid/w/kind
    groups = {}
    for k in main:
        layer, eid, w, kind = parse_expert_name(k)
        groups.setdefault((layer, eid), set()).add(w + "." + kind)
    assert len(groups) == N_LAYERS * N_EXPERTS, len(groups)
    assert all(len(v) == N_PER_EXPERT for v in groups.values())
    layer, eid, w, kind = parse_expert_name("layers.42.ffn.experts.255.w2.scale")
    assert (layer, eid, w, kind) == (42, 255, "w2", "scale")
    # mtp parses too, but is not consumed by default
    assert parse_expert_name("mtp.0.ffn.experts.7.w1.weight") == (-1, 7, "w1", "weight")
    print(f"[OK ] groups={len(groups)} blocks/expert={N_PER_EXPERT} parse ok")
    return True


def test_tee_split(wm):
    engine = ExpertOffloadEngine(slot_capacity=64, device="cpu")
    eng._ENGINE = engine

    received = []

    def stub_original(self, weights):
        for name, _ in weights:
            received.append(name)
        return set()

    saved = mod._ORIG_LOAD_WEIGHTS
    mod._ORIG_LOAD_WEIGHTS = stub_original
    try:
        it = ((k, torch.zeros(1, dtype=torch.uint8)) for k in wm)
        mod._offload_load_weights(object(), it)
    finally:
        mod._ORIG_LOAD_WEIGHTS = saved

    non_main = {k for k in wm if not is_expert_weight(k)}
    ok = (
        engine.captured == N_LAYERS * N_EXPERTS
        and set(received) == non_main
        and len(received) == len(wm) - N_LAYERS * N_EXPERTS * N_PER_EXPERT
    )
    print(f"[{'OK ' if ok else 'BAD'}] tee split: captured={engine.captured}/"
          f"{N_LAYERS * N_EXPERTS} passthrough={len(received)}")
    assert ok
    assert engine.stats()["parked"] == 0
    print(f"[OK ] bank experts={engine.bank.num_experts()} parked=0")
    return True


def _read_expert(reader_map, dir_, layer, eid):
    from safetensors import safe_open

    tensors = {}
    handles = {}
    for w in ("w1", "w2", "w3"):
        for kind in ("weight", "scale"):
            key = f"layers.{layer}.ffn.experts.{eid}.{w}.{kind}"
            shard = reader_map[key]
            h = handles.get(shard)
            if h is None:
                h = safe_open(os.path.join(dir_, shard), framework="pt")
                handles[shard] = h
            tensors[w + ("p" if kind == "weight" else "s")] = h.get_tensor(key)
    for h in handles.values():
        h.__exit__(None, None, None)
    return tensors


def test_real_shard(wm):
    engine = ExpertOffloadEngine(slot_capacity=64, device="npu:0")
    eng._ENGINE = engine

    layer, eid = 10, 0
    t = _read_expert(wm, CKPT, layer, eid)
    for w in ("w1", "w2", "w3"):
        engine.capture(f"layers.{layer}.ffn.experts.{eid}.{w}.weight", t[w + "p"])
        engine.capture(f"layers.{layer}.ffn.experts.{eid}.{w}.scale", t[w + "s"])
    assert engine.captured == 1 and engine.bank.has(layer, eid)

    d = engine.get(layer, eid)
    inter, dim = 2048, 4096
    ok = (
        d["w1"].shape == (inter, dim)
        and d["w2"].shape == (dim, inter)
        and d["w3"].shape == (inter, dim)
        and all(v.dtype == torch.bfloat16 for v in d.values())
        and all(v.device.type == "npu" for v in d.values())
        and all(bool(torch.isfinite(v).all()) for v in d.values())
    )
    print(f"[{'OK ' if ok else 'BAD'}] real shard: layer={layer} eid={eid} "
          f"w1={tuple(d['w1'].shape)}{d['w1'].dtype}@{d['w1'].device.type}")
    assert ok
    del engine
    torch.npu.empty_cache()
    return True


if __name__ == "__main__":
    wm = load_weight_map()
    all_ok = True
    all_ok &= test_name_coverage(wm)
    all_ok &= test_tee_split(wm)
    all_ok &= test_real_shard(wm)
    print("ALL PASS" if all_ok else "FAILED")
    raise SystemExit(0 if all_ok else 1)
