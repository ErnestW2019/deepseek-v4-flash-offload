# dsv4-plugin

**Single-card `DeepSeek-V4-Flash` inference via expert offloading, as an out-of-tree plugin for vLLM / vllm-ascend on Ascend NPU.**

[English](README.md) | [中文](README_zh.md)

---

## Overview

`dsv4-plugin` makes it possible to **serve `DeepSeek-V4-Flash` on a single Ascend 950PR (128 GB HBM)** without a fork of vLLM or vllm-ascend.

The checkpoint is ~159.6 GB, of which the routed MoE experts are ~150.6 GB (94%) — far beyond a single card's HBM. Instead of decompressing and placing every expert on device, this plugin keeps the experts **packed (MXFP4) in host RAM** and moves them to the NPU on demand, caching dequantized/quantized experts in an **LRU slot cache**.

It installs itself purely through the standard vLLM plugin entry points, replacing one MoE class and wrapping `load_weights` at runtime. **No `site-packages` modification, no fork.**

## Why

| Constraint | Value |
|---|---|
| Total checkpoint | ~159.6 GB |
| Routed experts | ~150.6 GB (94.4%) |
| Dense (non-expert) | ~9 GB |
| Hardware HBM | 128 GB |

Placing all experts on device is impossible, so the experts **must** live off-device while keeping the engine (PagedAttention, continuous batching, OpenAI server) intact.

## Architecture

![dsv4-plugin system architecture](docs/diagrams/architecture.svg)

Three-tier storage:

| Tier | Content | Budget |
|---|---|---|
| HBM (128 GB) | dense weights ~9 GB + KV/activations/graph 10–20 GB + hot expert slots 32–80 GB | ≤ 128 GB |
| Host pinned RAM (628 GB) | packed MXFP4 expert pool ~150 GB | ample |
| L3 NVMe | cold experts (optional) | — |

### Hot expert slots

Hot experts are cached in HBM slots and evicted by LRU; the hit rate climbs as the working set stabilizes.

![LRU expert-slot evolution](docs/diagrams/lru-slots.gif)

### Data flow

![Three-tier expert storage and data flow](docs/diagrams/storage-flow.gif)

### Execution flow (one layer, one step)

![Execution sequence for one MoE layer](docs/diagrams/sequence-reveal.gif)

The same step as a guided walkthrough, one phase at a time:

![Decode step walkthrough: gate & route, slot lookup, H2D fetch, grouped GEMM, write-back](docs/diagrams/decode-step.gif)

## Highlights

- **No fork.** Registered via `vllm.general_plugins`; only a MoE class and `load_weights` are patched at import time.
- **Packed experts on host.** FP4 experts are kept as int8 nibbles + E8M0 per-32 scales, so a miss only transfers 1/4 of the bf16 traffic before on-device dequantization.
- **Two precision profiles.** `bf16` (lossless reference path) and `a4w4` (MXFP4 resident, faster).
- **Ascend grouped-GEMM.** All activated experts of a layer are computed in a single `torch_npu.npu_grouped_matmul` call instead of a per-expert Python loop.
- **Pinned host memory.** H2D for packed experts goes from ~19.8 to ~51.3 GB/s.

## Results

All numbers measured on a single Ascend 950PR (128 GB HBM), `DeepSeek-V4-Flash`, greedy decoding. See [DESIGN.md](DESIGN.md) §9 / §10 for the full record.

![Decode throughput: MXFP4 (a4w4) vs bf16](docs/diagrams/performance.svg)

Single-stream (64-token greedy; experts are ~1/4 the size in FP4):

| Config | tok/s | Expert hit rate | Expert HBM ≈ |
|---|---|---|---|
| `a4w4` @512 packed slots | 6.55 | 0.42 | 6.9 GB |
| **`a4w4` @2048 packed slots** | **7.10** | **0.81** | 27.4 GB |
| `bf16` @512 slots (baseline) | 4.23 | 0.79 | 25.8 GB |

Equal-memory comparison: **`a4w4`@2048 vs `bf16`@512 = 1.68×**.

Serving (8 concurrent × 32 tokens = 256 tokens, `python accept.py`):

| Profile | Command | Aggregated tok/s |
|---|---|---|
| `bf16` | `./serve.sh` | 6.64 |
| `a4w4` | `PROFILE=a4w4 ./serve.sh` | **19.47** (~2.9×) |

Kernel-level: the grouped-GEMM path is **4.8–5.7×** faster than the per-expert loop, and is **bit-exact** (`max_abs_diff = 0.0`) on the bf16 path.

> The `a4w4` profile quantizes activations to MXFP4, which introduces a larger numerical deviation than the fp8 reference path. It is *not* bit-exact; the bf16 profile is the lossless baseline. See [DESIGN.md](DESIGN.md) §9 for the tolerance discussion.

## Requirements

| Component | Version / note |
|---|---|
| Hardware | Ascend 950PR, 128 GB HBM (single card) |
| Python | 3.12 |
| vLLM | 0.23.0 |
| vllm-ascend | 0.23.0 (950 variant, cp312 wheel) |
| torch / torch-npu | as required by the vllm-ascend wheel |
| CANN | matching the vllm-ascend 0.23.0 build |

Runtime dependencies (`torch`, `torch-npu`, `vllm`, `vllm-ascend`) are intentionally **not** declared in `pyproject.toml`, so installing this package never touches those heavy wheels. Install them separately following your vendor's instructions.

You also need the `DeepSeek-V4-Flash` checkpoint on disk (e.g. `/home/models/DeepSeek-V4-Flash`).

## Installation

```bash
# 1. Prepare your Ascend + vLLM + vllm-ascend environment (CANN, torch-npu, etc.)

# 2. Install this plugin (entry point registers it with vLLM automatically)
pip install -e .
```

Verify the plugin is picked up:

```bash
python -c "import vllm_ascend.models.deepseek_v4 as v4; import dsv4_plugin; \
dsv4_plugin.register_model(); print(v4.DeepseekV4MoE.__name__)"
# -> OffloadDeepseekV4MoE
```

## Usage

### Serve

```bash
# Required: point PYTHON at the interpreter that has `vllm` installed
export PYTHON=/path/to/venv/bin/python

# Lossless bf16 profile (default)
./serve.sh

# MXFP4-resident profile (faster; activations quantized, see Results)
PROFILE=a4w4 ./serve.sh
```

`serve.sh` exposes a running OpenAI-compatible server. Acceptance check:

```bash
python accept.py            # BASE_URL / MODEL / N / MAX_TOKENS overridable
```

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `PYTHON` | `python3` | Interpreter containing `vllm` (usually your venv) |
| `DSV4_MODEL` | `/home/models/DeepSeek-V4-Flash` | Checkpoint path |
| `PROFILE` | `bf16` | `bf16` or `a4w4` |
| `DSV4_EXPERT_SLOTS` | `512` | NPU bf16 expert slots |
| `DSV4_PACKED_SLOTS` | `4 × DSV4_EXPERT_SLOTS` | NPU packed-FP4 expert slots |
| `DSV4_PIN_HOST` | `1` | Pin packed experts in host page-locked memory |
| `DSV4_MXFP4_GMM` | `0` | `1` = a4w4 grouped-GEMM path |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Server bind address |
| `MEM_UTIL` | `0.75` | `gpu-memory-utilization` |
| `MAX_MODEL_LEN` / `MAX_NUM_SEQS` | `2048` / `8` | Server limits |
| `ASCEND_TOOLKIT_ROOT` | `/usr/local/Ascend/ascend-toolkit` | CANN toolkit root |
| `ASCEND_ATB_ROOT` | `/usr/local/Ascend/nnal/atb` | ATB root |
| `DSV4_EXTRA_LIBDIR` | *(unset)* | Optional extra `LD_LIBRARY_PATH` entry |

## Testing

Tests are plain scripts run from the repository root inside the Ascend environment. Tests that need the real checkpoint use `DSV4_MODEL`.

```bash
python test_npu_ops.py        # operator correctness vs torch references
python test_offload.py        # ExpertBank host bank + NPU slot cache
python test_loader.py         # weight loader (synthetic + real checkpoint)
python test_model.py          # model_npu forward (tiny + real config)
python test_moe_align.py      # MoE numerical alignment (bf16 bit-exact / a4w4 tolerance)
python test_plugin_load.py    # load_weights tee + expert capture
python test_e2e_vllm.py       # real vLLM engine end-to-end
```

## Repository layout

```
offload.py                  ExpertBank: packed-host experts + NPU LRU slot cache
npu_ops.py                  Ascend operator reimplementations (FP4/FP8 dequant, ...)
loader.py                   Streaming weight loader (reference implementation)
model_npu.py                Standalone reference model (per-token alignment baseline)
dsv4_plugin/
  __init__.py               register_model() — the vLLM plugin entry point
  core/engine.py            ExpertOffloadEngine (wraps ExpertBank)
  models/deepseek_v4.py     MoE takeover + load_weights tee
serve.sh                    Launch the OpenAI-compatible server
accept.py                   HTTP acceptance benchmark
test_*.py                   Test / self-check scripts
pyproject.toml              Packaging + vllm.general_plugins entry point
DESIGN.md                   Design doc + measured M4/M5 results (Chinese)
```

## Design documents

- [DESIGN.md](DESIGN.md) — full design rationale, takeover points, milestone records, and measured results (Chinese).

## Limitations & notes

- **MTP / speculative decoding is out of scope.** The `mtp.*.ffn.experts.*` draft-layer experts are not taken over by the plugin.
- **Single card only.** Expert offloading here assumes TP=1, no expert parallelism.
- **`a4w4` is not bit-exact** with the fp8 reference path (activation quantization differs); it is validated against tuned tolerances.
- Numbers are from a specific CANN / vllm-ascend build; different builds may vary.

## License

[Apache-2.0](LICENSE).
