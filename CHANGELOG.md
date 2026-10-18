# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-18

Initial release: serve `DeepSeek-V4-Flash` on a single Ascend 950PR (128 GB HBM)
by offloading the routed MoE experts to host memory, as an out-of-tree
vLLM / vllm-ascend plugin.

### Added

- **Out-of-tree vLLM plugin** (`dsv4_plugin`), registered through the
  `vllm.general_plugins` entry point. It replaces
  `vllm_ascend.models.deepseek_v4.DeepseekV4MoE` and wraps
  `AscendDeepseekV4ForCausalLM.load_weights` at import time — no `site-packages`
  modification and no fork.
- **`ExpertBank`** (`offload.py`): packed MXFP4 experts kept in pinned host RAM,
  with an NPU-side LRU slot cache for hot experts.
- **Ascend operator reimplementations** (`npu_ops.py`): FP4/FP8 dequantization
  and related operators, validated against torch references.
- **Two precision profiles**: `bf16` (lossless baseline) and `a4w4`
  (MXFP4 resident; faster, activation quantization is not bit-exact).
- **Ascend grouped-GEMM path** using `torch_npu.npu_grouped_matmul`, 4.8–5.7×
  faster than a per-expert loop and bit-exact on the bf16 path.
- **Reference implementations**: streaming weight `loader.py` and a standalone
  `model_npu.py` used as the per-token alignment baseline.
- **Serving tooling**: `serve.sh` (OpenAI-compatible server) and `accept.py`
  (HTTP acceptance benchmark).
- **Self-checks** (`test_*.py`) and micro-benchmarks (`_bench_dequant.py`,
  `_bench_gmm.py`).
- **Documentation**: bilingual `README.md` / `README_zh.md`, `DESIGN.md` with the
  measured results, and generated diagrams under `docs/diagrams/`.
- **Packaging**: `pyproject.toml` with the vLLM entry point, plus a hardware-free
  CI workflow that byte-compiles the sources and verifies the wheel contents.

[Unreleased]: https://github.com/ErnestW2019/deepseek-v4-flash-offload/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ErnestW2019/deepseek-v4-flash-offload/releases/tag/v0.1.0
