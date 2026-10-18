"""端到端：用真实 vLLM 引擎加载 DeepSeek-V4-Flash，验证卸载插件接管 MoE。

本脚本只做 M2 验收的前半段：
  1. 单进程（``VLLM_ENABLE_V1_MULTIPROCESSING=0``）起离线 ``LLM``，走通
     ``_offload_load_weights`` tee —— 全部主干专家应进 host 池；
  2. 打印 ``ExpertOffloadEngine`` 统计，校验 ``captured == 43*256``、``parked == 0``；
  3. 跑一次贪心生成，证明前向可服务（数值对齐在后续脚本单独做）。

用法（须先 source CANN/ATB 环境）：

    python test_e2e_vllm.py
"""

import os
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_V1", "1")

MODEL = os.environ.get("DSV4_MODEL", "/home/models/DeepSeek-V4-Flash")
SLOTS = int(os.environ.get("DSV4_EXPERT_SLOTS", "512"))
MEM_UTIL = float(os.environ.get("DSV4_GPU_MEM_UTIL", "0.75"))
MAX_TOKENS = int(os.environ.get("DSV4_MAX_TOKENS", "8"))
N_PROMPTS = int(os.environ.get("DSV4_N_PROMPTS", "1"))
MAX_NUM_SEQS = int(os.environ.get("DSV4_MAX_NUM_SEQS", "1"))
N_LAYERS = 43
N_EXPERTS = 256
N_MAIN_EXPERTS = N_LAYERS * N_EXPERTS  # 11008


def main() -> int:
    from vllm import LLM, SamplingParams

    t0 = time.time()
    print(f"[e2e] loading model={MODEL} slots={SLOTS} mem_util={MEM_UTIL} ...", flush=True)
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=2048,
        max_num_seqs=MAX_NUM_SEQS,
        gpu_memory_utilization=MEM_UTIL,
        trust_remote_code=False,
        disable_log_stats=True,
    )
    print(f"[e2e] engine built in {time.time() - t0:.1f}s", flush=True)

    from dsv4_plugin.core.engine import get_engine
    from dsv4_plugin.models import deepseek_v4 as mod
    import vllm_ascend.models.deepseek_v4 as v4

    print(f"[e2e] patch flag={getattr(v4, '_DSV4_OFFLOAD_PATCHED', False)} "
          f"moe_cls={v4.DeepseekV4MoE.__name__} "
          f"lw={v4.AscendDeepseekV4ForCausalLM.load_weights.__name__}", flush=True)

    engine = get_engine()
    stats = engine.stats()
    print(f"[e2e] engine stats: {stats}", flush=True)
    print(f"[e2e] bank num_experts={engine.bank.num_experts()} "
          f"captured={engine.captured} parked={stats['parked']}", flush=True)

    ok_capture = engine.captured == N_MAIN_EXPERTS and stats["parked"] == 0
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "0") != "0":
        # 多进程：权重在 EngineCore 进程加载，driver 进程的引擎单例为空，此项不适用。
        print(f"[SKIP] capture check (multiprocess): driver engine "
              f"captured={engine.captured}", flush=True)
        ok_capture = True
    else:
        print(f"[{'OK ' if ok_capture else 'BAD'}] capture: "
              f"{engine.captured}/{N_MAIN_EXPERTS} parked={stats['parked']}", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    base_prompts = [
        "The capital of France is",
        "The capital of Japan is",
        "Explain what a GPU is in one sentence:",
        "Write a Python function that adds two numbers:",
        "What is the boiling point of water?",
        "Name three primary colors:",
        "Translate 'hello world' into French:",
        "What year did the first man land on the moon?",
    ]
    prompts = (base_prompts * ((N_PROMPTS + len(base_prompts) - 1) // len(base_prompts)))[:N_PROMPTS]
    t1 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t1
    n_out = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[e2e] generate {dt:.2f}s for {n_out} tokens over {len(prompts)} prompts "
          f"-> {n_out / dt:.2f} tok/s (e2e incl. prefill)", flush=True)
    print(f"[e2e] sample output: {outs[0].outputs[0].text!r}", flush=True)
    print(f"[e2e] post-run stats: {engine.stats()}", flush=True)

    return 0 if ok_capture else 1


if __name__ == "__main__":
    sys.exit(main())
