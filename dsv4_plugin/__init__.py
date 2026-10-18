"""dsv4_plugin — 基于 vLLM / vllm-ascend 的 DeepSeek-V4 单卡专家卸载插件（out-of-tree）。

入口（见 pyproject.toml ``[project.entry-points."vllm.general_plugins"]``）：
    register_model()

它做两件事（幂等，可在多进程重复调用）：
  1. 把 ``vllm_ascend.models.deepseek_v4.DeepseekV4MoE`` 替换为卸载版 MoE；
  2. 在 ``AscendDeepseekV4ForCausalLM.load_weights`` 上挂专家权重拦截。

**不改 site-packages**：全部通过运行时替换模块符号实现。
"""

from dsv4_plugin.models.deepseek_v4 import install_patches

__all__ = ["register_model"]


def register_model():
    install_patches()
