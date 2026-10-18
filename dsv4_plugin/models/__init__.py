"""dsv4_plugin.models — 被接管的模型实现。"""

from dsv4_plugin.models.deepseek_v4 import (
    OffloadDeepseekV4MoE,
    install_patches,
)

__all__ = ["OffloadDeepseekV4MoE", "install_patches"]
