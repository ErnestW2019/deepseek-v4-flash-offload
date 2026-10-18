"""接管 vllm-ascend 的 DeepSeek-V4 MoE：把专家权重放到 host，按需上卡。

不改 site-packages，仅通过 ``install_patches()`` 在运行时替换两个符号：

* ``vllm_ascend.models.deepseek_v4.DeepseekV4MoE`` -> :class:`OffloadDeepseekV4MoE`
  （``DeepseekV2DecoderLayer.__init__`` 直接引用该模块全局，替换后所有层生效）；
* ``AscendDeepseekV4ForCausalLM.load_weights`` -> :func:`_offload_load_weights`
  （对专家权重做 tee 拦截，其余交回原实现，避免复刻 200 行加载逻辑）。

计算路径：路由（hash 层查表 / 非 hash 层带偏置 top-k，均按 DeepSeek-V4
``noaux_tc`` + ``sqrtsoftplus`` 语义）+ 引擎按专家批量取权重、用昇腾
grouped-GEMM 一次算完所有被激活专家（M4 生产内核），见 ../../DESIGN.md。
"""

import os

import torch
import torch.nn.functional as F
import torch_npu
from torch import nn

import vllm_ascend.models.deepseek_v4 as v4

from dsv4_plugin.core.engine import get_engine, is_expert_weight
from npu_ops import act_quant

_PATCHED = False
# p4-d：专家常驻 FP4、激活也转 FP4（a4w4 grouped-GEMM）。开启后数值口径偏离
# 参考 fp8 路径（激活量化从 FP8 变为 MXFP4），需按新基线对齐。
_MXFP4_GMM = os.environ.get("DSV4_MXFP4_GMM", "0") == "1"
_F4 = torch_npu.float4_e2m1fn_x2
_E8M0 = torch_npu.float8_e8m0fnu


class _ExpertsFacade:
    """占位 FusedMoE 的轻量替身：满足 ``set_moe_parameters`` 等对 ``moe.experts`` 的引用。"""

    is_internal_router = False

    def __init__(self, moe):
        self._moe = moe

    def update_expert_map(self):  # EPLB 关闭时不会被调用
        pass


class OffloadDeepseekV4MoE(v4.DeepseekV4MoE):
    """专家权重常驻 host、按需 H2D 的 MoE 层。"""

    def __init__(self, config, parallel_config, quant_config=None, prefix="", is_draft_layer=False):
        # 故意不调用 super().__init__：它会构造 FusedMoE 并把 ~150 GB 专家权重放到 NPU。
        nn.Module.__init__(self)

        self.tp_size = v4.get_tensor_model_parallel_world_size()
        self.tp_rank = v4.get_tensor_model_parallel_rank()
        self.layer_idx = int(prefix.split(sep=".")[-2])
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.5)
        self.swiglu_limit = getattr(config, "swiglu_limit", None)

        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.topk = config.num_experts_per_tok
        self.renormalize = config.norm_topk_prob
        self.num_expert_group = getattr(config, "n_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        self.scoring_func = getattr(config, "scoring_func", "softmax")

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        self.gate = v4.ReplicatedLinear(
            config.hidden_size, config.n_routed_experts, bias=False,
            quant_config=None, prefix=f"{prefix}.gate",
        )
        self.gate.precast_fp32_weight = True

        # EPLB 元数据（单卡默认关闭；保留字段以兼容 set_moe_parameters 的读取）。
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.ep_size = 1
        self.n_local_physical_experts = self.n_physical_experts
        self.physical_expert_start = 0
        self.physical_expert_end = self.n_local_physical_experts

        # 共享专家（dense）保持常驻 NPU，走原生路径。
        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            self.shared_experts = v4.DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.hash = self.layer_idx < config.num_hash_layers and not is_draft_layer
        if self.hash:
            self.gate.tid2eid = nn.Parameter(
                torch.zeros(config.vocab_size, config.num_experts_per_tok, dtype=torch.int32),
                requires_grad=False,
            )
            self.gate.e_score_correction_bias = None
        else:
            self.gate.tid2eid = None
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32)
            )

        self.engine = get_engine()
        self.experts = _ExpertsFacade(self)

    def _route(self, hs: torch.Tensor, input_ids: torch.Tensor | None):
        """DeepSeek-V4 路由（``noaux_tc`` + ``sqrtsoftplus`` 语义）。

        与 ``model_npu.Gate.forward``（及 vLLM ``fused_topk_bias``）一致：
        用*带偏置*的分数选专家，用*未带偏置*的原始分数作为路由权重，
        再 renormalize、最后乘 ``routed_scaling_factor``。
        """
        gating = F.linear(hs.float(), self.gate.weight)  # (T, E) logits
        if self.scoring_func == "softmax":
            scores = gating.softmax(dim=-1)
        elif self.scoring_func == "sigmoid":
            scores = gating.sigmoid()
        else:  # sqrtsoftplus
            scores = F.softplus(gating).sqrt()

        if self.hash:
            if input_ids is None:
                raise RuntimeError(
                    "hash 层路由需要 input_ids（forward_context.input_ids 缺失）"
                )
            indices = self.gate.tid2eid[input_ids.to(torch.long).flatten()]
        else:
            bias = self.gate.e_score_correction_bias
            scores_for_choice = scores if bias is None else scores + bias.to(scores.dtype)
            indices = torch.topk(scores_for_choice, k=self.topk, dim=-1, sorted=False)[1]

        weights = scores.gather(1, indices.to(torch.long))
        if self.renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.routed_scaling_factor
        return weights, indices

    def _routed(self, hs: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        weights, indices = self._route(hs, input_ids)
        T, K = indices.shape

        # 按专家 id 排序，使每个专家分到的 token 连续成组（等键必然相邻）。
        flat_e = indices.reshape(-1).to(torch.long)
        flat_w = weights.reshape(-1)
        order = torch.argsort(flat_e)
        s_e = flat_e[order]
        s_w = flat_w[order]
        row = order // K  # 每个 (token, top-k) 槽位对应的 token 行号

        counts = torch.bincount(s_e, minlength=self.n_routed_experts)
        active = torch.nonzero(counts > 0, as_tuple=False).squeeze(-1)
        group_list = counts[active].to(torch.int64)

        if _MXFP4_GMM:
            return self._routed_mxfp4(s_w, row, active, group_list, T, hs)

        # 与基线 ``_expert`` 一致：GEMM 前对激活做一次 fp8 round-trip。
        xq = act_quant(hs.clone(), 128, "ue8m0", None, True)
        x_sorted = xq[row]

        # grouped-GEMM：x 按专家分组排序，weight 为每专家 [K, N]（F.linear 权重转置）。
        w1 = [self.engine.get(self.layer_idx, int(e))["w1"].t() for e in active]
        w3 = [self.engine.get(self.layer_idx, int(e))["w3"].t() for e in active]

        gate = torch_npu.npu_grouped_matmul(
            x=[x_sorted], weight=w1, split_item=2, group_list_type=1,
            group_type=0, group_list=group_list)[0].float()
        up = torch_npu.npu_grouped_matmul(
            x=[x_sorted], weight=w3, split_item=2, group_list_type=1,
            group_type=0, group_list=group_list)[0].float()

        if self.swiglu_limit and self.swiglu_limit > 0:
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = gate.clamp(max=self.swiglu_limit)
        h = F.silu(gate) * up * s_w[:, None]
        hq = act_quant(h.to(hs.dtype).clone(), 128, "ue8m0", None, True)

        w2 = [self.engine.get(self.layer_idx, int(e))["w2"].t() for e in active]
        out = torch_npu.npu_grouped_matmul(
            x=[hq], weight=w2, split_item=2, group_list_type=1,
            group_type=0, group_list=group_list)[0]

        y = torch.zeros(T, hs.shape[-1], dtype=torch.float32, device=hs.device)
        y.index_add_(0, row, out.float())
        return y.type_as(hs)

    def _routed_mxfp4(self, s_w, row, active, group_list, T, hs):
        """a4w4 路径：激活与专家权重均为 MXFP4，grouped-GEMM 直接消费打包态。

        与 bf16 路径共用路由/排序/``s_w``；区别在于
        * 激活用 ``npu_dynamic_mx_quant`` 转 FP4（而非 fp8 round-trip）；
        * 专家权重取自 ``engine.get_packed``（打包 FP4 + E8M0 scale，不解压）。

        GMM 的 weight/scale 必须是 ``[g, N, K//2]`` 拼好后 ``transpose(1, 2)`` 的
        转置视图（保留 transpose stride，见 vllm-ascend ``w4a4_mxfp4`` 注释），
        经由 ``scale=`` 传入；用 ``antiquant_scale=`` 或对视图 ``contiguous()`` 均会
        报 161002。
        """
        Kd = hs.shape[-1]
        xq4, xs4 = torch_npu.npu_dynamic_mx_quant(hs, dst_type=_F4)
        x_sorted = xq4[row]
        xs_sorted = xs4[row].reshape(-1, Kd // 64, 2)

        packed = [self.engine.get_packed(self.layer_idx, int(e)) for e in active]

        def gmm(x4, xs3, wk, sk, k_dim, out_dtype=torch.bfloat16):
            n_dim = packed[0][wk].shape[0]
            wstack = torch.cat([p[wk].unsqueeze(0) for p in packed], 0).transpose(1, 2)
            sstack = torch.cat(
                [p[sk].reshape(n_dim, k_dim // 64, 2).unsqueeze(0) for p in packed], 0
            ).transpose(1, 2)
            return torch_npu.npu_grouped_matmul(
                x=[x4], weight=[wstack], bias=None, scale=[sstack],
                per_token_scale=[xs3], split_item=2, group_list_type=1,
                group_type=0, group_list=group_list, output_dtype=out_dtype,
                x_dtype=_F4, weight_dtype=_F4, scale_dtype=_E8M0,
                per_token_scale_dtype=_E8M0)[0]

        gate = gmm(x_sorted, xs_sorted, "w1p", "w1s", Kd).float()
        up = gmm(x_sorted, xs_sorted, "w3p", "w3s", Kd).float()

        if self.swiglu_limit and self.swiglu_limit > 0:
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = gate.clamp(max=self.swiglu_limit)
        h = F.silu(gate) * up * s_w[:, None]

        hid = h.shape[-1]
        hq4, hs4 = torch_npu.npu_dynamic_mx_quant(h.to(hs.dtype), dst_type=_F4)
        out = gmm(hq4, hs4.reshape(-1, hid // 64, 2), "w2p", "w2s", hid)

        y = torch.zeros(T, hs.shape[-1], dtype=torch.float32, device=hs.device)
        y.index_add_(0, row, out.float())
        return y.type_as(hs)

    def forward(self, hidden_states: torch.Tensor, input_ids=None) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hs = hidden_states.view(-1, hidden_dim)

        # DecoderLayer 调用 ``self.mlp(hidden_states)`` 时不传 input_ids；
        # hash 层需要从 forward_context 取（与 vllm-ascend 一致）。
        if self.hash and input_ids is None:
            from vllm.forward_context import get_forward_context

            input_ids = getattr(get_forward_context(), "input_ids", None)

        if self.is_sequence_parallel:
            hs = v4.sequence_parallel_chunk(hs)
            if input_ids is not None:
                input_ids = v4.sequence_parallel_chunk(input_ids)

        out = self._routed(hs, input_ids)
        if self.shared_experts is not None:
            out = out + self.shared_experts(hs)

        if self.is_sequence_parallel:
            out = v4.tensor_model_parallel_all_gather(out, 0)[:num_tokens]
        return out.view(num_tokens, hidden_dim)


def _offload_load_weights(self, weights):
    """tee 权重流：主干专家进 host 池，其余交回 vllm-ascend 原实现。

    ``is_expert_weight`` 只认主干 ``layers.*.ffn.experts.*``；``mtp.*``（spec
    decode draft）不在本插件接管范围，原样透传后由上游 ``load_weights`` 跳过。
    """
    engine = get_engine()
    it = iter(weights)

    def filtered():
        for name, w in it:
            if is_expert_weight(name):
                engine.capture(name, w)
            else:
                yield name, w

    return _ORIG_LOAD_WEIGHTS(self, filtered())


_ORIG_LOAD_WEIGHTS = v4.AscendDeepseekV4ForCausalLM.load_weights


def install_patches():
    global _PATCHED
    if _PATCHED:
        return
    if not getattr(v4, "_DSV4_OFFLOAD_PATCHED", False):
        v4.DeepseekV4MoE = OffloadDeepseekV4MoE
        v4.AscendDeepseekV4ForCausalLM.load_weights = _offload_load_weights
        v4._DSV4_OFFLOAD_PATCHED = True
    _PATCHED = True
