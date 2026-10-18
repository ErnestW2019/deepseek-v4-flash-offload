# 方案设计：基于 vLLM / vllm-ascend 的单卡专家卸载推理框架

> 目标：硬件不变（单卡 Ascend 950PR，128 GB HBM），在 **vLLM + vllm-ascend** 之上
> 做增强，形成可服务化、可维护的自研专家卸载推理框架。
> 状态：**方案已落地**（实现与实测见 §9 / §10）；本文档保留最初的方案设计脉络。

---

## 0. 结论先行

采用 **「out-of-tree 插件接管 MoE 组件」为主干，必要时对 vllm-ascend 做最小 patch** 的混合形态。

- **不 fork 为主干**：vllm-ascend 0.23 是 950/A5 定制 wheel，散点改动会导致与上游脱钩、升级失控。
- **不另起独立引擎**：会丢掉 vLLM 的 PagedAttention / 连续批处理 / 调度 / OpenAI 服务化
  —— 这些正是「生产级」的主体。
- **本仓库现有 `offload.py` / `npu_ops.py` / `loader.py`** 作为**卸载引擎与算子的复用源**，
  不重写；`model_npu.py` 保留为**逐 token 对齐基线**。

一句话：**「基于 vllm-ascend 新开发的框架」= 挂在它上面的一个卸载插件包，而不是它的一个 fork。**

---

## 1. 现状盘点（本仓库）

| 文件 | 性质 | 在新方案中的角色 |
|---|---|---|
| `offload.py` | `ExpertBank`：host 打包 FP4 专家 + NPU bf16 槽位 LRU | **复用**，演进为 `ExpertOffloadEngine` |
| `npu_ops.py` | tilelang kernel 的 torch/NPU 重实现（FP4/FP8 反量化、rope、sparse attn…） | **复用**，补 vLLM 布局适配 |
| `loader.py` | 流式加载：专家留 host，FP8 非专家解量化上卡 | **参考**，改为 vLLM `load_weights` 内实现 |
| `model_npu.py` | 独立参考实现（逐专家 `F.linear` 循环） | **保留为对齐基线**，不进主干 |
| `test_*.py` ×4 | 验证脚本 | 保留，作为引擎单测 |

接口现状（决定改造工作量）：

- `ExpertBank.register(layer, eid, w1p, w1s, w2p, w2s, w3p, w3s)` —— host 侧按打包张量存。
- `ExpertBank.get(layer, eid) -> {"w1": bf16, "w2": bf16, "w3": bf16}` —— NPU 侧取解量化专家。
- `MoE.forward`（`model_npu.py:518`）：`bincount` + 逐专家 `bank.get` + `index_add_`，**逐专家串行 GEMM**。

---

## 2. 技术依据（已核实）

1. **vLLM 有 4 个插件入口组**：`vllm.general_plugins` / `vllm.platform_plugins` /
   `vllm.io_processor_plugins` / `vllm.stat_logger_plugins`。
   `general_plugins` 在 process0 / engine core / worker 全部进程加载一次。
2. **vllm-ascend 本身就是插件**：其 `entry_points.txt` 声明
   `[vllm.platform_plugins] ascend = vllm_ascend:register` 与
   `[vllm.general_plugins] ascend_model = vllm_ascend:register_model`。
3. **模型类可覆盖**：`ModelRegistry.register_model(arch, "module:attr")` 支持惰性字符串；
   同 arch 重复注册仅打 debug 日志、后者生效。
   vllm-ascend 当前把 `DeepseekV4ForCausalLM -> vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM`。
4. **接管点明确**：`vllm_ascend/models/deepseek_v4.py` 的 `DeepseekV4MoE.__init__`
   第 433 行 `self.experts = FusedMoE(...)` 是「整块专家上卡」的唯一入口，
   用子类整体替换即可，**无需进 `FusedMoE` 内部**。
5. **现有 NPU 卸载设施不匹配**：`vllm_ascend/model_executor/offloader/prefetch.py` 的
   `NPUPrefetchOffloader` 是**层粒度静态预取**（按 group 预取整层），
   本方案需要的是**专家粒度、动态 LRU**（命中集由 router 输出决定），二者不同层。
6. **容量约束**：权重 159.6 GB > 128 GB HBM；其中 routed experts 约 150.6 GB（94.4%），
   dense 约 9 GB。**必须把 experts 移出 NPU**。

---

## 3. 总体架构

```
vLLM 调度 / PagedAttention / OpenAI 服务化
        │
        ▼
[vllm.general_plugins]  dsv4_plugin:register_model
        │  覆盖注册 arch
        ▼
OffloadDeepseekV4ForCausalLM        (子类，仅覆写 MoE 与 load_weights)
        │
        ▼
OffloadDeepseekV4MoE                (子类，替换 FusedMoE)
        │
        ▼
ExpertOffloadEngine  ← 由 offload.ExpertBank 演进
   ├─ host bank : 打包 FP4 专家 (host pinned RAM，约 150 GB)
   ├─ slot cache: NPU bf16 专家槽位 (LRU，容量可配)
   └─ prefetch  : 依据上一层 gate 预测下一层 top-k 预热
```

### 三级存储

| 层 | 内容 | 预算 |
|---|---|---|
| HBM（128 GB） | dense ~9 GB + KV/激活/图 10~20 GB + 热专家槽位 32~80 GB（含双缓冲预取） | ≤128 GB |
| host pinned RAM（628 GB） | 打包 FP4 专家池 ~150 GB | 充裕 |
| L3 NVMe | 冷专家兜底（可选） | — |

### 搬运预算（设计估值）

43 层 × 每层 top-6、槽位稳态命中后，B≥43 时约 **3.2 GB/步**，按实测 H2D ~57.6 GB/s ≈ **56 ms/步**，
计算约 107 ms/步 → **搬运可被计算掩盖**。该项需在 M3 用真实 profiler 复核。

---

## 4. 目录结构（拟）

```
dsv4_offload/                       # 本仓库根
  DESIGN.md                         # 本文档
  offload.py  npu_ops.py  loader.py  model_npu.py   # 既有，保留
  dsv4_plugin/                      # 【新增】out-of-tree 插件包
    __init__.py                     #   register() / register_model()
    core/
      engine.py                     #   ExpertOffloadEngine（host bank + 槽位 LRU + 预取）
      layout.py                     #   vLLM 张量布局适配（打包 FP4 <-> FusedMoE 期望）
    models/
      deepseek_v4.py                #   OffloadDeepseekV4ForCausalLM / OffloadDeepseekV4MoE
    kernels/
      grouped_gemm.py               #   昇腾 grouped-GEMM 路径（M4）
  pyproject.toml                    # 【新增】打包 + entry_points
```

---

## 5. 关键设计

### 5.1 插件注册（`dsv4_plugin/__init__.py`）

```python
def register_model():
    from vllm import ModelRegistry
    ModelRegistry.register_model(
        "DeepseekV4ForCausalLM",
        "dsv4_plugin.models.deepseek_v4:OffloadDeepseekV4ForCausalLM",
    )
```

打包后用 `[project.entry-points."vllm.general_plugins"]` 暴露 `register_model`，
即可在**不改 site-packages** 的前提下让 vLLM 用我们的模型类覆盖 vllm-ascend 的注册。

### 5.2 MoE 接管（`OffloadDeepseekV4MoE`）

- `__init__`：复刻父类字段（`layer_id` / `dim` / `gate` / hash 相关 `tid2eid`），
  但 **不构造 `FusedMoE`**，改为构造 `ExpertOffloadEngine`。
- `forward`：走 `self.gate(x)` 得到 `weights, indices`，交引擎按专家取权重计算。
  与现有 `model_npu.MoE.forward` 语义一致，便于对齐。
- **决策点（待确认）**：dense 的 `shared_experts` 保持原 `FusedMoE(shared_experts=...)` 路径常驻 NPU。

### 5.3 权重加载拦截

在 `OffloadDeepseekV4ForCausalLM.load_weights` 中分流：

- 命中 `layers.N.ffn.experts.M.w{1,2,3}.{weight,scale}` → **不解压、不上卡**，
  直接登记进 host bank（复用 `loader.py` 的 `_EXPERT_RE` 与分组逻辑）。
- FP8 非专家权重 → 解量化后上卡（现有 `dequant_fp8_weight` 路径）。
- 其余 → 按目标 dtype 搬迁。

> 需确认：vLLM 的权重迭代接口（`weights_iterator` 产出的 `(name, tensor)`）与
> 本仓库 `safe_open` 直读的差异，二者需在 M2 对齐。

### 5.4 计算内核

- **M2（先跑通）**：沿用逐专家 `F.linear`，保证数值正确、可服务。
- **M4（性能）**：替换为昇腾 grouped-GEMM（`npu_ops` + 可能的 aclnn/自定义算子），
  消除 Python 层逐专家循环与 kernel 启动开销。

---

## 6. 里程碑

| 阶段 | 目标 | 验收 |
|---|---|---|
| M0 | 环境与可行性：定位 vllm-ascend 源码、跑通插件加载 | `inspect.getfile(vllm_ascend)` 命中预期；插件被 `load_general_plugins` 调用 |
| M1 | 台阶验证：用一个中小 MoE 模型验证接管链路 | 插件接管后输出与原始一致 |
| M2 | 单机可跑：DeepSeek-V4-Flash 起服务 | 与 `model_npu.py` / `inference/generate.py` 逐 token 对齐 |
| M3 | 性能：预取 + 槽位调参 | 吞吐/首 token 延迟达标，搬运被掩盖 |
| M4 | 内核：grouped-GEMM 替换逐专家 | 数值一致，延迟下降 |
| M5 | 工程化：打包、监控、KV/Engram（后续） | 可重复部署 |

---

## 7. 风险与未决项

1. **vllm-ascend 源码未落盘**：`dsv4-flash-npu/third_party/README.md` 声称已定位，
   但磁盘上只有 README。需决定是否 clone gitee 镜像 tag v0.23.0 并校验
   （纯插件路线不强制需要源码，但对齐/调试时需要）。
2. ~~**原生 MXFP4 不支持**（aclnn 161002）~~ **已解决（p4-d）**：`npu_grouped_matmul`
   a4w4（`scale=` + transpose-stride 形式）可用，专家常驻 FP4；数值口径见 §9（非逐位对齐，
   激活量化固有误差 ≈5× 于 fp8）。
3. ~~**grouped-GEMM 在昇腾的可用性**未验证~~ **已验证（p4-a/p4-d）**：bf16 grouped-GEMM
   逐位对齐且 4.8–5.7×；MXFP4 a4w4 可用，e2e 等显存 1.68×。
4. **权重迭代接口差异**：vLLM `load_weights` 与本仓库直读 shard 的差异（见 §5.3）。
5. **与平行项目（SGLang 路线）的分工**未对齐，
   避免两边各写一套专家卸载。
6. **MTP（投机解码 draft）不在接管范围**：`mtp.*.ffn.experts.*`（本 checkpoint 有
   1 个 draft 层、1536 个专家键）由 `DeepSeekV4MTP` 单独加载，未挂本插件的 tee；
   插件 tee 默认只收主干 `layers.*` 专家，MTP 键原样透传后由上游 `load_weights`
   跳过（`get_spec_layer_idx_from_weight_name`）。M2 阶段不启用投机解码。

---

## 8. 需用户确认

1. 主干形态：**插件优先 + 受控派生兜底**（本稿）是否采纳？
2. 是否需要现在就 clone 并校验 vllm-ascend v0.23.0 源码到 `third_party/`？
3. M1 用哪个模型做台阶验证（旧工程建议 Qwen 系）。

---

## 9. M4 / p4 性能内核实现记录（实测）

> 落点 `dsv4_plugin/models/deepseek_v4.py`（`OffloadDeepseekV4MoE`）+ `offload.py`。
> 全部经 `test_moe_align.py` / `test_e2e_vllm.py` 验证。

### p4-a 设备侧 grouped-GEMM（bf16 路径）

- 由逐专家 `F.linear` 循环改为单次 **`torch_npu.npu_grouped_matmul`**（`split_item=2,
  group_list_type=1, group_type=0`, `output_dtype=bfloat16`）。
- 对齐：`max_abs_diff = 0.0`（逐位一致）；内核微基准 **4.8–5.7×**（对比逐专家循环）。

### p4-c 存储侧

- **c1**：NPU bf16 槽位 64 → 512（`DSV4_EXPERT_SLOTS`）。
- **c2**：host 打包专家 **pinned**（`DSV4_PIN_HOST=1`，默认开）——H2D 19.8→51.3 GB/s，
  miss 帧 43→23 ms；e2e 2.67→3.38 tok/s（8-token，1.27×）。

### p4-d 设备侧 MXFP4 GMM（a4w4，激活+权重均 MXFP4）

开关 `DSV4_MXFP4_GMM=1`（默认 `0`，关闭）。专家**常驻 FP4**，`npu_grouped_matmul`
直接吃打包 FP4（省去反量化 + bf16 写入/回读）。

**唯一可用的算子调用形式**（其余形式均报 `161002` / aicpu 异常）：

- 权重/scale 必须先拼**原始** `[g, N, K/2]`（scale `[g, N, K/64, 2]`）再 `.transpose(1, 2)`，
  **保留 transpose stride**（逻辑 `[g, K/2, N]`）；不可先转视图再 transpose、不可 `.contiguous()`。
- 经 **`scale=`**（非 `antiquant_scale=`）传入；`x_dtype=weight_dtype=float4_e2m1fn_x2(296)`、
  `scale_dtype=per_token_scale_dtype=float8_e8m0fnu(293)`、`output_dtype=bfloat16`；
  `per_token_scale` 须 3D `[M, K//64, 2]`；`split_item=2, group_list_type=1, group_type=0`。
- **multi-expert 必须堆叠为单张量**（列表形式失败）；n/k 须 64 整倍数。
- 激活量化用 `torch_npu.npu_dynamic_mx_quant(x, dst_type=F4)` → `(packed_u8, mxscale_u8)`。

**数值口径（重建基线）**：a4w4 与参考 fp8 路径共用**同一批 FP4 专家权重**（E2M1 值在 bf16
可精确表示），差异**只来自激活量化**：fp8-per128（E4M3）逐元素 rel≈4% vs MXFP4-per32
（E2M1，1 位尾数）rel≈22% → 比值≈5×，与位宽一致。GMM 本身近似精确（以同 FP4 激活的
float 参考比对，残差仅 mean=6.9e-3，来自二次量化 nibble 边界翻转）。
回归基线改用**双阈值**（`MXFP4_MAX_ABS=0.35` 且 `MXFP4_MEAN_ABS=0.08`），非逐元素 allclose。
`test_moe_align.py`：bf16 路径仍 `max_abs_diff=0`；a4w4 `max_abs≈0.28 mean≈0.05`，**ALL PASS**。

### e2e 实测（64-token 贪心，摊薄 prefill；本机单卡 950PR）

| 配置 | tok/s | 专家命中率 | 专家显存≈ |
|---|---|---|---|
| a4w4 @512 打包槽 | 6.55 | 0.42 | 6.9 GB |
| **a4w4 @2048 打包槽** | **7.10** | **0.81** | 27.4 GB |
| bf16 @512 槽（基线） | 4.23 | 0.79 | 25.8 GB |

按**等显存**公平对比：**a4w4@2048 vs bf16@512 = 1.68×**。
（此前 8-token 口径为 3.62/2.47/3.06 tok/s，prefill 未摊薄，已弃用。）

**默认槽位**：`DSV4_PACKED_SLOTS` 默认改为 **`4 × DSV4_EXPERT_SLOTS`**（FP4 打包约
bf16 的 1/4，等显存即最佳命中率），故 a4w4 默认走 @2048。

---

## 10. M5 工程化/服务化记录（实测）

### m5-0 服务模式跑通

- 多进程（`VLLM_ENABLE_V1_MULTIPROCESSING=1`）+ `max_num_seqs=8`、8 并发请求跑通，
  聚合 **19.56 tok/s**（256 tokens / 8 prompts）。
- 真实 `vllm serve` + OpenAI `/v1/chat/completions` 返回正常（`"The capital of France is Paris."`）。
- 注意：多进程下权重在 **EngineCore 进程**加载，driver 进程 `get_engine()` 单例为空
  （`captured=0` 属正常，非 bug）；`test_e2e_vllm.py` 对该情形已打 `[SKIP]`。

### m5-1 瓶颈剖析（关键结论）

单层 MoE（layer3，256 专家，T=8）前向分解：

| 项目 | ms | 说明 |
|---|---|---|
| route only | 0.07 | 路由可忽略 |
| baseline（逐专家循环） | 16.47 | 对比基线 |
| **bf16 hot（cap=256 全命中）** | **3.69** | 纯 grouped-GEMM + Python（≈4.5× 于基线） |
| **bf16 cold（cap=0 全 miss）** | **84.45** | 每次全量 H2D + **反量化** |
| a4w4 hot（pack=256） | 2.05 | 常驻 FP4 |
| a4w4 cold（pack=0） | 13.63 | 每次全量 H2D（无解压） |

**结论**：
1. **bf16 的 miss 代价主导于「反量化」**（cold−hot≈80.8 ms；其中反量化 fp4→bf16 写出
   体积远大于 H2D 本身），a4w4 消除反量化后 cold 降 **6.2×**（84.45→13.63 ms）。
2. **H2D 可与 NPU 计算重叠（side-stream 探针，overlap ratio 92%）** → 预取在原理上可行。
   但单层 compute（hot ≈3.7 ms）远小于「全量 miss 搬运」（≈9 ms+），故**预取只能部分掩盖
   miss**，收益上界有限；真正的大杠杆是 **降低 miss 率（提高槽位/命中率）** 与
   **a4w4 降低搬运体积**。
3. steady-state（命中率≈0.8）下，e2e 成本由 **hot grouped-GEMM（每层≈2.1–3.7 ms）** 与
   少量 miss 共同构成；e2e ≈ 43 层 × 每层成本。

### m5-4 服务化封装 + 验收（实测）

交付物：
* `serve.sh` —— 启动脚本；`PROFILE=bf16|a4w4` 切换专家精度，两套均封装。
* `accept.py` —— HTTP 验收（等 `/health` → 并发 `/v1/completions` → 聚合 tok/s），仅用标准库。

验收（8 并发 × 32 token = 256 tokens，`python accept.py`）：

| profile | 启动 | 聚合 tok/s | 结果 |
|---|---|---|---|
| bf16（默认，精度无损） | `./serve.sh` | 6.64 | PASS |
| a4w4（MXFP4 常驻） | `PROFILE=a4w4 ./serve.sh` | **19.47** | PASS |

a4w4 相对 bf16 在 8 并发下约 **2.9×**（单流口径为 1.68×，见 §9）。

用法与环境变量：

```bash
./serve.sh                  # bf16（精度无损）
PROFILE=a4w4 ./serve.sh     # a4w4（提速）
python accept.py            # 验收；BASE_URL / MODEL / N / MAX_TOKENS 可覆盖
```

`serve.sh` 可覆盖：`DSV4_MODEL PORT HOST SERVED_NAME MAX_MODEL_LEN MEM_UTIL
MAX_NUM_SEQS DSV4_EXPERT_SLOTS DSV4_PACKED_SLOTS DSV4_PIN_HOST`；脚本自带 CANN/ATB/
glibcxx 环境，使用 `vllm serve` + 已 `pip install -e .` 的 `dsv4_offload` 插件。
