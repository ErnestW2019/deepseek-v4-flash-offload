#!/usr/bin/env bash
# DeepSeek-V4-Flash 单卡专家卸载推理服务（vllm serve + dsv4_plugin 插件）。
#
# 用法:
#   ./serve.sh                 # 默认 bf16 精度无损
#   PROFILE=a4w4 ./serve.sh    # MXFP4 常驻提速（激活量化有误差，见 DESIGN.md §9）
#
# 可覆盖的环境变量:
#   PYTHON  VLLM_BIN  DSV4_MODEL  PORT  HOST  SERVED_NAME  MAX_MODEL_LEN  MEM_UTIL
#   MAX_NUM_SEQS  DSV4_EXPERT_SLOTS  DSV4_PACKED_SLOTS  DSV4_PIN_HOST
#   ASCEND_TOOLKIT_ROOT  ASCEND_ATB_ROOT  DSV4_EXTRA_LIBDIR
set -eo pipefail

# 运行用 python（须指向含 vllm 可执行文件的解释器，通常是你的 venv）：
#   export PYTHON=/path/to/venv/bin/python
PY="${PYTHON:-python3}"
VLLM_BIN="${VLLM_BIN:-$(dirname "$PY")/vllm}"

# ---- CANN / ATB 运行环境（须在 set -u 之前 source，厂商脚本引用未定义变量）----
ASCEND_TOOLKIT_ROOT="${ASCEND_TOOLKIT_ROOT:-/usr/local/Ascend/ascend-toolkit}"
ASCEND_ATB_ROOT="${ASCEND_ATB_ROOT:-/usr/local/Ascend/nnal/atb}"
if [ -f "${ASCEND_TOOLKIT_ROOT}/set_env.sh" ]; then
  source "${ASCEND_TOOLKIT_ROOT}/set_env.sh"
fi
if [ -f "${ASCEND_ATB_ROOT}/set_env.sh" ]; then
  source "${ASCEND_ATB_ROOT}/set_env.sh"
fi
# 可选：额外的动态库搜索目录（如自备 glibcxx），用 DSV4_EXTRA_LIBDIR 指定。
if [ -n "${DSV4_EXTRA_LIBDIR:-}" ]; then
  export LD_LIBRARY_PATH="${DSV4_EXTRA_LIBDIR}:${LD_LIBRARY_PATH:-}"
fi

set -u

export DSV4_MODEL="${DSV4_MODEL:-/home/models/DeepSeek-V4-Flash}"
export DSV4_PIN_HOST="${DSV4_PIN_HOST:-1}"

# ---- 专家精度 profile（两套均封装，环境变量切换）----
PROFILE="${PROFILE:-bf16}"
case "$PROFILE" in
  bf16)
    export DSV4_MXFP4_GMM=0
    export DSV4_EXPERT_SLOTS="${DSV4_EXPERT_SLOTS:-512}"
    ;;
  a4w4)
    export DSV4_MXFP4_GMM=1
    export DSV4_EXPERT_SLOTS="${DSV4_EXPERT_SLOTS:-512}"
    export DSV4_PACKED_SLOTS="${DSV4_PACKED_SLOTS:-2048}"
    ;;
  *)
    echo "[serve] 未知 PROFILE=$PROFILE（可选 bf16 | a4w4）" >&2
    exit 2
    ;;
esac

: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${SERVED_NAME:=DeepSeek-V4-Flash}"
: "${MAX_MODEL_LEN:=2048}"
: "${MEM_UTIL:=0.75}"
: "${MAX_NUM_SEQS:=8}"

echo "[serve] profile=$PROFILE model=$DSV4_MODEL slots=$DSV4_EXPERT_SLOTS" \
     "mxfp4_gmm=$DSV4_MXFP4_GMM packed_slots=${DSV4_PACKED_SLOTS:-auto}" \
     "host=$HOST:$PORT" >&2

exec "$VLLM_BIN" serve "$DSV4_MODEL" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --enforce-eager \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$MEM_UTIL" \
  --host "$HOST" \
  --port "$PORT"
