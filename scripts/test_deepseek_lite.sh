#!/usr/bin/env bash
# =============================================================================
# Capacity-Aware MoE — DeepSeek-V2-Lite-Chat 精度测试脚本
#
# 测试所有策略在不同 EXPERT_CAPACITY 下的 accuracy 表现
# =============================================================================
set -euo pipefail

# ── 默认参数 ──────────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-${HOME}/data/workSpace/models/DeepSeek-V2-Lite-Chat}"
OUTPUT_BASE="${OUTPUT_BASE:-${MODEL_PATH}}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
DEBUG="${DEBUG:-0}"
DEBUG_LIMIT="${DEBUG_LIMIT:-16}"

# ── 策略列表 ──────────────────────────────────────────────────────────────────
STRATEGIES=(score random first last overselect)

# ── 容量因子列表 ──────────────────────────────────────────────────────────────
CAPACITY_FACTORS=(1.0 0.8 0.6 0.5)

# ── 任务与 few-shot 的默认映射 ───────────────────────────────────────────────
declare -A TASK_FEWSHOT_MAP=(
    [openbookqa]=0
    [piqa]=0
    [rte]=0
    [winogrande]=5
    [boolq]=0
    [arc_challenge]=25
    [hellaswag]=10
    [mmlu]=5
    [gsm8k]=5
    [ifeval]=0
    [gpqa_diamond_cot_n_shot]=5
)
DEFAULT_FEWSHOT=0

# ── 9 个评测任务（与 eval_capacity.sh 一致） ─────────────────────────────────
TASKS=(openbookqa piqa rte winogrande boolq arc_challenge hellaswag mmlu gsm8k ifeval gpqa_diamond_cot_n_shot)
FEWSHOTS=()
for t in "${TASKS[@]}"; do
    FEWSHOTS+=("${TASK_FEWSHOT_MAP[$t]}")
done

# ── 环境变量导出 ─────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NLTK_DATA=~/data/workSpace/dataset/nltk_data/nltk_data
# ── 帮助 ─────────────────────────────────────────────────────────────────────
usage() {
    cat << EOF
Usage: bash $(basename "$0") [OPTIONS]

Options:
  -m, --model PATH      模型路径 (default: ${MODEL_PATH})
  -o, --output DIR      输出目录 (default: ${OUTPUT_BASE})
  -b, --batch SIZE      batch size (default: auto)
  -g, --gpus ID         CUDA_VISIBLE_DEVICES (default: 0)
  -s, --strategy STR   仅测试指定策略，逗号分隔 (default: 全部)
  -c, --capacity CAP   仅测试指定容量因子，逗号分隔 (default: 全部)
  -t, --task TASK      仅测试指定任务，逗号分隔 (default: 全部)
  -d, --debug [N]      启用调试模式: --log_samples + --limit N (default: 16)
                       不传 N 时默认 16 条; 传 0 则仅 --log_samples 不限条数
  -h, --help           显示帮助

Examples:
  # 运行全部策略和容量因子的完整评测
  bash $0

  # 只测试 score 策略 + single task
  bash $0 -s score -c 1.0 -t piqa

  # 指定模型和多 GPU
  bash $0 -m /path/to/model -g 0,1 -b 4
EOF
    exit 0
}

# ── 参数解析 ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)      MODEL_PATH="$2"; shift 2 ;;
        -o|--output)     OUTPUT_BASE="$2"; shift 2 ;;
        -b|--batch)      BATCH_SIZE="$2"; shift 2 ;;
        -g|--gpus)       CUDA_DEVICES="$2"; shift 2 ;;
        -s|--strategy)   IFS=',' read -ra STRATEGIES <<< "$2"; shift 2 ;;
        -c|--capacity)   IFS=',' read -ra CAPACITY_FACTORS <<< "$2"; shift 2 ;;
        -t|--task)
    TASKS=()
    FEWSHOTS=()
    IFS=',' read -ra INPUT_TASKS <<< "$2"
    for t in "${INPUT_TASKS[@]}"; do
        TASKS+=("$t")
        FEWSHOTS+=("${TASK_FEWSHOT_MAP[$t]:-$DEFAULT_FEWSHOT}")
    done
    shift 2
    ;;
        -d|--debug)
    DEBUG=1
    # 如果下一个参数是数字则作为 limit，否则使用默认值
    if [[ $# -gt 1 && "$2" =~ ^[0-9]+$ ]]; then
        DEBUG_LIMIT="$2"; shift
    fi
    shift
    ;;
        -h|--help)       usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── 路径检查 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="${SCRIPT_DIR}/lm-evaluation-harness"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[ERROR] Model path does not exist: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -d "${EVAL_DIR}" ]]; then
    echo "[ERROR] lm-evaluation-harness not found at: ${EVAL_DIR}" >&2
    exit 1
fi

cd "${EVAL_DIR}"

# ── 调试参数拼接 ────────────────────────────────────────────────────────────
DEBUG_ARGS=""
if [[ "${DEBUG}" -eq 1 ]]; then
    DEBUG_ARGS="--log_samples"
    if [[ "${DEBUG_LIMIT}" -gt 0 ]]; then
        DEBUG_ARGS="${DEBUG_ARGS} --limit ${DEBUG_LIMIT}"
    fi
fi

# ── 打印配置 ─────────────────────────────────────────────────────────────────
echo "========================================================="
echo " Capacity-Aware MoE — Accuracy Test"
echo " Model: ${MODEL_PATH}"
echo " Output: ${OUTPUT_BASE}"
echo " Batch: ${BATCH_SIZE}"
echo " GPUs: ${CUDA_DEVICES}"
echo " Strategies: ${STRATEGIES[*]}"
echo " Capacities: ${CAPACITY_FACTORS[*]}"
echo " Tasks: ${TASKS[*]}"
if [[ "${DEBUG}" -eq 1 ]]; then
    echo " Debug: ON (log_samples + limit=${DEBUG_LIMIT})"
fi
echo "========================================================="
echo ""

TOTAL=$(( (${#STRATEGIES[@]} + 1) * ${#CAPACITY_FACTORS[@]} * ${#TASKS[@]} ))
CURRENT=0

# =========================================================================
# Part 1: 基线测试（不使用容量感知）
# =========================================================================
if [[ " ${STRATEGIES[*]} " == *" baseline "* ]] || [[ " ${STRATEGIES[*]} " == *"all"* ]]; then
    echo ">>>> [BASELINE] Starting baseline evaluation ..."
    OUTPUT_PATH="${OUTPUT_BASE}/baseline"
    mkdir -p "${OUTPUT_PATH}"

    for i in "${!TASKS[@]}"; do
        task="${TASKS[$i]}"
        num_fewshot="${FEWSHOTS[$i]:-0}"
        CURRENT=$((CURRENT + 1))
        echo "  [${CURRENT}/${TOTAL}] Task: ${task} (fewshot=${num_fewshot})"

        lm_eval \
            --model hf \
            --model_args "pretrained=${MODEL_PATH},parallelize=True,trust_remote_code=True,dtype=bfloat16" \
            --tasks "${task}" \
            --num_fewshot "${num_fewshot}" \
            --batch_size "${BATCH_SIZE}" \
            --output_path "${OUTPUT_PATH}/${task}.json" \
            ${DEBUG_ARGS}
    done
    echo "[BASELINE] Done!"
    echo ""
fi

# =========================================================================
# Part 2: 容量感知测试
# =========================================================================
for STRATEGY in "${STRATEGIES[@]}"; do
    [[ "${STRATEGY}" == "baseline" ]] && continue

    for CAPACITY in "${CAPACITY_FACTORS[@]}"; do
        echo ">>>> [STRATEGY=${STRATEGY}, CAPACITY=${CAPACITY}] Starting ..."

        OUTPUT_PATH="${OUTPUT_BASE}/expert_capacity-${CAPACITY}/${STRATEGY}"
        mkdir -p "${OUTPUT_PATH}"

        for i in "${!TASKS[@]}"; do
            task="${TASKS[$i]}"
            num_fewshot="${FEWSHOTS[$i]:-0}"
            CURRENT=$((CURRENT + 1))
            echo "  [${CURRENT}/${TOTAL}] ${STRATEGY} γ=${CAPACITY} | Task: ${task} (fewshot=${num_fewshot})"

            lm_eval \
                --model hf \
                --model_args "pretrained=${MODEL_PATH},expert_capacity=${CAPACITY},strategy=${STRATEGY},parallelize=True,trust_remote_code=True,dtype=bfloat16" \
                --tasks "${task}" \
                --num_fewshot "${num_fewshot}" \
                --batch_size "${BATCH_SIZE}" \
                --output_path "${OUTPUT_PATH}/${task}.json" \
                ${DEBUG_ARGS}
        done
        echo "[OK] STRATEGY=${STRATEGY}, CAPACITY=${CAPACITY}"
        echo ""
    done
done

# ── 汇总 ──────────────────────────────────────────────────────────────────────
echo "========================================================="
echo " All tests completed!"
echo " Results saved to: ${OUTPUT_BASE}"
echo "========================================================="

for STRATEGY in "${STRATEGIES[@]}"; do
    [[ "${STRATEGY}" == "baseline" ]] && continue
    echo ""
    echo "── ${STRATEGY} ──────────────────────────────────"
    for CAPACITY in "${CAPACITY_FACTORS[@]}"; do
        RESULT_DIR="${OUTPUT_BASE}/expert_capacity-${CAPACITY}/${STRATEGY}"
        echo "  γ=${CAPACITY}:"
        for task in "${TASKS[@]}"; do
            json_file="${RESULT_DIR}/${task}.json"
            if [[ -f "${json_file}" ]]; then
                acc=$(grep -o '"acc","[^"]*"' "${json_file}" 2>/dev/null | head -1 || echo "N/A")
                echo "    ${task}: ${acc}"
            fi
        done
    done
done
