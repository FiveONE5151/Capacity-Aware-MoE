#!/usr/bin/env bash
# ============================================================
# Capacity-Aware MoE — Dataset Download Script
# Downloads all 9 evaluation datasets for DeepSeek-V2-Lite-Chat
# ============================================================
set -euo pipefail

DATASET_DIR="${1:-}"
if [[ -z "${DATASET_DIR}" ]]; then
    DATASET_DIR="${HOME}/data/workSpace/dataset"
    echo "Target directory not specified, using default:"
fi

echo "All datasets will be downloaded to: ${DATASET_DIR}"
echo ""

# ── Helper ────────────────────────────────────────────────────
download() {
    local repo_id="$1"
    shift
    echo "[DOWNLOAD] ${repo_id} ..."
    huggingface-cli download "${repo_id}" --repo-type dataset "$@" --local-dir "${DATASET_DIR}/${repo_id}"
    echo "[OK] ${repo_id}"
    echo ""
}

# ── Create root directory ─────────────────────────────────────
mkdir -p "${DATASET_DIR}"

# ── 1. openbookqa ─────────────────────────────────────────────
download openbookqa

# ── 2. piqa ───────────────────────────────────────────────────
download piqa

# ── 3 & 5. super_glue (rte + boolq) ───────────────────────────
echo "[DOWNLOAD] super_glue (rte + boolq) ..."
huggingface-cli download super_glue --repo-type dataset \
    --include "rte/*" "boolq/*" \
    --local-dir "${DATASET_DIR}/super_glue"
echo "[OK] super_glue"
echo ""

# ── 4. winogrande ─────────────────────────────────────────────
download winogrande

# ── 6. arc_challenge ───────────────────────────────────────────
echo "[DOWNLOAD] allenai/ai2_arc (ARC-Challenge) ..."
huggingface-cli download allenai/ai2_arc --repo-type dataset \
    --include "ARC-Challenge/*" \
    --local-dir "${DATASET_DIR}/allenai/ai2_arc"
echo "[OK] allenai/ai2_arc"
echo ""

# ── 7. hellaswag ──────────────────────────────────────────────
download hellaswag

# ── 8. mmlu (57 subjects) ──────────────────────────────────────
download hails/mmlu_no_train

# ── 9. gsm8k ──────────────────────────────────────────────────
echo "[DOWNLOAD] gsm8k (main) ..."
huggingface-cli download gsm8k --repo-type dataset \
    --include "main/*" \
    --local-dir "${DATASET_DIR}/gsm8k"
echo "[OK] gsm8k"
echo ""

# ── Summary ───────────────────────────────────────────────────
echo "========================================"
echo "All datasets downloaded successfully!"
echo "Target: ${DATASET_DIR}"
echo ""
echo "To use as local HF cache, run:"
echo "  export HF_DATASETS_CACHE=${DATASET_DIR}"
echo "========================================"
