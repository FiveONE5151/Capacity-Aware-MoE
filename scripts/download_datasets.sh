#!/usr/bin/env bash
# ============================================================
# Capacity-Aware MoE — Dataset Download Script
# Downloads evaluation datasets for DeepSeek-V2-Lite-Chat
# Usage: ./download_datasets.sh [target_directory]
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

# ── 3. super_glue ─────────────────────────────────────────────
download super_glue

# ── 4. winogrande ─────────────────────────────────────────────
download winogrande

# ── 5. allenai/ai2_arc ────────────────────────────────────────
download allenai/ai2_arc

# ── 6. hellaswag ──────────────────────────────────────────────
download hellaswag

# ── 7. mmlu (57 subjects) ──────────────────────────────────────
download hails/mmlu_no_train

# ── 8. gsm8k ──────────────────────────────────────────────────
download gsm8k

# ── Summary ───────────────────────────────────────────────────
echo "========================================"
echo "All datasets downloaded successfully!"
echo "Target: ${DATASET_DIR}"
echo ""
echo "To use as local HF cache, run:"
echo "  export HF_DATASETS_CACHE=${DATASET_DIR}"
echo "========================================"
