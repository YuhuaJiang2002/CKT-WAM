#!/bin/bash
# =============================================================================
# CKT-WAM LIBERO Evaluation Script
#
# Runs zero-shot evaluation of a trained CKT-WAM checkpoint on a LIBERO
# task suite, using the teacher WAM as a single-pass observation encoder
# and the student WAM as the denoising backbone.  The transferred context
# C_A is injected into the student's conditioning stream automatically
# by the CKT pipeline.
#
# Usage:
#   bash run_libero_ckt_eval.sh libero_10
#   bash run_libero_ckt_eval.sh libero_spatial
#
# Environment variables (no absolute paths are hard-coded):
#   CKT_WAM_ROOT         -- repository root (default: current directory)
#   COSMOS_POLICY_ROOT   -- cosmos-policy install directory
#   DREAMZERO_ROOT       -- dreamzero install directory
#   STUDENT_CONFIG_NAME  -- cosmos-policy experiment config name
#                           (e.g. cosmos_predict2_2b_480p_libero__inference_only)
#   STUDENT_CKPT         -- path/HF slug of the student WAM checkpoint
#   TEACHER_CKPT_PATH    -- teacher WAM checkpoint directory
#   CKT_CKPT             -- trained CKT adapter bank checkpoint
#   LOCAL_LOG_DIR        -- eval log output directory
# =============================================================================

set -euo pipefail

TASK_SUITE="${1:-libero_10}"

CKT_WAM_ROOT="${CKT_WAM_ROOT:-$(pwd)}"
COSMOS_POLICY_ROOT="${COSMOS_POLICY_ROOT:-${CKT_WAM_ROOT}/third_party/cosmos-policy}"
DREAMZERO_ROOT="${DREAMZERO_ROOT:-${CKT_WAM_ROOT}/third_party/dreamzero}"

STUDENT_CONFIG_NAME="${STUDENT_CONFIG_NAME:-cosmos_predict2_2b_480p_libero__inference_only}"
STUDENT_CKPT="${STUDENT_CKPT:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
TEACHER_CKPT_PATH="${TEACHER_CKPT_PATH:-${DREAMZERO_ROOT}/checkpoints}"
CKT_CKPT="${CKT_CKPT:-${CKT_WAM_ROOT}/outputs/libero_ckt_8gpu/checkpoint_00100000.pt}"

LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${CKT_WAM_ROOT}/outputs/eval/libero_ckt_${TASK_SUITE}}"

TEACHER_BLOCK_INDEX="${TEACHER_BLOCK_INDEX:-20}"
SEED="${SEED:-195}"
NUM_TRIALS="${NUM_TRIALS:-50}"
RUN_NOTE="${RUN_NOTE:-ckt_wam_middle_l20}"

export PYTHONPATH="${CKT_WAM_ROOT}:${COSMOS_POLICY_ROOT}:${DREAMZERO_ROOT}:${PYTHONPATH:-}"
export CKT_COSMOS_CONFIG_FILE="${CKT_COSMOS_CONFIG_FILE:-${COSMOS_POLICY_ROOT}/cosmos_policy/config/config.py}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "============================================================"
echo "  CKT-WAM LIBERO Evaluation"
echo "============================================================"
echo "  Task suite:      ${TASK_SUITE}"
echo "  Student config:  ${STUDENT_CONFIG_NAME}"
echo "  Student ckpt:    ${STUDENT_CKPT}"
echo "  Teacher ckpt:    ${TEACHER_CKPT_PATH}"
echo "  CKT ckpt:        ${CKT_CKPT}"
echo "  Teacher layer:   l^* = ${TEACHER_BLOCK_INDEX}"
echo "  Log dir:         ${LOCAL_LOG_DIR}"
echo "  Seed:            ${SEED}"
echo "============================================================"

mkdir -p "${LOCAL_LOG_DIR}"

python -m ckt_wam.experiments.libero.run_libero_ckt_eval \
    --student_config      "${STUDENT_CONFIG_NAME}" \
    --student_ckpt        "${STUDENT_CKPT}" \
    --teacher_path        "${TEACHER_CKPT_PATH}" \
    --ckt_ckpt            "${CKT_CKPT}" \
    --teacher_block_index "${TEACHER_BLOCK_INDEX}" \
    --task_suite_name     "${TASK_SUITE}" \
    --local_log_dir       "${LOCAL_LOG_DIR}" \
    --seed                "${SEED}" \
    --num_trials_per_task "${NUM_TRIALS}" \
    --run_id_note         "${RUN_NOTE}" \
    2>&1 | tee "${LOCAL_LOG_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"
