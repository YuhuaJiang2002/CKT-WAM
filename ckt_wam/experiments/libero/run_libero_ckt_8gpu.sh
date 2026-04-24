#!/bin/bash
# =============================================================================
# 8-GPU Distributed CKT-WAM Training Script (LIBERO-plus)
#
# Parameter-Efficient Context Knowledge Transfer Between World Action Models
#   Teacher WAM (frozen):   DreamZero-14B    -- intermediate-layer features
#   Student WAM (frozen):   Cosmos-Policy-2B -- native diffusion backbone
#   CKT Adapter Bank (trainable): generalized + top-k specialized adapters
#
# Architecture:
#   Teacher -> H_T [B, N, 5120] (from transformer block l^* = 20 of 40)
#   AdapterBank: H_T -> C_A [B, 2K, 2048]
#   Student: conditioning tokens augmented with C_A, native action/video
#            denoising objective unchanged.
#
# Loss:
#   L = L_CKT + lambda_bal * L_bal
#
# Usage:
#   bash run_libero_ckt_8gpu.sh
#   bash run_libero_ckt_8gpu.sh --resume <checkpoint.pt>
#
# All paths below are driven by environment variables; no absolute paths
# are hard-coded.  Set the following before running (or export them in
# your shell profile):
#
#   CKT_WAM_ROOT        -- root of the CKT-WAM repository.
#   COSMOS_POLICY_ROOT  -- root of the installed cosmos-policy checkout.
#   DREAMZERO_ROOT      -- root of the installed dreamzero checkout.
#   BASE_DATASETS_DIR   -- parent directory of the LIBERO-Cosmos-Policy
#                          and related datasets.
#   STUDENT_CONFIG      -- path to the student experiment config file.
#   TEACHER_CKPT_PATH   -- path to the teacher WAM checkpoint directory.
# =============================================================================

set -euo pipefail

# ========================== User Configuration ===============================

CKT_WAM_ROOT="${CKT_WAM_ROOT:-$(pwd)}"
COSMOS_POLICY_ROOT="${COSMOS_POLICY_ROOT:-${CKT_WAM_ROOT}/third_party/cosmos-policy}"
DREAMZERO_ROOT="${DREAMZERO_ROOT:-${CKT_WAM_ROOT}/third_party/dreamzero}"

STUDENT_CONFIG="${STUDENT_CONFIG:-${COSMOS_POLICY_ROOT}/cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py}"
TEACHER_CKPT_PATH="${TEACHER_CKPT_PATH:-${DREAMZERO_ROOT}/checkpoints}"

OUTPUT_DIR="${OUTPUT_DIR:-${CKT_WAM_ROOT}/outputs/libero_ckt_8gpu}"
WANDB_PROJECT="${WANDB_PROJECT:-ckt-wam}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-libero_ckt_8gpu_$(date +%Y%m%d_%H%M%S)}"

# -- Hardware --
NUM_GPUS="${NUM_GPUS:-8}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29500}"

# -- Training Hyperparameters --
MAX_ITERATIONS="${MAX_ITERATIONS:-100000}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-4}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
ADAPTER_LR="${ADAPTER_LR:-3e-4}"
WARMUP_ITERATIONS="${WARMUP_ITERATIONS:-1000}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
SEED="${SEED:-42}"

# -- CKT / Adapter Configuration --
TEACHER_BLOCK_INDEX="${TEACHER_BLOCK_INDEX:-20}"   # Intermediate-layer (paper default)
NUM_SPECIALIZED_EXPERTS="${NUM_SPECIALIZED_EXPERTS:-8}"
TOP_K="${TOP_K:-2}"
NUM_ADAPTER_OUTPUT_TOKENS="${NUM_ADAPTER_OUTPUT_TOKENS:-32}"
ADAPTER_BOTTLENECK_DIM="${ADAPTER_BOTTLENECK_DIM:-512}"

# -- Loss Configuration --
LOAD_BALANCE_WEIGHT="${LOAD_BALANCE_WEIGHT:-0.01}"

# -- Logging / Checkpointing --
LOG_EVERY="${LOG_EVERY:-50}"
SAVE_EVERY="${SAVE_EVERY:-5000}"

# ========================== Environment Setup ================================

export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONPATH="${CKT_WAM_ROOT}:${COSMOS_POLICY_ROOT}:${DREAMZERO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ========================== Resume Handling ==================================

RESUME_ARG=""
if [[ "${1:-}" == "--resume" ]] && [[ -n "${2:-}" ]]; then
    RESUME_ARG="--resume_from ${2}"
    echo "[INFO] Resuming from checkpoint: ${2}"
elif [[ -d "${OUTPUT_DIR}" ]]; then
    LATEST_CKPT=$(ls -t "${OUTPUT_DIR}"/checkpoint_*.pt 2>/dev/null | head -1 || true)
    if [[ -n "${LATEST_CKPT}" ]]; then
        echo "[INFO] Found existing checkpoint: ${LATEST_CKPT}"
        echo "[INFO] Pass --resume ${LATEST_CKPT} to resume, or delete it to start fresh."
    fi
fi

# ========================== Pre-flight Checks ================================

echo "============================================================"
echo "  CKT-WAM LIBERO Training (8 GPU)"
echo "============================================================"
echo ""
echo "  Student config:   ${STUDENT_CONFIG}"
echo "  Teacher ckpt:     ${TEACHER_CKPT_PATH}"
echo "  Output dir:       ${OUTPUT_DIR}"
echo "  GPUs:             ${NUM_GPUS} x ${GPUS}"
echo "  Batch size:       ${BATCH_SIZE_PER_GPU}/gpu * ${NUM_GPUS} gpus * ${GRAD_ACCUMULATION_STEPS} accum"
echo "  Max iterations:   ${MAX_ITERATIONS}"
echo "  LR (student):     ${LEARNING_RATE}"
echo "  LR (adapter):     ${ADAPTER_LR}"
echo "  Experts:          ${NUM_SPECIALIZED_EXPERTS} specialized, Top-${TOP_K}"
echo "  Query tokens K:   ${NUM_ADAPTER_OUTPUT_TOKENS}"
echo "  Bottleneck d_b:   ${ADAPTER_BOTTLENECK_DIM}"
echo "  Teacher layer:    l^* = ${TEACHER_BLOCK_INDEX}"
echo "  LB loss weight:   ${LOAD_BALANCE_WEIGHT}"
echo "  ${RESUME_ARG:+Resuming from: ${RESUME_ARG}}"
echo ""
echo "============================================================"

if [[ ! -f "${STUDENT_CONFIG}" ]]; then
    echo "[ERROR] Student config not found: ${STUDENT_CONFIG}"
    exit 1
fi
if [[ ! -d "${TEACHER_CKPT_PATH}" ]]; then
    echo "[WARNING] Teacher checkpoint directory not found: ${TEACHER_CKPT_PATH}"
    echo "          Make sure the path is correct before training starts."
fi

mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/run_script_backup.sh" 2>/dev/null || true

# ========================== Launch Training ==================================

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting 8-GPU distributed CKT-WAM training..."
echo ""

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    --nnodes=1 \
    --node_rank=0 \
    -m ckt_wam.scripts.train_ckt \
    --student_config "${STUDENT_CONFIG}" \
    --teacher_path "${TEACHER_CKPT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_iterations ${MAX_ITERATIONS} \
    --batch_size ${BATCH_SIZE_PER_GPU} \
    --learning_rate ${LEARNING_RATE} \
    --adapter_lr ${ADAPTER_LR} \
    --load_balance_weight ${LOAD_BALANCE_WEIGHT} \
    --num_specialized_experts ${NUM_SPECIALIZED_EXPERTS} \
    --top_k ${TOP_K} \
    --num_adapter_output_tokens ${NUM_ADAPTER_OUTPUT_TOKENS} \
    --adapter_bottleneck_dim ${ADAPTER_BOTTLENECK_DIM} \
    --teacher_block_index ${TEACHER_BLOCK_INDEX} \
    --grad_accumulation_steps ${GRAD_ACCUMULATION_STEPS} \
    --seed ${SEED} \
    --log_every ${LOG_EVERY} \
    --save_every ${SAVE_EVERY} \
    --use_amp \
    ${RESUME_ARG} \
    2>&1 | tee "${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed successfully."
    echo "  Checkpoints saved in: ${OUTPUT_DIR}"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training exited with code ${EXIT_CODE}."
    exit ${EXIT_CODE}
fi
