#!/bin/bash
# =============================================================================
# CKT-WAM Installation Script
#
# Sets up an editable CKT-WAM environment along with the two WAM backends
# it depends on:
#   - cosmos-policy (student WAM backend, ``cosmos-predict2``-based)
#   - dreamzero     (teacher WAM backend)
#
# Usage:
#   bash install.sh [cu128|cu130]
#
# Arguments:
#   cu128 - Install with CUDA 12.8 / PyTorch 2.7 support (default)
#   cu130 - Install with CUDA 13.0 / PyTorch 2.9 support
#
# All paths are resolved relative to CKT_WAM_ROOT (defaults to the
# directory containing this script); no absolute paths are hard-coded.
# =============================================================================

set -e

CUDA_VERSION="${1:-cu128}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CKT_WAM_ROOT="${CKT_WAM_ROOT:-$(cd -- "$(dirname -- "$0")" &>/dev/null && pwd)}"
THIRD_PARTY_DIR="${CKT_WAM_ROOT}/third_party"
COSMOS_POLICY_ROOT="${COSMOS_POLICY_ROOT:-${THIRD_PARTY_DIR}/cosmos-policy}"
DREAMZERO_ROOT="${DREAMZERO_ROOT:-${THIRD_PARTY_DIR}/dreamzero}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CKT-WAM Installation Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "  CKT_WAM_ROOT        = ${CKT_WAM_ROOT}"
echo "  COSMOS_POLICY_ROOT  = ${COSMOS_POLICY_ROOT}"
echo "  DREAMZERO_ROOT      = ${DREAMZERO_ROOT}"
echo "  CUDA_VERSION        = ${CUDA_VERSION}"
echo ""

if [ ! -f "${CKT_WAM_ROOT}/pyproject.toml" ]; then
    echo -e "${RED}Error: pyproject.toml not found in ${CKT_WAM_ROOT}!${NC}"
    exit 1
fi

if [ "${CUDA_VERSION}" != "cu128" ] && [ "${CUDA_VERSION}" != "cu130" ]; then
    echo -e "${RED}Error: Invalid CUDA version '${CUDA_VERSION}'${NC}"
    echo "Usage: bash install.sh [cu128|cu130]"
    exit 1
fi

# --- Check uv / fallback to pip --------------------------------------------
if command -v uv &>/dev/null; then
    echo -e "${GREEN}[OK] uv is installed${NC}"
    USE_UV=true
else
    echo -e "${YELLOW}[WARN] uv is not installed. Using pip instead.${NC}"
    echo "  For faster installation, consider installing uv:"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    USE_UV=false
fi

# --- Check Python version --------------------------------------------------
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "${PYTHON_VERSION}" | cut -d. -f1)
PYTHON_MINOR=$(echo "${PYTHON_VERSION}" | cut -d. -f2)

echo "Python version: ${PYTHON_VERSION}"
if [ "${PYTHON_MAJOR}" -lt 3 ] || ([ "${PYTHON_MAJOR}" -eq 3 ] && [ "${PYTHON_MINOR}" -lt 10 ]); then
    echo -e "${RED}Error: Python 3.10 or higher is required${NC}"
    exit 1
fi
echo -e "${GREEN}[OK] Python version OK${NC}"
echo ""

# --- Ensure third_party checkouts ------------------------------------------
mkdir -p "${THIRD_PARTY_DIR}"

if [ ! -d "${COSMOS_POLICY_ROOT}" ]; then
    echo -e "${YELLOW}[WARN] cosmos-policy checkout not found at:${NC}"
    echo "         ${COSMOS_POLICY_ROOT}"
    echo "       Please clone the cosmos-policy repository manually into"
    echo "       that location (or set COSMOS_POLICY_ROOT before re-running)."
fi
if [ ! -d "${DREAMZERO_ROOT}" ]; then
    echo -e "${YELLOW}[WARN] dreamzero checkout not found at:${NC}"
    echo "         ${DREAMZERO_ROOT}"
    echo "       Please clone the dreamzero repository manually into that"
    echo "       location (or set DREAMZERO_ROOT before re-running)."
fi

# --- Install CKT-WAM -------------------------------------------------------
echo ""
echo -e "${YELLOW}Installing CKT-WAM (${CUDA_VERSION}) ...${NC}"
cd "${CKT_WAM_ROOT}"

if [ "${USE_UV}" = true ]; then
    uv pip install -e ".[${CUDA_VERSION}]"
else
    if [ "${CUDA_VERSION}" = "cu128" ]; then
        pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
    else
        pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu130
    fi

    pip install -e .

    echo -e "${YELLOW}Installing GPU-accelerated packages...${NC}"
    if [ "${CUDA_VERSION}" = "cu128" ]; then
        pip install "flash-attn>=2.7.3" "transformer-engine>=2.2.0"
        pip install xformers --index-url https://download.pytorch.org/whl/cu128
    else
        pip install "flash-attn>=2.7.4" "transformer-engine>=2.8.0"
        pip install xformers --index-url https://download.pytorch.org/whl/cu130
    fi
fi
echo -e "${GREEN}[OK] CKT-WAM installed${NC}"

# --- Install cosmos-policy backend -----------------------------------------
if [ -d "${COSMOS_POLICY_ROOT}" ]; then
    echo ""
    echo -e "${YELLOW}Installing cosmos-policy (student WAM backend) ...${NC}"
    (cd "${COSMOS_POLICY_ROOT}" && \
     if [ "${USE_UV}" = true ]; then uv pip install -e ".[${CUDA_VERSION}]"; \
     else pip install -e .; fi)
    echo -e "${GREEN}[OK] cosmos-policy installed${NC}"
fi

# --- Install dreamzero backend (teacher) -----------------------------------
if [ -d "${DREAMZERO_ROOT}" ]; then
    echo ""
    echo -e "${YELLOW}Installing dreamzero (teacher WAM backend) ...${NC}"
    (cd "${DREAMZERO_ROOT}" && \
     if [ "${USE_UV}" = true ]; then uv pip install -e .; \
     else pip install -e .; fi)
    echo -e "${GREEN}[OK] dreamzero installed${NC}"
fi

# --- Install simulation environments (LIBERO / robosuite / RoboCasa) -------
echo ""
echo -e "${YELLOW}Installing simulation environments (if present) ...${NC}"
for env_dir in "LIBERO" "robosuite" "robocasa"; do
    if [ -d "${THIRD_PARTY_DIR}/${env_dir}" ]; then
        echo "Installing ${env_dir} ..."
        (cd "${THIRD_PARTY_DIR}/${env_dir}" && pip install -e .)
        echo -e "${GREEN}[OK] ${env_dir} installed${NC}"
    else
        echo -e "${YELLOW}[SKIP] ${env_dir} not found under ${THIRD_PARTY_DIR}${NC}"
    fi
done

# --- Final note ------------------------------------------------------------
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installation complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Export the environment variables documented in the README.md"
echo "     (CKT_WAM_ROOT, COSMOS_POLICY_ROOT, DREAMZERO_ROOT,"
echo "     BASE_DATASETS_DIR, STUDENT_CONFIG, TEACHER_CKPT_PATH)."
echo "  2. Launch training:  bash ckt_wam/experiments/libero/run_libero_ckt_8gpu.sh"
echo "  3. Launch evaluation: bash ckt_wam/experiments/libero/run_libero_ckt_eval.sh libero_10"
