# CKT-WAM: Parameter Efficient Context Knowledge Transfer Between World Action Models

CKT-WAM is a parameter-efficient framework for transferring knowledge between **World Action Models (WAMs)**. Instead of imitating the teacher's output logits or aligning every hidden state, CKT-WAM extracts an intermediate representation from a frozen **teacher WAM** and delivers it to a frozen **student WAM** as a compact, *transferable context*.

Only the CKT Module (≈ 187.4M parameters) is trained; both WAMs stay frozen. The denoising objective of the student and the auxiliary load-balancing loss are the only supervision signals.

---

## 1. Repository layout

```
CKT-WAM/
├── ckt_wam/                       # Installable package (src layout)
│   ├── models/
│   │   └── ckt_adapter_bank.py    # Adapter, DynamicRouter, AdapterBank
│   ├── losses/
│   │   └── ckt_losses.py          # CKTLoss, LoadBalancingLoss, ActionChunkLoss
│   ├── pipeline/
│   │   ├── ckt_pipeline.py        # CKT with teacher last-block features
│   │   └── ckt_pipeline_middle.py # CKT with teacher intermediate (l*=20) features
│   ├── scripts/
│   │   └── train_ckt.py           # Distributed training entry point
│   └── experiments/libero/
│       ├── run_libero_ckt_eval.py # LIBERO evaluation wrapper
│       ├── run_libero_ckt_8gpu.sh # 8-GPU training launcher
│       └── run_libero_ckt_eval.sh # Evaluation launcher
├── third_party/                   # (You provide) external WAM checkouts
│   ├── cosmos-policy/             #   student WAM backend
│   └── dreamzero/                 #   teacher WAM backend
├── pyproject.toml                 # Merged dependency manifest
├── requirements.txt               # Plain-pip mirror of the dependencies
├── install.sh                     # One-shot installer (uv / pip)
└── README.md
```

---

## 2. Environment setup

### 2.1 Hardware / software requirements

- Linux (x86-64, CUDA ≥ 12.8 recommended)
- Python 3.10 – 3.12
- NVIDIA GPU with ≥ 40 GB VRAM per device (for training); a single 24 GB GPU is enough for inference-only evaluation.
- 8 × GPUs for the default training recipe (LIBERO-plus).

### 2.2 Clone CKT-WAM and its WAM backends

```bash
# Clone this repository
git clone <ckt-wam-anonymous-url> CKT-WAM
cd CKT-WAM
export CKT_WAM_ROOT=$(pwd)

# Place the two backend checkouts under third_party/ (or anywhere you like,
# as long as you point COSMOS_POLICY_ROOT / DREAMZERO_ROOT to them).
mkdir -p third_party
cd third_party
git clone <cosmos-policy-url>   cosmos-policy
git clone <dreamzero-url>       dreamzero
cd "${CKT_WAM_ROOT}"

export COSMOS_POLICY_ROOT="${CKT_WAM_ROOT}/third_party/cosmos-policy"
export DREAMZERO_ROOT="${CKT_WAM_ROOT}/third_party/dreamzero"
```

### 2.3 Install

Pick the extra that matches your CUDA driver and run the installer:

```bash
# CUDA 12.8 / PyTorch 2.7 (default)
bash install.sh cu128
# or
# CUDA 13.0 / PyTorch 2.9
bash install.sh cu130
```

The installer will:

1. Verify the Python version.
2. Install `ckt-wam` in editable mode together with the `cu128`/`cu130` extras (PyTorch, xformers, flash-attn, transformer-engine, NATTEN).
3. Install `cosmos-policy` and `dreamzero` in editable mode (if their checkouts are present under `third_party/`).
4. Install any simulation environments (`LIBERO/`, `robosuite/`, `robocasa/`) that you have placed under `third_party/`.

If you prefer plain `pip`, you can install from the wheel dependency list:

```bash
pip install -e ".[cu128]"   # or .[cu130]
pip install -e "${COSMOS_POLICY_ROOT}"
pip install -e "${DREAMZERO_ROOT}"
```

### 2.4 Checkpoints

CKT-WAM itself only ships the adapter-bank weights (produced by training). You will need to download:

- **Teacher WAM** — DreamZero-14B checkpoint. Place it at `${DREAMZERO_ROOT}/checkpoints` or point `TEACHER_CKPT_PATH` to a directory of your choice.
- **Student WAM** — Cosmos-Policy-2B checkpoint. Any checkpoint supported by the `cosmos-policy` LIBERO inference config works (the default is the public HuggingFace slug `nvidia/Cosmos-Policy-LIBERO-Predict2-2B`, referenced through the `--student_ckpt` flag).

### 2.5 Datasets

CKT-WAM reuses the dataset loaders of `cosmos-policy` and expects the LIBERO / RoboCasa data to follow the same format. Set the parent directory once:

```bash
export BASE_DATASETS_DIR=/path/to/datasets
```

The training launcher uses this variable internally; no path is written into source files.

### 2.6 Summary of environment variables

| Variable | Required by | Description |
| :--- | :--- | :--- |
| `CKT_WAM_ROOT` | train & eval | Repository root (auto-detected by default). |
| `COSMOS_POLICY_ROOT` | train & eval | Student WAM backend (`cosmos-policy`) checkout. |
| `DREAMZERO_ROOT` | train & eval | Teacher WAM backend (`dreamzero`) checkout. |
| `BASE_DATASETS_DIR` | train | Root of the LIBERO / RoboCasa datasets. |
| `STUDENT_CONFIG` | train | LazyConfig file of the student experiment. |
| `TEACHER_CKPT_PATH` | train & eval | Directory containing the teacher WAM checkpoint. |
| `STUDENT_CKPT` | eval | Student WAM checkpoint path or HuggingFace slug. |
| `CKT_CKPT` | eval | Trained adapter-bank checkpoint (`*.pt`). |

---

## 3. Training

The reference recipe trains the CKT Adapter Bank on **LIBERO-plus** with 8 GPUs.

```bash
cd "${CKT_WAM_ROOT}"
bash ckt_wam/experiments/libero/run_libero_ckt_8gpu.sh
```

Key hyperparameters (overridable via environment variables):

| Variable | Default | Meaning |
| :--- | :---: | :--- |
| `MAX_ITERATIONS` | 100 000 | Total optimization steps. |
| `BATCH_SIZE_PER_GPU` | 4 | Micro-batch per device. |
| `GRAD_ACCUMULATION_STEPS` | 1 | Gradient accumulation. |
| `LEARNING_RATE` | 1e-4 | Base learning rate (unused parameters of the student). |
| `ADAPTER_LR` | 3e-4 | Learning rate for adapter bank. |
| `TEACHER_BLOCK_INDEX` | 20 | Teacher layer `l*` to extract (40 layers total). |
| `NUM_SPECIALIZED_EXPERTS` | 8 | `M` specialized adapters. |
| `TOP_K` | 2 | Active specialized adapters per instance. |
| `NUM_ADAPTER_OUTPUT_TOKENS` | 32 | Number of learnable query tokens `K`. |
| `ADAPTER_BOTTLENECK_DIM` | 512 | Bottleneck dim `d_b` per expert. |
| `LOAD_BALANCE_WEIGHT` | 0.01 | `λ_bal` for the auxiliary LB loss. |

Example overrides:

```bash
MAX_ITERATIONS=200000 ADAPTER_LR=2e-4 \
BASE_DATASETS_DIR=/data/libero-plus \
STUDENT_CONFIG="${COSMOS_POLICY_ROOT}/cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py" \
TEACHER_CKPT_PATH="${DREAMZERO_ROOT}/checkpoints" \
bash ckt_wam/experiments/libero/run_libero_ckt_8gpu.sh
```

Resuming an interrupted run:

```bash
bash ckt_wam/experiments/libero/run_libero_ckt_8gpu.sh --resume \
    "${CKT_WAM_ROOT}/outputs/libero_ckt_8gpu/checkpoint_00050000.pt"
```

Checkpoints, training logs, and a copy of the launching script are written to `${OUTPUT_DIR}` (default: `${CKT_WAM_ROOT}/outputs/libero_ckt_8gpu`).

### 3.1 Ablation: teacher last-block vs. intermediate layer

The default training uses `CKTPipelineMiddle`, which extracts the teacher hidden states at `TEACHER_BLOCK_INDEX` (layer 20 of 40). To reproduce the ablation that uses the teacher's final block instead, pass `--use_last_layer` to `train_ckt.py` (or set `USE_LAST_LAYER=true` when extending the launcher).

---

## 4. Evaluation

CKT-WAM reuses the standard `cosmos-policy` LIBERO evaluator. A thin wrapper (`run_libero_ckt_eval.py`) builds the CKT pipeline (teacher + student + adapter bank), hooks the student's conditioning stream, and then delegates to `cosmos_policy`'s `eval_libero` routine.

```bash
cd "${CKT_WAM_ROOT}"

export STUDENT_CKPT="nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
export TEACHER_CKPT_PATH="${DREAMZERO_ROOT}/checkpoints"
export CKT_CKPT="${CKT_WAM_ROOT}/outputs/libero_ckt_8gpu/checkpoint_00100000.pt"

# Evaluate on the long-horizon LIBERO-10 suite.
bash ckt_wam/experiments/libero/run_libero_ckt_eval.sh libero_10

# Other suites
bash ckt_wam/experiments/libero/run_libero_ckt_eval.sh libero_spatial
bash ckt_wam/experiments/libero/run_libero_ckt_eval.sh libero_object
bash ckt_wam/experiments/libero/run_libero_ckt_eval.sh libero_goal
```

The script prints per-task success rates and writes rollouts / JSON summaries to `${LOCAL_LOG_DIR}` (default: `${CKT_WAM_ROOT}/outputs/eval/libero_ckt_<task_suite>`).

### 4.1 Command-line arguments

`run_libero_ckt_eval.py` exposes the following flags (all have sensible defaults via the launcher):

| Flag | Meaning |
| :--- | :--- |
| `--student_config` | Name of the `cosmos-policy` inference config. |
| `--student_ckpt` | Student WAM checkpoint path or HuggingFace slug. |
| `--teacher_path` | Teacher WAM checkpoint directory. |
| `--ckt_ckpt` | Trained adapter-bank checkpoint. |
| `--teacher_block_index` | Teacher layer index (must match training). |
| `--use_last_layer` | Use the teacher's final block (ablation). |
| `--task_suite_name` | `libero_spatial` / `libero_object` / `libero_goal` / `libero_10` / `libero_90`. |
| `--num_trials_per_task` | Episodes per task (default 50). |
| `--local_log_dir` | Output directory for videos and JSON summaries. |
| `--seed` | Random seed. |

---

## 5. Citation

If you use this code, please cite our paper:

```bibtex
@article{ckt_wam_2026,
  title   = {CKT-WAM: Parameter Efficient Context Knowledge Transfer
             Between World Action Models},
  author  = {Anonymous},
  journal = {Under review},
  year    = {2026}
}
```

---

## 6. License

The CKT-WAM code is released under the Apache-2.0 license. The licences of the underlying student and teacher WAMs (`cosmos-policy`, `dreamzero`) apply to the respective backends and checkpoints; please consult those repositories for details.
