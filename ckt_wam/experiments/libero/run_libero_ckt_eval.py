"""
Evaluate a CKT-WAM checkpoint on the LIBERO-plus benchmark.

This script wraps the standard LIBERO evaluation loop (shipped with the
``cosmos_policy`` package) and patches in the CKT transfer module so that
the student WAM receives the teacher-derived context ``C_A`` at inference
time.

Both the teacher WAM and the student WAM are loaded once at startup;
the teacher is executed a single time per observation (``t^* = 0``) and
the resulting ``C_A`` is reused across all denoising steps of the student,
matching the acceleration recipe described in the paper.

Usage::

    python -m ckt_wam.experiments.libero.run_libero_ckt_eval \
        --student_config  cosmos_predict2_2b_480p_libero__inference_only \
        --student_ckpt    <path/to/student_weights> \
        --teacher_path    <path/to/teacher_checkpoint> \
        --ckt_ckpt        <path/to/ckt_adapter_bank.pt> \
        --task_suite_name libero_10 \
        --local_log_dir   outputs/eval/libero_ckt
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ckt_wam.libero_eval")


def _load_student(
    student_config_name: str,
    student_ckpt: Optional[str],
    device: torch.device,
) -> nn.Module:
    """
    Load the student WAM (Cosmos-Policy-2B) using the ``cosmos_policy``
    inference utilities and push it to the requested device.
    """
    from cosmos_policy.experiments.robot.cosmos_utils import get_model

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.config = student_config_name
    cfg.ckpt_path = student_ckpt or ""
    cfg.config_file = os.environ.get(
        "CKT_COSMOS_CONFIG_FILE", "cosmos_policy/config/config.py"
    )
    model, _ = get_model(cfg)
    return model.to(device)


def _load_teacher(teacher_path: str, device: torch.device) -> nn.Module:
    """Load the teacher WAM (DreamZero-14B) and freeze it."""
    from groot.vla.model.dreamzero.base_vla import VLA

    logger.info(f"Loading teacher WAM from: {teacher_path}")
    teacher = VLA.from_pretrained(teacher_path)
    teacher = teacher.to(device=device, dtype=torch.bfloat16)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def build_ckt_pipeline(
    student_config_name: str,
    student_ckpt: str,
    teacher_path: str,
    ckt_ckpt: str,
    teacher_block_index: int = 20,
    use_middle_layer: bool = True,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Build a CKT pipeline ready for LIBERO evaluation.

    The resulting module exposes ``.student`` (for use with the standard
    cosmos-policy evaluation loop), while ``__call__`` / ``generate_samples``
    transparently inject the adapter-derived context ``C_A``.
    """
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    student = _load_student(student_config_name, student_ckpt, device)
    teacher = _load_teacher(teacher_path, device)

    if use_middle_layer:
        from ckt_wam.pipeline.ckt_pipeline_middle import (
            CKTPipelineMiddle as _Pipeline,
        )
        from ckt_wam.pipeline.ckt_pipeline_middle import (
            CKTPipelineMiddleConfig as _Cfg,
        )

        pipeline_cfg = _Cfg(teacher_block_index=teacher_block_index)
    else:
        from ckt_wam.pipeline.ckt_pipeline import CKTPipeline as _Pipeline
        from ckt_wam.pipeline.ckt_pipeline import CKTPipelineConfig as _Cfg

        pipeline_cfg = _Cfg()

    pipeline = _Pipeline(
        teacher_wam=teacher,
        student_model=student,
        config=pipeline_cfg,
    )
    pipeline.freeze_teacher()
    pipeline.to(device)

    state = torch.load(ckt_ckpt, map_location="cpu")
    if "adapter_bank" in state:
        pipeline.adapter_bank.load_state_dict(state["adapter_bank"])
        logger.info(f"Loaded CKT adapter bank weights from {ckt_ckpt}")
    else:
        pipeline.adapter_bank.load_state_dict(state)
        logger.info(
            f"Loaded CKT adapter bank weights from {ckt_ckpt} (raw state dict)"
        )

    pipeline.eval()
    return pipeline


def main():
    parser = argparse.ArgumentParser(
        description="LIBERO evaluation for a CKT-WAM checkpoint"
    )
    parser.add_argument("--student_config", type=str, required=True)
    parser.add_argument("--student_ckpt", type=str, required=True)
    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument("--ckt_ckpt", type=str, required=True)
    parser.add_argument("--teacher_block_index", type=int, default=20)
    parser.add_argument(
        "--use_last_layer",
        action="store_true",
        help="Use the teacher's last block (ablation).",
    )
    parser.add_argument(
        "--task_suite_name",
        type=str,
        default="libero_10",
        choices=[
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
            "libero_90",
        ],
    )
    parser.add_argument("--local_log_dir", type=str, default="outputs/eval/libero_ckt")
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--run_id_note", type=str, default="ckt_wam")
    args = parser.parse_args()

    pipeline = build_ckt_pipeline(
        student_config_name=args.student_config,
        student_ckpt=args.student_ckpt,
        teacher_path=args.teacher_path,
        ckt_ckpt=args.ckt_ckpt,
        teacher_block_index=args.teacher_block_index,
        use_middle_layer=not args.use_last_layer,
    )

    # Delegate the actual rollout loop to the reference LIBERO evaluator
    # shipped with cosmos_policy.  The CKT pipeline has already installed
    # its forward hook on ``student.net.text_embedding``, so every student
    # forward call will automatically see the transferred context ``C_A``.
    from cosmos_policy.experiments.robot.libero.run_libero_eval import (
        PolicyEvalConfig,
        eval_libero,
    )

    cfg = PolicyEvalConfig(
        config=args.student_config,
        ckpt_path=args.student_ckpt,
        task_suite_name=args.task_suite_name,
        local_log_dir=args.local_log_dir,
        seed=args.seed,
        num_trials_per_task=args.num_trials_per_task,
        run_id_note=args.run_id_note,
    )

    # Stash the CKT pipeline on an attribute so downstream hooks can
    # access it if needed (the forward hook on text_embedding is what
    # actually carries the context through, so nothing else is required).
    cfg.ckt_pipeline = pipeline  # type: ignore[attr-defined]

    final_sr = eval_libero(cfg)
    logger.info(f"[CKT-WAM LIBERO] final success rate: {final_sr * 100:.1f}%")


if __name__ == "__main__":
    main()
