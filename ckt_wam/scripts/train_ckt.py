"""
Training entry point for CKT-WAM.

This script launches distributed CKT-WAM training with a frozen teacher
WAM and a frozen student WAM; only the sparse adapter bank is optimized.

Usage (single GPU)::

    python -m ckt_wam.scripts.train_ckt \
        --student_config <path/to/student/config.py> \
        --teacher_path   <path/to/teacher/checkpoint> \
        --output_dir     outputs/ckt_libero

Usage (multi-GPU with ``torchrun``)::

    torchrun --nproc_per_node=8 -m ckt_wam.scripts.train_ckt \
        --student_config <path/to/student/config.py> \
        --teacher_path   <path/to/teacher/checkpoint> \
        --output_dir     outputs/ckt_libero

Paths are resolved through environment variables / CLI arguments so that
no absolute paths are hard-coded inside the script.  See the project
README for the recommended environment-variable layout.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader, DistributedSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ckt_wam")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """CLI-configurable CKT-WAM training hyperparameters."""

    # --- Paths ---
    student_config: str = ""
    teacher_path: str = ""
    output_dir: str = "outputs/ckt_wam"
    resume_from: Optional[str] = None

    # --- Pipeline / Adapter ---
    teacher_hidden_dim: int = 5120
    student_hidden_dim: int = 2048
    teacher_block_index: int = 20
    adapter_bottleneck_dim: int = 512
    adapter_dropout: float = 0.1
    num_adapter_output_tokens: int = 32
    num_specialized_experts: int = 8
    top_k: int = 2
    gating_hidden_dim: int = 512

    # --- Training ---
    max_iterations: int = 100_000
    batch_size: int = 4
    learning_rate: float = 1e-4
    adapter_lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_iterations: int = 1000
    grad_clip_norm: float = 1.0
    grad_accumulation_steps: int = 1
    use_amp: bool = True

    # --- Loss ---
    load_balance_weight: float = 0.01
    action_loss_type: str = "l1"

    # --- Teacher-layer selection ---
    use_middle_layer: bool = True

    # --- Logging / Checkpointing ---
    log_every: int = 50
    save_every: int = 5000
    validate_every: int = 2000
    num_workers: int = 4
    seed: int = 42


# ---------------------------------------------------------------------------
# Optimizer / scheduler helpers
# ---------------------------------------------------------------------------
def build_optimizer(
    pipeline: nn.Module, config: TrainConfig
) -> torch.optim.Optimizer:
    """Build AdamW with a higher LR for the adapter bank."""
    adapter_params = []
    student_params = []

    for name, param in pipeline.named_parameters():
        if not param.requires_grad:
            continue
        if "adapter_bank" in name:
            adapter_params.append(param)
        else:
            student_params.append(param)

    param_groups = [
        {
            "params": adapter_params,
            "lr": config.adapter_lr,
            "weight_decay": config.weight_decay,
            "name": "adapter_bank",
        },
        {
            "params": student_params,
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
            "name": "student_backbone",
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)

    logger.info(
        "Optimizer: AdamW | "
        f"Adapter params: {sum(p.numel() for p in adapter_params):,} @ "
        f"lr={config.adapter_lr} | "
        f"Student params: {sum(p.numel() for p in student_params):,} @ "
        f"lr={config.learning_rate}"
    )
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainConfig
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup followed by cosine annealing."""

    def lr_lambda(step: int) -> float:
        if step < config.warmup_iterations:
            return step / max(config.warmup_iterations, 1)
        progress = (step - config.warmup_iterations) / max(
            config.max_iterations - config.warmup_iterations, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Model loading utilities
# ---------------------------------------------------------------------------
def load_teacher_model(
    teacher_path: Optional[str], device: torch.device
) -> nn.Module:
    """
    Load the teacher WAM (default: DreamZero-14B) in bfloat16 and freeze it.

    If ``teacher_path`` is ``None`` or empty, a lightweight dummy teacher
    is returned so that the pipeline can be smoke-tested without the full
    teacher checkpoint.
    """
    if not teacher_path:
        logger.warning(
            "No teacher path provided -- using a dummy teacher module."
        )

        class DummyTeacher(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy_param = nn.Linear(1, 1)

            def forward(self, *args, **kwargs):
                batch_size = 1
                for a in args:
                    if isinstance(a, torch.Tensor):
                        batch_size = a.shape[0]
                        break
                return torch.randn(batch_size, 769, 5120, dtype=torch.bfloat16)

        teacher = DummyTeacher().to(device=device, dtype=torch.bfloat16)
        teacher.eval()
        return teacher

    # Teacher WAM (DreamZero-14B) lives in the ``dreamzero`` package.
    # We import lazily so that the environment does not need dreamzero
    # available when only exercising the adapter code paths.
    from groot.vla.model.dreamzero.base_vla import VLA

    logger.info(f"Loading teacher WAM from: {teacher_path}")
    teacher = VLA.from_pretrained(teacher_path)
    teacher = teacher.to(device=device, dtype=torch.bfloat16)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    logger.info("Teacher WAM loaded and frozen.")
    return teacher


def load_student_model(
    student_config_path: str, device: torch.device
) -> nn.Module:
    """Load the student WAM (default: Cosmos-Policy-2B) from a LazyConfig."""
    from cosmos_policy._src.imaginaire.config import load_config
    from cosmos_policy._src.imaginaire.lazy_config import instantiate

    logger.info(f"Loading student WAM from config: {student_config_path}")
    config = load_config(student_config_path, [])
    student = instantiate(config.model)
    student = student.to(device=device)
    logger.info("Student WAM loaded.")
    return student


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------
def save_checkpoint(
    pipeline: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    iteration: int,
    config: TrainConfig,
    output_dir: str,
):
    """Save the trainable parts of the pipeline plus optimizer state."""
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"checkpoint_{iteration:08d}.pt")

    state = {
        "iteration": iteration,
        "adapter_bank": pipeline.adapter_bank.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": asdict(config),
    }
    student_trainable = {
        k: v
        for k, v in pipeline.student.state_dict().items()
        if any(
            p.data_ptr() == v.data_ptr()
            for p in pipeline.student.parameters()
            if p.requires_grad
        )
    }
    state["student_trainable"] = student_trainable

    torch.save(state, ckpt_path)
    logger.info(f"Checkpoint saved: {ckpt_path}")


def load_checkpoint(
    pipeline: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str,
) -> int:
    logger.info(f"Resuming from checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")

    pipeline.adapter_bank.load_state_dict(state["adapter_bank"])
    if "student_trainable" in state:
        pipeline.student.load_state_dict(
            state["student_trainable"], strict=False
        )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])

    iteration = state["iteration"]
    logger.info(f"Resumed from iteration {iteration}")
    return iteration


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(config: TrainConfig):
    """
    Main CKT-WAM training loop.

    Per iteration::

      1. Teacher (frozen): extract intermediate hidden states ``H_T``.
      2. AdapterBank:      ``H_T -> C_A`` + routing probabilities.
      3. Student (frozen): inject ``C_A`` into conditioning, run native
         diffusion ``training_step``.
      4. Loss:             ``L = L_CKT + lambda_bal * L_bal``.
      5. Backward + AdamW step.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0

    if is_main:
        logger.info(
            "Training config:\n" + json.dumps(asdict(config), indent=2)
        )

    torch.manual_seed(config.seed + local_rank)
    torch.cuda.manual_seed(config.seed + local_rank)

    teacher = load_teacher_model(config.teacher_path, device)
    student = load_student_model(config.student_config, device)

    # Build pipeline + loss (defer imports so the script works both from
    # a source checkout and from an installed wheel).
    if config.use_middle_layer:
        from ckt_wam.pipeline.ckt_pipeline_middle import (
            CKTPipelineMiddle as _Pipeline,
        )
        from ckt_wam.pipeline.ckt_pipeline_middle import (
            CKTPipelineMiddleConfig as _PipelineConfig,
        )

        pipeline_config = _PipelineConfig(
            teacher_hidden_dim=config.teacher_hidden_dim,
            student_hidden_dim=config.student_hidden_dim,
            teacher_block_index=config.teacher_block_index,
            adapter_bottleneck_dim=config.adapter_bottleneck_dim,
            adapter_dropout=config.adapter_dropout,
            num_adapter_output_tokens=config.num_adapter_output_tokens,
            num_specialized_experts=config.num_specialized_experts,
            top_k=config.top_k,
            gating_hidden_dim=config.gating_hidden_dim,
        )
    else:
        from ckt_wam.pipeline.ckt_pipeline import CKTPipeline as _Pipeline
        from ckt_wam.pipeline.ckt_pipeline import (
            CKTPipelineConfig as _PipelineConfig,
        )

        pipeline_config = _PipelineConfig(
            teacher_hidden_dim=config.teacher_hidden_dim,
            student_hidden_dim=config.student_hidden_dim,
            adapter_bottleneck_dim=config.adapter_bottleneck_dim,
            adapter_dropout=config.adapter_dropout,
            num_adapter_output_tokens=config.num_adapter_output_tokens,
            num_specialized_experts=config.num_specialized_experts,
            top_k=config.top_k,
            gating_hidden_dim=config.gating_hidden_dim,
        )

    from ckt_wam.losses.ckt_losses import CKTLoss, CKTLossConfig

    pipeline = _Pipeline(
        teacher_wam=teacher,
        student_model=student,
        config=pipeline_config,
    )
    pipeline.freeze_teacher()
    pipeline.to(device)

    if is_main:
        pipeline.print_param_summary()

    if is_distributed:
        pipeline = nn.parallel.DistributedDataParallel(
            pipeline,
            device_ids=[local_rank],
            find_unused_parameters=True,
        )
        pipeline_module = pipeline.module
    else:
        pipeline_module = pipeline

    loss_config = CKTLossConfig(
        load_balance_weight=config.load_balance_weight,
        action_loss_type=config.action_loss_type,
        num_experts=config.num_specialized_experts,
    )
    ckt_loss_fn = CKTLoss(loss_config).to(device)

    optimizer = build_optimizer(pipeline_module, config)
    scheduler = build_scheduler(optimizer, config)

    scaler = GradScaler("cuda", enabled=config.use_amp)

    start_iteration = 0
    if config.resume_from:
        start_iteration = load_checkpoint(
            pipeline_module, optimizer, scheduler, config.resume_from
        )

    # Dataset / dataloader construction is left to the downstream user --
    # see ``_build_placeholder_dataloader`` below for the expected format.
    logger.info(
        "NOTE: Dataset loading is provided as a placeholder. Replace the "
        "_build_placeholder_dataloader() call with a real dataset (e.g. "
        "LIBERODataset from cosmos_policy.datasets.libero_dataset). The "
        "data_batch must provide the student-side fields expected by the "
        "student WAM, plus teacher-side fields prefixed with 'teacher_'."
    )

    dataloader_train = _build_placeholder_dataloader(
        config, is_distributed, world_size, local_rank
    )

    pipeline.train()
    pipeline_module.freeze_teacher()

    running_loss = 0.0
    running_lb_loss = 0.0
    t_start = time.time()
    iteration = start_iteration
    epoch = 0

    while iteration < config.max_iterations:
        if is_distributed and hasattr(dataloader_train.sampler, "set_epoch"):
            dataloader_train.sampler.set_epoch(epoch)

        for data_batch in dataloader_train:
            if iteration >= config.max_iterations:
                break

            data_batch = {
                k: v.to(device, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in data_batch.items()
            }

            teacher_inputs = {
                k[len("teacher_"):]: v
                for k, v in data_batch.items()
                if k.startswith("teacher_")
            }
            if not teacher_inputs:
                raise ValueError(
                    "data_batch contains no 'teacher_*' keys. The dataset "
                    "must provide teacher-formatted fields prefixed with "
                    "'teacher_' (e.g. teacher_images, teacher_text, ...)."
                )

            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=config.use_amp
            ):
                (
                    output_batch,
                    student_loss,
                    expert_probs,
                ) = pipeline_module.training_step(
                    data_batch, iteration, teacher_inputs=teacher_inputs
                )

                total_loss, loss_dict = ckt_loss_fn(
                    student_loss=student_loss,
                    expert_probs=expert_probs,
                )

                scaled_loss = total_loss / config.grad_accumulation_steps

            scaler.scale(scaled_loss).backward()

            if (iteration + 1) % config.grad_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in pipeline.parameters() if p.requires_grad],
                    config.grad_clip_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running_loss += total_loss.item()
            running_lb_loss += loss_dict["load_balance_loss"].item()

            if is_main and (iteration + 1) % config.log_every == 0:
                avg_loss = running_loss / config.log_every
                avg_lb = running_lb_loss / config.log_every
                elapsed = time.time() - t_start
                it_per_sec = config.log_every / elapsed

                lr_adapter = optimizer.param_groups[0]["lr"]
                lr_student = optimizer.param_groups[1]["lr"]

                logger.info(
                    f"[Iter {iteration+1:>7d}/{config.max_iterations}] "
                    f"loss={avg_loss:.4f} "
                    f"edm={loss_dict['student_edm_loss']:.4f} "
                    f"lb={avg_lb:.4f} "
                    f"lr_adpt={lr_adapter:.2e} "
                    f"lr_stud={lr_student:.2e} "
                    f"it/s={it_per_sec:.2f}"
                )

                running_loss = 0.0
                running_lb_loss = 0.0
                t_start = time.time()

            if is_main and (iteration + 1) % config.save_every == 0:
                save_checkpoint(
                    pipeline_module,
                    optimizer,
                    scheduler,
                    iteration + 1,
                    config,
                    config.output_dir,
                )

            iteration += 1

        epoch += 1

    if is_main:
        save_checkpoint(
            pipeline_module,
            optimizer,
            scheduler,
            iteration,
            config,
            config.output_dir,
        )
        logger.info("Training complete.")

    pipeline_module.cleanup()

    if is_distributed:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Placeholder data loader
# ---------------------------------------------------------------------------
def _build_placeholder_dataloader(
    config: TrainConfig,
    is_distributed: bool,
    world_size: int,
    local_rank: int,
) -> DataLoader:
    """
    Return a DataLoader of dummy tensors matching the expected schema.

    Replace this with a real dataset (e.g. ``LIBERODataset``,
    ``RoboCasaDataset``, ``ALOHADataset`` from the ``cosmos_policy``
    package) to run real CKT-WAM training.  The expected schema is:

      Student-side keys (Cosmos-Policy-style)::

        video:                         [B, 3, T, H, W]
        t5_text_embeddings:            [B, L_text, 4096]
        t5_text_mask:                  [B, L_text]
        actions:                       [B, chunk_size, action_dim]
        proprio:                       [B, proprio_dim]
        rollout_data_mask:             [B]
        world_model_sample_mask:       [B]
        value_function_sample_mask:    [B]
        ...and latent index fields as required by the student WAM.

      Teacher-side keys (prefixed with ``teacher_``)::

        teacher_images:              [B, T, H, W, 3]
        teacher_text:                [B, L]
        teacher_text_attention_mask: [B, L]
        teacher_action:              [B, T_a, action_dim]
        teacher_state:               [B, T_s, state_dim]
        teacher_embodiment_id:       [B]
        teacher_has_real_action:     bool or [B]
        teacher_action_mask:         [B, T_a]
    """
    chunk_size = 16
    action_dim = 7
    proprio_dim = 9
    num_frames = 4
    text_len = 512
    H, W = 224, 224
    state_t = 9

    class _PlaceholderDataset(torch.utils.data.Dataset):
        def __init__(self, length: int = 1000):
            self.length = length

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            teacher_action_len = 16
            teacher_state_dim = 7

            return {
                "video": torch.randn(3, num_frames, H, W),
                "t5_text_embeddings": torch.randn(text_len, 4096),
                "t5_text_mask": torch.ones(text_len),
                "actions": torch.randn(chunk_size, action_dim).clamp(-1, 1),
                "action_latent_idx": torch.tensor(state_t - 2, dtype=torch.long),
                "proprio": torch.randn(proprio_dim),
                "current_proprio_latent_idx": torch.tensor(-1, dtype=torch.long),
                "future_proprio": torch.randn(proprio_dim),
                "future_proprio_latent_idx": torch.tensor(-1, dtype=torch.long),
                "future_image_latent_idx": torch.tensor(-1, dtype=torch.long),
                "future_image2_latent_idx": torch.tensor(-1, dtype=torch.long),
                "future_wrist_image_latent_idx": torch.tensor(-1, dtype=torch.long),
                "rollout_data_mask": torch.tensor(0, dtype=torch.long),
                "world_model_sample_mask": torch.tensor(0, dtype=torch.long),
                "value_function_sample_mask": torch.tensor(0, dtype=torch.long),
                "value_function_return": torch.tensor(0.0),
                "value_latent_idx": torch.tensor(state_t - 1, dtype=torch.long),
                "teacher_images": torch.randn(num_frames, H, W, 3),
                "teacher_text": torch.randint(0, 32000, (text_len,)),
                "teacher_text_attention_mask": torch.ones(text_len, dtype=torch.long),
                "teacher_action": torch.randn(teacher_action_len, action_dim).clamp(-1, 1),
                "teacher_state": torch.randn(num_frames - 1, teacher_state_dim),
                "teacher_embodiment_id": torch.tensor(0, dtype=torch.long),
                "teacher_has_real_action": torch.tensor(True),
                "teacher_action_mask": torch.ones(teacher_action_len, dtype=torch.bool),
            }

    dataset = _PlaceholderDataset()

    sampler = (
        DistributedSampler(
            dataset, num_replicas=world_size, rank=local_rank, shuffle=True
        )
        if is_distributed
        else None
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="CKT-WAM training")
    parser.add_argument("--student_config", type=str, required=True)
    parser.add_argument("--teacher_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/ckt_wam")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--teacher_block_index", type=int, default=20)
    parser.add_argument("--adapter_bottleneck_dim", type=int, default=512)
    parser.add_argument("--num_adapter_output_tokens", type=int, default=32)
    parser.add_argument("--max_iterations", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adapter_lr", type=float, default=3e-4)
    parser.add_argument("--load_balance_weight", type=float, default=0.01)
    parser.add_argument("--num_specialized_experts", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument(
        "--use_last_layer",
        action="store_true",
        default=False,
        help="Use teacher's last transformer block instead of the "
        "intermediate block (ablation variant).",
    )

    args = parser.parse_args()

    config = TrainConfig(
        student_config=args.student_config,
        teacher_path=args.teacher_path,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        teacher_block_index=args.teacher_block_index,
        adapter_bottleneck_dim=args.adapter_bottleneck_dim,
        num_adapter_output_tokens=args.num_adapter_output_tokens,
        max_iterations=args.max_iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        adapter_lr=args.adapter_lr,
        load_balance_weight=args.load_balance_weight,
        num_specialized_experts=args.num_specialized_experts,
        top_k=args.top_k,
        use_amp=args.use_amp,
        grad_accumulation_steps=args.grad_accumulation_steps,
        seed=args.seed,
        log_every=args.log_every,
        save_every=args.save_every,
        use_middle_layer=not args.use_last_layer,
    )
    return config


def main() -> None:
    """Zero-argument entry point used by both ``python -m`` and the
    ``ckt-train`` console script."""
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
