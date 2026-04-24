"""
CKT-WAM pipeline with **last-block** teacher features (ablation variant).

Identical to ``CKTPipelineMiddle`` except that the teacher's final
transformer block is used as the source of ``H_T``.  This variant is
retained for ablation studies (Section "Method"): the CKT-WAM paper
reports that an intermediate teacher layer (``l^* = 20``) provides a
better cost--utility trade-off than relying on the deepest layer, which
this pipeline mirrors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ckt_wam.models.ckt_adapter_bank import AdapterBank

logger = logging.getLogger(__name__)


@dataclass
class CKTPipelineConfig:
    """Configuration for the last-block CKT pipeline (ablation)."""

    teacher_hidden_dim: int = 5120
    teacher_text_dim: int = 4096
    teacher_clip_dim: int = 1280

    student_hidden_dim: int = 2048

    adapter_bottleneck_dim: int = 512
    adapter_dropout: float = 0.1
    num_adapter_output_tokens: int = 32
    num_specialized_experts: int = 8
    top_k: int = 2
    gating_hidden_dim: int = 512

    load_balance_loss_weight: float = 0.01


class _AttrDict(dict):
    """Dict subclass with attribute-style access (BatchFeature drop-in)."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class TeacherFeatureExtractor(nn.Module):
    """
    Extract hidden states from the **last** transformer block of the teacher
    WAM via a forward hook.

    See ``TeacherMiddleLayerFeatureExtractor`` for the gradient handling
    discussion -- the same caveats apply here.
    """

    def __init__(self, teacher_wam: nn.Module):
        super().__init__()
        self.teacher_wam = teacher_wam
        self.teacher_wam.eval()
        for p in self.teacher_wam.parameters():
            p.requires_grad = False

        self._base_model = self._resolve_base_model()

        self._last_block_output: Optional[torch.Tensor] = None
        self._hook_handle = self._base_model.blocks[-1].register_forward_hook(
            self._capture_last_block_hook
        )
        logger.info(
            "TeacherFeatureExtractor: registered hook on last transformer "
            "block (index %d of %d)",
            len(self._base_model.blocks) - 1,
            len(self._base_model.blocks),
        )

    def _resolve_base_model(self):
        model = self.teacher_wam.action_head.model
        if hasattr(model, "base_model"):
            return model.base_model.model
        return model

    def _capture_last_block_hook(
        self,
        module: nn.Module,
        input: Any,
        output: Any,
    ) -> None:
        x = output[0] if isinstance(output, tuple) else output
        self._last_block_output = x.detach()

    def forward(self, teacher_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        orig_gc = getattr(self._base_model, "gradient_checkpointing", False)
        self._base_model.gradient_checkpointing = True

        try:
            self.teacher_wam.action_head(
                _AttrDict(),
                _AttrDict(teacher_inputs),
            )
        finally:
            self._base_model.gradient_checkpointing = orig_gc

        h_teacher = self._last_block_output
        self._last_block_output = None

        if h_teacher is None:
            raise RuntimeError(
                "Forward hook on the last transformer block did not fire. "
                "Verify that teacher_inputs triggers the teacher's training "
                "forward path."
            )

        return h_teacher

    def cleanup(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


class CKTPipeline(nn.Module):
    """
    CKT-WAM pipeline using the teacher's last-block hidden states.

    See :class:`ckt_wam.pipeline.ckt_pipeline_middle.CKTPipelineMiddle` for
    the recommended intermediate-layer variant used in the CKT-WAM paper.
    """

    def __init__(
        self,
        teacher_wam: nn.Module,
        student_model: nn.Module,
        config: CKTPipelineConfig,
    ):
        super().__init__()
        self.config = config

        self.teacher_extractor = TeacherFeatureExtractor(teacher_wam)

        self.adapter_bank = AdapterBank(
            teacher_hidden_dim=config.teacher_hidden_dim,
            student_hidden_dim=config.student_hidden_dim,
            adapter_bottleneck_dim=config.adapter_bottleneck_dim,
            adapter_dropout=config.adapter_dropout,
            num_adapter_output_tokens=config.num_adapter_output_tokens,
            num_specialized_experts=config.num_specialized_experts,
            top_k=config.top_k,
            gating_hidden_dim=config.gating_hidden_dim,
        )

        self.student = student_model

        self._context_to_inject: Optional[torch.Tensor] = None
        self._hook_handle = self.student.net.text_embedding.register_forward_hook(
            self._inject_context_hook
        )

    def _inject_context_hook(
        self,
        module: nn.Module,
        input: Any,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if self._context_to_inject is not None:
            return torch.cat([self._context_to_inject, output], dim=1)
        return output

    def training_step(
        self,
        data_batch: Dict[str, torch.Tensor],
        iteration: int,
        teacher_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        if teacher_inputs is None:
            raise ValueError(
                "teacher_inputs is required. Provide the teacher WAM's "
                "formatted dict (images, text, text_attention_mask, action, "
                "state, embodiment_id, has_real_action, action_mask)."
            )

        h_teacher = self.teacher_extractor(teacher_inputs)

        c_agg, expert_probs = self.adapter_bank(h_teacher.float())

        self._context_to_inject = c_agg

        output_batch, student_loss = self.student.training_step(
            data_batch, iteration
        )

        self._context_to_inject = None

        output_batch["expert_probs"] = expert_probs
        output_batch["adapter_context_norm"] = c_agg.norm(dim=-1).mean()

        return output_batch, student_loss, expert_probs

    @torch.no_grad()
    def generate_samples(
        self,
        data_batch: Dict[str, torch.Tensor],
        teacher_inputs: Optional[Dict[str, torch.Tensor]] = None,
        **sampling_kwargs,
    ) -> torch.Tensor:
        if teacher_inputs is None:
            raise ValueError("teacher_inputs is required for generate_samples.")

        with torch.enable_grad():
            h_teacher = self.teacher_extractor(teacher_inputs)

        c_agg, _ = self.adapter_bank(h_teacher.float())

        self._context_to_inject = c_agg

        samples = self.student.generate_samples_from_batch(
            data_batch, **sampling_kwargs
        )

        self._context_to_inject = None
        return samples

    def get_trainable_parameters(self):
        params = []
        params.extend(self.adapter_bank.parameters())
        for p in self.student.parameters():
            if p.requires_grad:
                params.append(p)
        return params

    def freeze_teacher(self):
        self.teacher_extractor.teacher_wam.eval()
        for p in self.teacher_extractor.parameters():
            p.requires_grad = False

    def print_param_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        adapter_params = sum(p.numel() for p in self.adapter_bank.parameters())
        student_trainable = sum(
            p.numel() for p in self.student.parameters() if p.requires_grad
        )
        teacher_params = sum(
            p.numel() for p in self.teacher_extractor.parameters()
        )

        logger.info("CKT pipeline parameter summary (last-block teacher):")
        logger.info(f"  Total params:        {total:>12,}")
        logger.info(f"  Trainable params:    {trainable:>12,}")
        logger.info(f"  Teacher (frozen):    {teacher_params:>12,}")
        logger.info(f"  Adapter bank:        {adapter_params:>12,}")
        logger.info(f"  Student (frozen):    {student_trainable:>12,}")

    def cleanup(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self.teacher_extractor.cleanup()
