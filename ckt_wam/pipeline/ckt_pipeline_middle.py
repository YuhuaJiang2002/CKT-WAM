"""
CKT-WAM pipeline with **intermediate-layer** teacher features (default).

This is the main pipeline used in the CKT-WAM paper: hidden states are
extracted from a chosen intermediate transformer block ``l^*`` of the
frozen teacher WAM (``l^* = 20`` out of 40 layers for DreamZero-14B), which
provides a better cost--utility trade-off than relying on the deepest
teacher layer.

The extracted hidden states ``H_T : [B, N, d_tea]`` are passed through the
``AdapterBank`` to produce the compact transferred context
``C_A : [B, 2K, d_stu]`` (Eq. (10)).  A forward hook on the student's
text-embedding module then assembles the augmented conditioning sequence

    ``E_tilde = [ E_t ; <SEP> ; C_A ] in R^{B x (L_t + 2K + 1) x d_stu}``  (Eq. (11))

where ``E_t`` is the student's original textual embedding, ``<SEP>`` is a
learnable separator stored on the adapter bank, and ``C_A`` is the
teacher-derived context.  Both teacher and student backbones remain frozen;
only the adapter bank (including the ``<SEP>`` token) is optimized.
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
class CKTPipelineMiddleConfig:
    """Configuration for the intermediate-layer CKT pipeline."""

    teacher_hidden_dim: int = 5120
    teacher_text_dim: int = 4096
    teacher_clip_dim: int = 1280

    teacher_block_index: int = 20

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


class TeacherMiddleLayerFeatureExtractor(nn.Module):
    """
    Extract hidden states from an intermediate transformer block of the
    teacher WAM via a forward hook.

    The teacher's full forward pass (text / image encoding, VAE encoding,
    noise sampling, full transformer stack) is executed so that the
    captured hidden states ``H_T`` reflect the teacher's full
    representation at depth ``l^*``.
    """

    def __init__(self, teacher_wam: nn.Module, block_index: int = 20):
        super().__init__()
        self.block_index = block_index
        self.teacher_wam = teacher_wam
        self.teacher_wam.eval()
        for p in self.teacher_wam.parameters():
            p.requires_grad = False

        self._base_model = self._resolve_base_model()

        num_blocks = len(self._base_model.blocks)
        if not (0 <= self.block_index < num_blocks):
            raise ValueError(
                f"teacher_block_index must be in [0, {num_blocks}), got "
                f"{self.block_index}"
            )

        self._block_output: Optional[torch.Tensor] = None
        self._hook_handle = self._base_model.blocks[
            self.block_index
        ].register_forward_hook(self._capture_block_hook)
        logger.info(
            "TeacherMiddleLayerFeatureExtractor: registered hook on "
            "transformer block index %d of %d",
            self.block_index,
            num_blocks,
        )

    def _resolve_base_model(self):
        """Navigate through a possible PEFT wrapper to the underlying model."""
        model = self.teacher_wam.action_head.model
        if hasattr(model, "base_model"):
            return model.base_model.model
        return model

    def _capture_block_hook(
        self,
        module: nn.Module,
        input: Any,
        output: Any,
    ) -> None:
        """Capture block output (sever gradient flow via ``detach``)."""
        x = output[0] if isinstance(output, tuple) else output
        self._block_output = x.detach()

    def forward(self, teacher_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run the teacher's full forward pass and return the hooked block output.

        Gradient handling matches the sibling full-depth extractor: we do
        not wrap the call in ``torch.no_grad()`` so that gradient
        checkpointing unpacks ``(x, kv_cache)`` correctly inside the
        teacher transformer stack.  All teacher parameters have
        ``requires_grad=False``, so no real computation graph is built.
        """
        orig_gc = getattr(self._base_model, "gradient_checkpointing", False)
        self._base_model.gradient_checkpointing = True

        try:
            self.teacher_wam.action_head(
                _AttrDict(),
                _AttrDict(teacher_inputs),
            )
        finally:
            self._base_model.gradient_checkpointing = orig_gc

        h_teacher = self._block_output
        self._block_output = None

        if h_teacher is None:
            raise RuntimeError(
                f"Forward hook on transformer block {self.block_index} did "
                "not fire. Verify that teacher_inputs triggers the teacher "
                "backbone's training forward path."
            )

        return h_teacher

    def cleanup(self):
        """Remove the forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


class CKTPipelineMiddle(nn.Module):
    """
    Full CKT-WAM pipeline with intermediate-layer teacher features.

    Components::

        Teacher WAM (frozen)
            -> TeacherMiddleLayerFeatureExtractor -> H_T   [B, N, d_tea]
        AdapterBank (trainable)
            -> (C_g, C_s) -> C_A                           [B, 2K, d_stu]
            -> learnable <SEP> token                       [1, 1, d_stu]
        Student WAM (frozen)
            -> conditioning sequence assembled by a forward hook on the
               student's text_embedding module:
                   E_tilde = [ E_t ; <SEP> ; C_A ]         (Eq. (11))

    During training, only the adapter bank is optimized; both teacher and
    student backbones remain frozen.  At inference time the teacher is
    executed once per observation (``t^* = 0``) and the resulting ``C_A``
    is reused across the entire student denoising trajectory, keeping the
    transfer overhead negligible.
    """

    def __init__(
        self,
        teacher_wam: nn.Module,
        student_model: nn.Module,
        config: CKTPipelineMiddleConfig,
    ):
        super().__init__()
        self.config = config

        self.teacher_extractor = TeacherMiddleLayerFeatureExtractor(
            teacher_wam,
            block_index=config.teacher_block_index,
        )

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
        """
        Build the augmented conditioning sequence (Eq. (11))::

            E_tilde = [ E_t ; <SEP> ; C_A ]

        where ``E_t`` is the student's original textual embedding output,
        ``<SEP>`` is the learnable separator stored on the adapter bank,
        and ``C_A`` is the transferred context cached in
        ``self._context_to_inject``.
        """
        if self._context_to_inject is None:
            return output

        B = output.shape[0]
        sep = self.adapter_bank.sep_token.expand(B, -1, -1).to(
            dtype=output.dtype, device=output.device
        )
        c_a = self._context_to_inject.to(
            dtype=output.dtype, device=output.device
        )
        return torch.cat([output, sep, c_a], dim=1)

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

        # Re-enable grad so the teacher's gradient-checkpointing path
        # correctly unpacks block outputs.
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

        logger.info(
            "CKT pipeline parameter summary (intermediate teacher layer %s):",
            self.config.teacher_block_index,
        )
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
