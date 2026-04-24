"""
Loss functions for CKT-WAM training.

Implements the auxiliary objectives described in the CKT-WAM paper:

  1. ``ActionChunkLoss``   -- optional direct L1 / MSE action supervision on
                              extracted action chunks, used for ablations.
  2. ``LoadBalancingLoss`` -- ``L_bal = M * sum_m f_m P_m`` (Eq. (14)),
                              preventing routing collapse across the M
                              specialized experts.
  3. ``CKTLoss``           -- combines the student WAM's native EDM denoising
                              objective ``L_CKT = L_act + lambda_vid * L_vid``
                              with the auxiliary load-balancing term (Eq. (15)).

The student's native latent diffusion objective already encodes both
future-action and future-video supervision; this module only adds the
load-balancing regularizer and an optional action chunk loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CKTLossConfig:
    """Hyperparameters for ``CKTLoss``."""

    load_balance_weight: float = 0.01
    action_loss_type: str = "l1"
    num_experts: int = 4


class LoadBalancingLoss(nn.Module):
    """
    Switch-Transformer-style load balancing loss.

    For each batch of softmax routing probabilities ``p_{b,m}`` produced by the
    ``DynamicRouter``, this computes::

        f_m = fraction of instances whose top-1 expert is m
        P_m = 1/B * sum_b p_{b,m}
        L   = M * sum_m f_m P_m

    which is minimized (and equal to 1) when routing is uniform across all
    ``M`` experts, and grows whenever the router collapses onto a small
    subset of experts.
    """

    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, expert_probs: torch.Tensor) -> torch.Tensor:
        top_expert = expert_probs.argmax(dim=-1)
        expert_counts = torch.zeros(
            self.num_experts,
            device=expert_probs.device,
            dtype=expert_probs.dtype,
        )
        expert_counts.scatter_add_(
            0, top_expert, torch.ones_like(top_expert, dtype=expert_probs.dtype)
        )
        f_i = expert_counts / expert_probs.shape[0]

        p_i = expert_probs.mean(dim=0)

        loss = self.num_experts * (f_i * p_i).sum()
        return loss


class ActionChunkLoss(nn.Module):
    """
    Optional direct action chunk loss (``l1`` or ``mse``).

    The primary CKT-WAM training signal comes from the student's EDM
    denoising objective ``L_CKT``; this module provides an auxiliary term
    that can be enabled when explicit action supervision is available
    (e.g. for ablations or when the student exposes decoded action chunks).
    """

    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        assert loss_type in ("l1", "mse"), f"Unsupported loss_type: {loss_type}"
        self.loss_type = loss_type

    def forward(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.loss_type == "l1":
            per_elem_loss = F.l1_loss(
                pred_actions, target_actions, reduction="none"
            )
        else:
            per_elem_loss = F.mse_loss(
                pred_actions, target_actions, reduction="none"
            )

        if mask is not None:
            per_elem_loss = per_elem_loss * mask.unsqueeze(-1)
            return per_elem_loss.sum() / (
                mask.sum() * pred_actions.shape[-1] + 1e-8
            )

        return per_elem_loss.mean()


class CKTLoss(nn.Module):
    """
    Total CKT-WAM training objective.

    ::

        L = L_CKT + lambda_bal * L_bal

    where ``L_CKT`` is the student WAM's native EDM denoising loss on future
    actions (and, when enabled, future video) already returned by the
    student's ``training_step``.  An optional ``ActionChunkLoss`` can be
    added on top for ablations.
    """

    def __init__(self, config: CKTLossConfig):
        super().__init__()
        self.config = config
        self.load_balance_loss = LoadBalancingLoss(num_experts=config.num_experts)
        self.action_chunk_loss = ActionChunkLoss(loss_type=config.action_loss_type)

    def forward(
        self,
        student_loss: torch.Tensor,
        expert_probs: torch.Tensor,
        pred_actions: torch.Tensor | None = None,
        target_actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        lb_loss = self.load_balance_loss(expert_probs)

        total_loss = student_loss + self.config.load_balance_weight * lb_loss

        loss_dict = {
            "total_loss": total_loss.detach(),
            "student_edm_loss": student_loss.detach(),
            "load_balance_loss": lb_loss.detach(),
            "load_balance_weight": torch.tensor(self.config.load_balance_weight),
        }

        if pred_actions is not None and target_actions is not None:
            aux_action_loss = self.action_chunk_loss(pred_actions, target_actions)
            total_loss = total_loss + aux_action_loss
            loss_dict["aux_action_loss"] = aux_action_loss.detach()
            loss_dict["total_loss"] = total_loss.detach()

        return total_loss, loss_dict
