"""
CKT-WAM adapter modules.

This module implements the core sparse parameter-efficient transfer block
described in the CKT-WAM paper: a shared projection-compression trunk that
maps the teacher WAM's intermediate hidden states into the student feature
space, an always-on generalized adapter, a lightweight router, and a pool of
sparsely activated specialized adapters.

Core components:
  - ``Adapter``        : learnable-query cross-attention compressor + bottleneck
                         residual MLP.  Implements the Eq. (1)-(3) pipeline in
                         the paper, mapping teacher hidden states
                         ``H_T : [B, N, d_tea]`` to a compact context
                         ``C : [B, K, d_stu]``.
  - ``DynamicRouter``  : top-k routing gate over ``M`` specialized experts
                         (Eq. (6)-(7)).
  - ``AdapterBank``    : full transfer module combining one generalized branch
                         and ``M`` specialized branches with route-first sparse
                         execution (Eq. (8)-(9)), producing the transferred
                         context ``C_A : [B, 2K, d_stu]`` (Eq. (10)).

Default dimensions reflect a Cosmos-Policy-2B student (d_stu = 2048) and a
DreamZero-14B teacher (d_tea = 5120).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AdapterConfig:
    """Configuration for a single CKT adapter branch."""

    teacher_hidden_dim: int = 5120
    student_hidden_dim: int = 2048
    adapter_bottleneck_dim: int = 1024
    adapter_dropout: float = 0.1
    num_adapter_output_tokens: int = 16


class Adapter(nn.Module):
    """
    Learnable-query compressor + residual bottleneck adapter.

    Given teacher hidden states ``H_T : [B, N, d_tea]``, this module first
    projects them into the student feature space through a shared bottleneck
    trunk (``W_down`` / ``W_up`` with GELU, LayerNorm and Dropout), and then
    compresses the resulting variable-length sequence into a fixed number of
    transferable tokens via multi-head cross-attention with a learnable
    query bank.

    Output:  ``C : [B, K, d_stu]`` -- a compact context that can be routed
    through the generalized or specialized branch of the adapter bank.
    """

    def __init__(self, config: AdapterConfig):
        super().__init__()
        t_dim = config.teacher_hidden_dim
        s_dim = config.student_hidden_dim
        b_dim = config.adapter_bottleneck_dim
        n_out = config.num_adapter_output_tokens

        self.down_proj = nn.Linear(t_dim, b_dim)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(b_dim, s_dim)
        self.layer_norm = nn.LayerNorm(s_dim)
        self.dropout = nn.Dropout(config.adapter_dropout)

        self.query_tokens = nn.Parameter(torch.randn(1, n_out, s_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=s_dim,
            num_heads=8,
            dropout=config.adapter_dropout,
            batch_first=True,
        )
        self.post_attn_norm = nn.LayerNorm(s_dim)

    def forward(self, h_teacher: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_teacher: ``[B, N, d_tea]`` teacher intermediate-layer hidden states.
        Returns:
            context:   ``[B, K, d_stu]`` compressed transferable context tokens.
        """
        B = h_teacher.shape[0]

        h = self.down_proj(h_teacher)
        h = self.act(h)
        h = self.up_proj(h)
        h = self.layer_norm(h)
        h = self.dropout(h)

        queries = self.query_tokens.expand(B, -1, -1)
        context, _ = self.cross_attn(query=queries, key=h, value=h)

        context = self.post_attn_norm(context + queries)
        return context


class DynamicRouter(nn.Module):
    """
    Lightweight top-k router over specialized adapters.

    Operates on a mean-pooled teacher summary and returns the top-k expert
    indices together with their renormalized weights, plus the full softmax
    probabilities needed by the load-balancing auxiliary loss
    (Eq. (11)-(13) in the paper).
    """

    def __init__(
        self,
        teacher_hidden_dim: int = 5120,
        num_experts: int = 4,
        top_k: int = 2,
        gating_hidden_dim: int = 512,
    ):
        super().__init__()
        assert top_k <= num_experts, (
            f"top_k ({top_k}) must be <= num_experts ({num_experts})"
        )

        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Sequential(
            nn.Linear(teacher_hidden_dim, gating_hidden_dim),
            nn.GELU(),
            nn.Linear(gating_hidden_dim, num_experts),
        )
        self.noise_weight = nn.Parameter(torch.zeros(1))

    def forward(
        self, h_pooled: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_pooled: ``[B, d_tea]`` mean-pooled teacher hidden states.
        Returns:
            top_k_weights:  ``[B, top_k]``  renormalized weights for selected experts.
            top_k_indices:  ``[B, top_k]``  indices of selected experts.
            expert_probs:   ``[B, M]``      full softmax probabilities (for L_bal).
        """
        logits = self.gate(h_pooled)

        if self.training:
            noise = torch.randn_like(logits) * F.softplus(self.noise_weight)
            logits = logits + noise

        expert_probs = F.softmax(logits, dim=-1)

        top_k_weights, top_k_indices = torch.topk(
            expert_probs, self.top_k, dim=-1
        )
        top_k_weights = top_k_weights / (
            top_k_weights.sum(dim=-1, keepdim=True) + 1e-8
        )

        return top_k_weights, top_k_indices, expert_probs


class AdapterBank(nn.Module):
    """
    Sparse parameter-efficient CKT transfer module.

    Structure:
      1) one always-on **generalized adapter** that captures task-agnostic
         transferable structure;
      2) ``M`` **specialized adapters** that model input-dependent transfer
         patterns;
      3) a lightweight **router** that selects the top-k specialized branches
         per instance.

    Given teacher hidden states ``H_T : [B, N, d_tea]``, the module returns
    the transferred context

        ``C_A = [ C_g ; C_s ] in R^{B x 2K x d_stu}``     (Eq. (10))

    together with the full routing probabilities used by the load-balancing
    auxiliary loss.
    """

    def __init__(
        self,
        teacher_hidden_dim: int = 5120,
        student_hidden_dim: int = 2048,
        adapter_bottleneck_dim: int = 1024,
        adapter_dropout: float = 0.1,
        num_adapter_output_tokens: int = 16,
        num_specialized_experts: int = 4,
        top_k: int = 2,
        gating_hidden_dim: int = 512,
    ):
        super().__init__()
        self.num_specialized_experts = num_specialized_experts
        self.top_k = top_k

        adapter_cfg = AdapterConfig(
            teacher_hidden_dim=teacher_hidden_dim,
            student_hidden_dim=student_hidden_dim,
            adapter_bottleneck_dim=adapter_bottleneck_dim,
            adapter_dropout=adapter_dropout,
            num_adapter_output_tokens=num_adapter_output_tokens,
        )

        self.generalized_adapter = Adapter(adapter_cfg)

        self.specialized_adapters = nn.ModuleList(
            [Adapter(adapter_cfg) for _ in range(num_specialized_experts)]
        )

        self.router = DynamicRouter(
            teacher_hidden_dim=teacher_hidden_dim,
            num_experts=num_specialized_experts,
            top_k=top_k,
            gating_hidden_dim=gating_hidden_dim,
        )

    def forward(
        self, h_teacher: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_teacher: ``[B, N, d_tea]`` teacher hidden states from the chosen
                       intermediate transformer block.
        Returns:
            c_agg:        ``[B, 2K, d_stu]`` transferred context ``C_A``.
            expert_probs: ``[B, M]`` full routing probabilities.
        """
        B = h_teacher.shape[0]

        c_g = self.generalized_adapter(h_teacher)

        c_specialists = torch.stack(
            [adapter(h_teacher) for adapter in self.specialized_adapters], dim=1
        )

        h_pooled = h_teacher.mean(dim=1)
        top_k_weights, top_k_indices, expert_probs = self.router(h_pooled)

        num_tokens = c_g.shape[1]
        student_dim = c_g.shape[2]

        gather_idx = top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(
            B, self.top_k, num_tokens, student_dim
        )

        selected_c_s = torch.gather(c_specialists, dim=1, index=gather_idx)

        weighted_c_s = (
            selected_c_s * top_k_weights.unsqueeze(-1).unsqueeze(-1)
        )
        c_s_combined = weighted_c_s.sum(dim=1)

        c_agg = torch.cat([c_g, c_s_combined], dim=1)

        return c_agg, expert_probs
