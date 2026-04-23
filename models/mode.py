"""
models/mode.py
──────────────────────────────────────────────────────────────
★ NOVELTY 2 — Mixture of Degradation-aware Experts (MoDE)

Key idea:
  - Prompt P_hat from OT-DPL conditions a filter activation branch
    that computes group-wise correlation f_i = x_a^T · P_hat_i
  - A router selects the top-1 expert (non-shared conv blocks)
    based on global average pooling of base features
  - Different R-level inputs route to different experts →
    avoids gradient conflict between acceleration factors

Pipeline:
  x ──┬── proj_a ──────────────────────────── filter activation f_i
      │                                              ↓
      └── proj_b → dw_conv → base Fx ──── L(f_i) modulation
                                                     ↓
                              router (gap + FC) → top-1 expert
                                                     ↓
                              weighted expert output + residual

Reference: DaPT (Wei et al., TIP 2025) — Eq. 10-11
Extended with: explicit top-k routing, load balance awareness

Fix: n_groups must evenly divide C at every decoder level.
     We default n_groups=1 (safe for all channel widths).
     Users can increase it as long as C % n_groups == 0.
Fix: scatter_() dtype must match under AMP — cast softmax output
     to scores.dtype before scatter_.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertBlock(nn.Module):
    """
    Single expert: lightweight conv block with non-shared parameters.
    Each expert specializes in a different R-level degradation pattern.
    """
    def __init__(self, C: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(C, C, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoDE(nn.Module):
    """
    Mixture of Degradation-aware Experts module.

    Args:
        C          : input channel dimension
        n_experts  : number of expert networks  (default 5)
        n_groups   : groups for group-wise filter correlation (default 1)
                     Must divide C evenly.  Default=1 (always safe).
        top_k      : number of experts to activate per forward pass (default 1)

    Forward inputs:
        x     : [B, C, H, W]  feature map from previous block
        P_hat : [B, N, C]     degradation-aware prompts from OT-DPL

    Forward output:
        out   : [B, C, H, W]  modulated + expert-routed features
    """

    def __init__(
        self,
        C:         int = 48,
        n_experts: int = 5,
        n_groups:  int = 1,
        top_k:     int = 1,
    ):
        super().__init__()
        # Ensure n_groups divides C — fall back to 1 if not
        if C % n_groups != 0:
            n_groups = 1
        self.C         = C
        self.n_experts = n_experts
        self.n_groups  = n_groups
        self.group_dim = C // n_groups
        self.top_k     = top_k

        # Branch A: filter activation (prompt-conditioned)
        self.proj_a = nn.Conv2d(C, C, 1, bias=False)

        # Branch B: base feature extraction
        self.proj_b = nn.Conv2d(C, C, 1, bias=False)
        self.dw_b   = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False)

        # Linear alignment: group-wise correlation → C channels
        self.fi_align = nn.Conv2d(n_groups, C, 1, bias=False)

        # Expert bank (non-shared parameters)
        self.experts = nn.ModuleList([ExpertBlock(C) for _ in range(n_experts)])

        # Router: global context → expert weights
        self.router = nn.Linear(C, n_experts)

    def _filter_activation(
        self,
        xa:    torch.Tensor,   # [B, C, H, W]
        P_hat: torch.Tensor,   # [B, N, C]  — already projected to this level's C
    ) -> torch.Tensor:
        """
        Compute group-wise prompt-conditioned filter activation.

        f_i(u) = x̂_a(u)^T · P̂_i(u)   for each group u  (DaPT Eq. 10)

        xa    : [B, C, H, W]
        P_hat : [B, N, C]
        Returns fi : [B, C, H, W]  (after linear alignment)
        """
        B, C, H, W = xa.shape

        # Avg prompts across N → [B, C]
        p_avg = P_hat.mean(dim=1)                                # [B, C]

        # Reshape to groups: [B, n_groups, group_dim, H*W]
        xa_g = xa.view(B, self.n_groups, self.group_dim, H * W)
        p_g  = p_avg.view(B, self.n_groups, self.group_dim)

        # Group-wise dot product → [B, n_groups, H, W]
        fi = torch.einsum("bgc,bgcn->bgn", p_g, xa_g)
        fi = fi.view(B, self.n_groups, H, W)

        # Align to C channels
        fi = self.fi_align(fi)                                   # [B, C, H, W]
        return fi

    def forward(
        self,
        x:     torch.Tensor,   # [B, C, H, W]
        P_hat: torch.Tensor,   # [B, N, C]  — projected to match x's C
    ) -> torch.Tensor:
        B, C, H, W = x.shape

        # ── Branch A: prompt-conditioned filter ───────────────
        xa = self.proj_a(x)                                      # [B, C, H, W]
        fi = self._filter_activation(xa, P_hat)                  # [B, C, H, W]

        # ── Branch B: base feature with local enrichment ──────
        xb = self.dw_b(self.proj_b(x))                          # [B, C, H, W]

        # ── Modulate base features with filter activation ──────
        Fx = xb + fi                                             # [B, C, H, W]

        # ── Routing: global pooling → expert scores ───────────
        gap    = F.adaptive_avg_pool2d(Fx, 1).squeeze(-1).squeeze(-1)  # [B, C]
        scores = self.router(gap)                                # [B, n_experts]

        topk_val, topk_idx = scores.topk(self.top_k, dim=-1)    # [B, k]
        G = torch.zeros_like(scores)                             # [B, n_experts]

        # FIX: cast softmax output to scores.dtype before scatter_
        # Under AMP scores is float16; torch.compile enforces that
        # scatter_ src dtype == self dtype exactly — no silent cast.
        G.scatter_(1, topk_idx,
                   F.softmax(topk_val, dim=-1).to(dtype=scores.dtype))

        # ── Aggregate expert outputs ───────────────────────────
        out = torch.zeros_like(Fx)
        for i, expert in enumerate(self.experts):
            weight = G[:, i].view(B, 1, 1, 1)                   # [B,1,1,1]
            # Only compute expert if at least one sample routes to it
            if weight.sum() > 0:
                out = out + weight * expert(Fx)

        return out + x                                           # residual