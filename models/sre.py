"""
models/sre.py
──────────────────────────────────────────────────────────────
Spatial Refinement Expert (SRE)

P_hat passed here is already projected to match x's channel dim C
(done in net.py via p_proj3 / p_proj2).
"""

from __future__ import annotations
import torch
import torch.nn as nn


class SRE(nn.Module):
    """
    Spatial Refinement Expert.

    Args:
        C           : channel dimension (must match feature map)
        kernel_size : large depthwise conv kernel (default 11)
    """

    def __init__(self, C: int, kernel_size: int = 11):
        super().__init__()
        pad = kernel_size // 2

        # Channel gating: prompt → per-channel scalar gate
        self.fc_gate = nn.Sequential(
            nn.Linear(C, C),
            nn.Sigmoid(),
        )

        # K, V point-wise projections
        self.kv_proj = nn.Conv2d(C, C * 2, 1, bias=False)

        # Large-kernel depthwise conv for long-range spatial context
        self.dconv = nn.Conv2d(
            C, C,
            kernel_size=kernel_size,
            padding=pad,
            groups=C,
            bias=False,
        )

        # Output projection
        self.out_proj = nn.Conv2d(C, C, 1, bias=False)

        # Normalization
        self.norm = nn.GroupNorm(1, C)

    def forward(
        self,
        x:     torch.Tensor,   # [B, C, H, W]
        P_hat: torch.Tensor,   # [B, N, C]  — projected to this level's C
    ) -> torch.Tensor:
        B, C, H, W = x.shape

        # Average prompts → [B, C] → gate
        p_avg = P_hat.mean(dim=1)                   # [B, C]
        Q = self.fc_gate(p_avg)                     # [B, C]
        Q = Q.view(B, C, 1, 1)                      # broadcast over H, W

        # K, V from input features
        kv   = self.kv_proj(x)                      # [B, 2C, H, W]
        K, V = kv.chunk(2, dim=1)                   # each [B, C, H, W]

        # Prompt-modulated spatial refinement
        K_refined = self.dconv(Q * K)               # [B, C, H, W]

        # Fuse and project
        out = self.out_proj(K_refined * V)          # [B, C, H, W]

        return self.norm(out) + x