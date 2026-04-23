"""
models/net.py
──────────────────────────────────────────────────────────────
OTDRNet — OT-Guided Dynamic Routing Network for CS-MRI
"""

from __future__ import annotations
import torch
import torch.nn as nn

from .restormer_block import (
    RestormerBlock, OverlapPatchEmbed, Downsample, Upsample
)
from .ot_dpl import OT_DPL
from .mode  import MoDE
from .sre   import SRE


# ── Prompt-Guided Transformer Block (PTB) ────────────────────

class PTB(nn.Module):
    """
    One decoder block = MoDE (degradation routing) + SRE (spatial refine).
    Receives skip connection from encoder via channel concat + conv.

    P_hat fed here is already projected to match channel dim C.
    """
    def __init__(self, C: int, n_blocks: int,
                 n_experts: int = 5, top_k: int = 1):
        super().__init__()
        self.skip_conv = nn.Conv2d(C * 2, C, 1, bias=False)
        self.modes = nn.ModuleList([
            MoDE(C, n_experts=n_experts, top_k=top_k)
            for _ in range(n_blocks)
        ])
        self.sres = nn.ModuleList([
            SRE(C) for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor,
                skip: torch.Tensor,
                P_hat: torch.Tensor) -> torch.Tensor:
        # Fuse encoder skip connection
        x = self.skip_conv(torch.cat([x, skip], dim=1))
        for mode, sre in zip(self.modes, self.sres):
            x = mode(x, P_hat)
            x = sre(x, P_hat)
        return x


# ── Full Network ──────────────────────────────────────────────

class OTDRNet(nn.Module):
    """
    OT-Guided Dynamic Routing Network.

    Args:
        in_c        : input channels (1 for grayscale MRI)
        C           : base channel width (default 48)
        enc_blocks  : Restormer block counts per level [L0, L1, L2, L3]
        num_heads   : attention heads per level
        n_prompts   : number of OT-DPL prompts
        n_experts   : MoDE expert count
        top_k       : top-k expert routing
        eps_ot      : Sinkhorn entropy parameter
        iters_ot    : Sinkhorn iterations
    """

    def __init__(
        self,
        in_c:       int   = 1,
        C:          int   = 48,
        enc_blocks: list  = [2, 3, 3, 4],
        num_heads:  list  = [1, 2, 4, 8],
        n_prompts:  int   = 5,
        n_experts:  int   = 5,
        top_k:      int   = 1,
        eps_ot:     float = 0.05,
        iters_ot:   int   = 100,
    ):
        super().__init__()
        L = enc_blocks

        # ── Shallow feature extraction ────────────────────────
        self.patch_embed = OverlapPatchEmbed(in_c, C)

        # ── Encoder ───────────────────────────────────────────
        self.enc1  = nn.Sequential(*[RestormerBlock(C,     num_heads[0]) for _ in range(L[0])])
        self.down1 = Downsample(C)        # C   → 2C,  H/2

        self.enc2  = nn.Sequential(*[RestormerBlock(C * 2, num_heads[1]) for _ in range(L[1])])
        self.down2 = Downsample(C * 2)   # 2C  → 4C,  H/4

        self.enc3  = nn.Sequential(*[RestormerBlock(C * 4, num_heads[2]) for _ in range(L[2])])
        self.down3 = Downsample(C * 4)   # 4C  → 8C,  H/8

        # ── Bottleneck ────────────────────────────────────────
        self.bottleneck = nn.Sequential(*[
            RestormerBlock(C * 8, num_heads[3]) for _ in range(L[3])
        ])

        # ── OT-DPL ────────────────────────────────────────────
        # Reduce bottleneck C*8 → C before DPL for efficiency.
        # P_hat output: [B, N, C]
        self.dpl_proj = nn.Conv2d(C * 8, C, 1, bias=False)
        self.ot_dpl   = OT_DPL(C, N=n_prompts, eps=eps_ot, iters=iters_ot)

        # ── Per-level prompt projections ──────────────────────
        # P_hat is [B, N, C].  Each PTB level needs prompts
        # matching its own channel width so MoDE / SRE can work.
        #   ptb3 → C*4   ptb2 → C*2   ptb1 → C (no proj)
        self.p_proj3 = nn.Linear(C, C * 4)
        self.p_proj2 = nn.Linear(C, C * 2)
        # ptb1 consumes C directly — no extra projection

        # ── Decoder with PTBs ─────────────────────────────────
        self.up3  = Upsample(C * 8)      # 8C → 4C
        self.ptb3 = PTB(C * 4, L[2], n_experts, top_k)

        self.up2  = Upsample(C * 4)      # 4C → 2C
        self.ptb2 = PTB(C * 2, L[1], n_experts, top_k)

        self.up1  = Upsample(C * 2)      # 2C → C
        self.ptb1 = PTB(C,     L[0], n_experts, top_k)

        # ── Output projection ─────────────────────────────────
        self.out_conv = nn.Conv2d(C, in_c, 3, padding=1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """
        x    : [B, 1, H, W]   LQ undersampled MRI
        Returns:
            pred  [B, 1, H, W]   restored MRI
            L_ot  scalar          OT regularization loss
        """
        inp = x

        # ── Encoder ───────────────────────────────────────────
        x  = self.patch_embed(x)   # [B, C,   H,   W]
        s1 = self.enc1(x)          # [B, C,   H,   W]
        x  = self.down1(s1)        # [B, 2C,  H/2, W/2]
        s2 = self.enc2(x)          # [B, 2C,  H/2, W/2]
        x  = self.down2(s2)        # [B, 4C,  H/4, W/4]
        s3 = self.enc3(x)          # [B, 4C,  H/4, W/4]
        x  = self.down3(s3)        # [B, 8C,  H/8, W/8]

        # ── Bottleneck ────────────────────────────────────────
        Fd = self.bottleneck(x)    # [B, 8C,  H/8, W/8]

        # ── OT-DPL → degradation-aware prompts ───────────────
        Fd_r        = self.dpl_proj(Fd)    # [B, C,   H/8, W/8]
        P_hat, L_ot = self.ot_dpl(Fd_r)   # [B, N,   C],  scalar

        # ── Project prompts to each decoder level's channel dim ─
        # p_proj3/2 operate on last dim (C), broadcast over N
        P3 = self.p_proj3(P_hat)   # [B, N, 4C]
        P2 = self.p_proj2(P_hat)   # [B, N, 2C]
        P1 = P_hat                 # [B, N, C]   (no projection needed)

        # ── Decoder ───────────────────────────────────────────
        x = self.up3(Fd)                   # [B, 4C,  H/4, W/4]
        x = self.ptb3(x, s3, P3)          # [B, 4C,  H/4, W/4]

        x = self.up2(x)                    # [B, 2C,  H/2, W/2]
        x = self.ptb2(x, s2, P2)          # [B, 2C,  H/2, W/2]

        x = self.up1(x)                    # [B, C,   H,   W]
        x = self.ptb1(x, s1, P1)          # [B, C,   H,   W]

        # ── Global residual output ─────────────────────────────
        residual = self.out_conv(x)        # [B, 1, H, W]
        pred     = inp + residual

        return pred, L_ot


# ── Quick sanity check ────────────────────────────────────────
if __name__ == "__main__":
    model = OTDRNet(C=48, enc_blocks=[2, 3, 3, 4])
    x = torch.randn(1, 1, 256, 256)
    pred, L_ot = model(x)
    print(f"Output : {pred.shape}")
    print(f"L_ot   : {L_ot.item():.4f}")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params : {n/1e6:.2f} M")