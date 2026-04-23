from __future__ import annotations

"""
models/restormer_block.py
──────────────────────────────────────────────────────────────
Restormer Transformer Block used as the encoder backbone.
Reference: Zamir et al., CVPR 2022 — "Restormer: Efficient
Transformer for High-Resolution Image Restoration"

Components:
  - MDTA  : Multi-Dconv Head Transposed Attention
  - GDFN  : Gated-Dconv Feed-Forward Network
  - RestormerBlock : MDTA + GDFN with LayerNorm
  - OverlapPatchEmbed : shallow feature extraction
  - Downsample / Upsample : encoder-decoder transitions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ── Helpers ───────────────────────────────────────────────────

class LayerNorm(nn.Module):
    """Channel-first LayerNorm: input [B, C, H, W]."""
    def __init__(self, C, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(C))
        self.bias   = nn.Parameter(torch.zeros(C))
        self.eps    = eps

    def forward(self, x):
        # x: [B, C, H, W]
        mean = x.mean(1, keepdim=True)
        var  = (x - mean).pow(2).mean(1, keepdim=True)
        x    = (x - mean) / (var + self.eps).sqrt()
        return self.weight[:, None, None] * x + self.bias[:, None, None]


# ── MDTA : Multi-Dconv Head Transposed Attention ──────────────

class MDTA(nn.Module):
    """
    Transposed self-attention across channels (not spatial tokens).
    Complexity O(C²·HW) instead of O((HW)²·C) — efficient for
    high-resolution medical images.
    """
    def __init__(self, C, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv      = nn.Conv2d(C, C * 3, 1, bias=bias)
        self.qkv_dw   = nn.Conv2d(C * 3, C * 3, 3, padding=1,
                                   groups=C * 3, bias=bias)
        self.proj_out = nn.Conv2d(C, C, 1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_dw(self.qkv(x))                # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)                 # each [B, C, H, W]

        # Reshape to heads
        q = rearrange(q, "b (h c) x y -> b h c (x y)", h=self.num_heads)
        k = rearrange(k, "b (h c) x y -> b h c (x y)", h=self.num_heads)
        v = rearrange(v, "b (h c) x y -> b h c (x y)", h=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Transposed attention: C×C instead of (HW)×(HW)
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # [B, h, C, C]
        attn = attn.softmax(dim=-1)
        out  = attn @ v                                # [B, h, C, HW]

        out = rearrange(out, "b h c (x y) -> b (h c) x y", x=H, y=W)
        return self.proj_out(out)


# ── GDFN : Gated-Dconv Feed-Forward Network ──────────────────

class GDFN(nn.Module):
    """
    Gated depth-wise conv feed-forward network.
    Replaces standard FFN with gating for richer feature modulation.
    """
    def __init__(self, C, ffn_expansion=2.66, bias=False):
        super().__init__()
        hidden = int(C * ffn_expansion)
        self.proj_in  = nn.Conv2d(C, hidden * 2, 1, bias=bias)
        self.dw_conv  = nn.Conv2d(hidden * 2, hidden * 2, 3,
                                   padding=1, groups=hidden * 2, bias=bias)
        self.proj_out = nn.Conv2d(hidden, C, 1, bias=bias)

    def forward(self, x):
        x1, x2 = self.dw_conv(self.proj_in(x)).chunk(2, dim=1)
        return self.proj_out(F.gelu(x1) * x2)


# ── Restormer Block ───────────────────────────────────────────

class RestormerBlock(nn.Module):
    """One Restormer Transformer Block: LayerNorm → MDTA → LayerNorm → GDFN."""
    def __init__(self, C, num_heads, ffn_expansion=2.66, bias=False):
        super().__init__()
        self.norm1 = LayerNorm(C)
        self.attn  = MDTA(C, num_heads, bias)
        self.norm2 = LayerNorm(C)
        self.ffn   = GDFN(C, ffn_expansion, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ── Overlap Patch Embed ───────────────────────────────────────

class OverlapPatchEmbed(nn.Module):
    """3×3 conv to extract shallow features from input image."""
    def __init__(self, in_c=1, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


# ── Downsample / Upsample ─────────────────────────────────────

class Downsample(nn.Module):
    """Pixel-unshuffle: [B, C, H, W] → [B, 2C, H/2, W/2]."""
    def __init__(self, C):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(C, C // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """Pixel-shuffle: [B, C, H, W] → [B, C//2, 2H, 2W]."""
    def __init__(self, C):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(C, C * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)