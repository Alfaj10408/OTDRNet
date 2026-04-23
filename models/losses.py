"""
models/losses.py
──────────────────────────────────────────────────────────────
Loss functions for OT-guided CS-MRI reconstruction.

Total loss:
    L = L1(pred, target) + λ · L_ot

  L1    : pixel-wise absolute difference — stable for MRI restoration
  L_ot  : OT regularization from OT-DPL — enforces prompt discrimination

Optional metrics (not used in backprop):
  PSNR, SSIM — computed at eval time via scripts/compute_metrics.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class TotalLoss(nn.Module):
    """
    Combined training loss.

    Args:
        lambda_ot : weight for OT regularization term (default 0.1)
    """
    def __init__(self, lambda_ot: float = 0.1):
        super().__init__()
        self.lambda_ot = lambda_ot
        self.l1 = L1Loss()

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        L_ot:   torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        l1   = self.l1(pred, target)
        loss = l1 + self.lambda_ot * L_ot

        return loss, {
            "loss_total": loss.item(),
            "loss_l1":    l1.item(),
            "loss_ot":    L_ot.item(),
        }