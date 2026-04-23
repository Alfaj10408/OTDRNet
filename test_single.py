"""
test_single.py
──────────────────────────────────────────────────────────────
Quick sanity check: load one LQ slice, run forward pass,
print PSNR, and save a side-by-side PNG.

Usage:
    # With no checkpoint (random weights — just tests shapes):
    python test_single.py --lq data/processed/test/LQ/IXI002-Guys-0828-T2_slice050.npy
                          --hq data/processed/test/HQ/IXI002-Guys-0828-T2_slice050.npy

    # With checkpoint:
    python test_single.py --lq ... --hq ... \
        --ckpt experiments/checkpoints/otdr_mri_default/ckpt_0200000.pth
"""

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import OTDRNet
from scripts.compute_metrics import compute_psnr, compute_ssim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lq",   type=str, required=True)
    parser.add_argument("--hq",   type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--out",  type=str, default="test_output.png")
    # Model defaults (match default.yaml)
    parser.add_argument("--C",         type=int,   default=48)
    parser.add_argument("--n_prompts", type=int,   default=5)
    parser.add_argument("--n_experts", type=int,   default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load data ─────────────────────────────────────────────
    lq_np = np.load(args.lq).astype(np.float32)
    lq    = torch.from_numpy(lq_np).unsqueeze(0).unsqueeze(0).to(device)
    print(f"[test] Input  shape : {lq.shape}")

    hq_np = None
    if args.hq and os.path.exists(args.hq):
        hq_np = np.load(args.hq).astype(np.float32)

    # ── Model ─────────────────────────────────────────────────
    model = OTDRNet(
        C         = args.C,
        n_prompts = args.n_prompts,
        n_experts = args.n_experts,
    ).to(device)

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[test] Loaded weights: {args.ckpt}")
    else:
        print("[test] No checkpoint — using random weights (shape test only)")

    model.eval()

    # ── Forward ───────────────────────────────────────────────
    with torch.no_grad():
        pred, L_ot = model(lq)

    pred_np = pred[0, 0].cpu().clamp(0, 1).numpy()
    print(f"[test] Output shape : {pred.shape}")
    print(f"[test] L_ot         : {L_ot.item():.6f}")

    if hq_np is not None:
        psnr = compute_psnr(pred_np, hq_np)
        ssim = compute_ssim(pred_np, hq_np)
        print(f"[test] PSNR = {psnr:.4f} dB")
        print(f"[test] SSIM = {ssim:.4f}")

    # ── Visualize ─────────────────────────────────────────────
    n_panels = 3 if hq_np is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    fig.patch.set_facecolor("#0b0f1a")

    panels = [("LQ (input)", lq_np), ("Prediction", pred_np)]
    if hq_np is not None:
        panels.append(("HQ (ground truth)", hq_np))

    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, color="white", fontsize=11)
        ax.axis("off")

    if hq_np is not None:
        axes[-1].set_title(
            f"HQ (GT)\nPSNR={psnr:.2f}dB  SSIM={ssim:.4f}",
            color="lightgreen", fontsize=10
        )

    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight",
                facecolor="#0b0f1a")
    plt.close()
    print(f"[test] Saved figure → {args.out}")


if __name__ == "__main__":
    main()