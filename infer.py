"""
infer.py
──────────────────────────────────────────────────────────────
Run OTDRNet inference on:
  - a single LQ .npy slice
  - a folder of LQ .npy slices
  - the full test split in data/processed/test/LQ/

Outputs reconstructed images as .npy and/or .png side-by-side grids.

Usage examples:
  # Single slice (with ground truth for PSNR)
  python infer.py \
    --ckpt experiments/checkpoints/otdr_mri_default/ckpt_0190000.pth \
    --lq   data/processed/test/LQ/IXI012-HH-1211-T2_s050.npy \
    --hq   data/processed/test/HQ/IXI012-HH-1211-T2_s050.npy

  # Whole test folder
  python infer.py \
    --ckpt   experiments/checkpoints/otdr_mri_default/ckpt_0190000.pth \
    --lq_dir data/processed/test/LQ \
    --hq_dir data/processed/test/HQ \
    --out_dir experiments/results/infer_190k \
    --save_png \
    --n_vis 16

  # Quick comparison across checkpoints
  python infer.py \
    --ckpt   experiments/checkpoints/otdr_mri_default/ckpt_0100000.pth \
              experiments/checkpoints/otdr_mri_default/ckpt_0190000.pth \
    --lq   data/processed/test/LQ/IXI012-HH-1211-T2_s050.npy \
    --hq   data/processed/test/HQ/IXI012-HH-1211-T2_s050.npy
"""

import os
import sys
import argparse
import numpy as np
import torch
import yaml
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from models import OTDRNet
from scripts.compute_metrics import compute_psnr, compute_ssim


# ── Helpers ───────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> OTDRNet:
    """Load OTDRNet from checkpoint. Reads config saved inside the .pth if available."""
    ckpt = torch.load(ckpt_path, map_location=device)

    # Use config saved in checkpoint if present, else fall back to defaults
    cfg = ckpt.get("config", None)
    if cfg is not None:
        mc = cfg["model"]
        model = OTDRNet(
            in_c       = mc["in_c"],
            C          = mc["C"],
            enc_blocks = mc["enc_blocks"],
            num_heads  = mc["num_heads"],
            n_prompts  = mc["n_prompts"],
            n_experts  = mc["n_experts"],
            top_k      = mc["top_k"],
            eps_ot     = cfg["ot"]["eps"],
            iters_ot   = cfg["ot"].get("iters", 100),
        ).to(device)
    else:
        # Safe fallback — default architecture
        model = OTDRNet(C=48, enc_blocks=[2,3,3,4], num_heads=[1,2,4,8],
                        n_prompts=5, n_experts=5, top_k=1,
                        eps_ot=0.05, iters_ot=100).to(device)

    model.load_state_dict(ckpt["model"])
    iteration = ckpt.get("iteration", "?")
    print(f"  Loaded {Path(ckpt_path).name}  (iter {iteration})")
    return model


@torch.no_grad()
def reconstruct(model: OTDRNet,
                lq_np: np.ndarray,
                device: torch.device) -> np.ndarray:
    """
    Run one forward pass.
    lq_np : [H, W]  float32  in [0, 1]
    Returns pred : [H, W] float32 in [0, 1]
    """
    model.eval()
    lq = torch.from_numpy(lq_np).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        pred, _ = model(lq)
    return pred[0, 0].cpu().float().clamp(0, 1).numpy()


def error_map(pred: np.ndarray, hq: np.ndarray) -> np.ndarray:
    err = np.abs(pred - hq)
    if err.max() > 1e-8:
        err = err / err.max()
    return err


# ── Single-image mode ─────────────────────────────────────────

def infer_single(args, device):
    lq_np = np.load(args.lq).astype(np.float32)
    hq_np = np.load(args.hq).astype(np.float32) if args.hq else None

    ckpt_paths = args.ckpt if isinstance(args.ckpt, list) else [args.ckpt]
    n_ckpts    = len(ckpt_paths)
    n_panels   = 2 + n_ckpts + (1 if hq_np is not None else 0)

    fig, axes = plt.subplots(1, n_panels,
                             figsize=(4 * n_panels, 4.5),
                             facecolor="#0b0f1a")

    # LQ
    axes[0].imshow(lq_np, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("LQ Input", color="white", fontsize=10)
    axes[0].axis("off")

    pred_last = None
    for idx, ckpt_path in enumerate(ckpt_paths):
        model = load_model(ckpt_path, device)
        pred  = reconstruct(model, lq_np, device)

        pred_last = pred   # keep reference for error map
        ax = axes[1 + idx]
        ax.imshow(pred, cmap="gray", vmin=0, vmax=1)
        label = Path(ckpt_path).stem

        if hq_np is not None:
            psnr = compute_psnr(pred, hq_np)
            ssim = compute_ssim(pred, hq_np)
            ax.set_title(f"{label}\nPSNR={psnr:.2f}  SSIM={ssim:.4f}",
                         color="#34d399", fontsize=8)
        else:
            ax.set_title(label, color="#38bdf8", fontsize=9)
        ax.axis("off")

        # Save .npy
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            stem = Path(args.lq).stem
            np.save(os.path.join(args.out_dir, f"{stem}_{label}_pred.npy"), pred)

    if hq_np is not None:
        # Second-to-last panel: error map (abs difference, hot colormap)
        # dark = low error, bright = high error
        ax_err = axes[-2]
        err = error_map(pred_last, hq_np)
        ax_err.imshow(err, cmap="hot", vmin=0, vmax=1)
        ax_err.set_title("Error Map\n(brighter = worse)", color="#fb923c", fontsize=9)
        ax_err.axis("off")

        ax_hq = axes[-1]
        ax_hq.imshow(hq_np, cmap="gray", vmin=0, vmax=1)
        ax_hq.set_title("Ground Truth", color="white", fontsize=10)
        ax_hq.axis("off")

    plt.tight_layout()
    out_png = args.out or "infer_result.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="#0b0f1a")
    plt.close()
    print(f"\nSaved figure → {out_png}")


# ── Folder / test-split mode ──────────────────────────────────

def infer_folder(args, device):
    lq_dir = Path(args.lq_dir)
    hq_dir = Path(args.hq_dir) if args.hq_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lq_files = sorted(lq_dir.glob("*.npy"))
    print(f"Found {len(lq_files)} LQ slices in {lq_dir}")

    ckpt_path = args.ckpt[0] if isinstance(args.ckpt, list) else args.ckpt
    model = load_model(ckpt_path, device)

    psnr_list, ssim_list = [], []
    vis_samples = []

    for i, lq_path in enumerate(lq_files):
        lq_np = np.load(lq_path).astype(np.float32)
        pred  = reconstruct(model, lq_np, device)

        # Save .npy prediction
        np.save(out_dir / f"{lq_path.stem}_pred.npy", pred)

        if hq_dir:
            hq_path = hq_dir / lq_path.name
            if hq_path.exists():
                hq_np = np.load(hq_path).astype(np.float32)
                psnr  = compute_psnr(pred, hq_np)
                ssim  = compute_ssim(pred, hq_np)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                # Stash for visualisation grid
                if len(vis_samples) < args.n_vis:
                    vis_samples.append((lq_np, pred, hq_np, lq_path.stem))

        if (i + 1) % 500 == 0 or (i + 1) == len(lq_files):
            print(f"  Processed {i+1}/{len(lq_files)} slices", flush=True)

    # ── Metrics ───────────────────────────────────────────────
    if psnr_list:
        mean_psnr = np.mean(psnr_list)
        mean_ssim = np.mean(ssim_list)
        print(f"\n{'='*50}")
        print(f"  Test Results  ({len(psnr_list)} samples)")
        print(f"  PSNR : {mean_psnr:.4f} ± {np.std(psnr_list):.4f} dB")
        print(f"  SSIM : {mean_ssim:.4f} ± {np.std(ssim_list):.4f}")
        print(f"{'='*50}")

        # Save metrics to txt
        with open(out_dir / "metrics.txt", "w") as f:
            f.write(f"Checkpoint : {ckpt_path}\n")
            f.write(f"N samples  : {len(psnr_list)}\n")
            f.write(f"PSNR mean  : {mean_psnr:.4f}\n")
            f.write(f"PSNR std   : {np.std(psnr_list):.4f}\n")
            f.write(f"SSIM mean  : {mean_ssim:.4f}\n")
            f.write(f"SSIM std   : {np.std(ssim_list):.4f}\n")
        print(f"  Metrics saved → {out_dir / 'metrics.txt'}")

    # ── Visual grid ────────────────────────────────────────────
    if args.save_png and vis_samples:
        n      = len(vis_samples)
        n_cols = 4   # LQ | Pred | HQ | Error
        fig    = plt.figure(figsize=(4*n_cols, 4*n), facecolor="#0b0f1a")
        gs     = gridspec.GridSpec(n, n_cols, figure=fig,
                                   hspace=0.05, wspace=0.05)

        col_labels = ["LQ Input", "Prediction", "Ground Truth", "Error Map"]
        cmaps      = ["gray", "gray", "gray", "hot"]

        for r, (lq_np, pred, hq_np, stem) in enumerate(vis_samples):
            imgs = [lq_np, pred, hq_np, error_map(pred, hq_np)]
            psnr = compute_psnr(pred, hq_np)
            ssim = compute_ssim(pred, hq_np)

            for c, (img, cmap) in enumerate(zip(imgs, cmaps)):
                ax = fig.add_subplot(gs[r, c])
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
                ax.axis("off")
                if r == 0:
                    ax.set_title(col_labels[c], color="white", fontsize=9, pad=3)
                if c == 1:
                    ax.set_title(f"PSNR={psnr:.2f}  SSIM={ssim:.3f}",
                                 color="#34d399", fontsize=8, pad=2)

        grid_path = out_dir / "reconstruction_grid.png"
        fig.savefig(grid_path, dpi=130, bbox_inches="tight", facecolor="#0b0f1a")
        plt.close()
        print(f"  Grid saved → {grid_path}")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OTDRNet inference — single slice or full folder"
    )

    # Mode A: single slice
    parser.add_argument("--lq",      type=str, default=None,
                        help="Path to single LQ .npy slice")
    parser.add_argument("--hq",      type=str, default=None,
                        help="Path to HQ .npy slice (optional, for PSNR)")

    # Mode B: folder
    parser.add_argument("--lq_dir",  type=str, default=None,
                        help="Folder of LQ .npy slices")
    parser.add_argument("--hq_dir",  type=str, default=None,
                        help="Folder of HQ .npy slices (optional)")

    # Common
    parser.add_argument("--ckpt",    type=str, nargs="+", required=True,
                        help="Checkpoint path(s). Multiple → side-by-side comparison")
    parser.add_argument("--out_dir", type=str, default="experiments/results/inference",
                        help="Directory to save .npy predictions and metrics")
    parser.add_argument("--out",     type=str, default=None,
                        help="Output PNG path for single-slice mode")
    parser.add_argument("--save_png",action="store_true",
                        help="Save reconstruction grid PNG (folder mode)")
    parser.add_argument("--n_vis",   type=int, default=16,
                        help="Number of slices to include in the visual grid")
    parser.add_argument("--config",  type=str, default="configs/default.yaml",
                        help="Config yaml (only used if checkpoint has no saved config)")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.lq:
        # Single-slice mode
        infer_single(args, device)

    elif args.lq_dir:
        # Folder mode — use only first checkpoint
        infer_folder(args, device)

    else:
        parser.error("Provide either --lq (single slice) or --lq_dir (folder)")


if __name__ == "__main__":
    main()