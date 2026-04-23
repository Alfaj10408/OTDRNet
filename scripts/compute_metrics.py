"""
scripts/compute_metrics.py
──────────────────────────────────────────────────────────────
PSNR and SSIM computation utilities.
Used by train.py (val loop), eval.py, and standalone for tables.

Standalone usage:
    python scripts/compute_metrics.py \
        --pred_dir experiments/results/otdr_mri_default \
        --split test
"""

import os
import argparse
import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio as sk_psnr,
    structural_similarity   as sk_ssim,
)


def compute_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """PSNR between two float images in [0, data_range]."""
    return float(sk_psnr(target, pred, data_range=data_range))


def compute_ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """SSIM between two float images in [0, data_range]."""
    return float(sk_ssim(target, pred, data_range=data_range))


def evaluate_dir(pred_dir: str, split: str = "test"):
    """
    Load all pred/hq pairs from result directory and compute metrics.
    Expects files named:  <fname>_pred.npy  and  <fname>_hq.npy
    """
    pred_files = sorted([
        f for f in os.listdir(pred_dir) if f.endswith("_pred.npy")
    ])

    if not pred_files:
        print(f"[metrics] No *_pred.npy files found in {pred_dir}")
        return

    psnr_list, ssim_list = [], []

    for pf in pred_files:
        stem  = pf.replace("_pred.npy", "")
        hf    = stem + "_hq.npy"
        p_path = os.path.join(pred_dir, pf)
        h_path = os.path.join(pred_dir, hf)

        if not os.path.exists(h_path):
            print(f"  [warn] Missing HQ: {hf}")
            continue

        pred   = np.load(p_path).astype(np.float32)
        target = np.load(h_path).astype(np.float32)

        psnr_list.append(compute_psnr(pred, target))
        ssim_list.append(compute_ssim(pred, target))

    print(f"\n{'='*50}")
    print(f"  Metrics on {len(psnr_list)} samples  [{split}]")
    print(f"  PSNR : {np.mean(psnr_list):.4f} ± {np.std(psnr_list):.4f} dB")
    print(f"  SSIM : {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--split",    type=str, default="test")
    args = parser.parse_args()
    evaluate_dir(args.pred_dir, args.split)