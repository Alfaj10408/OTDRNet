"""
data/masks/generate_masks.py
──────────────────────────────────────────────────────────────
Generate and save Cartesian k-space masks for different R values.

Usage:
    python data/masks/generate_masks.py --H 256 --W 256 --R 4 8 16
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.transforms import KSpaceMask


def generate_and_save(H: int, W: int, R_list: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(R_list), figsize=(4 * len(R_list), 4))
    if len(R_list) == 1:
        axes = [axes]

    for ax, R in zip(axes, R_list):
        km   = KSpaceMask(H, W, R)
        path = os.path.join(out_dir, f"kspace_mask_R{R}.npy")
        km.save(path)

        ax.imshow(km.mask, cmap="gray")
        ax.set_title(f"R={R}  ({km.mask.mean()*100:.1f}% coverage)")
        ax.axis("off")

    fig_path = os.path.join(out_dir, "masks_overview.png")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=100)
    plt.close()
    print(f"[generate_masks] Saved overview → {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--H",       type=int,   default=256)
    parser.add_argument("--W",       type=int,   default=256)
    parser.add_argument("--R",       type=int,   nargs="+", default=[4])
    parser.add_argument("--out_dir", type=str,   default="data/masks")
    args = parser.parse_args()

    generate_and_save(args.H, args.W, args.R, args.out_dir)