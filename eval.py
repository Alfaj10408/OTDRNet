"""
eval.py
──────────────────────────────────────────────────────────────
Full test-set evaluation for OTDRNet.

Usage:
    python eval.py --config configs/default.yaml \
                   --ckpt experiments/checkpoints/otdr_mri_default/ckpt_0200000.pth \
                   --save_images
"""

import os
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from models import OTDRNet
from data.mri_dataset import MRIDataset
from scripts.compute_metrics import compute_psnr, compute_ssim


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model, test_loader, device, result_dir=None, save_images=False):
    model.eval()

    if save_images and result_dir:
        os.makedirs(result_dir, exist_ok=True)

    psnr_list, ssim_list = [], []

    for batch in test_loader:
        lq    = batch["lq"].to(device)
        hq    = batch["hq"].to(device)
        fnames = batch["fname"]

        pred, _ = model(lq)
        pred    = pred.clamp(0.0, 1.0)

        for b in range(pred.shape[0]):
            p = pred[b, 0].cpu().numpy()
            h = hq[b, 0].cpu().numpy()
            l = lq[b, 0].cpu().numpy()

            psnr = compute_psnr(p, h)
            ssim = compute_ssim(p, h)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            if save_images and result_dir:
                fname = fnames[b]
                np.save(os.path.join(result_dir, f"{fname}_pred.npy"), p)
                np.save(os.path.join(result_dir, f"{fname}_hq.npy"),   h)
                np.save(os.path.join(result_dir, f"{fname}_lq.npy"),   l)

    mean_psnr = np.mean(psnr_list)
    mean_ssim = np.mean(ssim_list)
    std_psnr  = np.std(psnr_list)
    std_ssim  = np.std(ssim_list)

    print(f"\n{'='*50}")
    print(f"  Test Results")
    print(f"  PSNR : {mean_psnr:.4f} ± {std_psnr:.4f} dB")
    print(f"  SSIM : {mean_ssim:.4f} ± {std_ssim:.4f}")
    print(f"  N    : {len(psnr_list)} samples")
    print(f"{'='*50}\n")

    return {"psnr": mean_psnr, "ssim": mean_ssim,
            "psnr_std": std_psnr, "ssim_std": std_ssim}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt",        type=str, required=True)
    parser.add_argument("--save_images", action="store_true")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    test_ds = MRIDataset(
        root       = cfg["data"]["root"],
        split      = "test",
        patch_size = 256,
        augment    = False,
    )
    test_loader = DataLoader(test_ds, batch_size=1,
                             shuffle=False, num_workers=2)

    # Model
    mc = cfg["model"]
    model = OTDRNet(
        in_c      = mc["in_c"],
        C         = mc["C"],
        enc_blocks= mc["enc_blocks"],
        num_heads = mc["num_heads"],
        n_prompts = mc["n_prompts"],
        n_experts = mc["n_experts"],
        top_k     = mc["top_k"],
        eps_ot    = cfg["ot"]["eps"],
        iters_ot  = cfg["ot"]["iters"],
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"[eval] Loaded weights from {args.ckpt}")

    result_dir = os.path.join(
        cfg["log"]["result_dir"], cfg["log"]["exp_name"]
    ) if args.save_images else None

    evaluate(model, test_loader, device, result_dir, args.save_images)


if __name__ == "__main__":
    main()