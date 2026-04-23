"""
data/mri_dataset.py
──────────────────────────────────────────────────────────────
PyTorch Dataset for paired (LQ, HQ) MRI slices.

Expects directory structure produced by prepare_ixi.py:
  data/processed/
    train/
      HQ/  *.npy   float32 [256, 256]  normalized [0,1]
      LQ/  *.npy   float32 [256, 256]  k-space undersampled
    val/
      HQ/  *.npy
      LQ/  *.npy
    test/
      HQ/  *.npy
      LQ/  *.npy

Returns:
  dict with keys:
    "lq"    : torch.FloatTensor [1, H, W]
    "hq"    : torch.FloatTensor [1, H, W]
    "fname" : str  (stem of file for saving results)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class MRIDataset(Dataset):

    def __init__(
        self,
        root:       str,
        split:      str  = "train",
        patch_size: int  = 128,
        augment:    bool = True,
    ):
        self.patch_size = patch_size
        self.augment    = augment and (split == "train")

        hq_dir = os.path.join(root, split, "HQ")
        lq_dir = os.path.join(root, split, "LQ")

        assert os.path.isdir(hq_dir), f"HQ dir not found: {hq_dir}"
        assert os.path.isdir(lq_dir), f"LQ dir not found: {lq_dir}"

        self.hq_files = sorted(Path(hq_dir).glob("*.npy"))
        self.lq_files = sorted(Path(lq_dir).glob("*.npy"))

        assert len(self.hq_files) > 0, f"No .npy files in {hq_dir}"
        assert len(self.hq_files) == len(self.lq_files), (
            f"Mismatch: {len(self.hq_files)} HQ vs {len(self.lq_files)} LQ"
        )

        print(f"[MRIDataset] split={split} | samples={len(self.hq_files)} "
              f"| patch_size={patch_size} | augment={self.augment}")

    def __len__(self):
        return len(self.hq_files)

    def __getitem__(self, idx: int) -> dict:
        hq = np.load(self.hq_files[idx]).astype(np.float32)   # [H, W]
        lq = np.load(self.lq_files[idx]).astype(np.float32)   # [H, W]

        # Random crop during training
        if self.augment:
            hq, lq = self._random_crop(hq, lq)
            hq, lq = self._random_flip_rotate(hq, lq)

        # Add channel dim: [1, H, W]
        hq = torch.from_numpy(hq.copy()).unsqueeze(0)
        lq = torch.from_numpy(lq.copy()).unsqueeze(0)

        return {
            "lq":    lq,
            "hq":    hq,
            "fname": self.hq_files[idx].stem,
        }

    def _random_crop(
        self,
        hq: np.ndarray,
        lq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        H, W = hq.shape
        ps   = self.patch_size
        if H <= ps or W <= ps:
            # Pad if image smaller than patch
            pad_h = max(0, ps - H)
            pad_w = max(0, ps - W)
            hq = np.pad(hq, ((0, pad_h), (0, pad_w)))
            lq = np.pad(lq, ((0, pad_h), (0, pad_w)))
            H, W = hq.shape
        i = np.random.randint(0, H - ps + 1)
        j = np.random.randint(0, W - ps + 1)
        return hq[i:i+ps, j:j+ps], lq[i:i+ps, j:j+ps]

    def _random_flip_rotate(
        self,
        hq: np.ndarray,
        lq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Horizontal flip
        if np.random.rand() > 0.5:
            hq = np.fliplr(hq)
            lq = np.fliplr(lq)
        # Vertical flip
        if np.random.rand() > 0.5:
            hq = np.flipud(hq)
            lq = np.flipud(lq)
        # 90° rotation
        k = np.random.randint(0, 4)
        hq = np.rot90(hq, k)
        lq = np.rot90(lq, k)
        return hq, lq