"""
data/transforms.py
──────────────────────────────────────────────────────────────
K-space undersampling transform and normalization utilities.

Used by both prepare_ixi.py (offline) and optionally at runtime.
"""

import numpy as np


def normalize_minmax(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize image to [0, 1] using min-max scaling."""
    mn, mx = img.min(), img.max()
    if mx - mn < eps:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


class KSpaceMask:
    """
    Generates and applies a central Cartesian k-space mask.

    Args:
        H, W : image dimensions
        R    : undersampling factor. Retains central 1/R of each
               k-space dimension → coverage = 1/R² × 100%
               e.g. R=4 → 6.25% coverage (AMIR protocol)
    """

    def __init__(self, H: int = 256, W: int = 256, R: int = 4):
        self.H    = H
        self.W    = W
        self.R    = R
        self.mask = self._build_mask(H, W, R)

    @staticmethod
    def _build_mask(H: int, W: int, R: int) -> np.ndarray:
        mask = np.zeros((H, W), dtype=bool)
        ch   = H // (2 * R)
        cw   = W // (2 * R)
        mh, mw = H // 2, W // 2
        mask[mh - ch: mh + ch, mw - cw: mw + cw] = True
        coverage = mask.mean() * 100
        print(f"[KSpaceMask] R={R} | shape=({H},{W}) | coverage={coverage:.2f}%")
        return mask

    def apply(self, img_hq: np.ndarray) -> np.ndarray:
        """
        Undersample a 2D image in k-space.

        img_hq : [H, W] float32 in [0,1]
        Returns: [H, W] float32 in [0,1]  (LQ image)
        """
        kspace    = np.fft.fftshift(np.fft.fft2(img_hq))
        kspace_us = kspace * self.mask
        img_lq    = np.abs(
            np.fft.ifft2(np.fft.ifftshift(kspace_us))
        ).astype(np.float32)
        return normalize_minmax(img_lq)

    def save(self, path: str):
        """Save mask as .npy for reference / reproducibility."""
        np.save(path, self.mask.astype(np.uint8))
        print(f"[KSpaceMask] Saved → {path}")