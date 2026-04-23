"""
data/prepare_ixi.py
──────────────────────────────────────────────────────────────
IXI T2 MRI Dataset Preparation for CS-MRI Reconstruction.

Following AMIR (Yang et al., MICCAI 2024) protocol:
  - 578 T2 weighted MRI volumes (256 × 256 × n slices)
  - Extract central 100 2D slices per volume
  - Undersample k-space with factor R=4 (retains 6.25% of data)
  - Split: train=405 / val=59 / test=114

Pipeline:
  1. Scan raw dir for .nii.gz files
  2. Shuffle with fixed seed for reproducibility
  3. For each volume:
       a. Load via nibabel
       b. Extract central 100 slices → [100, H, W]
       c. Resize to 256×256 if needed
       d. Normalize each slice to [0, 1]
       e. Skip near-empty (air-only) slices
       f. Apply k-space undersampling → LQ
       g. Save paired (HQ, LQ) as float32 .npy

Usage:
    python data/prepare_ixi.py \
        --raw_dir  data/raw/IXI \
        --out_dir  data/processed \
        --R        4 \
        --n_slices 100 \
        --seed     42
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.transforms import KSpaceMask, normalize_minmax


# ── Constants ─────────────────────────────────────────────────
TARGET_H   = 256
TARGET_W   = 256
SPLIT_RATIOS = {"train": 405, "val": 59, "test": 114}   # sums to 578
AIR_THRESHOLD = 1e-3    # slices with max < this are skipped


# ── Volume loading ────────────────────────────────────────────

def load_volume(path: str) -> np.ndarray:
    """
    Load a .nii or .nii.gz MRI volume.
    Returns float32 array of shape [H, W, D].
    """
    img = nib.load(path)
    vol = np.asarray(img.dataobj, dtype=np.float32)

    # Some IXI volumes are [W, H, D] — ensure at least 3D
    if vol.ndim == 4:
        vol = vol[..., 0]       # drop extra dim if present
    assert vol.ndim == 3, f"Unexpected volume shape: {vol.shape}"
    return vol


def extract_central_slices(
    vol: np.ndarray,
    n_slices: int = 100,
) -> np.ndarray:
    """
    Extract n_slices central slices along the last (depth) axis.

    Input  : [H, W, D]
    Output : [n_slices, H, W]
    """
    D   = vol.shape[-1]
    mid = D // 2
    half = n_slices // 2
    start = max(0, mid - half)
    end   = min(D, start + n_slices)
    start = max(0, end - n_slices)       # adjust if near end

    slices = vol[:, :, start:end]        # [H, W, n_slices]
    return np.transpose(slices, (2, 0, 1))  # [n_slices, H, W]


def resize_slice(sl: np.ndarray, H: int = 256, W: int = 256) -> np.ndarray:
    """
    Resize 2D slice to (H, W) using simple zoom if necessary.
    Most IXI T2 slices are already 256×256; this is a safety net.
    """
    if sl.shape == (H, W):
        return sl
    from scipy.ndimage import zoom
    zoom_h = H / sl.shape[0]
    zoom_w = W / sl.shape[1]
    return zoom(sl, (zoom_h, zoom_w), order=1).astype(np.float32)


# ── Split assignment ──────────────────────────────────────────

def assign_splits(
    all_paths: list,
    ratios: dict,
    seed: int = 42,
) -> dict:
    """
    Shuffle and split volume paths.

    Returns dict: {"train": [...], "val": [...], "test": [...]}
    """
    rng   = np.random.default_rng(seed)
    paths = list(all_paths)
    rng.shuffle(paths)

    n_train = ratios["train"]
    n_val   = ratios["val"]

    splits = {
        "train": paths[:n_train],
        "val":   paths[n_train: n_train + n_val],
        "test":  paths[n_train + n_val:],
    }

    for k, v in splits.items():
        print(f"  {k:5s}: {len(v)} volumes")

    return splits


# ── Per-split processing ──────────────────────────────────────

def process_split(
    vol_paths: list,
    split_name: str,
    out_dir: str,
    mask: KSpaceMask,
    n_slices: int = 100,
) -> dict:
    """
    Process a list of .nii.gz volumes into paired (HQ, LQ) .npy slices.

    Directory layout:
        out_dir/{split_name}/HQ/  *.npy   float32 [256, 256]
        out_dir/{split_name}/LQ/  *.npy   float32 [256, 256]

    Returns stats dict.
    """
    hq_dir = os.path.join(out_dir, split_name, "HQ")
    lq_dir = os.path.join(out_dir, split_name, "LQ")
    os.makedirs(hq_dir, exist_ok=True)
    os.makedirs(lq_dir, exist_ok=True)

    n_saved   = 0
    n_skipped = 0
    n_failed  = 0

    for vol_path in tqdm(vol_paths, desc=f"  [{split_name}]"):
        # Derive a clean volume ID from filename
        vol_id = Path(vol_path).name.replace(".nii.gz", "").replace(".nii", "")

        try:
            vol    = load_volume(vol_path)           # [H, W, D]
            slices = extract_central_slices(vol, n_slices)  # [N, H, W]

            for s_idx, sl in enumerate(slices):
                # Resize to target resolution
                sl = resize_slice(sl, TARGET_H, TARGET_W)

                # Skip near-empty (air-dominated) slices
                if sl.max() < AIR_THRESHOLD:
                    n_skipped += 1
                    continue

                # Normalize HQ to [0, 1]
                sl_hq = normalize_minmax(sl)         # [H, W]

                # Generate LQ via k-space undersampling
                sl_lq = mask.apply(sl_hq)            # [H, W]

                # Save as float32 .npy
                fname = f"{vol_id}_s{s_idx:03d}.npy"
                np.save(os.path.join(hq_dir, fname), sl_hq.astype(np.float32))
                np.save(os.path.join(lq_dir, fname), sl_lq.astype(np.float32))
                n_saved += 1

        except Exception as exc:
            print(f"\n  [WARN] Failed to process {vol_path}: {exc}")
            n_failed += 1
            continue

    stats = {
        "split":   split_name,
        "saved":   n_saved,
        "skipped": n_skipped,
        "failed":  n_failed,
    }
    print(
        f"  [{split_name}] saved={n_saved}  "
        f"skipped={n_skipped}  failed={n_failed}"
    )
    return stats


# ── Verification ─────────────────────────────────────────────

def verify_output(out_dir: str):
    """
    Quick sanity check: count files and verify one pair loads correctly.
    """
    print("\n[verify] Checking output...")
    for split in ["train", "val", "test"]:
        hq_dir = os.path.join(out_dir, split, "HQ")
        lq_dir = os.path.join(out_dir, split, "LQ")
        if not os.path.isdir(hq_dir):
            continue
        hq_files = list(Path(hq_dir).glob("*.npy"))
        lq_files = list(Path(lq_dir).glob("*.npy"))
        print(f"  {split:5s}: {len(hq_files)} HQ  /  {len(lq_files)} LQ")

        # Load one sample and check shapes / value range
        if hq_files:
            hq_s = np.load(hq_files[0])
            lq_s = np.load(
                os.path.join(lq_dir, hq_files[0].name)
            )
            assert hq_s.shape == (TARGET_H, TARGET_W), \
                f"Unexpected HQ shape: {hq_s.shape}"
            assert hq_s.min() >= 0.0 and hq_s.max() <= 1.0 + 1e-5, \
                f"HQ out of range: [{hq_s.min():.3f}, {hq_s.max():.3f}]"
            print(
                f"         sample HQ shape={hq_s.shape}  "
                f"range=[{hq_s.min():.3f}, {hq_s.max():.3f}]"
            )
            print(
                f"         sample LQ shape={lq_s.shape}  "
                f"range=[{lq_s.min():.3f}, {lq_s.max():.3f}]"
            )

    print("[verify] OK\n")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare IXI T2 MRI dataset for CS-MRI reconstruction."
    )
    parser.add_argument(
        "--raw_dir",  type=str, default="data/raw/IXI",
        help="Directory containing IXI-T2 .nii.gz files"
    )
    parser.add_argument(
        "--out_dir",  type=str, default="data/processed",
        help="Output directory for processed .npy files"
    )
    parser.add_argument(
        "--mask_dir", type=str, default="data/masks",
        help="Directory to save k-space mask"
    )
    parser.add_argument(
        "--R",        type=int, default=4,
        help="K-space undersampling factor (4 → 6.25%% retained)"
    )
    parser.add_argument(
        "--n_slices", type=int, default=100,
        help="Number of central slices to extract per volume"
    )
    parser.add_argument(
        "--seed",     type=int, default=42,
        help="Random seed for train/val/test split"
    )
    parser.add_argument(
        "--splits",   type=str, nargs="+",
        default=["train", "val", "test"],
        help="Which splits to process (default: all)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  IXI T2 MRI Data Preparation")
    print(f"  raw_dir  : {args.raw_dir}")
    print(f"  out_dir  : {args.out_dir}")
    print(f"  R        : {args.R}  ({100 / args.R**2:.2f}% k-space retained)")
    print(f"  n_slices : {args.n_slices}")
    print(f"  seed     : {args.seed}")
    print("=" * 60)

    # ── Find all .nii.gz volumes ──────────────────────────────
    all_vols = sorted([
        str(p) for p in Path(args.raw_dir).rglob("*.nii.gz")
    ])
    # Also accept uncompressed .nii
    all_vols += sorted([
        str(p) for p in Path(args.raw_dir).rglob("*.nii")
        if not str(p).endswith(".nii.gz")
    ])

    print(f"\nFound {len(all_vols)} volumes in {args.raw_dir}")

    if len(all_vols) == 0:
        print("""
  [ERROR] No MRI volumes found.
  Please download IXI-T2 first:

      bash scripts/download_ixi.sh

  Or manually from:
      http://brain-development.org/ixi-dataset/
        """)
        sys.exit(1)

    # ── Build k-space mask ────────────────────────────────────
    os.makedirs(args.mask_dir, exist_ok=True)
    mask = KSpaceMask(TARGET_H, TARGET_W, args.R)
    mask.save(os.path.join(args.mask_dir, f"kspace_mask_R{args.R}.npy"))

    # ── Split volumes ─────────────────────────────────────────
    print("\nSplitting volumes:")
    # Clip to available volumes if fewer than 578
    ratios = {k: v for k, v in SPLIT_RATIOS.items()}
    total_expected = sum(ratios.values())
    if len(all_vols) < total_expected:
        print(f"  [WARN] Found {len(all_vols)} volumes, expected {total_expected}.")
        print("          Adjusting split ratios proportionally.")
        frac   = len(all_vols) / total_expected
        ratios = {k: max(1, int(v * frac)) for k, v in ratios.items()}
        # Assign remainder to train
        diff = len(all_vols) - sum(ratios.values())
        ratios["train"] += diff

    splits = assign_splits(all_vols, ratios, seed=args.seed)

    # ── Process each split ────────────────────────────────────
    print()
    all_stats = []
    for split_name in args.splits:
        if split_name not in splits:
            print(f"  [WARN] Unknown split '{split_name}', skipping.")
            continue
        stats = process_split(
            vol_paths  = splits[split_name],
            split_name = split_name,
            out_dir    = args.out_dir,
            mask       = mask,
            n_slices   = args.n_slices,
        )
        all_stats.append(stats)

    # ── Summary ───────────────────────────────────────────────
    total_saved = sum(s["saved"] for s in all_stats)
    print(f"\n{'='*60}")
    print(f"  Preparation complete!")
    print(f"  Total slice pairs saved : {total_saved:,}")
    print(f"  Output directory        : {args.out_dir}/")
    print(f"    train/HQ/  &  train/LQ/   (~{ratios['train'] * args.n_slices:,} pairs)")
    print(f"    val/HQ/    &  val/LQ/     (~{ratios['val']   * args.n_slices:,} pairs)")
    print(f"    test/HQ/   &  test/LQ/    (~{ratios['test']  * args.n_slices:,} pairs)")
    print(f"  Estimated disk usage    : ~{total_saved * 256 * 256 * 4 * 2 / 1e9:.1f} GB")
    print(f"{'='*60}\n")

    # ── Verify ────────────────────────────────────────────────
    verify_output(args.out_dir)

    print("Next step:")
    print("  python test_single.py \\")
    print(f"    --lq data/processed/test/LQ/<any_file>.npy \\")
    print(f"    --hq data/processed/test/HQ/<same_file>.npy\n")


if __name__ == "__main__":
    main()