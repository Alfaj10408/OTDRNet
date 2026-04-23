#!/bin/bash
# scripts/download_ixi.sh
# ──────────────────────────────────────────────────────────────
# Download IXI T2 MRI Dataset
# Source: http://brain-development.org/ixi-dataset/
# Size  : ~4.5 GB compressed

OUT_DIR="data/raw/IXI"
mkdir -p "$OUT_DIR"

echo ">>> Downloading IXI-T2 (~4.5 GB)..."
wget -c \
  "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar" \
  -O "$OUT_DIR/IXI-T2.tar"

echo ">>> Extracting..."
tar -xf "$OUT_DIR/IXI-T2.tar" -C "$OUT_DIR"

N=$(ls "$OUT_DIR"/*.nii.gz 2>/dev/null | wc -l)
echo ">>> Done. Found $N .nii.gz volumes in $OUT_DIR"

if [ "$N" -lt 500 ]; then
  echo "[WARN] Expected ~578 volumes but only found $N."
  echo "       Check your internet connection or re-run this script."
fi