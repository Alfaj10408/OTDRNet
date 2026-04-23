"""
train.py  — OTDRNet CS-MRI  (AMP + torch.compile optimized)
──────────────────────────────────────────────────────────────
Speedups applied:
  ① AMP (torch.cuda.amp)  — ~2x on A100 tensor cores, halves VRAM
  ② torch.compile()       — ~10-20% extra via kernel fusion (PyTorch 2.0+)
  ③ num_workers tuned     — avoid CPU 100% bottleneck on shared servers
  ④ Sinkhorn iters in cfg — keep at 20 during training, 100 for eval
  ⑤ GradScaler            — safe FP16 gradient scaling

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --resume <ckpt.pth>
    python train.py --config configs/default.yaml --no_compile   # if torch<2.0
"""

import os
import time
import argparse
import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import OTDRNet
from models.losses import TotalLoss
from data.mri_dataset import MRIDataset
from scripts.compute_metrics import compute_psnr, compute_ssim


# ── Config ────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Checkpoint ────────────────────────────────────────────────

# ── run_tag is set once at startup and reused for every save ──
_RUN_TAG = None

def get_run_tag(cfg):
    """
    Returns a stable run identifier for this process:
        <exp_name>/<exp_name>_<timestamp>
    Created once on first call, reused for every subsequent checkpoint
    so all checkpoints from a single training run share the same folder.
    """
    global _RUN_TAG
    if _RUN_TAG is None:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _RUN_TAG = os.path.join(
            cfg["log"]["exp_name"],
            f"{cfg['log']['exp_name']}_{ts}"
        )
    return _RUN_TAG


def save_checkpoint(model, optimizer, scheduler, scaler, iteration, cfg):
    run_dir  = os.path.join(cfg["log"]["checkpoint_dir"], get_run_tag(cfg))
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"ckpt_{iteration:07d}.pth")
    # If model was compiled, save the underlying module
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "iteration": iteration,
        "model":     raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler":    scaler.state_dict(),
        "run_tag":   _RUN_TAG,           # saved for reference
        "config":    cfg,                # save full config alongside weights
    }, path)
    print(f"[train] Saved checkpoint → {path}", flush=True)


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    global _RUN_TAG
    ckpt = torch.load(path, map_location="cpu")
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    # Restore run_tag so resumed checkpoints go into the same folder
    if "run_tag" in ckpt:
        _RUN_TAG = ckpt["run_tag"]
        print(f"[train] Resumed run_tag: {_RUN_TAG}", flush=True)
    print(f"[train] Resumed from iteration {ckpt['iteration']}", flush=True)
    return ckpt["iteration"]


# ── Validation  (runs in FP32 for accuracy) ───────────────────

@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    psnr_sum, ssim_sum, count = 0.0, 0.0, 0
    for batch in val_loader:
        lq = batch["lq"].to(device)
        hq = batch["hq"].to(device)
        # Validation always in FP32 — more stable PSNR/SSIM
        with autocast(enabled=False):
            pred, _ = model(lq.float())
        pred = pred.clamp(0.0, 1.0)
        for b in range(pred.shape[0]):
            p = pred[b, 0].cpu().numpy()
            h = hq[b, 0].cpu().numpy()
            psnr_sum += compute_psnr(p, h)
            ssim_sum += compute_ssim(p, h)
            count += 1
    model.train()
    return psnr_sum / count, ssim_sum / count


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str,  default="configs/default.yaml")
    parser.add_argument("--resume",     type=str,  default=None)
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile (use if PyTorch < 2.0)")
    parser.add_argument("--no_amp",     action="store_true",
                        help="Disable AMP (use for debugging NaN losses)")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and (device.type == "cuda")

    # ── Hardware tuning ───────────────────────────────────────
    # benchmark=True: auto-selects fastest cuDNN conv algorithm
    # for fixed input sizes — free 5-15% speedup after warmup
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True   # A100: TF32 for matmul
        torch.backends.cudnn.allow_tf32 = True          # A100: TF32 for conv

    print(f"[train] device={device}  AMP={use_amp}  "
          f"compile={not args.no_compile}  "
          f"cudnn.benchmark={torch.backends.cudnn.benchmark}", flush=True)

    # ── Datasets ──────────────────────────────────────────────
    train_ds = MRIDataset(
        root       = cfg["data"]["root"],
        split      = "train",
        patch_size = cfg["data"]["patch_size"],
        augment    = cfg["data"]["augment"],
    )
    val_ds = MRIDataset(
        root       = cfg["data"]["root"],
        split      = "val",
        patch_size = cfg["data"]["patch_size"],
        augment    = False,
    )

    nw = cfg["train"]["num_workers"]
    print(f"[train] num_workers={nw}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["train"]["batch_size"],
        shuffle     = True,
        num_workers = nw,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["val"]["batch_size"],
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True,
    )

    # ── Model ─────────────────────────────────────────────────
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
        iters_ot   = cfg["ot"]["iters"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Parameters: {n_params/1e6:.2f} M", flush=True)

    # ── ② torch.compile — fuses kernels, ~10-20% speedup ──────
    if not args.no_compile and hasattr(torch, "compile"):
        print("[train] Compiling model with torch.compile(, flush=True) ...")
        model = torch.compile(model, mode="reduce-overhead")
        print("[train] Compile done  (first iter will be slow — normal, flush=True)")
    else:
        print("[train] torch.compile skipped", flush=True)

    # ── Loss / Optimizer / Scheduler ──────────────────────────
    print("[train] Setting up loss/optimizer/scheduler...", flush=True)
    criterion = TotalLoss(lambda_ot=cfg["ot"]["lambda_ot"])
    optimizer = Adam(
        model.parameters(),
        lr    = cfg["train"]["lr_init"],
        betas = (cfg["train"]["beta1"], cfg["train"]["beta2"]),
    )
    total_iters = cfg["train"]["iterations"]
    scheduler   = CosineAnnealingLR(
        optimizer,
        T_max   = total_iters,
        eta_min = cfg["train"]["lr_min"],
    )
    print("[train] Optimizer ready", flush=True)

    # ── ① AMP GradScaler ──────────────────────────────────────
    scaler = GradScaler(enabled=use_amp)
    print(f"[train] GradScaler ready (enabled={use_amp}, flush=True)")

    # ── Resume ────────────────────────────────────────────────
    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler
        )

    # ── TensorBoard ───────────────────────────────────────────
    print("[train] Setting up TensorBoard...", flush=True)
    log_dir = os.path.join(cfg["log"]["log_dir"], cfg["log"]["exp_name"])
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir) if cfg["log"]["use_tensorboard"] else None
    print(f"[train] TensorBoard ready → {log_dir}", flush=True)

    # ── Training Loop ─────────────────────────────────────────
    model.train()
    iteration     = start_iter
    timer_start   = time.time()
    window_start  = time.time()
    window_losses = []

    print(f"[train] Starting from iteration {start_iter} / {total_iters}", flush=True)
    print(f"[train] Batch size={cfg['train']['batch_size']}  "
          f"Patch={cfg['data']['patch_size']}  "
          f"OT iters={cfg['ot']['iters']}", flush=True)
    print(f"[train] Initializing DataLoader workers ({cfg['train']['num_workers']}, flush=True)...")
    data_iter = iter(train_loader)
    print(f"[train] DataLoader ready — entering loop", flush=True)

    while iteration < total_iters:

        # ── Data ──────────────────────────────────────────────
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        lq = batch["lq"].to(device, non_blocking=True)   # non_blocking with pin_memory
        hq = batch["hq"].to(device, non_blocking=True)

        # ── ① AMP forward pass ────────────────────────────────
        optimizer.zero_grad(set_to_none=True)             # faster than zero_grad()

        with autocast(enabled=use_amp):
            pred, L_ot      = model(lq)
            loss, loss_dict = criterion(pred, hq, L_ot)

        # ── ① AMP backward + step ─────────────────────────────
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["train"]["grad_clip"]
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        iteration += 1
        window_losses.append(loss_dict["loss_total"])

        # ── Logging every 100 iters ───────────────────────────
        if iteration % 100 == 0:
            lr  = optimizer.param_groups[0]["lr"]
            now = time.time()

            elapsed_win     = now - window_start
            it_per_sec      = 100.0 / max(elapsed_win, 1e-6)
            iters_left      = total_iters - iteration
            eta_sec         = iters_left / it_per_sec
            eta_h           = int(eta_sec // 3600)
            eta_m           = int((eta_sec % 3600) // 60)
            elapsed_total_h = (now - timer_start) / 3600
            avg_loss        = sum(window_losses) / len(window_losses)
            amp_scale       = scaler.get_scale() if use_amp else 1.0
            window_losses.clear()
            window_start = now

            print(
                f"[{iteration:7d}/{total_iters}] "
                f"loss={avg_loss:.4f}  "
                f"l1={loss_dict['loss_l1']:.4f}  "
                f"ot={loss_dict['loss_ot']:.4f}  "
                f"lr={lr:.2e}  scale={amp_scale:.0f}"
                f"  | {it_per_sec:.1f} it/s  "
                f"elapsed={elapsed_total_h:.2f}h  "
                f"ETA={eta_h:02d}h{eta_m:02d}m"
            , flush=True)

            if writer:
                for k, v in loss_dict.items():
                    writer.add_scalar(f"train/{k}",     v,           iteration)
                writer.add_scalar("train/lr",           lr,          iteration)
                writer.add_scalar("train/it_per_sec",   it_per_sec,  iteration)
                writer.add_scalar("train/eta_hours",    eta_sec/3600,iteration)
                writer.add_scalar("train/amp_scale",    amp_scale,   iteration)

        # ── Validation ────────────────────────────────────────
        if iteration % cfg["val"]["every_iter"] == 0:
            psnr, ssim = validate(model, val_loader, device)
            print(f"  [val] PSNR={psnr:.2f} dB  SSIM={ssim:.4f}", flush=True)
            if writer:
                writer.add_scalar("val/PSNR", psnr, iteration)
                writer.add_scalar("val/SSIM", ssim, iteration)

        # ── Checkpoint ────────────────────────────────────────
        if iteration % cfg["log"]["save_every"] == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, iteration, cfg
            )

    # ── Final checkpoint ──────────────────────────────────────
    save_checkpoint(model, optimizer, scheduler, scaler, iteration, cfg)
    print("[train] Done.", flush=True)
    if writer:
        writer.close()


if __name__ == "__main__":
    main()