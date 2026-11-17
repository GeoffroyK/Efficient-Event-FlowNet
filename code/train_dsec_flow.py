"""
Training script for optical flow on DSEC event-camera dataset using SiLIFFlowNetScaled.

Usage:
    python train_dsec_flow.py
    python train_dsec_flow.py --epochs 100 --lr 1e-4 --batch_size 4
"""

import argparse
import os
import tempfile
from contextlib import nullcontext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import hdf5plugin  # noqa: F401 – needed to open DSEC HDF5 files

from dsec_optical.dsec_utils import RepresentationType
from dsec_optical.flow_cache import FLOW_TRAIN_RECORDINGS, _FLOW_VAL_RECORDINGS
from dsec_optical.loader_dsec import assemble_dsec_sequences
from dsec_optical.transformers import downsample_spatial_flow
from models.dsecconvflow import SiLIFFlowNetScaled
from models.spikeflownet import SpikeFlowNet, FullSpikeFlowNet


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

DATA_ROOT = "/home/geoffroyk/idnet/data/"
# 13 train / 5 val split (train = every GT recording not in val = FLOW_TRAIN_RECORDINGS)
DEFAULT_VAL_SEQ = _FLOW_VAL_RECORDINGS


def train_collate(sample_list):
    batch = {}
    for field_name in sample_list[0]:
        if field_name == "seq_name":
            batch["seq_name"] = [s[field_name] for s in sample_list]
        elif field_name == "new_sequence":
            batch["new_sequence"] = [s[field_name] for s in sample_list]
        elif field_name.startswith("event_volume"):
            batch[field_name] = torch.stack([s[field_name] for s in sample_list])
        elif field_name.startswith("flow_gt"):
            if all(field_name in s for s in sample_list):
                batch[field_name] = torch.stack([s[field_name][0] for s in sample_list])
                batch[field_name + "_valid_mask"] = torch.stack(
                    [s[field_name][1] for s in sample_list]
                )
    return batch


def make_loaders(args, native_gt_override=None):
    native_gt = getattr(args, "native_upsample", False) if native_gt_override is None else native_gt_override
    base = {
        "sequence_length": 1,
        "downsample_ratio": args.downsample_ratio,
        "val_seq": args.val_seq,
        "native_gt": native_gt,
        "only_new": True,
    }
    train_config = OmegaConf.create({**base, "horizontal_flip": 0.5})
    val_config   = OmegaConf.create(base)

    train_set = assemble_dsec_sequences(
        dataset_root=DATA_ROOT,
        config=train_config,
        exclude_seq=args.val_seq,
        representation_type=RepresentationType.VOXEL,
    )
    val_set = assemble_dsec_sequences(
        dataset_root=DATA_ROOT,
        config=val_config,
        include_seq=args.val_seq,
        representation_type=RepresentationType.VOXEL,
    )

    loader_train = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_collate,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )
    loader_val = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_collate,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )
    return loader_train, loader_val


def make_native_val_loader(args):
    """Build a validation DataLoader whose GT flow/mask remain at native 480×640."""
    val_config = OmegaConf.create({
        "sequence_length": 1,
        "downsample_ratio": args.downsample_ratio,
        "val_seq": args.val_seq,
        "native_gt": True,
        "only_new": True,
    })
    val_set = assemble_dsec_sequences(
        dataset_root=DATA_ROOT,
        config=val_config,
        include_seq=args.val_seq,
        representation_type=RepresentationType.VOXEL,
    )
    return DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_collate,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=False,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Loss / metric
# ---------------------------------------------------------------------------

def epe_loss(flow_pred, flow_gt, valid_mask):
    """
    Masked endpoint error loss.

    flow_pred:  (B, 2, H, W)
    flow_gt:    (B, 2, H, W)
    valid_mask: (B, 1, H, W) – 1 where GT is valid
    """
    epe = torch.sqrt(((flow_pred - flow_gt) ** 2).sum(dim=1, keepdim=True) + 1e-8)
    denom = valid_mask.sum().clamp(min=1.0)
    return (epe * valid_mask).sum() / denom


def multiscale_loss(final_flow, coarse_flow, medium_flow, flow_gt, valid_mask,
                    weights=(1.0, 0.5, 0.25)):
    """Supervise all three pyramid outputs; GT/mask are mask-downsampled to match each scale."""
    def downsample_gt(gt, mask, factor):
        gt_ds, mask_ds = downsample_spatial_flow(gt, mask, factor)
        return gt_ds, mask_ds.float()

    loss_fine = epe_loss(final_flow[:, -1], flow_gt, valid_mask)
    loss_med = epe_loss(medium_flow[:, -1], *downsample_gt(flow_gt, valid_mask, 2))
    loss_crs = epe_loss(coarse_flow[:, -1], *downsample_gt(flow_gt, valid_mask, 4))

    return weights[0] * loss_fine + weights[1] * loss_med + weights[2] * loss_crs


@torch.no_grad()
def compute_epe(flow_pred, flow_gt, valid_mask):
    """Mean EPE over valid pixels (px/frame)."""
    epe = torch.sqrt(((flow_pred - flow_gt) ** 2).sum(dim=1, keepdim=True) + 1e-8)
    denom = valid_mask.sum().clamp(min=1.0)
    return ((epe * valid_mask).sum() / denom).item()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def flow_to_rgb(flow_2hw: np.ndarray) -> np.ndarray:
    """Convert (2, H, W) optical flow to an HSV-based RGB image."""
    u, v = flow_2hw[0], flow_2hw[1]
    mag = np.sqrt(u ** 2 + v ** 2)
    angle = np.arctan2(v, u)
    H, W = u.shape
    hsv = np.zeros((H, W, 3), dtype=np.float32)
    hsv[:, :, 0] = (angle + np.pi) / (2 * np.pi)
    hsv[:, :, 1] = 1.0
    hsv[:, :, 2] = np.clip(mag / max(mag.max(), 1e-6), 0, 1)
    from matplotlib.colors import hsv_to_rgb
    return hsv_to_rgb(hsv)


def make_flow_figure(events_thw, flow_gt_2hw, flow_pred_2hw, valid_mask_hw):
    """
    Build a 1-row matplotlib figure for a single val sample.
    events_thw:    (T, H, W)  – summed event volume
    flow_gt_2hw:   (2, H, W)
    flow_pred_2hw: (2, H, W)
    valid_mask_hw: (H, W)
    Returns a plt.Figure.
    """
    event_frame = events_thw.sum(0)
    error = np.sqrt(
        (flow_pred_2hw[0] - flow_gt_2hw[0]) ** 2
        + (flow_pred_2hw[1] - flow_gt_2hw[1]) ** 2
    )
    error[valid_mask_hw < 0.5] = 0.0

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Validation flow prediction", fontsize=12)

    axes[0].imshow(event_frame, cmap="gray")
    axes[0].set_title("Events (sum)")
    axes[0].axis("off")

    axes[1].imshow(flow_to_rgb(flow_gt_2hw))
    axes[1].set_title("GT flow")
    axes[1].axis("off")

    axes[2].imshow(flow_to_rgb(flow_pred_2hw))
    axes[2].set_title("Predicted flow")
    axes[2].axis("off")

    axes[3].imshow(error, cmap="hot", vmin=0, vmax=max(error.max(), 0.1))
    axes[3].set_title("EPE error")
    axes[3].axis("off")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, accum_steps=1, scaler=None):
    model.train()
    total_loss, n = 0.0, 0
    optimizer.zero_grad()
    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.float16) if scaler else nullcontext()

    for step, batch in enumerate(loader):
        x = batch["event_volume_new"].to(device, non_blocking=True)
        flow_gt = batch["flow_gt_event_volume_new"].to(device, non_blocking=True)
        mask = batch["flow_gt_event_volume_new_valid_mask"].to(device, non_blocking=True)

        with amp_ctx:
            final_flow, coarse_flow, medium_flow = model(x)
            loss = multiscale_loss(final_flow, coarse_flow, medium_flow, flow_gt, mask)

        scaled = loss / accum_steps
        if scaler:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_epe, n = 0.0, 0.0, 0
    for batch in loader:
        x = batch["event_volume_new"].to(device, non_blocking=True)
        flow_gt = batch["flow_gt_event_volume_new"].to(device, non_blocking=True)
        mask = batch["flow_gt_event_volume_new_valid_mask"].to(device, non_blocking=True)

        final_flow, coarse_flow, medium_flow = model(x)
        pred = final_flow[:, -1]  # last timestep → (B, 2, H, W)

        if pred.shape[-2:] != flow_gt.shape[-2:]:
            scale = flow_gt.shape[-1] / pred.shape[-1]
            pred_scaled = F.interpolate(pred, size=flow_gt.shape[-2:], mode="bilinear", align_corners=False) * scale
            total_epe += compute_epe(pred_scaled, flow_gt, mask) * x.size(0)
            total_loss += epe_loss(pred_scaled, flow_gt, mask).item() * x.size(0)
        else:
            total_loss += multiscale_loss(final_flow, coarse_flow, medium_flow, flow_gt, mask).item() * x.size(0)
            total_epe += compute_epe(pred, flow_gt, mask) * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1), total_epe / max(n, 1)


@torch.no_grad()
def evaluate_full(model, loader, device):
    """
    Full evaluation pass: AEE mean/median, outlier %, firing rates.

    Collects per-pixel EPE at every valid GT location across the whole loader,
    so median and outlier thresholds are computed over the full distribution.
    Firing rates are recorded via forward hooks on every spiking ConvSiLIFLayer.

    Returns a flat dict of floats (MLflow-ready).
    """
    from models.convsnn import ConvSiLIFLayer

    model.eval()

    # --- attach firing-rate hooks on every spiking (non-membrane) layer ---
    rate_sums: dict[str, float] = {}
    rate_counts: dict[str, int] = {}
    handles = []

    for name, module in model.named_modules():
        if isinstance(module, ConvSiLIFLayer) and not module.return_membrane:
            rate_sums[name] = 0.0
            rate_counts[name] = 0

            def _make_hook(n: str):
                def _hook(mod, inp, output):
                    # output: (B, T, C, H, W) binary spikes
                    rate_sums[n] += output.float().mean().item()
                    rate_counts[n] += 1
                return _hook

            handles.append(module.register_forward_hook(_make_hook(name)))

    # --- collect per-pixel flow values at valid GT locations ---
    all_epe:    list[torch.Tensor] = []
    all_pred_u: list[torch.Tensor] = []
    all_pred_v: list[torch.Tensor] = []
    all_gt_u:   list[torch.Tensor] = []
    all_gt_v:   list[torch.Tensor] = []

    for batch in loader:
        x       = batch["event_volume_new"].to(device, non_blocking=True)
        flow_gt = batch["flow_gt_event_volume_new"].to(device, non_blocking=True)
        mask    = batch["flow_gt_event_volume_new_valid_mask"].to(device, non_blocking=True)

        final_flow, _, _ = model(x)
        pred = final_flow[:, -1]  # (B, 2, H, W)
        if pred.shape[-2:] != flow_gt.shape[-2:]:
            scale = flow_gt.shape[-1] / pred.shape[-1]
            pred = F.interpolate(pred, size=flow_gt.shape[-2:], mode="bilinear", align_corners=False) * scale

        epe  = torch.sqrt(((pred - flow_gt) ** 2).sum(dim=1, keepdim=True) + 1e-8)
        valid = (mask > 0.5).squeeze(1)  # (B, H, W) bool

        all_epe.append(epe.squeeze(1)[valid].cpu())
        all_pred_u.append(pred[:, 0][valid].cpu())
        all_pred_v.append(pred[:, 1][valid].cpu())
        all_gt_u.append(flow_gt[:, 0][valid].cpu())
        all_gt_v.append(flow_gt[:, 1][valid].cpu())

    for h in handles:
        h.remove()

    all_epe_t = torch.cat(all_epe)  # (N_valid_pixels,)

    # Angular error (Barron et al. 1994): treat flow as 3-D vector [u, v, 1],
    # compute angle between prediction and GT in degrees.
    # Robust to magnitude: a 180° direction flip at 0.1 px/frame scores the same
    # as at 10 px/frame, unlike EPE.
    pu = torch.cat(all_pred_u)
    pv = torch.cat(all_pred_v)
    gu = torch.cat(all_gt_u)
    gv = torch.cat(all_gt_v)
    dot       = pu * gu + pv * gv + 1.0
    norm_pred = torch.sqrt(pu ** 2 + pv ** 2 + 1.0)
    norm_gt   = torch.sqrt(gu ** 2 + gv ** 2 + 1.0)
    ae_rad    = torch.acos((dot / (norm_pred * norm_gt)).clamp(-1.0, 1.0))
    all_ae_t  = torch.rad2deg(ae_rad)  # (N_valid_pixels,) in degrees

    # AE breakdown by GT flow magnitude — diagnoses whether direction errors
    # come from near-static regions (direction is undefined when flow ≈ 0)
    # or from genuinely fast-moving areas (real prediction failure).
    gt_mag    = torch.sqrt(gu ** 2 + gv ** 2)
    low_mask  = gt_mag < 1.0
    high_mask = gt_mag >= 3.0

    def _masked_mean(t: torch.Tensor, m: torch.Tensor) -> float:
        return t[m].mean().item() if m.any() else float("nan")

    metrics: dict[str, float] = {
        # EPE = AEE mean: mean Euclidean distance over valid pixels (px)
        "epe":            all_epe_t.mean().item(),
        "aee_median":     all_epe_t.median().item(),
        # AE: mean angular error in degrees (direction-sensitive, magnitude-agnostic)
        "ae":             all_ae_t.mean().item(),
        "ae_low_flow":    _masked_mean(all_ae_t, low_mask),   # GT mag < 1px
        "ae_high_flow":   _masked_mean(all_ae_t, high_mask),  # GT mag >= 3px
        # nPE: % of valid pixels with EPE > n px  (outlier fractions)
        "1pe":            (all_epe_t > 1.0).float().mean().item() * 100,
        "2pe":            (all_epe_t > 2.0).float().mean().item() * 100,
        "3pe":            (all_epe_t > 3.0).float().mean().item() * 100,
        "n_valid_pixels": float(len(all_epe_t)),
        "n_low_flow_px":  float(low_mask.sum().item()),
        "n_high_flow_px": float(high_mask.sum().item()),
    }

    for name, s in rate_sums.items():
        key = "firing_rate_" + name.replace(".", "_")
        metrics[key] = s / max(rate_counts[name], 1)

    return metrics


@torch.no_grad()
def log_val_prediction(model, loader, device, epoch):
    """Run model on the first val batch and log a flow figure to MLflow."""
    model.eval()
    batch = next(iter(loader))
    x = batch["event_volume_new"].to(device, non_blocking=True)
    flow_gt = batch["flow_gt_event_volume_new"].to(device, non_blocking=True)
    mask = batch["flow_gt_event_volume_new_valid_mask"].to(device, non_blocking=True)

    final_flow, _, _ = model(x)
    pred = final_flow[:, -1]  # (B, 2, H, W)

    # Take first sample in batch
    events_np = x[0].cpu().numpy()             # (T, H, W)
    gt_np = flow_gt[0].cpu().numpy()           # (2, H, W)
    pred_np = pred[0].cpu().numpy()            # (2, H, W)
    mask_np = mask[0, 0].cpu().numpy()         # (H, W)

    fig = make_flow_figure(events_np, gt_np, pred_np, mask_np)

    with tempfile.NamedTemporaryFile(suffix=f"_epoch{epoch:03d}.png", delete=False) as f:
        fig.savefig(f.name, dpi=100, bbox_inches="tight")
        mlflow.log_artifact(f.name, artifact_path="flow_predictions")
        os.unlink(f.name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",       choices=["spikeflownet", "full_spike", "silif_scaled"],
                        default="spikeflownet",
                        help="Model architecture (default: spikeflownet)")
    parser.add_argument("--val_seq",        nargs="+",  default=DEFAULT_VAL_SEQ)
    parser.add_argument("--downsample_ratio", type=int, default=2,
                        help="Spatial downsample factor (2 → 240×320, 1 → full 480×640)")
    parser.add_argument("--channels",   nargs=3, type=int, default=[32, 64, 128])
    parser.add_argument("--v_threshold", type=float, default=0.5,
                        help="SiLIF firing threshold (only used by spikeflownet)")
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=2)
    parser.add_argument("--ckpt",       default=None,
                        help="Checkpoint path (default: best_<arch>.pt)")
    parser.add_argument("--viz_every",   type=int,   default=5,
                        help="Log flow prediction image every N epochs")
    parser.add_argument("--accum_steps", type=int,   default=1,
                        help="Gradient accumulation steps (effective batch = batch_size × accum_steps)")
    parser.add_argument("--amp",          action="store_true",
                        help="Enable automatic mixed precision (fp16 forward, fp32 params)")
    parser.add_argument("--native_upsample", action="store_true",
                        help="Upsample predictions to native 480×640 inside SpikeFlowNet and supervise with native GT")
    parser.add_argument("--resume",      default=None,
                        help="Path to checkpoint (.pt) to initialize/resume model weights from")
    parser.add_argument("--run_name",    default=None,
                        help="Custom MLflow run name (default: <arch> or <arch>_native_upsample)")
    parser.add_argument("--seed",        type=int, default=3407,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.ckpt is None:
        suffix = "_native_upsample" if args.native_upsample else "_fix_gt_ds"
        args.ckpt = f"best_{args.arch}{suffix}.pt"
    if args.run_name is None:
        args.run_name = f"{args.arch}_native_upsample" if args.native_upsample else f"{args.arch}_fix_gt_ds"
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device} | Run: {args.run_name} | Native upsample: {args.native_upsample}", flush=True)

    loader_train, loader_val = make_loaders(args)
    loader_val_native = loader_val if args.native_upsample else make_native_val_loader(args)
    print(f"Train batches: {len(loader_train)} | Val batches: {len(loader_val)}", flush=True)

    if args.arch == "spikeflownet":
        model = SpikeFlowNet(
            in_channels=1,
            channels=tuple(args.channels),
            v_threshold=args.v_threshold,
            dropout=args.dropout,
            native_upsample=args.native_upsample,
        ).to(device)
    elif args.arch == "full_spike":
        model = FullSpikeFlowNet(
            in_channels=1,
            channels=tuple(args.channels),
            v_threshold=args.v_threshold,
            dropout=args.dropout,
        ).to(device)
    else:  # silif_scaled
        model = SiLIFFlowNetScaled(
            in_channels=1,
            channels=args.channels,
            dropout=args.dropout,
            v_threshold=args.v_threshold,
        ).to(device)

    if args.resume and os.path.isfile(args.resume):
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Resumed weights from: {args.resume}", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None
    print(f"Mixed precision: {'ON (fp16)' if scaler else 'OFF (fp32)'}")

    mlflow.set_experiment("DSEC_SiLIF_Flow")
    with mlflow.start_run(run_name=args.run_name):
        out_res = "480x640" if args.native_upsample else f"{480 // args.downsample_ratio}x{640 // args.downsample_ratio}"
        mlflow.log_params({
            "arch":             args.arch,
            "run_name":         args.run_name,
            "native_upsample":  args.native_upsample,
            "gt_downsampling":  "native_480x640" if args.native_upsample else "mask_normalized_avgpool",
            "channels":         args.channels,
            "v_threshold":      args.v_threshold,
            "dropout":          args.dropout,
            "batch_size":       args.batch_size,
            "epochs":           args.epochs,
            "lr":               args.lr,
            "n_params":         n_params,
            "device":           str(device),
            "val_seq":          str(args.val_seq),
            "downsample_ratio":  args.downsample_ratio,
            "input_resolution":  f"{480 // args.downsample_ratio}x{640 // args.downsample_ratio}",
            "output_resolution": out_res,
            "accum_steps":       args.accum_steps,
            "effective_batch":   args.batch_size * args.accum_steps,
            "amp":               args.amp,
            "seed":              args.seed,
        })

        best_val_epe = float("inf")
        best_val_epe_native = float("inf")

        for epoch in range(1, args.epochs + 1):
            tr_loss = train_one_epoch(model, loader_train, optimizer, device, args.accum_steps, scaler)
            val_loss, val_epe = evaluate(model, loader_val, device)
            if args.native_upsample:
                val_epe_native = val_epe
            else:
                _, val_epe_native = evaluate(model, loader_val_native, device)
            scheduler.step()

            lr_now = optimizer.param_groups[0]["lr"]
            best_val_epe_native = min(best_val_epe_native, val_epe_native)
            mlflow.log_metrics({
                "train_loss":            tr_loss,
                "val_loss":              val_loss,
                "val_epe":               val_epe,
                "val_epe_native_480x640": val_epe_native,
                "lr":                    lr_now,
            }, step=epoch)

            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"train {tr_loss:.4f} | val {val_loss:.4f} | EPE {val_epe:.4f} px "
                f"(native 480x640: {val_epe_native:.4f} px) | lr {lr_now:.2e}",
                flush=True,
            )

            if val_epe < best_val_epe:
                best_val_epe = val_epe
                torch.save(model.state_dict(), args.ckpt)
                print(f"  -> new best ({best_val_epe:.4f} px), saved to {args.ckpt}", flush=True)

            if epoch % args.viz_every == 0 or epoch == 1:
                log_val_prediction(model, loader_val, device, epoch)

        mlflow.log_metric("best_val_epe", best_val_epe)
        mlflow.log_metric("best_val_epe_native_480x640", best_val_epe_native)
        mlflow.log_artifact(args.ckpt)
        print(f"\nDone. Best val EPE: {best_val_epe:.4f} px (native 480x640: {best_val_epe_native:.4f} px)")

        # --- comprehensive final evaluation on the best checkpoint ---
        print("\nRunning final evaluation on best checkpoint...")
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        final_metrics = evaluate_full(model, loader_val, device)
        mlflow.log_metrics({f"final_{k}": v for k, v in final_metrics.items()})

        if args.native_upsample:
            final_native = final_metrics
        else:
            final_native = evaluate_full(model, loader_val_native, device)
        mlflow.log_metrics({f"final_native_{k}": v for k, v in final_native.items() if not k.startswith("firing_rate_")})

        n = int(final_metrics["n_valid_pixels"])
        print(f"\n{'='*60}")
        print(f"  Final Evaluation ({out_res} space, {n:,} valid pixels)")
        print(f"{'='*60}")
        print(f"  EPE (AEE mean)    {final_metrics['epe']:.4f} px  [Native 480x640: {final_native['epe']:.4f} px]")
        print(f"  AEE median        {final_metrics['aee_median']:.4f} px  [Native 480x640: {final_native['aee_median']:.4f} px]")
        print(f"  AE (angular err)  {final_metrics['ae']:.4f} deg [Native 480x640: {final_native['ae']:.4f} deg]")
        print(f"  1PE (EPE >1px)    {final_metrics['1pe']:.2f}%  [Native 480x640: {final_native['1pe']:.2f}%]")
        print(f"  2PE (EPE >2px)    {final_metrics['2pe']:.2f}%  [Native 480x640: {final_native['2pe']:.2f}%]")
        print(f"  3PE (EPE >3px)    {final_metrics['3pe']:.2f}%  [Native 480x640: {final_native['3pe']:.2f}%]")
        firing_keys = sorted(k for k in final_metrics if k.startswith("firing_rate_"))
        if firing_keys:
            print(f"{'─'*60}")
            print("  Firing rates:")
            for k in firing_keys:
                layer = k[len("firing_rate_"):]
                print(f"    {layer:<22} {final_metrics[k]*100:.2f}%")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()