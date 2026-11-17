"""
Generate a validation video for SpikeFlowNet (--native_upsample) on DSEC (zurich_city_01_a).

Displays:
  1. Event Frame (ON/OFF polarity accumulation)
  2. Model Inference — Unmasked Dense Flow (native 480x640, standard downstream usage)
  3. Model Inference — Masked to Valid GT Pixels (native 480x640)
  4. Ground Truth Flow — Masked (native 480x640 LiDAR GT)
  5. Continuous CPU Inference Latency (ms/frame) & EPE trace across the sequence

Usage:
    python visualize_dsec_flow_video.py
    # train_fast_flow.py checkpoint (QAT included), one validation recording, run as the timed
    # deployment path (BN folded + compiled, fp16, one CUDA graph per frame):
    python visualize_dsec_flow_video.py --fast_ckpt checkpoints/fast_flow/h200_p3_int8_D_j33273.pt \
        --val_seq zurich_city_08_a --out_mp4 figures/x.mp4 --out_gif figures/x.gif
"""

import argparse
import os
import subprocess
import time

import cv2
import hdf5plugin  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from dsec_optical.dsec_utils import RepresentationType
from dsec_optical.loader_dsec import assemble_dsec_sequences
from models.spikeflownet import SpikeFlowNet, _up
from train_dsec_flow import DATA_ROOT, DEFAULT_VAL_SEQ


def flow_to_rgb_uint8(flow_2hw: np.ndarray, max_flow: float = None, dark_bg: bool = False) -> np.ndarray:
    """
    Convert (2, H, W) optical flow to (H, W, 3) RGB uint8 using HSV color wheel.
    If dark_bg=True, zero-magnitude pixels are rendered on a dark slate background.
    """
    u, v = flow_2hw[0], flow_2hw[1]
    mag = np.sqrt(u ** 2 + v ** 2)
    ang = np.arctan2(-v, -u)  # [-pi, pi]

    if max_flow is None or max_flow <= 0:
        max_flow = max(float(np.percentile(mag, 99)), 1.0)

    hue = (ang + np.pi) / (2.0 * np.pi)
    sat = np.clip(mag / max(max_flow, 1e-6), 0.0, 1.0)

    if dark_bg:
        val = np.where(mag > 1e-5, 0.25 + 0.75 * sat, 0.08)
        sat_hsv = np.where(mag > 1e-5, 0.85, 0.0)
        hsv = np.stack([hue, sat_hsv, val], axis=-1).astype(np.float32)
    else:
        val = np.ones_like(sat)
        hsv = np.stack([hue, sat, val], axis=-1).astype(np.float32)

    rgb = mcolors.hsv_to_rgb(hsv)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def make_color_wheel_uint8(size: int = 56) -> np.ndarray:
    """Generate a circular HSV optical flow legend (size, size, 3) uint8 + alpha mask."""
    y, x = np.mgrid[-1:1:complex(0, size), -1:1:complex(0, size)]
    r = np.sqrt(x ** 2 + y ** 2)
    ang = np.arctan2(-y, -x)
    hue = (ang + np.pi) / (2.0 * np.pi)
    sat = np.clip(r, 0.0, 1.0)
    val = np.ones_like(sat)
    rgb = (mcolors.hsv_to_rgb(np.stack([hue, sat, val], axis=-1)) * 255.0).astype(np.uint8)
    mask = r <= 1.0
    return rgb, mask


def events_to_rgb_uint8(voxel_thw: np.ndarray, out_h: int = 480, out_w: int = 640) -> np.ndarray:
    """
    Render (T, H, W) event voxel grid into a dark-background ON/OFF RGB image (out_h, out_w, 3).
    Positive polarity -> Red/Coral, Negative polarity -> Cyan/Blue.
    """
    pos = np.maximum(voxel_thw, 0.0).sum(axis=0)
    neg = np.maximum(-voxel_thw, 0.0).sum(axis=0)

    p99 = max(float(np.percentile(pos + neg, 98)), 1e-3)
    pos_n = np.clip(pos / p99, 0.0, 1.0)
    neg_n = np.clip(neg / p99, 0.0, 1.0)

    bg = 18.0
    r = bg + (255.0 - bg) * pos_n + 30.0 * neg_n
    g = bg + 65.0 * pos_n + 190.0 * neg_n
    b = bg + 55.0 * pos_n + (255.0 - bg) * neg_n

    rgb = np.stack([np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)], axis=-1).astype(np.uint8)
    if rgb.shape[0] != out_h or rgb.shape[1] != out_w:
        rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return rgb


def forward_with_submodule_timing(model: SpikeFlowNet, x: torch.Tensor, device: torch.device):
    """
    Run SpikeFlowNet forward pass while timing:
      1. Total model forward time (ms)
      2. Spiking encoder time (enc1 + enc2 + enc3 over T=15) (ms)
      3. 2D decoder + native upsampling time (dec3..0) (ms)
    Uses the requested synchronize + time.perf_counter() pattern.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    s_enc = time.perf_counter()
    x_seq = x.unsqueeze(2)
    s1 = model.enc1(x_seq)
    s2 = model.enc2(s1)
    s3 = model.enc3(s2)
    f1 = s1.mean(dim=1)
    f2 = s2.mean(dim=1)
    f3 = s3.mean(dim=1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_mid = time.perf_counter()

    d3 = model.dec3(f3)
    coarse_flow = model.flow_head3(d3)
    coarse_up = _up(coarse_flow)
    d2 = model.dec2(torch.cat([_up(d3), f2], dim=1))
    medium_flow = model.flow_head2(d2) + coarse_up
    medium_up = _up(medium_flow)
    d1 = model.dec1(torch.cat([_up(d2), f1], dim=1))
    fine_flow = model.flow_head1(d1) + medium_up
    if model.native_upsample:
        fine_up = _up(fine_flow) * 2.0
        d0 = model.dec0(torch.cat([_up(d1), fine_up], dim=1))
        pred_flow_2hw = (model.flow_head0(d0) + fine_up)[0]
    else:
        pred_flow_2hw = fine_flow[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    enc_ms = (t_mid - s_enc) * 1000.0
    dec_ms = (t_end - t_mid) * 1000.0
    total_ms = (t_end - s_enc) * 1000.0
    return pred_flow_2hw, total_ms, enc_ms, dec_ms


def draw_panel_header(img_bgr: np.ndarray, title: str, badge_text: str = ""):
    """Draw a top header bar for the title and a bottom-left pill badge for metrics/info."""
    h, w = img_bgr.shape[:2]
    # Top header banner (title only — never collides with right side)
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, 34), (15, 18, 24), -1)
    cv2.addWeighted(overlay, 0.80, img_bgr, 0.20, 0, img_bgr)
    cv2.putText(img_bgr, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 247, 250), 1, cv2.LINE_AA)

    # Bottom-left badge pill for subtitle/metrics
    if badge_text:
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        bx0, by0 = 10, h - 36
        bx1, by1 = bx0 + tw + 18, h - 8
        sub_ov = img_bgr.copy()
        cv2.rectangle(sub_ov, (bx0, by0), (bx1, by1), (15, 18, 24), -1)
        cv2.addWeighted(sub_ov, 0.82, img_bgr, 0.18, 0, img_bgr)
        cv2.rectangle(img_bgr, (bx0, by0), (bx1, by1), (70, 80, 100), 1)
        cv2.putText(
            img_bgr, badge_text, (bx0 + 9, by1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (130, 220, 255), 1, cv2.LINE_AA
        )


def draw_telemetry_panel(
    width: int,
    height: int,
    frame_idx: int,
    n_frames: int,
    total_ms_hist: list,
    enc_ms_hist: list,
    dec_ms_hist: list,
    epe_hist: list,
    device_label: str,
    seq_label: str = "zurich_city_01_a (Validation Split)",
) -> np.ndarray:
    """
    Draw the bottom continuous CPU inference time & EPE telemetry dashboard.
    All titles and legends sit in dedicated rows above the plot boxes with zero overlap.
    """
    panel = np.full((height, width, 3), (20, 23, 30), dtype=np.uint8)
    cv2.line(panel, (0, 0), (width, 0), (65, 72, 90), 2)

    pad_l, pad_r, pad_t, pad_b = 62, 24, 58, 32
    mid_x = int(width * 0.56)

    # --- Plot 1: Continuous CPU Inference Latency (ms) ---
    x0, x1 = pad_l, mid_x - 24
    y0, y1 = pad_t, height - pad_b
    cv2.rectangle(panel, (x0, y0), (x1, y1), (28, 32, 42), -1)
    cv2.rectangle(panel, (x0, y0), (x1, y1), (55, 62, 78), 1)

    cur_tot = total_ms_hist[-1]
    split = bool(enc_ms_hist)  # (no encoder/decoder split for the fused fast-flow models)
    mean_tot = float(np.mean(total_ms_hist))
    fps = 1000.0 / max(mean_tot, 1e-3)

    # Row 1 (y=22): Title only (strictly within [x0, x1])
    title_lat = f"Continuous {device_label} Inference Time  |  Mean: {mean_tot:.1f} ms ({fps:.1f} FPS)"
    cv2.putText(panel, title_lat, (x0, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 244, 250), 1, cv2.LINE_AA)

    # Row 2 (y=44): Dedicated Legend Bar above the plot box
    leg_items = [(f"Total: {cur_tot:.2f} ms", (80, 210, 255))]
    if split:
        leg_items += [(f"Spiking Enc (T=15): {enc_ms_hist[-1]:.1f} ms", (255, 140, 60)),
                      (f"2D Dec+Up: {dec_ms_hist[-1]:.1f} ms", (255, 190, 90))]
    leg_items += [("Mean", (120, 220, 140))]
    lx = x0
    for text, color in leg_items:
        cv2.circle(panel, (lx + 5, 40), 4, color, -1, cv2.LINE_AA)
        cv2.putText(panel, text, (lx + 14, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        lx += tw + 30

    y_max = max(float(np.percentile(total_ms_hist, 98)) * 1.30, 5.0 if split else 1.0)
    y_min = 0.0

    for frac in [0.25, 0.5, 0.75, 1.0]:
        val = y_min + frac * (y_max - y_min)
        gy = int(y1 - frac * (y1 - y0))
        cv2.line(panel, (x0, gy), (x1, gy), (42, 47, 60), 1)
        lbl = f"{val:.1f}ms" if y_max < 25.0 else f"{val:.0f}ms"
        cv2.putText(panel, lbl, (6, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 160, 175), 1, cv2.LINE_AA)

    mean_y = int(np.clip(y1 - (mean_tot / y_max) * (y1 - y0), y0, y1))
    for gx in range(x0, x1 - 6, 12):
        cv2.line(panel, (gx, mean_y), (gx + 6, mean_y), (120, 220, 140), 1)

    def to_pts(vals):
        pts = []
        for i, v in enumerate(vals):
            px = int(x0 + (i / max(n_frames - 1, 1)) * (x1 - x0))
            py = int(np.clip(y1 - (v / y_max) * (y1 - y0), y0 + 1, y1 - 1))
            pts.append([px, py])
        return np.array(pts, dtype=np.int32)

    if len(total_ms_hist) >= 2:
        pts_tot = to_pts(total_ms_hist)
        if split:
            cv2.polylines(panel, [to_pts(dec_ms_hist)], False, (255, 190, 90), 1, cv2.LINE_AA)
            cv2.polylines(panel, [to_pts(enc_ms_hist)], False, (255, 140, 60), 2, cv2.LINE_AA)
        cv2.polylines(panel, [pts_tot], False, (80, 210, 255), 2, cv2.LINE_AA)
        cv2.circle(panel, tuple(pts_tot[-1]), 4, (80, 210, 255), -1, cv2.LINE_AA)

    cv2.putText(
        panel, f"Frame {frame_idx + 1}/{n_frames} (t = {(frame_idx + 1) * 0.1:.1f} s)",
        ((x0 + x1) // 2 - 75, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 180, 195), 1, cv2.LINE_AA
    )

    # --- Plot 2: Continuous Native 480x640 Validation EPE (px) ---
    rx0, rx1 = mid_x + 46, width - pad_r
    ry0, ry1 = pad_t, height - pad_b
    cv2.rectangle(panel, (rx0, ry0), (rx1, ry1), (28, 32, 42), -1)
    cv2.rectangle(panel, (rx0, ry0), (rx1, ry1), (55, 62, 78), 1)

    cur_epe = epe_hist[-1]
    mean_epe = float(np.mean(epe_hist))

    # Row 1 (y=22): Title only
    cv2.putText(panel, "Native 480x640 Validation EPE (px)", (rx0, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 244, 250), 1, cv2.LINE_AA)

    # Row 2 (y=44): Dedicated Legend Bar above Plot 2
    cv2.circle(panel, (rx0 + 5, 40), 4, (130, 130, 255), -1, cv2.LINE_AA)
    cv2.putText(panel, f"Frame EPE: {cur_epe:.2f} px", (rx0 + 14, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 255), 1, cv2.LINE_AA)
    cv2.circle(panel, (rx0 + 195, 40), 4, (120, 220, 140), -1, cv2.LINE_AA)
    cv2.putText(panel, f"Running Mean: {mean_epe:.2f} px", (rx0 + 204, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 220, 140), 1, cv2.LINE_AA)

    epe_max = max(float(np.max(epe_hist)) * 1.15, 4.5)
    for frac in [0.25, 0.5, 0.75, 1.0]:
        val = frac * epe_max
        gy = int(ry1 - frac * (ry1 - ry0))
        cv2.line(panel, (rx0, gy), (rx1, gy), (42, 47, 60), 1)
        cv2.putText(panel, f"{val:.1f}px", (rx0 - 44, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 160, 175), 1, cv2.LINE_AA)

    mean_epe_y = int(np.clip(ry1 - (mean_epe / epe_max) * (ry1 - ry0), ry0, ry1))
    for gx in range(rx0, rx1 - 6, 12):
        cv2.line(panel, (gx, mean_epe_y), (gx + 6, mean_epe_y), (120, 220, 140), 1)

    if len(epe_hist) >= 2:
        pts_epe = []
        for i, v in enumerate(epe_hist):
            px = int(rx0 + (i / max(n_frames - 1, 1)) * (rx1 - rx0))
            py = int(np.clip(ry1 - (v / epe_max) * (ry1 - ry0), ry0 + 1, ry1 - 1))
            pts_epe.append([px, py])
        pts_epe = np.array(pts_epe, dtype=np.int32)
        cv2.polylines(panel, [pts_epe], False, (130, 130, 255), 2, cv2.LINE_AA)
        cv2.circle(panel, tuple(pts_epe[-1]), 4, (130, 130, 255), -1, cv2.LINE_AA)

    cv2.putText(
        panel, seq_label,
        ((rx0 + rx1) // 2 - 110, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 180, 195), 1, cv2.LINE_AA
    )

    return panel


def load_fast_flow_runner(ckpt_path: str, device: torch.device):
    """train_fast_flow.py checkpoint (QAT included) -> the path timed in train_fast_flow.py
    (latency_fp16_fused_graph_ms): BN folded + SiLIF custom op + torch.compile, fp16, CUDA graph.
    QAT fake-quant weights are baked in (INT8-valued) before folding. Returns run(x) -> (pred, ms), name."""
    from types import SimpleNamespace
    from torch.nn.utils import parametrize
    import train_fast_flow as tff
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = SimpleNamespace(**ck["args"])
    model = tff.build_model(a)
    if a.qat_bits:
        tff.apply_qat(model, a.qat_bits, a.qat_act_bits)
    model.load_state_dict(ck["model"])
    for m in model.modules():
        if parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
    n_params = sum(p.numel() for p in model.parameters())
    fused = torch.compile(tff.fuse_for_inference(tff.inference_copy(model.to(device))).half(), dynamic=False)
    x_static = torch.zeros(1, tff.BINS, tff.H, tff.W, device=device, dtype=torch.half)
    with torch.inference_mode():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                fused(x_static)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fused(x_static)[0]

    def run(x):
        with torch.inference_mode():  # (the graph was captured in inference mode)
            x_static.copy_(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0
            return out[0].float().clone(), ms

    quant = f" INT{a.qat_bits}" if a.qat_bits else ""
    name = f"Spiking T={a.T}{quant} ({n_params / 1e3:.0f}k)"
    print(f"{ckpt_path}: {name}, val EPE {ck['val']['epe']:.4f}, spike rate {ck['val'].get('rate_avg', 0):.4f}", flush=True)
    return run, name


def main():
    parser = argparse.ArgumentParser(description="Create DSEC validation optical flow video with live CPU latency")
    parser.add_argument("--ckpt", default="checkpoints/dsec_flow/best_spikeflownet_native_upsample.pt")
    parser.add_argument("--val_seq", nargs="+", default=DEFAULT_VAL_SEQ)
    parser.add_argument("--fast_ckpt", default=None,
                        help="train_fast_flow.py checkpoint instead of --ckpt (GPU; cached voxels of ONE --val_seq)")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--out_mp4", default="figures/dsec_val_spikeflownet_native.mp4")
    parser.add_argument("--out_gif", default="figures/dsec_val_spikeflownet_native.gif")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_mp4) or ".", exist_ok=True)
    torch.set_num_threads(args.num_threads)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    device_label = f"CPU ({args.num_threads}T)" if device.type == "cpu" else "GPU"

    if args.fast_ckpt:
        from dsec_optical.flow_cache import CachedFlowDataset
        assert device.type == "cuda" and len(args.val_seq) == 1, "--fast_ckpt: GPU and a single --val_seq"
        run_fast, model_name = load_fast_flow_runner(args.fast_ckpt, device)
        device_label = "GPU fused fp16 graph"
        cached = CachedFlowDataset(args.val_seq)
    else:
        model_name = "SpikeFlowNet"
    val_config = OmegaConf.create({
        "sequence_length": 1,
        "downsample_ratio": 2,
        "val_seq": args.val_seq,
        "native_gt": True,
        "only_new": True,
    })
    val_set = cached if args.fast_ckpt else assemble_dsec_sequences(
        dataset_root=DATA_ROOT,
        config=val_config,
        include_seq=args.val_seq,
        representation_type=RepresentationType.VOXEL,
    )

    if not args.fast_ckpt:
        model = SpikeFlowNet(
            in_channels=1,
            channels=(32, 64, 128),
            v_threshold=0.5,
            dropout=0.0,
            native_upsample=True,
        ).to(device)
        state = torch.load(args.ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        # Warmup 3 passes so initial threadpool/allocator overhead does not skew frame 1
        with torch.no_grad():
            dummy = torch.zeros(1, 15, 240, 320, device=device)
            for _ in range(3):
                _ = model(dummy)

    # Panel dimensions: 2x2 grid of (640 x 440) panels -> 1280 x 880 + 220px telemetry = 1280 x 1100
    pw, ph = 640, 480
    disp_w, disp_h = 640, 440
    telem_h = 220
    canvas_w = disp_w * 2
    canvas_h = disp_h * 2 + telem_h

    wheel_rgb, wheel_mask = make_color_wheel_uint8(size=52)
    wheel_bgr = cv2.cvtColor(wheel_rgb, cv2.COLOR_RGB2BGR)

    n_frames = len(val_set)
    print(f"Rendering {n_frames} validation frames on {device_label} from {args.fast_ckpt or args.ckpt} ...", flush=True)

    tmp_mp4 = args.out_mp4 + ".raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_mp4, fourcc, float(args.fps), (canvas_w, canvas_h))

    total_ms_hist = []
    enc_ms_hist = []
    dec_ms_hist = []
    epe_hist = []
    gif_frames_rgb = []

    with torch.no_grad():
        for idx in range(n_frames):
            if args.fast_ckpt:
                vox, flow, mask, _ = val_set[idx]
                sample = {"event_volume_new": vox}
                x = vox[None].to(device)
                flow_gt = flow.numpy()
                valid_mask = mask[0].numpy()
                pred_flow_t, total_ms = run_fast(x)
            else:
                sample = val_set[idx]
                x = sample["event_volume_new"].unsqueeze(0).to(device)        # (1, 15, 240, 320)
                flow_gt = sample["flow_gt_event_volume_new"][0].numpy()       # (2, 480, 640)
                valid_mask = sample["flow_gt_event_volume_new"][1][0].numpy().astype(bool)  # (480, 640)
                pred_flow_t, total_ms, enc_ms, dec_ms = forward_with_submodule_timing(model, x, device)
            pred_flow = pred_flow_t.cpu().numpy()  # (2, 480, 640)

            # Compute frame EPE & 3PE on valid LiDAR pixels
            diff = pred_flow - flow_gt
            epe_map = np.sqrt(np.sum(diff ** 2, axis=0) + 1e-8)
            valid_epe = epe_map[valid_mask]
            frame_epe = float(valid_epe.mean()) if valid_epe.size > 0 else 0.0
            frame_3pe = float((valid_epe > 3.0).mean() * 100.0) if valid_epe.size > 0 else 0.0

            total_ms_hist.append(total_ms)
            if not args.fast_ckpt:
                enc_ms_hist.append(enc_ms)
                dec_ms_hist.append(dec_ms)
            epe_hist.append(frame_epe)

            # Shared flow normalization scale from GT valid pixels
            gt_mag = np.sqrt(flow_gt[0] ** 2 + flow_gt[1] ** 2)
            max_flow = max(float(np.percentile(gt_mag[valid_mask], 98)) if valid_mask.any() else 10.0, 2.0)

            # 1. Event Frame
            ev_rgb = events_to_rgb_uint8(sample["event_volume_new"].numpy(), out_h=ph, out_w=pw)
            ev_bgr = cv2.cvtColor(ev_rgb, cv2.COLOR_RGB2BGR)

            # 2. Dense Predicted Flow (Unmasked - standard optical flow usage)
            pred_dense_rgb = flow_to_rgb_uint8(pred_flow, max_flow=max_flow, dark_bg=False)
            pred_dense_bgr = cv2.cvtColor(pred_dense_rgb, cv2.COLOR_RGB2BGR)
            # Stamp color wheel in bottom-right corner
            wy0, wx0 = ph - 62, pw - 62
            pred_dense_bgr[wy0:wy0 + 52, wx0:wx0 + 52][wheel_mask] = wheel_bgr[wheel_mask]

            # 3. Masked Predicted Flow (on valid GT pixels)
            pred_masked = np.where(valid_mask[None, :, :], pred_flow, 0.0)
            pred_mask_rgb = flow_to_rgb_uint8(pred_masked, max_flow=max_flow, dark_bg=True)
            pred_mask_bgr = cv2.cvtColor(pred_mask_rgb, cv2.COLOR_RGB2BGR)

            # 4. Masked Ground Truth Flow
            gt_masked = np.where(valid_mask[None, :, :], flow_gt, 0.0)
            gt_mask_rgb = flow_to_rgb_uint8(gt_masked, max_flow=max_flow, dark_bg=True)
            gt_mask_bgr = cv2.cvtColor(gt_mask_rgb, cv2.COLOR_RGB2BGR)
            gt_mask_bgr[wy0:wy0 + 52, wx0:wx0 + 52][wheel_mask] = wheel_bgr[wheel_mask]

            # Resize 4 panels to display size
            p1 = cv2.resize(ev_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            p2 = cv2.resize(pred_dense_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            p3 = cv2.resize(pred_mask_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            p4 = cv2.resize(gt_mask_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

            draw_panel_header(p1, f"1. Event Frame (T=15 Voxel, Frame {idx + 1}/{n_frames})", "ON (Red) / OFF (Cyan)")
            draw_panel_header(p2, f"2. {model_name} Inference (Dense 480x640)", f"Latency: {total_ms:.2f} ms")
            draw_panel_header(p3, f"3. {model_name} Inference (Masked to GT)", f"EPE: {frame_epe:.2f} px | 3PE: {frame_3pe:.1f}%")
            draw_panel_header(p4, "4. Ground Truth Optical Flow (Native 480x640)", f"Max scale: {max_flow:.1f} px")

            # Assemble 2x2 grid + bottom telemetry panel
            top_row = np.hstack([p1, p2])
            mid_row = np.hstack([p3, p4])
            cv2.line(top_row, (disp_w, 0), (disp_w, disp_h), (55, 62, 78), 1)
            cv2.line(mid_row, (disp_w, 0), (disp_w, disp_h), (55, 62, 78), 1)
            cv2.line(mid_row, (0, 0), (canvas_w, 0), (55, 62, 78), 1)

            telem = draw_telemetry_panel(
                canvas_w, telem_h, idx, n_frames,
                total_ms_hist, enc_ms_hist, dec_ms_hist, epe_hist, device_label,
                **({"seq_label": f"{args.val_seq[0]} (validation split)"} if args.fast_ckpt else {})
            )

            frame_bgr = np.vstack([top_row, mid_row, telem])
            writer.write(frame_bgr)
            if idx == 50:
                cv2.imwrite(os.path.splitext(args.out_mp4)[0] + "_preview.png" if args.fast_ckpt
                            else "figures/dsec_val_preview_frame.png", frame_bgr)

            # Save every 2nd frame at 800px width for lightweight inline GIF
            if idx % 2 == 0:
                gif_w = 840
                gif_h = int(canvas_h * (gif_w / canvas_w))
                small_rgb = cv2.cvtColor(
                    cv2.resize(frame_bgr, (gif_w, gif_h), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2RGB,
                )
                gif_frames_rgb.append(small_rgb)

            if (idx + 1) % 25 == 0 or (idx + 1) == n_frames:
                print(
                    f"  Frame {idx + 1:3d}/{n_frames} | "
                    f"Latency: {total_ms:.2f} ms (mean {np.mean(total_ms_hist):.2f} ms) | "
                    f"EPE: {frame_epe:.3f} px (mean {np.mean(epe_hist):.3f} px)",
                    flush=True,
                )

    writer.release()

    # Re-encode to browser-compatible H.264 MP4 using conda ffmpeg
    ffmpeg_bin = "/home/geoffroyk/miniconda3/envs/IDNet/bin/ffmpeg"
    if os.path.exists(ffmpeg_bin):
        subprocess.run(
            [
                ffmpeg_bin, "-y", "-i", tmp_mp4,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "20", "-preset", "fast",
                args.out_mp4,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.remove(tmp_mp4)
    else:
        os.replace(tmp_mp4, args.out_mp4)

    # Save animated GIF using imageio
    import imageio
    imageio.mimsave(args.out_gif, gif_frames_rgb, fps=max(args.fps // 2, 5), loop=0)

    print(f"\nSaved H.264 MP4 video : {args.out_mp4}")
    print(f"Saved animated GIF    : {args.out_gif}")
    print(f"Overall Mean Latency  : {np.mean(total_ms_hist):.2f} ms/frame ({1000.0 / np.mean(total_ms_hist):.1f} FPS)")
    if enc_ms_hist:
        print(f"  - Spiking Encoder   : {np.mean(enc_ms_hist):.2f} ms/frame ({100.0 * np.mean(enc_ms_hist) / np.mean(total_ms_hist):.1f}%)")
        print(f"  - 2D Dec + Upsample : {np.mean(dec_ms_hist):.2f} ms/frame ({100.0 * np.mean(dec_ms_hist) / np.mean(total_ms_hist):.1f}%)")
    print(f"Overall Mean Val EPE  : {np.mean(epe_hist):.4f} px")


if __name__ == "__main__":
    main()
