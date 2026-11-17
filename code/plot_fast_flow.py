"""
Qualitative DSEC validation figure for a train_fast_flow.py checkpoint (QAT checkpoints included):
one row per validation recording (the sample at the middle of the recording), columns
events | prediction (dense) | prediction on valid GT pixels | GT | EPE map.
Pred and GT of a row share the same colour scale (GT 99th-percentile magnitude).
Reuses the colour helpers of visualize_dsec_flow_video.py.

    python plot_fast_flow.py --ckpt checkpoints/fast_flow/h200_p3_int8_D_j33273.pt
"""
import argparse
import os
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from train_fast_flow import apply_qat, build_model, BINS
from dsec_optical.flow_cache import CachedFlowDataset, _FLOW_VAL_RECORDINGS
from visualize_dsec_flow_video import events_to_rgb_uint8, flow_to_rgb_uint8, make_color_wheel_uint8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/fast_flow/h200_p3_int8_D_j33273.pt")
    ap.add_argument("--out", default="figures/fast_flow_qualitative")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = SimpleNamespace(**ck["args"])
    model = build_model(a)
    if a.qat_bits:
        apply_qat(model, a.qat_bits, a.qat_act_bits)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    quant = f"INT{a.qat_bits} weights" if a.qat_bits else "FP"
    title = (f"{a.run_name} ({a.arch}, T={a.T}, {quant}, {sum(p.numel() for p in model.parameters()):,} params) | "
             f"val EPE {ck['val']['epe']:.3f}, spike rate {100 * ck['val'].get('rate_avg', 0):.1f}%")
    print(title, flush=True)

    ds = CachedFlowDataset(_FLOW_VAL_RECORDINGS, bins=BINS)
    rows = []
    for r, seq in enumerate(ds.recordings):
        idx = [i for i, (rr, _) in enumerate(ds.items) if rr == r]
        vox, flow, mask, _ = ds[idx[len(idx) // 2]]
        with torch.no_grad():
            pred = model(vox[None].to(device))[0].float()
            pred = F.interpolate(pred, size=flow.shape[-2:], mode="bilinear",
                                 align_corners=False) * (flow.shape[-1] / pred.shape[-1])
        pred, flow, mask = pred[0].cpu().numpy(), flow.numpy(), mask[0].numpy()
        epe = np.linalg.norm(pred - flow, axis=0)
        rows.append((seq, vox.numpy(), pred, flow, mask, epe))

    cols = ["Events", "Prediction (dense)", "Prediction (valid GT px)", "Ground truth", "EPE (px)"]
    fig, axes = plt.subplots(len(rows), 5, figsize=(5 * 3.2, len(rows) * 2.6 + 0.6))
    for i, (seq, vox, pred, flow, mask, epe) in enumerate(rows):
        vmax = max(float(np.percentile(np.linalg.norm(flow, axis=0)[mask], 99)), 1.0)
        pred_rgb = flow_to_rgb_uint8(pred, max_flow=vmax)
        masked = pred_rgb.copy()
        masked[~mask] = 255
        gt_rgb = flow_to_rgb_uint8(flow, max_flow=vmax)
        gt_rgb[~mask] = 255
        imgs = [events_to_rgb_uint8(vox), pred_rgb, masked, gt_rgb]
        for j, im in enumerate(imgs):
            axes[i, j].imshow(im)
        e = np.where(mask, epe, np.nan)
        h = axes[i, 4].imshow(e, cmap="magma", vmin=0, vmax=max(3.0, float(np.nanpercentile(e, 95))))
        fig.colorbar(h, ax=axes[i, 4], fraction=0.035, pad=0.02)
        axes[i, 0].set_ylabel(f"{seq}\nEPE {epe[mask].mean():.2f}", fontsize=9)
        for j in range(5):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(cols[j], fontsize=10)
    wheel, wmask = make_color_wheel_uint8(64)
    wheel = np.dstack([wheel, (wmask * 255).astype(np.uint8)])
    fig.figimage(wheel, xo=10, yo=10, origin="upper")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=130)
    print(f"saved {args.out}.png/.pdf", flush=True)


if __name__ == "__main__":
    main()
