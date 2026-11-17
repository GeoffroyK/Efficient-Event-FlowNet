"""
Voxel-grid cache + dataset for DSEC optical flow (fast SpikeFlowNet pipeline).

The on-the-fly loader (``loader_dsec.Sequence``) re-reads ~2M events per sample
from the compressed events.h5 and voxelises them at 480x640 on the CPU every
epoch (~0.2-0.6 s/sample, the bottleneck of ``train_dsec_flow.py``). This module
does it once. Per recording it writes

    <cache_root>/<seq>/voxel<BINS>_<H>x<W>.npy   (N, BINS, H, W) float16
    <cache_root>/<seq>/valid.npy                  (N,) bool   (False: no events)

for every flow-GT file j: events in [from_ts_j, to_ts_j), the same normalised
``VoxelGrid`` as the on-the-fly loader, then AvgPool to H x W (= the old
``downsample_ratio`` transform). GT flow is read from the original 16-bit PNGs
at train time (native 480x640, exact).

Build (CPU only, ~10 min):
    python -m dsec_optical.flow_cache --bins 15 --downsample 2 --workers 18
"""
import argparse
import os
import random
from multiprocessing import Pool
from pathlib import Path
import sys
from typing import List

import numpy as np
import torch  # before cv2/hdf5plugin: loads the env's libstdc++ (the node's /lib64 one is too old)
import torch.nn.functional as F
_sys_path = list(sys.path)
try:
    import cv2
except ImportError:  # nodes without libGL (e.g. the H200 node): imageio FreeImage reads the same 16-bit PNG
    # cv2's bootstrap already put its package dir on sys.path (shadows stdlib `typing` in the
    # torch.compile worker subprocesses, which inherit sys.path): undo it
    sys.path[:] = _sys_path
    for _m in [m for m in sys.modules if m == "cv2" or m.startswith("cv2.")]:
        del sys.modules[_m]
    cv2 = None
    import imageio
import h5py
import hdf5plugin  # noqa: F401 - registers the blosc filter used by events.h5
from torch.utils.data import Dataset

from dsec_optical.dsec_utils import VoxelGrid

DATA_ROOT = Path("/home/geoffroyk/idnet/data")
CACHE_ROOT = DATA_ROOT / "flow_cache"

FLOW_TRAIN_RECORDINGS: List[str] = [
    "zurich_city_01_a",
    "zurich_city_02_a",
    "zurich_city_02_c",
    "zurich_city_02_e",
    "zurich_city_05_a",
    "zurich_city_05_b",
    "zurich_city_06_a",
    "zurich_city_07_a",
    "zurich_city_09_a",
    "zurich_city_10_a",
    "zurich_city_10_b",
    "zurich_city_11_a",
    "zurich_city_11_c",
]

_FLOW_VAL_RECORDINGS: List[str] = [
    "thun_00_a",
    "zurich_city_02_d",
    "zurich_city_03_a",
    "zurich_city_08_a",
    "zurich_city_11_b",
]

H_NATIVE, W_NATIVE = 480, 640


def _flow_dir(seq: str, data_root: Path = DATA_ROOT) -> Path:
    return data_root / "train_optical_flow" / seq / "flow"


def voxel_path(seq: str, bins: int, downsample: int, cache_root: Path = CACHE_ROOT) -> Path:
    return cache_root / seq / f"voxel{bins}_{H_NATIVE // downsample}x{W_NATIVE // downsample}.npy"


def build_recording(seq: str, bins: int, downsample: int,
                    data_root: Path = DATA_ROOT, cache_root: Path = CACHE_ROOT) -> str:
    torch.set_num_threads(2)
    out = voxel_path(seq, bins, downsample, cache_root)
    if out.exists():
        return f"{seq}: exists, skipped"
    out.parent.mkdir(parents=True, exist_ok=True)

    ts = np.loadtxt(_flow_dir(seq, data_root) / "forward_timestamps.txt", delimiter=",", dtype="int64")
    n = len(ts)
    assert n == len(list((_flow_dir(seq, data_root) / "forward").glob("*.png"))), seq
    ev_dir = data_root / "train_events" / seq / "events" / "left"
    h, w = H_NATIVE // downsample, W_NATIVE // downsample

    tmp = out.with_suffix(".tmp.npy")
    vox = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, bins, h, w))
    valid = np.zeros(n, dtype=bool)
    from dsec_optical.loader_dsec import EventSlicer  # (imports cv2: builder only)
    grid = VoxelGrid((bins, H_NATIVE, W_NATIVE), normalize=True)
    with h5py.File(ev_dir / "events.h5", "r") as f, h5py.File(ev_dir / "rectify_map.h5", "r") as fr:
        slicer = EventSlicer(f)
        rect = fr["rectify_map"][()]
        for j in range(n):
            ev = slicer.get_events(int(ts[j, 0]), int(ts[j, 1]))
            if ev is None or len(ev["t"]) < 2 or ev["t"][-1] == ev["t"][0]:
                vox[j] = 0
                continue
            xy = rect[ev["y"], ev["x"]]
            t = (ev["t"] - ev["t"][0]).astype("float32")
            g = grid.convert({
                "p": torch.from_numpy(ev["p"].astype("float32")),
                "t": torch.from_numpy(t / t[-1]),
                "x": torch.from_numpy(xy[:, 0].astype("float32")),
                "y": torch.from_numpy(xy[:, 1].astype("float32")),
            })
            if downsample > 1:
                g = F.avg_pool2d(g[None], downsample)[0]
            vox[j] = g.numpy().astype(np.float16)
            valid[j] = True
    vox.flush()
    del vox
    np.save(out.parent / "valid.npy", valid)
    os.replace(tmp, out)
    return f"{seq}: {valid.sum()}/{n} valid samples"


class CachedFlowDataset(Dataset):
    """Samples of (voxel (BINS, h, w) float32, flow (2, 480, 640), mask (1, 480, 640) bool, recording idx).

    ``hflip``: random horizontal flip (p=0.5) of voxel + GT, negating u — the only
    augmentation of the old pipeline. ``vflip``: random vertical flip (p=0.5), negating v
    (both together also give 180-degree rotations). ``strong_aug``: also polarity swap (p=0.5,
    flow unchanged), event gain U(0.7, 1.3), and zoom-in (p=0.5): a random crop of 1/s of the
    frame, s ~ U(1, 1.3), resized back (voxel bilinear, GT nearest, flow * s).
    ``cont[k]``: item k directly follows item k-1 in time (same recording, no gap).
    """

    def __init__(self, recordings: List[str], bins: int = 15, downsample: int = 2,
                 hflip: bool = False, vflip: bool = False, strong_aug: bool = False,
                 data_root: Path = DATA_ROOT, cache_root: Path = CACHE_ROOT):
        self.recordings = list(recordings)
        self.paths = [voxel_path(s, bins, downsample, cache_root) for s in self.recordings]
        self.pngs = [sorted((_flow_dir(s, data_root) / "forward").glob("*.png")) for s in self.recordings]
        self.items, self.cont = [], []
        for r, s in enumerate(self.recordings):
            valid = np.load(cache_root / s / "valid.npy")
            assert len(valid) == len(self.pngs[r]), s
            ts = np.loadtxt(_flow_dir(s, data_root) / "forward_timestamps.txt", delimiter=",", dtype="int64")
            for j in np.nonzero(valid)[0]:
                self.cont.append(bool(self.items) and self.items[-1] == (r, j - 1) and ts[j, 0] == ts[j - 1, 1])
                self.items.append((r, j))
        self.hflip, self.vflip, self.strong_aug = hflip, vflip, strong_aug
        self._mm = {}

    def __len__(self):
        return len(self.items)

    def draw_aug(self):
        """Random augmentation parameters (one draw is shared by all windows of a sequence)."""
        a = {"hflip": self.hflip and random.random() < 0.5, "vflip": self.vflip and random.random() < 0.5}
        if self.strong_aug:
            a["pol"] = random.random() < 0.5
            a["gain"] = random.uniform(0.7, 1.3)
            a["zoom"] = (random.uniform(1.0, 1.3), random.random(), random.random()) if random.random() < 0.5 else None
        return a

    def __getitem__(self, i):
        return self.load(i, self.draw_aug())

    def load(self, i, aug):
        r, j = self.items[i]
        if r not in self._mm:  # opened lazily, once per worker process
            self._mm[r] = np.load(self.paths[r], mmap_mode="r")
        vox = np.asarray(self._mm[r][j], dtype=np.float32)
        if cv2 is not None:
            png = cv2.imread(str(self.pngs[r][j]), cv2.IMREAD_UNCHANGED)[..., ::-1]  # BGR -> RGB
        else:
            png = imageio.imread(str(self.pngs[r][j]), format="PNG-FI")
        mask = png[..., 2] == 1
        flow = (png[..., :2].astype(np.float32) - 32768.0) / 128.0
        flow[~mask] = 0.0
        flow = flow.transpose(2, 0, 1)
        mask = mask[None]
        if aug["hflip"]:
            vox, flow, mask = vox[..., ::-1], flow[..., ::-1].copy(), mask[..., ::-1]
            flow[0] = -flow[0]
        if aug["vflip"]:
            vox, flow, mask = vox[..., ::-1, :], flow[..., ::-1, :].copy(), mask[..., ::-1, :]
            flow[1] = -flow[1]
        vox, flow, mask = (torch.from_numpy(np.ascontiguousarray(a)) for a in (vox, flow, mask))
        if aug.get("pol"):
            vox = -vox
        if "gain" in aug:
            vox = vox * aug["gain"]
        if aug.get("zoom"):
            s, oy, ox = aug["zoom"]
            (h, w), (H, W) = vox.shape[-2:], flow.shape[-2:]
            ch, cw = round(h / s), round(w / s)
            y0, x0 = int(oy * (h - ch)), int(ox * (w - cw))
            f = H // h
            vox = F.interpolate(vox[None, :, y0:y0 + ch, x0:x0 + cw], size=(h, w), mode="bilinear",
                                align_corners=False)[0]
            crop = lambda t: F.interpolate(t[None, :, f * y0:f * (y0 + ch), f * x0:f * (x0 + cw)].float(),  # noqa: E731
                                           size=(H, W), mode="nearest")[0]
            flow, mask = crop(flow) * (W / (f * cw)), crop(mask).bool()
        return vox, flow, mask, r


class FlowSequenceDataset(Dataset):
    """K temporally consecutive windows of a CachedFlowDataset (for carrying the SNN state across
    windows), all with the same augmentation draw. Items: (K, ...) stacked tensors + recording idx."""

    def __init__(self, base: CachedFlowDataset, K: int):
        self.base, self.K = base, K
        run, self.starts = 0, []
        for k, c in enumerate(base.cont):   # run = number of consecutive windows ending at k
            run = run + 1 if c else 1
            if run >= K:
                self.starts.append(k - K + 1)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        aug = self.base.draw_aug()
        out = [self.base.load(self.starts[i] + k, aug) for k in range(self.K)]
        return tuple(torch.stack([o[n] for o in out]) for n in range(3)) + (out[0][3],)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--seqs", nargs="*", default=FLOW_TRAIN_RECORDINGS + _FLOW_VAL_RECORDINGS)
    args = ap.parse_args()
    # biggest recordings first so the pool tail is short
    seqs = sorted(args.seqs, key=lambda s: -len(np.loadtxt(_flow_dir(s) / "forward_timestamps.txt", delimiter=",")))
    with Pool(args.workers) as pool:
        for msg in pool.starmap(build_recording, [(s, args.bins, args.downsample) for s in seqs]):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
