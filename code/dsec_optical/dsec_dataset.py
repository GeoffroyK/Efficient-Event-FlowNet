"""
PyTorch Dataset for pre-processed DSEC optical flow data.

Expects the directory layout produced by preprocess.py:

    <root>/
        event_tensors/<N>frames/   — {sequence}_{idx:04d}.npy  (N, 2, 480, 640)
        gt_tensors/                — {sequence}_{idx:04d}.npy  (2, 480, 640)
        mask_tensors/              — {sequence}_{idx:04d}.npy  (480, 640) bool
        sequence_lists/
            train_split.csv
            val_split.csv

Each CSV row contains two consecutive filenames (no header):
    zurich_city_01_a_0001.npy,zurich_city_01_a_0002.npy
    zurich_city_01_a_0002.npy,zurich_city_01_a_0003.npy
    ...

The dataset returns **two consecutive event windows concatenated** along
the temporal axis, dropping the oldest frame so that the network always
sees a fixed context window of (2*n_bins - 1) bins.  The flow GT and
validity mask correspond to the second (more recent) timestamp.

    events:   (2*n_bins - 1, 2, H, W)  float32
    flow_gt:  (2, H, W)                float32  pixels / interval
    mask:     (H, W)                   bool     True = valid pixel
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DSECDataset(Dataset):
    """
    DSEC optical-flow dataset (pre-processed NPY format).

    Args:
        root:         path to the pre-processed data root
        split_csv:    CSV filename inside ``<root>/sequence_lists/``
                      (e.g. ``"train_split.csv"`` or ``"val_split.csv"``)
        n_bins:       number of temporal bins per flow interval used
                      during preprocessing (default: 11)
        transform:    optional callable applied jointly to
                      ``(events, flow_gt, mask)`` — see augmentation.py
    """

    def __init__(
        self,
        root: str,
        split_csv: str,
        n_bins: int = 11,
        transform=None,
    ):
        self.root = Path(root)
        self.events_dir = self.root / "event_tensors" / f"{n_bins:02d}frames"
        self.flow_dir = self.root / "gt_tensors"
        self.mask_dir = self.root / "mask_tensors"
        self.transform = transform

        csv_path = self.root / "sequence_lists" / split_csv
        self.pairs = pd.read_csv(csv_path, header=None)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        f1 = self.pairs.iloc[idx, 0]
        f2 = self.pairs.iloc[idx, 1]

        # Load and concatenate two consecutive event windows
        ev1 = torch.from_numpy(
            np.load(self.events_dir / f1).astype(np.float32)
        )  # (n_bins, 2, H, W)
        ev2 = torch.from_numpy(
            np.load(self.events_dir / f2).astype(np.float32)
        )  # (n_bins, 2, H, W)
        events = torch.cat([ev1, ev2], dim=0)[1:]  # (2*n_bins-1, 2, H, W)

        # GT and mask from the second (target) timestamp
        flow_gt = torch.from_numpy(
            np.load(self.flow_dir / f2).astype(np.float32)
        )  # (2, H, W)
        mask = torch.from_numpy(
            np.load(self.mask_dir / f2)
        )  # (H, W) bool

        if self.transform is not None:
            events, flow_gt, mask = self.transform((events, flow_gt, mask))

        return events, flow_gt, mask
