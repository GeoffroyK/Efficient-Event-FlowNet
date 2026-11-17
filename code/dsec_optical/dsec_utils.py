import torch
import numpy as np
from enum import Enum, auto
from time import time


class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()


class EventRepresentation:
    def __init__(self):
        pass

    def convert(self, events):
        raise NotImplementedError


class VoxelGrid(EventRepresentation):
    def __init__(self, input_size: tuple, normalize: bool):
        assert len(input_size) == 3
        self.voxel_grid = torch.zeros(
            (input_size), dtype=torch.float, requires_grad=False)
        self.nb_channels = input_size[0]
        self.normalize = normalize

    def convert(self, events):
        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(events['p'].device)
            voxel_grid = self.voxel_grid.clone()

            t_norm = events['t']
            t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])

            x0 = events['x'].int()
            y0 = events['y'].int()
            t0 = t_norm.int()

            value = 2*events['p']-1

            wx1 = events['x'] - x0
            wx0 = 1.0 - wx1
            wy1 = events['y'] - y0
            wy0 = 1.0 - wy1
            wt1 = t_norm - t0
            wt0 = 1.0 - wt1

            x0_l = x0.long()
            y0_l = y0.long()
            t0_l = t0.long()

            x_valid0 = (x0_l >= 0) & (x0_l < W)
            x_valid1 = (x0_l + 1 >= 0) & (x0_l + 1 < W)
            y_valid0 = (y0_l >= 0) & (y0_l < H)
            y_valid1 = (y0_l + 1 >= 0) & (y0_l + 1 < H)
            t_valid0 = (t0_l >= 0) & (t0_l < self.nb_channels)
            t_valid1 = (t0_l + 1 >= 0) & (t0_l + 1 < self.nb_channels)

            base_idx = H * W * t0_l + W * y0_l + x0_l

            for dx, wx, xv in ((0, wx0, x_valid0), (1, wx1, x_valid1)):
                vwx = value * wx
                for dy, wy, yv in ((0, wy0, y_valid0), (W, wy1, y_valid1)):
                    xy_mask = xv & yv
                    vwxy = vwx * wy
                    xy_offset = dx + dy
                    for dt, wt, tv in ((0, wt0, t_valid0), (H * W, wt1, t_valid1)):
                        mask = xy_mask & tv
                        interp_weights = vwxy * wt
                        index = base_idx + (xy_offset + dt)
                        voxel_grid.put_(
                            index[mask], interp_weights[mask], accumulate=True)

            if self.normalize:
                mask = voxel_grid != 0
                if mask.any():
                    nz = voxel_grid[mask]
                    mean = nz.mean()
                    std = nz.std()
                    if std > 0:
                        voxel_grid[mask] = (nz - mean) / std
                    else:
                        voxel_grid[mask] = nz - mean

        return voxel_grid


class PolarityCount(EventRepresentation):
    def __init__(self, input_size: tuple):
        assert len(input_size) == 3
        self.voxel_grid = torch.zeros(
            (input_size), dtype=torch.float, requires_grad=False)
        self.nb_channels = input_size[0]

    def convert(self, events):
        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(events['p'].device)
            voxel_grid = self.voxel_grid.clone()

            x0 = events['x'].int()
            y0 = events['y'].int()

            #start_t = time()
            for xlim in [x0, x0+1]:
                for ylim in [y0, y0+1]:
                    mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (
                        ylim >= 0)
                    interp_weights = (1 - (xlim-events['x']).abs()) * (
                        1 - (ylim-events['y']).abs())
                    index = H * W * events['p'].long() + \
                        W * ylim.long() + \
                        xlim.long()

                    voxel_grid.put_(
                        index[mask], interp_weights[mask], accumulate=True)

        return voxel_grid


def flow_16bit_to_float(flow_16bit: np.ndarray):
    assert flow_16bit.dtype == np.uint16
    assert flow_16bit.ndim == 3
    h, w, c = flow_16bit.shape
    assert c == 3

    valid2D = flow_16bit[..., 2] == 1
    assert valid2D.shape == (h, w)
    assert np.all(flow_16bit[~valid2D, -1] == 0)

    flow_map = np.zeros((h, w, 2), dtype=np.float32)
    flow_map[valid2D, 0] = (flow_16bit[valid2D, 0].astype(np.float32) - 32768.0) / 128.0
    flow_map[valid2D, 1] = (flow_16bit[valid2D, 1].astype(np.float32) - 32768.0) / 128.0
    return flow_map, valid2D
