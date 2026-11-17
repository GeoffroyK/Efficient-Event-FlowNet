"""
Data augmentation transforms for DSEC optical flow training.

Each transform takes and returns a tuple ``(events, flow_gt, mask)`` where:
    events:   (T, 2, H, W) float32
    flow_gt:  (2, H, W)    float32
    mask:     (H, W)       bool

All spatial transforms are applied consistently across the three tensors.
Flow components are sign-corrected to remain physically valid after flips.

Compose with torchvision.transforms.Compose:
    from torchvision import transforms as T
    aug = T.Compose([
        RandomHorizontalFlip(p=0.4),
        RandomEventDrop(p=0.5),
        RandomPatch(p=0.5),
    ])
"""

import random

import numpy as np
import torch
import torchvision.transforms.functional as TF


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x):
        events, flow_gt, mask = x
        if torch.rand(1).item() < self.p:
            events = TF.hflip(events)
            flow_gt = TF.hflip(flow_gt)
            flow_gt[0] *= -1  # negate vx
            mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)
        return events, flow_gt, mask


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x):
        events, flow_gt, mask = x
        if torch.rand(1).item() < self.p:
            events = TF.vflip(events)
            flow_gt = TF.vflip(flow_gt)
            flow_gt[1] *= -1  # negate vy
            mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)
        return events, flow_gt, mask


class RandomEventDrop:
    """
    Randomly zero out a fraction of events (simulates sensor drop-outs).

    Args:
        p:             probability of applying the transform
        min_drop_rate: minimum fraction of events to drop
        max_drop_rate: maximum fraction of events to drop
    """

    def __init__(self, p: float = 0.5, min_drop_rate: float = 0.0, max_drop_rate: float = 0.6):
        self.p = p
        self.min_drop_rate = min_drop_rate
        self.max_drop_rate = max_drop_rate

    def __call__(self, x):
        events, flow_gt, mask = x
        if torch.rand(1).item() < self.p:
            q = (self.max_drop_rate - self.min_drop_rate) * torch.rand(1) + self.min_drop_rate
            keep = torch.rand_like(events) > q
            events = events * keep
        return events, flow_gt, mask


class RandomPatch:
    """
    Randomly occlude rectangular patches in the event tensor.

    Args:
        p:           probability of applying any patches at a given time step
        min_size:    minimum patch side length (pixels)
        max_size:    maximum patch side length (pixels)
        max_patches: maximum number of patches per time step
    """

    def __init__(self, p: float = 0.65, min_size: int = 20, max_size: int = 50, max_patches: int = 6):
        self.p = p
        self.sizes = list(range(min_size, max_size + 1))
        self.n_patches = list(range(1, max_patches + 1))

    def __call__(self, x):
        events, flow_gt, mask = x
        T, C, H, W = events.shape
        for t in range(T):
            if torch.rand(1).item() < self.p:
                for _ in range(random.choice(self.n_patches)):
                    size = random.choice(self.sizes)
                    y0 = random.randint(0, H - size)
                    x0 = random.randint(0, W - size)
                    events[t, :, y0:y0 + size, x0:x0 + size] = 0
        return events, flow_gt, mask


class RandomRotate90:
    """
    Randomly rotate events and flow by ±90°.
    Flow components are permuted/sign-corrected accordingly.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x):
        events, flow_gt, mask = x
        if torch.rand(1).item() < self.p:
            angle = 90 if torch.rand(1).item() < 0.5 else -90

            events = TF.rotate(events, angle)
            flow_gt = TF.rotate(flow_gt, angle)
            mask = TF.rotate(mask.unsqueeze(0), angle).squeeze(0)

            flow_np = np.flip(flow_gt.numpy(), axis=0).copy()
            flow_gt = torch.from_numpy(flow_np)
            if angle == -90:
                flow_gt[1] *= -1
            else:
                flow_gt[0] *= -1

        return events, flow_gt, mask
