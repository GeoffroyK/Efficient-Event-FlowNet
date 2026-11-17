"""
SpikeFlowNet: clean SNN-based optical flow estimator.

Design principles:
  - SiLIF encoder: pure spiking chain that processes T event bins sequentially.
    Each "a" layer fires spikes that flow into the next downsampling layer.
    No layer simultaneously acts as a skip-connection membrane reader AND
    a spike source for the next layer (the root cause of the previous collapse).
  - Spike aggregation: mean over T converts (B,T,C,H,W) spikes to (B,C,H,W)
    rate maps. This is differentiable, interpretable, and dense.
  - CNN decoder: standard Conv+BN+ReLU blocks. Gradients flow cleanly with
    no surrogate-gradient path in the backward pass of the decoder.
  - Multi-scale output: coarse (H/4), medium (H/2), fine (H) flow heads with
    residual upsampling, compatible with the existing multiscale_loss.

Input:  (B, T, H, W)  — single-channel event bins (polarity merged / voxel bin)
Output: (fine_flow, coarse_flow, medium_flow) each (B, 1, 2, *, *)
        The singleton T-dim makes this a drop-in for the existing training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convsnn import ConvSiLIFLayer


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _up(x: torch.Tensor, scale: int = 2) -> torch.Tensor:
    return F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)


class SpikeFlowNet(nn.Module):
    """
    SiLIF encoder + CNN decoder for event-based optical flow.

    Encoder:  three spiking ConvSiLIFLayers (stride-2 downsampling at enc2/enc3).
              Each layer receives spikes from the previous one — no inf-threshold
              layers in the encoding path.

    Aggregation: mean over T bins → spike-rate feature maps.

    Decoder:  three plain Conv+BN+ReLU blocks with skip connections from the
              encoder, plus 1×1 flow heads with zero-weight init and residual
              upsampling between scales.

    Args:
        in_channels:  channels per event bin (1 for polarity-merged, 2 for ±)
        channels:     (c1, c2, c3) feature widths for enc1/enc2/enc3
        v_threshold:  SiLIF firing threshold (0.5 works well after BN)
        dropout:      spatial dropout rate in the encoder
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple = (32, 64, 128),
        v_threshold: float = 0.5,
        dropout: float = 0.1,
        native_upsample: bool = False,
    ):
        super().__init__()
        c1, c2, c3 = channels
        self.native_upsample = native_upsample

        # ── SiLIF encoder (spiking chain) ──────────────────────────────────
        self.enc1 = ConvSiLIFLayer(in_channels, c1, stride=1,
                                   v_threshold=v_threshold, dropout=dropout)
        self.enc2 = ConvSiLIFLayer(c1, c2, stride=2,
                                   v_threshold=v_threshold, dropout=dropout)
        self.enc3 = ConvSiLIFLayer(c2, c3, stride=2,
                                   v_threshold=v_threshold, dropout=dropout)

        # ── CNN decoder ────────────────────────────────────────────────────
        # dec3: process bottleneck spike rates
        self.dec3 = _conv_block(c3, c2)
        # dec2: upsample dec3 + skip from enc2
        self.dec2 = _conv_block(c2 + c2, c1)
        # dec1: upsample dec2 + skip from enc1
        self.dec1 = _conv_block(c1 + c1, c1)

        # ── Flow heads (1×1 conv, zero-weight init) ─────────────────────
        self.flow_head3 = nn.Conv2d(c2, 2, 1, bias=True)
        self.flow_head2 = nn.Conv2d(c1, 2, 1, bias=True)
        self.flow_head1 = nn.Conv2d(c1, 2, 1, bias=True)

        heads = [self.flow_head1, self.flow_head2, self.flow_head3]
        if self.native_upsample:
            c0 = max(c1 // 2, 16)
            self.dec0 = _conv_block(c1 + 2, c0)
            self.flow_head0 = nn.Conv2d(c0, 2, 1, bias=True)
            heads.append(self.flow_head0)

        for head in heads:
            # Keep default kaiming weight init so upstream gradients are non-zero.
            # Zero bias gives near-zero initial predictions without killing the Jacobian.
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, H, W)  — event voxel grid (single channel per bin)
        Returns: (fine_flow, coarse_flow, medium_flow)
                 each (B, 1, 2, H*, W*)  — singleton T dim for training-loop compat
        """
        B, T, H, W = x.shape
        x = x.unsqueeze(2)  # (B, T, 1, H, W)

        # ── Encoder: spikes propagate through the chain ─────────────────
        s1 = self.enc1(x)   # (B, T, c1, H,   W)
        s2 = self.enc2(s1)  # (B, T, c2, H/2, W/2)
        s3 = self.enc3(s2)  # (B, T, c3, H/4, W/4)

        # ── Aggregate spikes → spike-rate feature maps ──────────────────
        f1 = s1.mean(dim=1)  # (B, c1, H,   W)
        f2 = s2.mean(dim=1)  # (B, c2, H/2, W/2)
        f3 = s3.mean(dim=1)  # (B, c3, H/4, W/4)

        # ── Coarse flow at H/4 ──────────────────────────────────────────
        d3 = self.dec3(f3)                              # (B, c2, H/4, W/4)
        coarse_flow = self.flow_head3(d3)               # (B, 2,  H/4, W/4)

        # ── Medium flow at H/2 ──────────────────────────────────────────
        coarse_up = _up(coarse_flow)                    # (B, 2,  H/2, W/2)
        d2 = self.dec2(torch.cat([_up(d3), f2], dim=1))  # (B, c1, H/2, W/2)
        medium_flow = self.flow_head2(d2) + coarse_up  # residual

        # ── Fine flow at H ───────────────────────────────────────────────
        medium_up = _up(medium_flow)                    # (B, 2, H, W)
        d1 = self.dec1(torch.cat([_up(d2), f1], dim=1))  # (B, c1, H, W)
        fine_flow = self.flow_head1(d1) + medium_up    # residual

        if self.native_upsample:
            # ── Native full-resolution flow at 2H × 2W (e.g. 480×640) ───
            fine_up = _up(fine_flow) * 2.0                          # (B, 2,  2H, 2W)
            d0 = self.dec0(torch.cat([_up(d1), fine_up], dim=1))    # (B, c0, 2H, 2W)
            native_flow = self.flow_head0(d0) + fine_up             # residual at 2H×2W
            return (
                native_flow.unsqueeze(1),  # (B, 1, 2, 2H,  2W) -> 1.0x native scale
                medium_flow.unsqueeze(1),  # (B, 1, 2, H/2, W/2) -> 0.25x native scale
                fine_flow.unsqueeze(1),    # (B, 1, 2, H,   W)   -> 0.50x native scale
            )

        # Add singleton T dim so the existing multiscale_loss (which does [:, -1])
        # works without any changes to the training loop.
        return (
            fine_flow.unsqueeze(1),    # (B, 1, 2, H,   W)
            coarse_flow.unsqueeze(1),  # (B, 1, 2, H/4, W/4)
            medium_flow.unsqueeze(1),  # (B, 1, 2, H/2, W/2)
        )


def _up_t(x: torch.Tensor) -> torch.Tensor:
    """Bilinear ×2 upsample on (B, T, C, H, W) — spatial dims only."""
    B, T, C, H, W = x.shape
    return F.interpolate(
        x.flatten(0, 1), scale_factor=2, mode="bilinear", align_corners=False
    ).view(B, T, C, H * 2, W * 2)


def _cat_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Concatenate two (B, T, C, H, W) tensors along the channel dim."""
    return torch.cat([a, b], dim=2)


def _head_t(head: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """Apply a 1×1 conv head to (B, T, C, H, W), return (B, T, 2, H, W)."""
    B, T, C, H, W = x.shape
    return head(x.flatten(0, 1)).view(B, T, 2, H, W)


class FullSpikeFlowNet(nn.Module):
    """
    Fully spiking optical flow estimator.

    Every hidden layer is a SiLIF neuron with a real firing threshold — no
    temporal aggregation, no non-spiking membrane-integrator layers anywhere
    in the hidden stack.  The SiLIF dynamics (learnable α, β, a, b, dt) do
    all the temporal integration; flow is read from the membrane of the final
    decoder layer only.

    Data flow
    ---------
    Encoder (spiking chain, threshold = vt):
      enc1 → s1  (B, T, c1, H,   W)
      enc2 → s2  (B, T, c2, H/2, W/2)   receives s1
      enc3 → s3  (B, T, c3, H/4, W/4)   receives s2

    Decoder:
      dec3 (spiking, vt)  → d3  (B, T, c2, H/4, W/4)   receives s3
      dec2 (spiking, vt)  → d2  (B, T, c1, H/2, W/2)   receives cat(up(d3), s2)
      dec1 (membrane, ∞)  → m1  (B, T, c1, H,   W)     receives cat(up(d2), s1)

    Flow heads (1×1 Conv applied per-timestep):
      coarse → from d3 spikes  (B, T, 2, H/4, W/4)
      medium → from d2 spikes  (B, T, 2, H/2, W/2)  + upsampled coarse (residual)
      fine   → from m1 membrane (B, T, 2, H,   W)   + upsampled medium (residual)

    The coarse/medium heads project sparse binary spikes, so they are noisy
    early in training but improve as the spiking layers become more active.
    The fine head projects a continuous membrane potential, giving a smooth
    output at full resolution.

    Args:
        in_channels:  event bins per timestep (1 for polarity-merged voxel)
        channels:     (c1, c2, c3) feature widths
        v_threshold:  SiLIF firing threshold for all hidden spiking layers
        dropout:      spatial dropout in all SiLIF layers
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple = (32, 64, 128),
        v_threshold: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        c1, c2, c3 = channels
        vt = v_threshold

        # ── Encoder (fully spiking) ─────────────────────────────────────
        self.enc1 = ConvSiLIFLayer(in_channels, c1, stride=1,
                                   v_threshold=vt, dropout=dropout)
        self.enc2 = ConvSiLIFLayer(c1, c2, stride=2,
                                   v_threshold=vt, dropout=dropout)
        self.enc3 = ConvSiLIFLayer(c2, c3, stride=2,
                                   v_threshold=vt, dropout=dropout)

        # ── Decoder ──────────────────────────────────────────────────────
        # dec3: spiking, receives bottleneck spikes s3
        self.dec3 = ConvSiLIFLayer(c3, c2, stride=1,
                                   v_threshold=vt, dropout=dropout)
        # dec2: spiking, receives cat(up(d3), s2)
        self.dec2 = ConvSiLIFLayer(c2 + c2, c1, stride=1,
                                   v_threshold=vt, dropout=dropout)
        # dec1: membrane readout, receives cat(up(d2), s1)
        # return_membrane=True sets threshold=inf → integrates but never fires
        self.dec1 = ConvSiLIFLayer(c1 + c1, c1, stride=1,
                                   return_membrane=True, dropout=dropout)

        # ── Flow heads (1×1 conv) ────────────────────────────────────────
        self.flow_head3 = nn.Conv2d(c2, 2, 1, bias=True)  # coarse, from d3 spikes
        self.flow_head2 = nn.Conv2d(c1, 2, 1, bias=True)  # medium, from d2 spikes
        self.flow_head1 = nn.Conv2d(c1, 2, 1, bias=True)  # fine,   from m1 membrane

        for head in [self.flow_head1, self.flow_head2, self.flow_head3]:
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, H, W)
        Returns: (fine_flow, coarse_flow, medium_flow) each (B, T, 2, H*, W*)
        """
        B, T, H, W = x.shape
        x = x.unsqueeze(2)  # (B, T, 1, H, W)

        # ── Encoder ─────────────────────────────────────────────────────
        s1 = self.enc1(x)    # (B, T, c1, H,   W)
        s2 = self.enc2(s1)   # (B, T, c2, H/2, W/2)
        s3 = self.enc3(s2)   # (B, T, c3, H/4, W/4)

        # ── Decoder ─────────────────────────────────────────────────────
        d3 = self.dec3(s3)                        # (B, T, c2, H/4, W/4)  spikes
        d2 = self.dec2(_cat_t(_up_t(d3), s2))    # (B, T, c1, H/2, W/2)  spikes
        _, m1 = self.dec1(_cat_t(_up_t(d2), s1)) # (B, T, c1, H,   W)    membrane

        # ── Coarse flow (H/4) ────────────────────────────────────────────
        coarse_flow = _head_t(self.flow_head3, d3)          # (B, T, 2, H/4, W/4)

        # ── Medium flow (H/2) — residual on upsampled coarse ────────────
        coarse_up   = _up_t(coarse_flow)
        medium_flow = _head_t(self.flow_head2, d2) + coarse_up  # (B, T, 2, H/2, W/2)

        # ── Fine flow (H) — residual on upsampled medium ─────────────────
        medium_up = _up_t(medium_flow)
        fine_flow = _head_t(self.flow_head1, m1) + medium_up    # (B, T, 2, H, W)

        return fine_flow, coarse_flow, medium_flow


if __name__ == "__main__":
    model = SpikeFlowNet(in_channels=1, channels=(32, 64, 128))
    x1 = torch.randn(2, 15, 60, 80)
    x2 = torch.randn(2, 15, 60, 80)

    fine, coarse, medium = model(x1)
    print(f"fine:   {fine.shape}")    # (2, 1, 2, 60,  80)
    print(f"coarse: {coarse.shape}")  # (2, 1, 2, 15,  20)
    print(f"medium: {medium.shape}")  # (2, 1, 2, 30,  40)

    fine2, _, _ = model(x2)
    print(f"outputs differ: {not torch.equal(fine, fine2)}")  # True

    # Check gradient flow
    loss = fine.mean()
    loss.backward()
    enc1_grad = model.enc1.conv.weight.grad
    print(f"enc1 grad norm: {enc1_grad.norm():.4f}")  # should be non-zero
