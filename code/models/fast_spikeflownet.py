"""
FastSpikeFlowNet: low-latency, sparse spiking optical flow for DSEC.

Same family as SpikeFlowNet (models/spikeflownet.py): SiLIF spiking encoder +
light Conv-BN-ReLU decoder. Re-designed for few SNN steps (T <= 5) and fast
GPU inference:

  * input (B, T*k, H, W) voxel bins in time order -> T SNN steps of k bins each,
    so cutting T does not discard input temporal resolution;
  * encoder kept in (T, B, C, H, W) layout: convs batched over T*B, the neuron
    recurrence is one fused Triton SiLIF scan per layer (silif_triton);
  * SiLIF dt initialised to the physical step 0.1 s / T (100 ms window);
  * learned per-channel temporal readout of each spike train (init = mean over
    T), so spike timing reaches the decoder instead of only spike counts;
  * coarse-to-fine residual flow, upsampled flows scaled x2 (pixel units);
  * convex x2 upsampling (RAFT) to the native 480x640 resolution.

forward(x) returns flows finest first:
    [native (B,2,2H,2W), level0 (B,2,H,W), level1 (B,2,H/2,W/2), ...]
``q_*`` modules are identity placeholders where train_fast_flow.py inserts
activation fake-quant for INT8 QAT.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional

from silif_triton import SiLIFNodeTriton


def _up(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


def convex_upsample(flow: torch.Tensor, mask: torch.Tensor, r: int = 2) -> torch.Tensor:
    """RAFT convex upsampling: each fine pixel = softmax-weighted 3x3 neighbourhood of r*flow."""
    B, _, H, W = flow.shape
    mask = mask.view(B, 1, 9, r, r, H, W).softmax(dim=2)
    nb = F.unfold(r * flow, 3, padding=1).view(B, 2, 9, 1, 1, H, W)
    up = (mask * nb).sum(dim=2)                       # (B, 2, r, r, H, W)
    return up.permute(0, 1, 4, 2, 5, 3).reshape(B, 2, r * H, r * W)


class SpikingConv(nn.Module):
    """3x3 conv (stride s) -> BN -> SiLIF scan, on a (T, B, C, H, W) sequence."""

    def __init__(self, cin, cout, stride, v_threshold, dt_init, dropout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        self.norm = nn.BatchNorm2d(cout, momentum=0.05)
        self.neuron = SiLIFNodeTriton(hidden_size=cout, v_threshold=v_threshold,
                                      dt_init=dt_init, step_mode="m")
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.carry = False   # True: keep the SiLIF state from the previous forward (see set_carry)

    def forward(self, x):
        T, B = x.shape[:2]
        y = self.norm(self.conv(x.flatten(0, 1)))
        if not self.carry:
            functional.reset_net(self.neuron)
        s = self.neuron(y.view(T, B, *y.shape[1:]))   # (T, B, C, H', W') binary spikes
        return self.drop(s.flatten(0, 1)).view_as(s)


class TemporalReadout(nn.Module):
    """Per-channel weighted sum of the T spike maps, initialised to the mean (1/T)."""

    def __init__(self, T, C):
        super().__init__()
        self.w = nn.Parameter(torch.full((T, 1, C, 1, 1), 1.0 / T))

    def forward(self, s):
        return (s * self.w).sum(0)


def _conv_bn_relu(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class FastSpikeFlowNet(nn.Module):
    def __init__(self, T=5, bins_per_step=3, channels=(32, 64, 128), dec_channels=(32, 32, 64),
                 v_threshold=0.5, dropout=0.1, upsample="convex"):
        super().__init__()
        assert len(channels) == len(dec_channels)
        self.T, self.k = T, bins_per_step
        L = len(channels)
        strides = (1,) + (2,) * (L - 1)
        self.q_in = nn.Identity()
        self.enc = nn.ModuleList()
        cin = bins_per_step
        for c, s in zip(channels, strides):
            self.enc.append(SpikingConv(cin, c, s, v_threshold, 0.1 / T, dropout))
            cin = c
        self.readout = nn.ModuleList([TemporalReadout(T, c) for c in channels])
        self.q_feat = nn.ModuleList([nn.Identity() for _ in channels])
        self.dec = nn.ModuleList()
        self.q_dec = nn.ModuleList([nn.Identity() for _ in channels])
        self.heads = nn.ModuleList()
        for l in range(L):
            skip = dec_channels[l + 1] if l < L - 1 else 0
            self.dec.append(_conv_bn_relu(channels[l] + skip, dec_channels[l]))
            head = nn.Conv2d(dec_channels[l], 2, 1)
            nn.init.zeros_(head.bias)
            self.heads.append(head)
        self.up_mask = nn.Conv2d(dec_channels[0], 9 * 4, 1) if upsample == "convex" else None

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.q_in(x).view(B, self.T, self.k, H, W).transpose(0, 1)   # (T, B, k, H, W)
        feats = []
        for enc, ro, q in zip(self.enc, self.readout, self.q_feat):
            x = enc(x)
            feats.append(q(ro(x)))
        flows, d, flow = [], None, None
        for l in reversed(range(len(self.enc))):
            inp = feats[l] if d is None else torch.cat([feats[l], _up(d)], dim=1)
            d = self.q_dec[l](self.dec[l](inp))
            f = self.heads[l](d)
            flow = f if flow is None else f + 2.0 * _up(flow)
            flows.append(flow)
        flows = flows[::-1]
        if self.up_mask is not None:
            native = convex_upsample(flow, self.up_mask(d), 2)
        else:
            native = 2.0 * _up(flow)
        return [native] + flows


# ---------------------------------------------------------------------------
# Fully spiking variant (neuromorphic deployment)
# ---------------------------------------------------------------------------

def _up_spikes(s: torch.Tensor) -> torch.Tensor:
    """Nearest x2 upsampling of a (T, B, C, H, W) spike train: stays binary (synaptic fan-out)."""
    T, B = s.shape[:2]
    y = F.interpolate(s.flatten(0, 1), scale_factor=2, mode="nearest")
    return y.view(T, B, *y.shape[1:])


class MembraneReadout(nn.Module):
    """Non-firing leaky integrator neurons: v_T = sum_t lam^(T-1-t) * conv(s_t), per-output-channel
    leak lam = sigmoid(leak_logit) (init ~0.95, close to a pure integrator). The flow is v_T."""

    def __init__(self, cin, cout, kernel=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel, padding=kernel // 2)
        nn.init.zeros_(self.conv.bias)
        self.leak_logit = nn.Parameter(torch.full((cout,), 3.0))

    def forward(self, s):
        T, B = s.shape[:2]
        y = self.conv(s.flatten(0, 1))
        y = y.view(T, B, *y.shape[1:])
        lam = torch.sigmoid(self.leak_logit).view(1, 1, -1, 1, 1)
        k = torch.arange(T - 1, -1, -1, device=s.device, dtype=y.dtype).view(T, 1, 1, 1, 1)
        return (y * lam.to(y.dtype) ** k).sum(0)


class SpikingResBlock(nn.Module):
    """Residual spiking block (membrane shortcut): the input spikes add to the neurons' input
    current (one unit-weight synapse per neuron), so the output stays binary. Dilated 3x3 conv
    for extra context at the coarse levels."""

    def __init__(self, c, v_threshold, dt_init, dropout, dilation=2):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        self.norm = nn.BatchNorm2d(c, momentum=0.05)
        self.neuron = SiLIFNodeTriton(hidden_size=c, v_threshold=v_threshold, dt_init=dt_init, step_mode="m")
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.carry = False

    def forward(self, s):
        y = self.norm(self.conv(s.flatten(0, 1))).view_as(s) + s
        if not self.carry:
            functional.reset_net(self.neuron)
        out = self.neuron(y)
        return self.drop(out.flatten(0, 1)).view_as(out)


class SpikingFastFlowNet(nn.Module):
    """
    Fully spiking flow network: every hidden layer (encoder AND decoder) is conv -> BN ->
    SiLIF emitting binary spikes over all T steps; between layers only spikes travel
    (skip = spike concat, upsampling = nearest spike copy). Flow is read from the membrane
    of non-firing leaky integrators. At inference only the native-resolution readout is
    computed; the per-level readouts are deep-supervision heads used in training only.
    BN folds into conv weight + bias for deployment. Input: the voxel bins of each step as
    the first layer's input current (direct encoding).

    forward(x) -> [native (B,2,2H,2W)] in eval; [native, level0, level1, ...] in training.
    """

    def __init__(self, T=5, bins_per_step=3, channels=(32, 64, 96, 128), dec_channels=(32, 64, 96, 128),
                 v_threshold=0.5, dropout=0.1, res_blocks=None):
        super().__init__()
        assert len(channels) == len(dec_channels)
        self.T, self.k = T, bins_per_step
        L = len(channels)
        dt = 0.1 / T
        res_blocks = res_blocks or (0,) * L     # residual spiking blocks after each encoder level
        self.q_in = nn.Identity()
        self.q_feat, self.q_dec = nn.ModuleList(), nn.ModuleList()   # (activations are spikes)
        self.enc = nn.ModuleList()
        cin = bins_per_step
        for c, s in zip(channels, (1,) + (2,) * (L - 1)):
            self.enc.append(SpikingConv(cin, c, s, v_threshold, dt, dropout))
            cin = c
        self.res = nn.ModuleList([nn.Sequential(*[SpikingResBlock(c, v_threshold, dt, dropout) for _ in range(n)])
                                  for c, n in zip(channels, res_blocks)])
        self.dec = nn.ModuleList()
        for l in range(L):
            skip = dec_channels[l + 1] if l < L - 1 else 0
            self.dec.append(SpikingConv(channels[l] + skip, dec_channels[l], 1, v_threshold, dt, dropout))
        self.heads = nn.ModuleList([MembraneReadout(c, 2) for c in dec_channels])
        self.out = MembraneReadout(dec_channels[0], 2, kernel=3)    # on nearest-upsampled spikes

    def set_carry(self, on: bool):
        """on: the SiLIF states (v, w, last spike) are NOT reset between forwards, so consecutive
        windows are processed as one continuous stream; call reset_state() at each new stream.
        The membrane readouts stay per window (the flow of each window is read independently)."""
        for m in self.modules():
            if isinstance(m, (SpikingConv, SpikingResBlock)):
                m.carry = on

    def reset_state(self):
        functional.reset_net(self)

    def forward(self, x):
        B, _, H, W = x.shape
        s = self.q_in(x).view(B, self.T, self.k, H, W).transpose(0, 1)    # (T, B, k, H, W)
        feats = []
        for enc, res in zip(self.enc, self.res):
            s = res(enc(s))
            feats.append(s)
        d, aux = None, []
        for l in reversed(range(len(self.enc))):
            d = self.dec[l](feats[l] if d is None else torch.cat([feats[l], _up_spikes(d)], dim=2))
            if self.training:
                aux.append(self.heads[l](d))
        native = self.out(_up_spikes(d))
        return [native] + aux[::-1]
