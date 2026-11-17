"""
Fast / sparse SpikeFlowNet training on DSEC optical flow (cached voxels).

  * split: FLOW_TRAIN_RECORDINGS (13) / _FLOW_VAL_RECORDINGS (5) from dsec_optical.flow_cache
  * input: cached 15-bin voxel grid at 240x320, run as T <= 5 SNN steps
  * loss : multiscale EPE + rate_penalty * (average spike rate of all spiking layers)
  * eval : EPE at native 480x640 (+1PE/3PE/AE, per recording), per-layer spike rates
  * speed: batch-1 GPU forward latency in inference mode (eager fp32 / fp16 / CUDA-graph
           fp16 / fused: BN folded + torch.compile + CUDA graph)
  * --qat_bits: INT-k weight (+ --qat_act_bits activation) fake-quant fine-tuning

Archs:
  spikeflownet  original models/spikeflownet.SpikeFlowNet; the k = bins/T bins of a
                step are summed into its single input channel (neurons swapped for the
                numerically identical Triton SiLIF kernel unless --torch_neurons)
  fast          models/fast_spikeflownet.FastSpikeFlowNet (spiking encoder, ANN decoder)
  spiking       models/fast_spikeflownet.SpikingFastFlowNet (fully spiking, membrane readout)
  idnet         ANN teacher for --teacher distillation only: IDNet id-4x (Wu et al., ICRA'24,
                /home/geoffroyk/idnet), trained from scratch on the same 13 recordings

    python train_fast_flow.py --arch fast --T 5 --epochs 60 --amp
"""
import argparse
import copy
import math
import os
import random
import statistics
import time
from types import SimpleNamespace

import numpy as np
import torch  # before dsec_optical (pandas/hdf5plugin need the env's libstdc++)
import torch.nn as nn
import torch.nn.functional as F
import mlflow
from torch.nn.utils import parametrize
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, RandomSampler
from spikingjelly.activation_based import quantize

from dsec_optical.flow_cache import CachedFlowDataset, FlowSequenceDataset, FLOW_TRAIN_RECORDINGS, _FLOW_VAL_RECORDINGS
from dsec_optical.transformers import downsample_spatial_flow
from linear_silif.qat import FakeQuantWeight
from models.convsnn import ConvSiLIFLayer
from models.fast_spikeflownet import FastSpikeFlowNet, SpikingFastFlowNet
from models.spikeflownet import SpikeFlowNet
from neurons.silif import SiLIFNode
from silif_triton import SiLIFMultiStepTriton, SiLIFNodeTriton, silif_scan_torch

BINS = 15
H, W = 240, 320


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SpikeFlowNetT(nn.Module):
    """Original SpikeFlowNet fed T steps of summed voxel bins; returns [fine, medium, coarse]."""

    def __init__(self, T, triton=True, **kw):
        super().__init__()
        self.T, self.k = T, BINS // T
        self.net = SpikeFlowNet(in_channels=1, **kw)
        if triton:
            for m in self.net.modules():
                if isinstance(m, ConvSiLIFLayer):
                    old = m.neuron
                    new = SiLIFNodeTriton(hidden_size=old.hidden_size, v_threshold=old.v_threshold,
                                          step_mode="m", store_v_seq=old.store_v_seq)
                    new.load_state_dict(old.state_dict())
                    m.neuron = new

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, self.T, self.k, *x.shape[-2:]).sum(2)
        out = self.net(x)
        if getattr(self.net, "native_upsample", False):   # (native, medium, fine)
            return [o[:, -1] for o in out]
        fine, coarse, medium = out
        return [fine[:, -1], medium[:, -1], coarse[:, -1]]


class IDNetTeacher(nn.Module):
    """IDNet id-4x on the cached 240x320 voxels; returns [flow at 240x320 (input pixel units)]."""

    def __init__(self):
        super().__init__()
        import sys
        sys.path.insert(0, "/home/geoffroyk/idnet")
        from idn.model.idedeq import IDEDEQIDO
        self.net = IDEDEQIDO(SimpleNamespace(hidden_dim=128, downsample=4, update_iters=4, zero_init=True,
                                             deq_mode=False, input_flowmap=True, deblur=True, add_delta=True))

    def forward(self, x):
        return [self.net({"event_volume_new": x})["final_prediction"]]


def build_model(args, arch=None, T=None, triton=True):
    arch, T = arch or args.arch, T or args.T
    if arch == "idnet":
        return IDNetTeacher()
    assert BINS % T == 0, f"T={T} must divide the {BINS} cached bins"
    if arch == "spikeflownet":
        return SpikeFlowNetT(T, triton=triton, channels=tuple(args.channels),
                             v_threshold=args.v_threshold, dropout=args.dropout)
    if arch == "spiking":
        return SpikingFastFlowNet(T=T, bins_per_step=BINS // T, channels=tuple(args.channels),
                                  dec_channels=tuple(args.dec_channels), v_threshold=args.v_threshold,
                                  dropout=args.dropout, res_blocks=tuple(getattr(args, "res_blocks", None) or ()))
    return FastSpikeFlowNet(T=T, bins_per_step=BINS // T, channels=tuple(args.channels),
                            dec_channels=tuple(args.dec_channels), v_threshold=args.v_threshold,
                            dropout=args.dropout, upsample=args.upsample)


def freeze_adaptation(model):
    """--neuron lif: a = b = 0 frozen in every SiLIF, so w stays 0 and each neuron is a LIF
    (soft reset) with its learned per-channel leak alpha (lambda_alpha, dt). Same kernel, same cost."""
    n = 0
    for m in model.modules():
        if isinstance(m, SiLIFNode):
            for p in (m.a, m.b):
                p.data.zero_()
                p.requires_grad_(False)
            n += 1
    return n


# ---------------------------------------------------------------------------
# Spike-rate monitor (differentiable, used for the penalty)
# ---------------------------------------------------------------------------

class SpikeMonitor:
    """Hooks every firing SiLIF neuron; average rate = total spikes / total neuron-steps.
    Also records the non-zero fraction of every (decoder) ReLU output, report-only."""

    def __init__(self, model):
        self.out, self.dens = {}, {}
        self.names, self.relu_names = [], []
        for name, m in model.named_modules():
            if isinstance(m, SiLIFNode) and math.isfinite(m.v_threshold):
                self.names.append(name)
                m.register_forward_hook(self._hook(name))
            elif isinstance(m, nn.ReLU):
                self.relu_names.append(name)
                m.register_forward_hook(self._relu_hook(name))

    def _hook(self, name):
        def hook(mod, inp, out):
            self.out[name] = (out.mean(dtype=torch.float32), out.numel())
        return hook

    def _relu_hook(self, name):
        def hook(mod, inp, out):
            if not mod.training:
                self.dens[name] = ((out > 0).sum(dtype=torch.float32), out.numel())
        return hook

    def average(self):
        if not self.out:  # (no spiking layer: the ANN teacher)
            return torch.zeros(())
        num = sum(r * n for r, n in self.out.values())
        return num / sum(n for _, n in self.out.values())


# ---------------------------------------------------------------------------
# Loss / eval
# ---------------------------------------------------------------------------

def epe_loss(pred, gt, mask):
    epe = torch.sqrt(((pred - gt) ** 2).sum(dim=1, keepdim=True) + 1e-8)
    return (epe * mask).sum() / mask.sum().clamp(min=1.0)


def multiscale_epe(preds, flow, mask):
    """preds finest first; scale i weighted 1/2^i; GT mask-avg-pooled to each scale (pixel units)."""
    total, gts = 0.0, {}
    for i, p in enumerate(preds):
        f = flow.shape[-2] // p.shape[-2]
        if f not in gts:
            if f == 1:
                gts[f] = (flow, mask.float())
            else:
                g, m = downsample_spatial_flow(flow, mask.float(), f)
                gts[f] = (g, m.float())
        total = total + epe_loss(p.float(), *gts[f]) / 2 ** i
    return total


def gt_support(ds, vflip, every=20, min_freq=0.005):
    """Pixels where the GT is ever valid (never on the car hood, rows >~415), symmetrised under the
    training flips. Distillation is restricted to it: the teacher is unsupervised and arbitrary outside."""
    flips, ds.hflip, ds.vflip = (ds.hflip, ds.vflip), False, False   # (un-augmented masks)
    m = torch.stack([ds[i][2][0] for i in range(0, len(ds), every)]).float().mean(0) > min_freq
    ds.hflip, ds.vflip = flips
    m = m & m.flip(1)
    return m & m.flip(0) if vflip else m


@torch.no_grad()
def evaluate(model, loader, monitor, device, half=False, stream=False):
    """stream: batch-1 loader in temporal order; the SNN state is carried from window to window and
    reset only at the start of each recording / after a time gap (dataset.cont)."""
    model.eval()
    assert not stream or loader.batch_size == 1
    n_rec = len(loader.dataset.recordings)
    acc = torch.zeros(5, dtype=torch.float64, device=device)      # epe, n, >1, >3, ae
    rec = torch.zeros(2, n_rec, dtype=torch.float64, device=device)
    rate_num = {n: 0.0 for n in monitor.names}
    rate_den = {n: 0 for n in monitor.names}
    dens_num, dens_den = {}, {}
    for i, (vox, flow, mask, r) in enumerate(loader):
        if stream and not loader.dataset.cont[i]:
            model.reset_state()
        vox = vox.to(device, non_blocking=True)
        flow = flow.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)[:, 0]
        r = r.to(device)
        pred = model(vox.half() if half else vox)[0].float()
        if pred.shape[-2:] != flow.shape[-2:]:
            pred = F.interpolate(pred, size=flow.shape[-2:], mode="bilinear",
                                 align_corners=False) * (flow.shape[-1] / pred.shape[-1])
        epe = (pred - flow).norm(dim=1)
        e = epe[mask]
        pu, pv, gu, gv = pred[:, 0][mask], pred[:, 1][mask], flow[:, 0][mask], flow[:, 1][mask]
        cos = (pu * gu + pv * gv + 1) / torch.sqrt((pu ** 2 + pv ** 2 + 1) * (gu ** 2 + gv ** 2 + 1))
        acc += torch.stack([e.sum(), torch.tensor(e.numel(), device=device), (e > 1).sum(), (e > 3).sum(),
                            torch.rad2deg(torch.acos(cos.clamp(-1, 1))).sum()]).double()
        rec[0].index_add_(0, r, (epe * mask).flatten(1).sum(1).double())
        rec[1].index_add_(0, r, mask.flatten(1).sum(1).double())
        for n, (rate, k) in monitor.out.items():
            rate_num[n] += rate * k
            rate_den[n] += k
        for n, (nz, k) in monitor.dens.items():
            dens_num[n] = dens_num.get(n, 0.0) + nz
            dens_den[n] = dens_den.get(n, 0) + k
    acc, rec = acc.cpu(), rec.cpu()
    m = {"epe": acc[0] / acc[1], "1pe": 100 * acc[2] / acc[1], "3pe": 100 * acc[3] / acc[1], "ae": acc[4] / acc[1]}
    for i, s in enumerate(loader.dataset.recordings):
        m[f"epe_{s}"] = rec[0, i] / rec[1, i]
    for n in monitor.names:
        m["rate_" + n.replace(".", "_")] = float(rate_num[n]) / rate_den[n]
    if monitor.names:  # (the fused inference model is not monitored)
        m["rate_avg"] = float(sum(rate_num.values())) / sum(rate_den.values())
    if dens_den:
        m["dec_density_avg"] = float(sum(dens_num.values())) / sum(dens_den.values())
    return {k: float(v) for k, v in m.items()}


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

@torch.library.custom_op("fastflow::silif_scan", mutates_args=())
def silif_scan(x_seq: torch.Tensor, lla: torch.Tensor, llb: torch.Tensor, dt: torch.Tensor,
               a: torch.Tensor, b: torch.Tensor, v_threshold: float, a_min: float, a_max: float,
               b_min: float, b_max: float) -> torch.Tensor:
    """Spikes of a SiLIF scan from the reset state; opaque to torch.compile (one Triton kernel)."""
    if not x_seq.is_cuda:
        return silif_scan_torch(x_seq, lla, llb, dt, a, b, v_threshold=v_threshold,
                                a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max)[0]
    z = x_seq.new_zeros(x_seq.shape[1:])
    return SiLIFMultiStepTriton.apply(x_seq, z, z, z, lla, llb, dt, a, b, v_threshold,
                                      a_min, a_max, b_min, b_max, 0.0)[0]  # 0.0: surrogate slope, backward only


@silif_scan.register_fake
def _(x_seq, *args):
    return torch.empty_like(x_seq)


class InferSiLIF(nn.Module):
    """Stateless stand-in for a SiLIF layer that is reset before every forward."""

    def __init__(self, n):
        super().__init__()
        for k in ("lambda_log_alpha", "lambda_log_beta", "dt", "a", "b"):
            self.register_buffer(k, getattr(n, k).detach().clone())
        self.consts = tuple(float(c) for c in (n.v_threshold, n.a_min, n.a_max, n.b_min, n.b_max))

    def forward(self, x_seq):
        return silif_scan(x_seq.contiguous(), self.lambda_log_alpha, self.lambda_log_beta,
                          self.dt, self.a, self.b, *self.consts)


def fuse_for_inference(m):
    """Eval model, in place: BN folded into the bias-free conv before it, SiLIF -> InferSiLIF."""
    for mod in list(m.modules()):
        for name, child in mod.named_children():
            if isinstance(child, SiLIFNode):
                setattr(mod, name, InferSiLIF(child))
        if isinstance(getattr(mod, "norm", None), nn.BatchNorm2d):          # spiking conv layers
            mod.conv, mod.norm = fuse_conv_bn_eval(mod.conv, mod.norm), nn.Identity()
        if isinstance(mod, nn.Sequential):                                 # decoder Conv-BN-ReLU
            for i in range(len(mod) - 1):
                if isinstance(mod[i], nn.Conv2d) and isinstance(mod[i + 1], nn.BatchNorm2d):
                    mod[i], mod[i + 1] = fuse_conv_bn_eval(mod[i], mod[i + 1]), nn.Identity()
    return m


def inference_copy(model):
    """Bare eval copy: no SpikeMonitor hooks, no autograd state / backward buffers."""
    m = copy.deepcopy(model).eval().requires_grad_(False)
    for mod in m.modules():
        mod._forward_hooks.clear()
    return m


def bench_latency(model, device, mode, iters=200, warmup=50):
    """Median batch-1 forward latency (ms) in inference mode, input already on GPU, synced per call."""
    m = inference_copy(model)
    x = torch.randn(1, BINS, H, W, device=device)
    with torch.inference_mode():
        if mode == "fp16_fused_graph":
            x = x.half()
            ref = copy.deepcopy(m).half()(x)[0].float()
            m = torch.compile(fuse_for_inference(m).half(), dynamic=False)
            err = ((m(x)[0].float() - ref).abs().mean() / ref.abs().mean().clamp_min(1e-6)).item()
            print(f"  fused vs plain fp16 output: mean |diff| / mean |flow| = {err:.1e}", flush=True)
        elif mode != "fp32":
            m, x = m.half(), x.half()
        fn = lambda: m(x)  # noqa: E731
        if mode.endswith("graph"):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(5):
                    m(x)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                m(x)
            fn = g.replay
        times = []
        for i in range(warmup + iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return 1000 * statistics.median(times[warmup:])


def bench_all(model, device, prefix):
    out = {}
    modes = ["fp32", "fp16", "fp16_graph"]
    if all(isinstance(n, SiLIFNodeTriton) for n in model.modules() if isinstance(n, SiLIFNode)):
        modes.append("fp16_fused_graph")   # fused path runs the Triton scan
    for mode in modes:
        try:
            out[f"{prefix}_{mode}_ms"] = bench_latency(model, device, mode)
        except Exception as e:  # report and keep going (e.g. capture unsupported)
            print(f"  latency {prefix} {mode}: failed ({type(e).__name__}: {e})", flush=True)
    for k, v in out.items():
        print(f"  {k:<45} {v:7.3f} ms", flush=True)
    return out


# ---------------------------------------------------------------------------
# QAT
# ---------------------------------------------------------------------------

class FakeQuantAct(nn.Module):
    """Per-tensor INT-`bits` activation fake-quant (STE); scale from an EMA of the batch absmax."""

    def __init__(self, bits, signed, momentum=0.01):
        super().__init__()
        self.bits, self.signed, self.momentum = bits, signed, momentum
        self.register_buffer("absmax", torch.zeros(()))

    def forward(self, x):
        if self.training:
            cur = x.detach().abs().amax().float()
            self.absmax.copy_(torch.where(self.absmax > 0, (1 - self.momentum) * self.absmax + self.momentum * cur, cur))
        qmax = 2 ** (self.bits - 1) - 1 if self.signed else 2 ** self.bits - 1
        lo = -qmax - 1 if self.signed else 0
        scale = (self.absmax / qmax).clamp(min=1e-8).to(x.dtype)
        return quantize.round(x / scale).clamp(lo, qmax) * scale


def apply_qat(model, w_bits, a_bits):
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            parametrize.register_parametrization(m, "weight", FakeQuantWeight(w_bits))
            n += 1
    if a_bits:
        model.q_in = FakeQuantAct(a_bits, signed=True)
        for l in range(len(model.q_feat)):
            model.q_feat[l] = FakeQuantAct(a_bits, signed=True)
            model.q_dec[l] = FakeQuantAct(a_bits, signed=False)
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["spikeflownet", "fast", "spiking", "idnet"], default="fast")
    ap.add_argument("--T", type=int, default=5, help="SNN timesteps (must divide 15)")
    ap.add_argument("--channels", nargs="+", type=int, default=[32, 64, 128])
    ap.add_argument("--dec_channels", nargs="+", type=int, default=[32, 32, 64])
    ap.add_argument("--upsample", choices=["convex", "bilinear"], default="convex")
    ap.add_argument("--res_blocks", nargs="+", type=int, default=None,
                    help="spiking arch: residual spiking blocks after each encoder level")
    ap.add_argument("--neuron", choices=["silif", "lif"], default="silif",
                    help="lif: SiLIF adaptation (a, b) frozen at 0 -> LIF with learned per-channel leak")
    ap.add_argument("--v_threshold", type=float, default=0.5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--rate_penalty", type=float, default=0.0, help="loss += lambda * average spike rate")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_epochs", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--vflip", action="store_true", help="also random vertical flips (hflip is always on)")
    ap.add_argument("--strong_aug", action="store_true", help="also polarity swap, event gain, zoom-in (flow_cache)")
    ap.add_argument("--carry", type=int, default=0,
                    help="K > 0 (spiking arch): SNN state carried across K consecutive windows in training "
                         "(BPTT through all K; --batch_size = sequences), streaming val (state reset at gaps)")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights (0 = off); EMA is evaluated + saved")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--teacher", default=None, help="checkpoint of a trained teacher (e.g. --arch idnet) to distil from")
    ap.add_argument("--distill_weight", type=float, default=0.5,
                    help="loss += w * multiscale EPE to the teacher flow on the pixels with events, "
                         "without GT, inside the GT support (with --teacher)")
    ap.add_argument("--init", default=None, help="checkpoint to start from (e.g. FP model for QAT)")
    ap.add_argument("--qat_bits", type=int, default=0, help="INT-k per-channel weight fake-quant (0 = FP)")
    ap.add_argument("--qat_act_bits", type=int, default=0, help="INT-k activation fake-quant (fast arch)")
    ap.add_argument("--bench_refs", action="store_true", help="also time the original SpikeFlowNet variants")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--max_batches", type=int, default=0, help="debug: cap batches per epoch")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    job = os.environ.get("SLURM_JOB_ID", "local")
    if args.run_name is None:
        args.run_name = f"{args.arch}_T{args.T}_rp{args.rate_penalty:g}" + (f"_qat{args.qat_bits}a{args.qat_act_bits}" if args.qat_bits else "")
    args.run_name += f"_j{job}"
    ckpt = f"checkpoints/fast_flow/{args.run_name}.pt"
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    print(f"Run {args.run_name} | device {device} | ckpt {ckpt}", flush=True)

    model = build_model(args).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device, weights_only=False)["model"])
        print(f"Init from {args.init}", flush=True)
    if args.neuron == "lif":
        print(f"LIF: adaptation frozen at 0 in {freeze_adaptation(model)} neuron layers", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    if args.qat_bits:
        nq = apply_qat(model, args.qat_bits, args.qat_act_bits)
        model.to(device)
        print(f"QAT: {nq} conv weights -> INT{args.qat_bits}, activations INT{args.qat_act_bits or '-'}", flush=True)
    monitor = SpikeMonitor(model)
    print(f"Params: {n_params:,} | spiking layers: {monitor.names}", flush=True)

    teacher = None
    if args.teacher:
        tc = torch.load(args.teacher, map_location=device, weights_only=False)
        teacher = build_model(SimpleNamespace(**tc["args"])).to(device).eval().requires_grad_(False)
        teacher.load_state_dict(tc["model"])
        print(f"Teacher {args.teacher} ({tc['args']['arch']}, val EPE {tc['val']['epe']:.4f}), "
              f"distill weight {args.distill_weight}", flush=True)

    latency = {}
    if device.type == "cuda" and not args.qat_bits and args.arch != "idnet":
        print("Latency (batch 1, 240x320 input, random init):", flush=True)
        if args.bench_refs:
            latency.update(bench_all(SpikeFlowNetT(15, triton=False, native_upsample=True).to(device), device, "ref_spikeflownet_native_T15_torch"))
            latency.update(bench_all(SpikeFlowNetT(15, triton=False).to(device), device, "ref_spikeflownet_T15_torch"))
            latency.update(bench_all(SpikeFlowNetT(15).to(device), device, "ref_spikeflownet_T15_triton"))
            for T in (5, 3):
                latency.update(bench_all(SpikeFlowNetT(T, triton=False).to(device), device, f"ref_spikeflownet_T{T}_torch"))
                latency.update(bench_all(SpikeFlowNetT(T).to(device), device, f"ref_spikeflownet_T{T}_triton"))
                latency.update(bench_all(build_model(args, "fast", T).to(device), device, f"ref_fast_T{T}"))

    train_set = CachedFlowDataset(FLOW_TRAIN_RECORDINGS, bins=BINS, hflip=True, vflip=args.vflip,
                                  strong_aug=args.strong_aug)
    val_set = CachedFlowDataset(_FLOW_VAL_RECORDINGS, bins=BINS)
    kw = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
              prefetch_factor=4 if args.num_workers > 0 else None)
    if args.carry:
        assert args.arch == "spiking" and not args.teacher, "--carry: spiking arch, no teacher"
        model.set_carry(True)
        seq_set = FlowSequenceDataset(train_set, args.carry)   # ~ one pass over the windows per epoch:
        train_loader = DataLoader(seq_set, batch_size=args.batch_size, drop_last=True,
                                  sampler=RandomSampler(seq_set, num_samples=len(seq_set) // args.carry), **kw)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kw)
        print(f"Carried state: {len(seq_set)} sequences of {args.carry} windows, streaming val", flush=True)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kw)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kw)
    stream = bool(args.carry)
    print(f"Train {len(train_set)} samples ({len(train_loader)} it/epoch) | val {len(val_set)}", flush=True)
    if teacher is not None:
        kd_support = gt_support(train_set, args.vflip).to(device)
        print(f"Distillation support: {100 * kd_support.float().mean():.1f}% of the pixels", flush=True)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() == 4 else no_decay).append(p)   # conv weights only
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    steps = args.epochs * (args.max_batches or len(train_loader))
    warm = int(args.warmup_epochs * (args.max_batches or len(train_loader)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True) if args.ema else None
    eval_model = ema.module if ema else model     # (deep copy keeps the SpikeMonitor hooks)

    mlflow.set_experiment("DSEC_FastSpikeFlow")
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({**{k: (str(v) if isinstance(v, list) else v) for k, v in vars(args).items()},
                           "n_params": n_params, "bins": BINS, "input_res": f"{H}x{W}", "eval_res": "480x640",
                           "train_recordings": len(FLOW_TRAIN_RECORDINGS), "val_recordings": str(_FLOW_VAL_RECORDINGS),
                           "slurm_job": job})
        if latency:
            mlflow.log_metrics(latency)

        best = float("inf")
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0 = time.time()
            tot = torch.zeros(4, device=device)
            for it, (vox, flow, mask, _) in enumerate(train_loader):
                if args.max_batches and it >= args.max_batches:
                    break
                vox = vox.to(device, non_blocking=True)
                flow = flow.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                with torch.autocast(device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
                    windows = range(vox.shape[1]) if args.carry else [None]   # (B, K, ...) with --carry
                    if args.carry:
                        model.reset_state()
                    l_flow, rate = 0.0, 0.0
                    for k in windows:
                        v, f, m = (vox, flow, mask) if k is None else (vox[:, k], flow[:, k], mask[:, k])
                        preds = model(v)
                        l_flow = l_flow + multiscale_epe(preds, f, m) / len(windows)
                        rate = rate + monitor.average() / len(windows)
                    loss = l_flow + args.rate_penalty * rate
                    l_kd = torch.zeros((), device=device)
                    if teacher is not None:
                        with torch.no_grad():
                            t = teacher(vox)[0].float()
                            t = F.interpolate(t, size=flow.shape[-2:], mode="bilinear",
                                              align_corners=False) * (flow.shape[-1] / t.shape[-1])
                        ev = F.interpolate((vox.abs().amax(1, keepdim=True) > 0).float(), size=flow.shape[-2:],
                                           mode="nearest").bool()
                        l_kd = multiscale_epe(preds, t, ev & ~mask & kd_support)
                        loss = loss + args.distill_weight * l_kd
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                if ema:
                    ema.update_parameters(model)
                tot += torch.stack([l_flow.detach().float(), rate.detach().float().to(device),
                                    l_kd.detach().float(), torch.ones((), device=device)])
            tot = (tot[:3] / tot[3]).tolist()
            t_train = time.time() - t0
            val = evaluate(eval_model, val_loader, monitor, device, stream=stream)
            mlflow.log_metrics({"train_flow_loss": tot[0], "train_rate": tot[1], "train_kd_loss": tot[2], "lr": opt.param_groups[0]["lr"],
                                "epoch_time_s": t_train, **{f"val_{k}": v for k, v in val.items()}}, step=epoch)
            print(f"Epoch {epoch:3d} | loss {tot[0]:.4f} rate {tot[1]:.4f} | val EPE {val['epe']:.4f} "
                  f"1PE {val['1pe']:.1f}% 3PE {val['3pe']:.1f}% | rate {val.get('rate_avg', 0):.4f} | {t_train:.0f}s", flush=True)
            if val["epe"] < best:
                best = val["epe"]
                torch.save({"model": eval_model.state_dict(), "args": vars(args), "epoch": epoch, "val": val}, ckpt)

        mlflow.log_metric("best_val_epe", best)
        print(f"\nBest val EPE {best:.4f} -> {ckpt}", flush=True)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
        final = evaluate(model, val_loader, monitor, device, stream=stream)
        mlflow.log_metrics({f"final_{k}": v for k, v in final.items()})
        print("Final (best ckpt, fp32): " + " ".join(f"{k}={v:.4f}" for k, v in final.items()), flush=True)
        if device.type == "cuda" and not args.qat_bits and args.arch != "idnet":
            final16 = evaluate(copy.deepcopy(model).half(), val_loader, monitor, device, half=True, stream=stream)
            mlflow.log_metrics({"final_fp16_epe": final16["epe"], "final_fp16_rate_avg": final16["rate_avg"]})
            print(f"Final fp16: EPE {final16['epe']:.4f} rate {final16['rate_avg']:.4f}", flush=True)
            if stream:   # the fused inference layers are stateless (reset every window): time only
                model.set_carry(False)
                print("Latency of the stateless fused path (the carried state adds 3 state reads/writes "
                      "per neuron per window):", flush=True)
            else:
                fused = torch.compile(fuse_for_inference(inference_copy(model)).half(), dynamic=False)
                fused_epe = evaluate(fused, val_loader, SimpleNamespace(names=[], out={}, dens={}), device, half=True)["epe"]
                mlflow.log_metric("final_fused_epe", fused_epe)
                print(f"Final fused fp16 (BN folded + compiled, the timed path): EPE {fused_epe:.4f}", flush=True)
                print("Latency (trained model):", flush=True)
            mlflow.log_metrics(bench_all(model, device, "latency"))
        mlflow.log_artifact(ckpt)


if __name__ == "__main__":
    main()
