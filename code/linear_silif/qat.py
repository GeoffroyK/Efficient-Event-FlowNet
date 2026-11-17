'''P1/P2 quantization-aware training: weights + neuron-param INT-k fake-quant.

Neuromorphic-chip (Loihi) target.
  P1 (`apply_weight_qat`): SYNAPTIC weights (Linear + Embedding). Spikes are
      already 1-bit so there is no activation-quantization cost.
  P2 (`apply_neuron_qat`): the SSM dynamics params of the neuron itself --
      decay/frequency (lambda_log_real, lambda_img), time step (dt), input
      gain (b_real, b_img) and axonal delay -- previously left FP.
Norms stay FP for now (P3). Uses SpikingJelly's straight-through `quantize.round`
so gradients flow to the underlying FP param; the quantized value is used in
forward. Implemented as a torch parametrization so the trainable parameter stays
FP and `module.weight` (or `.dt`, `.lambda_log_real`, ...) transparently returns
its INT-k reconstruction.
'''

import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from spikingjelly.activation_based import quantize


class FakeQuantWeight(nn.Module):
    '''Symmetric per-output-channel INT-`bits` fake-quant (STE). Scale = |w|max / qmax
    computed per slice along `ch_dim` (output channels for Linear, rows for Embedding),
    so each filter/embedding-row gets its own scale -- standard, more accurate than
    per-tensor. Scale is detached (gradient flows only through the STE round).'''

    def __init__(self, bits: int, ch_dim: int = 0):
        super().__init__()
        self.bits = bits
        self.ch_dim = ch_dim
        self.qmax = 2 ** (bits - 1) - 1

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        dims = [d for d in range(w.dim()) if d != self.ch_dim]
        absmax = w.detach().abs().amax(dim=dims, keepdim=True).clamp(min=1e-12)
        scale = absmax / self.qmax
        q = quantize.round(w / scale).clamp(-self.qmax - 1, self.qmax)   # STE round + clamp
        return q * scale


def apply_weight_qat(model: nn.Module, bits: int) -> int:
    '''Register INT-`bits` fake-quant on every Linear/Embedding weight in the model.
    Call AFTER loading the FP checkpoint and BEFORE building the optimizer (the trainable
    param becomes `...parametrizations.weight.original`, still in model.parameters()).
    Returns the number of quantized layers.'''
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            parametrize.register_parametrization(m, "weight", FakeQuantWeight(bits))
            n += 1
    return n


# SSM dynamics params living directly on the neuron block (AsyncCSiLIFBlock /
# ChannelDelay, both plain nn.Parameter attributes -- see linear_silif/async_model.py
# and evttc_ttc's reuse of the same block). Matches the `ssm_names` set already used
# in train_async.py to give these params their own (lower) LR / no weight decay.
NEURON_QAT_PARAMS = ("lambda_log_real", "lambda_img", "dt", "b_real", "b_img", "delay")


class FakeQuantVector(nn.Module):
    '''Symmetric per-TENSOR INT-`bits` fake-quant (STE) for a 1-D neuron parameter
    vector. Unlike `FakeQuantWeight`, one scale for the WHOLE vector: each entry is
    already a single neuron's own scalar (the "channel" dim), so a per-element scale
    would let every value round to itself exactly and quantize nothing.'''

    def __init__(self, bits: int):
        super().__init__()
        self.bits = bits
        self.qmax = 2 ** (bits - 1) - 1

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        absmax = w.detach().abs().amax().clamp(min=1e-12)
        scale = absmax / self.qmax
        q = quantize.round(w / scale).clamp(-self.qmax - 1, self.qmax)   # STE round + clamp
        return q * scale


def apply_neuron_qat(model: nn.Module, bits: int, param_names=NEURON_QAT_PARAMS) -> int:
    '''Register INT-`bits` fake-quant on the named SSM dynamics params of every
    neuron block in the model (P2). Duck-typed on `nn.Parameter` presence rather than
    importing AsyncCSiLIFBlock, so this works unchanged for any block (linear_silif,
    evttc_ttc, ...) that reuses those param names. Call AFTER loading the FP checkpoint
    and BEFORE building the optimizer, same as `apply_weight_qat`. Returns the number
    of quantized param tensors.'''
    n = 0
    for m in model.modules():
        for name in param_names:
            if m._parameters.get(name) is not None:
                parametrize.register_parametrization(m, name, FakeQuantVector(bits))
                n += 1
    return n
