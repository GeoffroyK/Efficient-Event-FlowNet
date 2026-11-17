import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional

from neurons.silif import SiLIFNode


class ConvSiLIFNode(SiLIFNode):
    """SiLIFNode adapted for conv feature maps: params are per-channel, broadcast over H×W."""

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, H, W)
        self.v_float_to_tensor(x)

        alpha = torch.exp(-torch.exp(self.lambda_log_alpha) * torch.exp(self.dt))
        beta  = torch.exp(-torch.exp(self.lambda_log_beta)  * torch.exp(self.dt))

        # Reshape (channels,) → (1, channels, 1, 1) for spatial broadcasting
        def ch(t):
            return t.view(1, -1, 1, 1)

        a = self.a.clamp(self.a_min, self.a_max)
        b = self.b.clamp(self.b_min, self.b_max)

        self.w = ch(beta) * self.w + ch(a) * self.v + ch(b) * self.s_prev
        self.v = ch(alpha) * (self.v - self.s_prev) + (1 - ch(alpha)) * (x - self.w)

        spike = self.neuronal_fire()
        self.s_prev = spike.detach() if self.detach_reset else spike
        return spike


class ConvSiLIFLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.0,
        return_membrane: bool = False,
        v_threshold: float = 0.5,
    ):
        super().__init__()
        self.return_membrane = return_membrane

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')

        self.norm = nn.BatchNorm2d(out_channels, momentum=0.05)
        v_thresh = float('inf') if return_membrane else v_threshold
        self.neuron = ConvSiLIFNode(
            hidden_size=out_channels,
            v_threshold=v_thresh,
            step_mode='m',
            store_v_seq=return_membrane,  # only pay the cost when we need membrane readout
        )
        self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor):
        # x: (B, T, C_in, H, W)
        B, T = x.shape[:2]
        functional.reset_net(self.neuron)

        # Conv2d + BN for all timesteps in one batched call instead of T separate ones.
        # BN also benefits: statistics are computed over B*T samples instead of B.
        out = self.norm(self.conv(x.flatten(0, 1)))        # (B*T, C_out, H', W')
        out = out.view(B, T, *out.shape[1:])               # (B,   T, C_out, H', W')

        # SpikingJelly multi-step mode expects (T, B, ...) and loops internally
        spikes = self.neuron(out.permute(1, 0, 2, 3, 4))  # (T, B, C_out, H', W')
        spikes = spikes.permute(1, 0, 2, 3, 4)            # (B, T, C_out, H', W')
        spikes = self.drop(spikes.flatten(0, 1)).reshape_as(spikes)

        if self.return_membrane:
            # v_seq is stored by SpikingJelly when store_v_seq=True: (T, B, C_out, H', W')
            membrane = self.neuron.v_seq.permute(1, 0, 2, 3, 4)  # (B, T, C_out, H', W')
            return spikes, membrane
        return spikes


class MembraneReadout(nn.Module):
    """
    Reads membrane potential instead of spikes.
    Applies global average pooling over spatial dims, then a leaky integrator
    over time, and projects to output classes.
    """

    def __init__(self, in_channels: int, output_size: int, tau: float = 10.0):
        super().__init__()
        self.W = nn.Linear(in_channels, output_size, bias=True)
        self.register_buffer('alpha', torch.tensor(1.0 - 1.0 / tau))

    def forward(self, membrane: torch.Tensor) -> torch.Tensor:
        # membrane: (batch, time, C, H, W)
        # Pool spatial dims → (batch, time, C)
        b, t, c = membrane.shape[:3]
        pooled = F.adaptive_avg_pool2d(membrane.flatten(0, 1), 1).view(b, t, c)

        Wv = self.W(pooled)  # (batch, time, output_size)

        v = torch.zeros(b, Wv.shape[2], device=membrane.device)
        out = torch.zeros_like(v)
        for i in range(t):
            v = self.alpha * v + (1.0 - self.alpha) * Wv[:, i]
            out = out + torch.softmax(v, dim=-1)

        return out  # (batch, output_size)


class ConvSNN(nn.Module):
    """
    Small conv-based SNN using SiLIF neurons.
    All hidden layers emit spikes; the final layer exposes its membrane potential
    to a non-spiking MembraneReadout for classification.

    Expected input: (batch, time, C, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        output_size: int,
        channels: list[int] = [16, 32],
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        ch_in = in_channels
        for i, ch_out in enumerate(channels):
            is_last = i == len(channels) - 1
            layers.append(
                ConvSiLIFLayer(
                    in_channels=ch_in,
                    out_channels=ch_out,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    dropout=dropout,
                    return_membrane=is_last,
                )
            )
            ch_in = ch_out

        self.layers = nn.ModuleList(layers)
        self.readout = MembraneReadout(channels[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, C, H, W)
        for layer in self.layers[:-1]:
            x = layer(x)

        # Last layer has v_threshold=inf — neuron never fires, v accumulates freely
        _, membrane = self.layers[-1](x)

        return self.readout(membrane)  # (batch, output_size)

    def forward_with_recording(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Run a forward pass and collect per-layer membrane potentials and spikes
        at every timestep via hooks on each ConvSiLIFNode.

        Returns:
            logits:     (batch, output_size)
            layer_data: dict mapping layer name → (membrane, spikes)
                        each array is (batch, time, C, H, W)
        """
        hooks: dict[str, dict[str, list]] = {
            f"layer_{i}": {"mem": [], "spk": []} for i in range(len(self.layers))
        }
        handles = []

        for i, layer in enumerate(self.layers):
            name = f"layer_{i}"

            def _hook(module, _, output, _name=name):
                hooks[_name]["mem"].append(module.v.detach().cpu())
                hooks[_name]["spk"].append(output.detach().cpu())

            handles.append(layer.neuron.register_forward_hook(_hook))

        logits = self.forward(x)

        for h in handles:
            h.remove()

        layer_data = {}
        for name, data in hooks.items():
            # list of T tensors (B, C, H, W) → stack to (B, T, C, H, W)
            layer_data[name] = (
                torch.stack(data["mem"], dim=1).numpy(),
                torch.stack(data["spk"], dim=1).numpy(),
            )

        return logits, layer_data

