"""
``SiLIFNodeTriton`` — a drop-in SiLIF neuron with a Triton-accelerated
multi-step path.

It subclasses the project's :class:`neurons.silif.SiLIFNode` (same parameters,
same recurrence, same ``functional.reset_net`` semantics) and overrides
``multi_step_forward`` to use the fused Triton kernel when:

  * ``backend == 'triton'`` and Triton is installed,
  * the input is on CUDA,
  * ``detach_reset`` is ``True`` (the SiLIF default), and
  * the surrogate exposes ``triton_codes()`` (e.g. ``surrogate.ATan``).

Otherwise it falls back to the exact torch recurrence (the parent's single-step
loop), so the module is safe to import and run on CPU. The single-step
(``step_mode='s'``) path always uses the torch implementation.
"""
import torch
from spikingjelly.activation_based import surrogate as sj_surrogate

from neurons.silif import SiLIFNode
from silif_triton.silif_triton_kernel import SiLIFMultiStepTriton, _HAS_TRITON


class SiLIFNodeTriton(SiLIFNode):
    def __init__(self, *args, backend: str = "triton", **kwargs):
        kwargs.setdefault("step_mode", "m")
        super().__init__(*args, **kwargs)
        self.backend = backend

    @property
    def supported_backends(self):
        if self.step_mode == "s":
            return ("torch",)
        return ("torch", "triton")

    def _can_use_triton(self, x_seq: torch.Tensor) -> bool:
        # The kernel inlines the ATan surrogate (triton 3.2 cannot take a
        # surrogate passed as a callable), so the Triton path requires ATan.
        return (
            self.backend == "triton"
            and _HAS_TRITON
            and x_seq.is_cuda
            and self.detach_reset
            and isinstance(self.surrogate_function, sj_surrogate.ATan)
        )

    def _ensure_state_tensors(self, x0: torch.Tensor):
        """Promote scalar (post-reset) states to (B, C, *spatial) tensors."""
        if not isinstance(self.v, torch.Tensor):
            self.v = torch.zeros_like(x0)
        if not isinstance(self.w, torch.Tensor):
            self.w = torch.zeros_like(x0)
        if not isinstance(self.s_prev, torch.Tensor):
            self.s_prev = torch.zeros_like(x0)

    def _torch_multistep(self, x_seq: torch.Tensor):
        """
        Exact torch recurrence over the whole sequence, with per-channel params
        broadcast for any input rank ((T, B, C) linear or (T, B, C, *spatial)
        conv). Updates v / w / s_prev / v_seq, matching the Triton path.
        """
        self._ensure_state_tensors(x_seq[0])
        T, C = x_seq.shape[0], x_seq.shape[2]
        view = (1, C) + (1,) * (x_seq.dim() - 3)
        D = torch.exp(self.dt)
        alpha = torch.exp(-torch.exp(self.lambda_log_alpha) * D).view(view)
        beta = torch.exp(-torch.exp(self.lambda_log_beta) * D).view(view)
        a = self.a.clamp(self.a_min, self.a_max).view(view)
        b = self.b.clamp(self.b_min, self.b_max).view(view)

        v, w, s_prev = self.v, self.w, self.s_prev
        s_list, v_list = [], []
        for t in range(T):
            w = beta * w + a * v + b * s_prev
            v = alpha * (v - s_prev) + (1.0 - alpha) * (x_seq[t] - w)
            s = self.surrogate_function(v - self.v_threshold)
            s_list.append(s)
            v_list.append(v)
            s_prev = s.detach() if self.detach_reset else s

        self.v, self.w, self.s_prev = v, w, s_prev
        s_seq = torch.stack(s_list, 0)
        if self.store_v_seq:
            self.v_seq = torch.stack(v_list, 0)
        return s_seq

    def multi_step_forward(self, x_seq: torch.Tensor):
        if not self._can_use_triton(x_seq):
            return self._torch_multistep(x_seq)

        self._ensure_state_tensors(x_seq[0])
        s_seq, v_seq, w_seq = SiLIFMultiStepTriton.apply(
            x_seq, self.v, self.w, self.s_prev,
            self.lambda_log_alpha, self.lambda_log_beta, self.dt, self.a, self.b,
            self.v_threshold, self.a_min, self.a_max, self.b_min, self.b_max,
            float(self.surrogate_function.alpha),
        )
        self.v = v_seq[-1].clone()
        self.w = w_seq[-1].clone()
        self.s_prev = s_seq[-1].detach()
        if self.store_v_seq:
            self.v_seq = v_seq
        return s_seq
