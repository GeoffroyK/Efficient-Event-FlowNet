"""
Triton kernels for the multi-step SiLIF neuron.

The torch SiLIF runs a Python ``for t in range(T)`` loop, launching ~6 small
elementwise CUDA kernels per step and materialising every intermediate. This
module fuses the whole temporal scan into a single Triton kernel (forward) and a
single reverse-time kernel (backward), keeping the per-neuron state (v, w) in
registers across all T steps — the same idea spikingjelly uses for its
LIF/PLIF triton kernels.

Design notes
------------
* **Layout.** A multi-step tensor ``(T, B, C, *spatial)`` is flattened to
  ``(T, N)`` with ``N = B·C·S`` and ``S = prod(spatial)``. The per-channel
  parameters ``(C,)`` are gathered per element with ``chan = (n // S) % C``.
* **Per-channel parameter gradients** are accumulated per element inside the
  kernel, written out as ``(N,)`` buffers, then reduced over batch+spatial in
  PyTorch (avoids cross-program atomics). The reparametrisation / clamp chain
  rule (``α = exp(-exp(λ_α)·exp(dt))`` etc.) is also applied in PyTorch — it is
  exactly the math validated against autograd in ``silif_reference``.
* **detach_reset=True** is assumed (the default for SiLIF). The previous spike
  carries no gradient into future states, which is why the backward only needs
  to propagate through the continuous (v, w) states.

The autograd-facing entry point is ``SiLIFMultiStepTriton.apply``.
"""
import torch
from torch import autograd

# Plain ``@triton.jit`` kernels read their own source at decoration time but do
# not touch the GPU driver until launched, so this module imports fine on a
# CPU-only / driverless machine; ``_HAS_TRITON`` gates actual use.
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
    _TL_DTYPE = {torch.float32: tl.float32, torch.float16: tl.float16,
                 torch.bfloat16: tl.bfloat16}
except BaseException:  # pragma: no cover - triton optional
    triton = None
    tl = None
    _HAS_TRITON = False
    _TL_DTYPE = {}

# Fixed launch configuration (a sane default; tune per GPU if desired).
_BLOCK = 256
_NUM_WARPS = 4


if _HAS_TRITON:

    @triton.jit
    def _atan_surrogate(x, alpha):
        # ATan surrogate gradient: alpha / 2 / (1 + (pi/2 * alpha * x)^2).
        # Inlined as a device function — triton 3.2 cannot take a surrogate
        # passed as a callable constexpr, so we hardcode SiLIF's default (ATan)
        # and receive only its scalar `alpha` at runtime.
        s = 1.5707963267948966 * alpha * x
        return alpha * 0.5 / (s * s + 1.0)

    @triton.jit
    def _silif_fwd_kernel(
        x_ptr, v_init_ptr, w_init_ptr, s_init_ptr,
        lla_ptr, llb_ptr, dt_ptr, a_ptr, b_ptr,
        s_seq_ptr, v_seq_ptr, w_seq_ptr,
        v_threshold, a_min, a_max, b_min, b_max,
        T: tl.constexpr, N: tl.constexpr, S: tl.constexpr, C: tl.constexpr,
        BLOCK: tl.constexpr, DT: tl.constexpr, save_w: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        chan = (offs // S) % C  # per-element channel index for the param gather

        lla = tl.load(lla_ptr + chan, mask=mask, other=0.0).to(tl.float32)
        llb = tl.load(llb_ptr + chan, mask=mask, other=0.0).to(tl.float32)
        dt = tl.load(dt_ptr + chan, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + chan, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + chan, mask=mask, other=0.0).to(tl.float32)
        a = tl.minimum(tl.maximum(a, a_min), a_max)
        b = tl.minimum(tl.maximum(b, b_min), b_max)

        D = tl.exp(dt)
        alpha = tl.exp(-tl.exp(lla) * D)
        beta = tl.exp(-tl.exp(llb) * D)
        one_m_alpha = 1.0 - alpha

        v = tl.load(v_init_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_init_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        s_prev = tl.load(s_init_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        for t in tl.static_range(0, T, 1):
            x = tl.load(x_ptr + t * N + offs, mask=mask, other=0.0).to(tl.float32)
            w = beta * w + a * v + b * s_prev
            v = alpha * (v - s_prev) + one_m_alpha * (x - w)
            s = (v >= v_threshold).to(tl.float32)
            tl.store(s_seq_ptr + t * N + offs, s.to(DT), mask=mask)
            tl.store(v_seq_ptr + t * N + offs, v.to(DT), mask=mask)
            if save_w:
                tl.store(w_seq_ptr + t * N + offs, w.to(DT), mask=mask)
            s_prev = s

    @triton.jit
    def _silif_bwd_kernel(
        grad_s_ptr, grad_v_ptr, x_ptr, v_seq_ptr, w_seq_ptr, s_seq_ptr,
        v_init_ptr, w_init_ptr, s_init_ptr,
        lla_ptr, llb_ptr, dt_ptr, a_ptr, b_ptr,
        grad_x_ptr, dalpha_ptr, dbeta_ptr, da_ptr, db_ptr,
        gv_init_ptr, gw_init_ptr,
        v_threshold, a_min, a_max, b_min, b_max, sg_alpha,
        T: tl.constexpr, N: tl.constexpr, S: tl.constexpr, C: tl.constexpr,
        BLOCK: tl.constexpr, DT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        chan = (offs // S) % C

        lla = tl.load(lla_ptr + chan, mask=mask, other=0.0).to(tl.float32)
        llb = tl.load(llb_ptr + chan, mask=mask, other=0.0).to(tl.float32)
        dt = tl.load(dt_ptr + chan, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + chan, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + chan, mask=mask, other=0.0).to(tl.float32)
        a = tl.minimum(tl.maximum(a, a_min), a_max)
        b = tl.minimum(tl.maximum(b, b_min), b_max)
        D = tl.exp(dt)
        alpha = tl.exp(-tl.exp(lla) * D)
        beta = tl.exp(-tl.exp(llb) * D)
        one_m_alpha = 1.0 - alpha

        gv = tl.zeros([BLOCK], dtype=tl.float32)
        gw = tl.zeros([BLOCK], dtype=tl.float32)
        dalpha = tl.zeros([BLOCK], dtype=tl.float32)
        dbeta = tl.zeros([BLOCK], dtype=tl.float32)
        da = tl.zeros([BLOCK], dtype=tl.float32)
        db = tl.zeros([BLOCK], dtype=tl.float32)

        for t in tl.static_range(T - 1, -1, -1):
            v_t = tl.load(v_seq_ptr + t * N + offs, mask=mask, other=0.0).to(tl.float32)
            w_t = tl.load(w_seq_ptr + t * N + offs, mask=mask, other=0.0).to(tl.float32)
            x_t = tl.load(x_ptr + t * N + offs, mask=mask, other=0.0).to(tl.float32)
            if t > 0:
                vp = tl.load(v_seq_ptr + (t - 1) * N + offs, mask=mask, other=0.0).to(tl.float32)
                wp = tl.load(w_seq_ptr + (t - 1) * N + offs, mask=mask, other=0.0).to(tl.float32)
                sp = tl.load(s_seq_ptr + (t - 1) * N + offs, mask=mask, other=0.0).to(tl.float32)
            else:
                vp = tl.load(v_init_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                wp = tl.load(w_init_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                sp = tl.load(s_init_ptr + offs, mask=mask, other=0.0).to(tl.float32)

            grad_s = tl.load(grad_s_ptr + t * N + offs, mask=mask, other=0.0).to(tl.float32)
            grad_v = tl.load(grad_v_ptr + t * N + offs, mask=mask, other=0.0).to(tl.float32)
            sigma = _atan_surrogate(v_t - v_threshold, sg_alpha)

            gV = grad_v + grad_s * sigma + gv
            gW = gw - gV * one_m_alpha

            tl.store(grad_x_ptr + t * N + offs, (gV * one_m_alpha).to(DT), mask=mask)
            dalpha += gV * (vp - sp - x_t + w_t)
            dbeta += gW * wp
            da += gW * vp
            db += gW * sp

            gv = gV * alpha + gW * a
            gw = gW * beta

        tl.store(dalpha_ptr + offs, dalpha, mask=mask)
        tl.store(dbeta_ptr + offs, dbeta, mask=mask)
        tl.store(da_ptr + offs, da, mask=mask)
        tl.store(db_ptr + offs, db, mask=mask)
        tl.store(gv_init_ptr + offs, gv, mask=mask)
        tl.store(gw_init_ptr + offs, gw, mask=mask)


def _shapes(x_seq):
    """(T, B, C, *spatial) -> (T, N, S, C, B)."""
    T, B, C = x_seq.shape[0], x_seq.shape[1], x_seq.shape[2]
    S = 1
    for d in x_seq.shape[3:]:
        S *= d
    N = B * C * S
    return T, N, S, C, B


class SiLIFMultiStepTriton(autograd.Function):
    """Triton-backed multi-step SiLIF (forward + backward), ``detach_reset=True``."""

    @staticmethod
    def forward(ctx, x_seq, v_init, w_init, s_init,
                lambda_log_alpha, lambda_log_beta, dt, a_raw, b_raw,
                v_threshold, a_min, a_max, b_min, b_max, sg_alpha):
        if not _HAS_TRITON:
            raise RuntimeError("Triton is not available in this environment.")
        x_seq = x_seq.contiguous()
        T, N, S, C, B = _shapes(x_seq)
        dt_tl = _TL_DTYPE[x_seq.dtype]

        x_flat = x_seq.reshape(T, N)
        v_init = v_init.reshape(N).contiguous()
        w_init = w_init.reshape(N).contiguous()
        s_init = s_init.reshape(N).contiguous()

        s_seq = torch.empty_like(x_flat)
        v_seq = torch.empty_like(x_flat)
        need_grad = any(ctx.needs_input_grad[:9])
        w_seq = torch.empty_like(x_flat) if need_grad else x_flat[:1]

        grid = (triton.cdiv(N, _BLOCK),)
        _silif_fwd_kernel[grid](
            x_flat, v_init, w_init, s_init,
            lambda_log_alpha, lambda_log_beta, dt, a_raw, b_raw,
            s_seq, v_seq, w_seq,
            float(v_threshold), float(a_min), float(a_max), float(b_min), float(b_max),
            T=T, N=N, S=S, C=C, BLOCK=_BLOCK, DT=dt_tl, save_w=need_grad,
            num_warps=_NUM_WARPS,
        )

        if need_grad:
            ctx.save_for_backward(x_flat, v_seq, w_seq, s_seq, v_init, w_init,
                                  s_init, lambda_log_alpha, lambda_log_beta,
                                  dt, a_raw, b_raw)
            ctx.params = (float(v_threshold), float(a_min), float(a_max),
                          float(b_min), float(b_max), float(sg_alpha),
                          T, N, S, C, B, x_seq.shape, dt_tl)

        s_out = s_seq.reshape(x_seq.shape)
        v_out = v_seq.reshape(x_seq.shape)
        w_out = w_seq.reshape(x_seq.shape) if need_grad else w_seq
        ctx.mark_non_differentiable(w_out)
        return s_out, v_out, w_out

    @staticmethod
    def backward(ctx, grad_s_seq, grad_v_seq, grad_w_seq):
        (x_flat, v_seq, w_seq, s_seq, v_init, w_init, s_init,
         lla, llb, dt, a_raw, b_raw) = ctx.saved_tensors
        (v_threshold, a_min, a_max, b_min, b_max, sg_alpha,
         T, N, S, C, B, orig_shape, dt_tl) = ctx.params

        grad_s_seq = grad_s_seq.reshape(T, N).contiguous() if grad_s_seq is not None \
            else torch.zeros_like(x_flat)
        grad_v_seq = grad_v_seq.reshape(T, N).contiguous() if grad_v_seq is not None \
            else torch.zeros_like(x_flat)

        grad_x = torch.empty_like(x_flat)
        dalpha = torch.empty(N, dtype=torch.float32, device=x_flat.device)
        dbeta = torch.empty_like(dalpha)
        da = torch.empty_like(dalpha)
        db = torch.empty_like(dalpha)
        gv_init = torch.empty_like(dalpha)
        gw_init = torch.empty_like(dalpha)

        grid = (triton.cdiv(N, _BLOCK),)
        _silif_bwd_kernel[grid](
            grad_s_seq, grad_v_seq, x_flat, v_seq, w_seq, s_seq,
            v_init, w_init, s_init,
            lla, llb, dt, a_raw, b_raw,
            grad_x, dalpha, dbeta, da, db, gv_init, gw_init,
            float(v_threshold), float(a_min), float(a_max), float(b_min), float(b_max),
            float(sg_alpha),
            T=T, N=N, S=S, C=C, BLOCK=_BLOCK, DT=dt_tl,
            num_warps=_NUM_WARPS,
        )

        # Reduce per-element param grads over batch + spatial -> per channel (C,).
        # Flat index n maps to (b, c, s); reshape to (B, C, S) and sum over b, s.
        def per_channel(buf):
            return buf.reshape(B, C, S).sum(dim=(0, 2))

        dalpha_c, dbeta_c = per_channel(dalpha), per_channel(dbeta)
        da_c, db_c = per_channel(da), per_channel(db)

        # Chain rule through α = exp(-exp(λ_α)·exp(dt)), β = exp(-exp(λ_β)·exp(dt)).
        D = torch.exp(dt)
        alpha = torch.exp(-torch.exp(lla) * D)
        beta = torch.exp(-torch.exp(llb) * D)
        alpha_ln = alpha * torch.log(alpha)
        beta_ln = beta * torch.log(beta)
        grad_lla = (dalpha_c * alpha_ln).to(lla.dtype)
        grad_llb = (dbeta_c * beta_ln).to(llb.dtype)
        grad_dt = (dalpha_c * alpha_ln + dbeta_c * beta_ln).to(dt.dtype)

        a_mask = ((a_raw >= a_min) & (a_raw <= a_max)).to(da_c.dtype)
        b_mask = ((b_raw >= b_min) & (b_raw <= b_max)).to(db_c.dtype)
        grad_a = (da_c * a_mask).to(a_raw.dtype)
        grad_b = (db_c * b_mask).to(b_raw.dtype)

        grad_x = grad_x.reshape(orig_shape)
        grad_v_in = gv_init.reshape(orig_shape[1:]).to(v_init.dtype) \
            if ctx.needs_input_grad[1] else None
        grad_w_in = gw_init.reshape(orig_shape[1:]).to(w_init.dtype) \
            if ctx.needs_input_grad[2] else None

        # inputs: x_seq, v_init, w_init, s_init, lla, llb, dt, a, b,
        #         v_threshold, a_min, a_max, b_min, b_max, sg_alpha
        return (grad_x, grad_v_in, grad_w_in, None,
                grad_lla, grad_llb, grad_dt, grad_a, grad_b,
                None, None, None, None, None, None)
