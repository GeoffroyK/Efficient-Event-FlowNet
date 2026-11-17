"""
Pure-PyTorch reference for the multi-step SiLIF recurrence.

This module is the *ground truth* the Triton kernel is validated against, and the
CPU / autograd fallback for the Triton neuron. It contains:

  * ``silif_scan_torch`` — an autograd-friendly multi-step scan that reproduces
    exactly the recurrence of ``neurons.silif.SiLIFNode`` (per-channel learnable
    decay/adaptation), but over a whole ``(T, ...)`` sequence at once.
  * ``silif_manual_backward`` — the *same* analytic gradients the Triton backward
    kernel computes, written in plain PyTorch so the (delicate) math can be
    gradchecked on CPU, independently of any GPU.

SiLIF recurrence (per time step, per channel; ``s`` is the previous spike):

    w_t = β·w_{t-1} + a·v_{t-1} + b·s_{t-1}
    v_t = α·(v_{t-1} - s_{t-1}) + (1-α)·(x_t - w_t)
    s_t = Θ(v_t - v_threshold)            (surrogate gradient on the backward)

with the reparametrisation (``D = exp(dt)``)

    α = exp(-exp(λ_α)·D),   β = exp(-exp(λ_β)·D),   a = clamp(a),  b = clamp(b)

Note there is no explicit membrane reset: the ``-s_{t-1}`` term inside ``v_t`` is
the (soft) reset, exactly as in the original ``SiLIFNode``. ``detach_reset=True``
(the default everywhere SiLIF is used) means the previous spike carries no
gradient into future states — this is what the Triton path supports.
"""
from typing import Optional

import torch


def derive_alpha_beta(lambda_log_alpha, lambda_log_beta, dt):
    """Map the log-space parameters to the (α, β) decay factors."""
    D = torch.exp(dt)
    alpha = torch.exp(-torch.exp(lambda_log_alpha) * D)
    beta = torch.exp(-torch.exp(lambda_log_beta) * D)
    return alpha, beta


def silif_scan_torch(x_seq, lambda_log_alpha, lambda_log_beta, dt, a_raw, b_raw,
                     v_init=None, w_init=None, s_init=None,
                     v_threshold=1.0, a_min=0.0, a_max=1.0, b_min=0.0, b_max=2.0,
                     surrogate=None, detach_reset=True):
    """
    Autograd-friendly SiLIF scan.

    Args:
        x_seq: input current, shape ``(T, B, C, *spatial)``. Channel is dim 2.
        lambda_log_alpha, lambda_log_beta, dt, a_raw, b_raw: per-channel
            parameters, shape ``(C,)``.
        v_init/w_init/s_init: optional initial states, shape ``(B, C, *spatial)``.
        surrogate: spiking surrogate (e.g. ``spikingjelly...surrogate.ATan()``).
            Defaults to ATan if ``None``.

    Returns:
        ``(s_seq, v_seq)`` each shaped like ``x_seq``.
    """
    if surrogate is None:
        from spikingjelly.activation_based import surrogate as sg
        surrogate = sg.ATan()

    T = x_seq.shape[0]
    C = x_seq.shape[2]
    # Broadcast per-channel params to (1, C, 1, ...) for the data layout.
    view = (1, C) + (1,) * (x_seq.dim() - 3)
    alpha, beta = derive_alpha_beta(lambda_log_alpha, lambda_log_beta, dt)
    alpha = alpha.view(view)
    beta = beta.view(view)
    a = a_raw.clamp(a_min, a_max).view(view)
    b = b_raw.clamp(b_min, b_max).view(view)

    v = torch.zeros_like(x_seq[0]) if v_init is None else v_init
    w = torch.zeros_like(x_seq[0]) if w_init is None else w_init
    s_prev = torch.zeros_like(x_seq[0]) if s_init is None else s_init

    s_list, v_list = [], []
    for t in range(T):
        x = x_seq[t]
        w = beta * w + a * v + b * s_prev
        v = alpha * (v - s_prev) + (1.0 - alpha) * (x - w)
        s = surrogate(v - v_threshold)
        s_list.append(s)
        v_list.append(v)
        s_prev = s.detach() if detach_reset else s
    return torch.stack(s_list, 0), torch.stack(v_list, 0)


@torch.no_grad()
def silif_manual_backward(grad_s_seq, grad_v_seq, x_seq, v_seq, w_seq, s_seq,
                          lambda_log_alpha, lambda_log_beta, dt, a_raw, b_raw,
                          v_init=None, w_init=None, s_init=None,
                          v_threshold=1.0, a_min=0.0, a_max=1.0,
                          b_min=0.0, b_max=2.0, surrogate=None):
    """
    Analytic SiLIF backward — the *exact* computation the Triton kernel performs,
    written in PyTorch so it can be gradchecked on CPU.

    Returns gradients ``(grad_x_seq, grad_lla, grad_llb, grad_dt, grad_a, grad_b,
    grad_v_init, grad_w_init)``. ``detach_reset=True`` is assumed.
    """
    if surrogate is None:
        from spikingjelly.activation_based import surrogate as sg
        surrogate = sg.ATan()

    T, B, C = x_seq.shape[0], x_seq.shape[1], x_seq.shape[2]
    view = (1, C) + (1,) * (x_seq.dim() - 3)
    alpha, beta = derive_alpha_beta(lambda_log_alpha, lambda_log_beta, dt)
    alpha_b = alpha.view(view)
    beta_b = beta.view(view)
    a_b = a_raw.clamp(a_min, a_max).view(view)

    if v_init is None:
        v_init = torch.zeros_like(x_seq[0])
    if w_init is None:
        w_init = torch.zeros_like(x_seq[0])
    if s_init is None:
        s_init = torch.zeros_like(x_seq[0])

    grad_x = torch.zeros_like(x_seq)
    gv = torch.zeros_like(x_seq[0])   # grad flowing back into v_{t-1}
    gw = torch.zeros_like(x_seq[0])   # grad flowing back into w_{t-1}
    # per-element accumulators for the (natural) param grads
    dalpha = torch.zeros_like(x_seq[0])
    dbeta = torch.zeros_like(x_seq[0])
    da = torch.zeros_like(x_seq[0])
    db = torch.zeros_like(x_seq[0])

    one_m_alpha = 1.0 - alpha_b
    for t in range(T - 1, -1, -1):
        v_t = v_seq[t]
        w_t = w_seq[t]
        x_t = x_seq[t]
        vp = v_seq[t - 1] if t > 0 else v_init
        wp = w_seq[t - 1] if t > 0 else w_init
        sp = s_seq[t - 1] if t > 0 else s_init

        # surrogate gradient of the spike at this step
        with torch.enable_grad():
            h = (v_t - v_threshold).detach().requires_grad_(True)
            sigma = torch.autograd.grad(surrogate(h).sum(), h)[0]

        gV = grad_v_seq[t] + grad_s_seq[t] * sigma + gv
        gW = gw - gV * one_m_alpha

        grad_x[t] = gV * one_m_alpha
        dalpha += gV * (vp - sp - x_t + w_t)
        dbeta += gW * wp
        da += gW * vp
        db += gW * sp

        gv = gV * alpha_b + gW * a_b
        gw = gW * beta_b

    # reduce per-element param grads over batch + spatial -> per channel (C,).
    # The accumulators have the per-step shape (B, C, *spatial): channel is dim 1.
    reduce_dims = (0,) + tuple(range(2, dalpha.dim()))
    dalpha_c = dalpha.sum(reduce_dims)
    dbeta_c = dbeta.sum(reduce_dims)
    da_c = da.sum(reduce_dims)
    db_c = db.sum(reduce_dims)

    # chain rule through the reparametrisation  α = exp(-exp(λ_α)·exp(dt)):
    #   ∂α/∂λ_α = α·ln α ,  ∂α/∂dt = α·ln α  (dt is shared between α and β)
    alpha_ln = alpha * torch.log(alpha)
    beta_ln = beta * torch.log(beta)
    grad_lla = dalpha_c * alpha_ln
    grad_llb = dbeta_c * beta_ln
    grad_dt = dalpha_c * alpha_ln + dbeta_c * beta_ln

    # clamp on a, b: gradient passes only inside the valid range
    a_mask = ((a_raw >= a_min) & (a_raw <= a_max)).to(da_c.dtype)
    b_mask = ((b_raw >= b_min) & (b_raw <= b_max)).to(db_c.dtype)
    grad_a = da_c * a_mask
    grad_b = db_c * b_mask

    return grad_x, grad_lla, grad_llb, grad_dt, grad_a, grad_b, gv, gw
