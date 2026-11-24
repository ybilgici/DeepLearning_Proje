"""
Paper-faithful Adam and AMSGrad (Reddi, Kale & Kumar, ICLR 2018).

Update rule (Algorithm 2 of the paper; Adam is the same without the max):

    m_t     = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t     = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    v_hat_t = max(v_hat_{t-1}, v_t)                     # AMSGrad only
    alpha_t = alpha            (lr_schedule="const")
            = alpha / sqrt(t)  (lr_schedule="sqrt")
    x_{t+1} = Proj_F( x_t - alpha_t * m_t / (sqrt(v_hat_t or v_t) + eps) )

Defaults follow the paper's analysis, not PyTorch:
    * bias_correction=False  (the paper drops Adam's debiasing step, footnote 1)
    * eps=0.0                (the paper's analysis has no epsilon)
    * clamp=None             (no projection; use (-1, 1) for the synthetic problems)

With bias_correction=True and eps=1e-8 the update is written exactly the way
torch.optim.Adam computes it, so PaperAdam / PaperAMSGrad match
torch.optim.Adam(amsgrad=False/True) to floating-point precision
(checked in tests/test_optimizers.py).

Projection: F is a box [lo, hi]^d, which is separable per coordinate, so the
weighted projection Proj_{F, sqrt(V_t)} of the paper equals a plain clamp.

Division safety: with eps=0 a coordinate whose gradient has always been exactly
zero (e.g. a dead ReLU unit) has m = v = 0. Its update is defined as 0 instead
of 0/0 = NaN.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple, Union

import torch
from torch.optim import Optimizer

Number = Union[float, torch.Tensor]

__all__ = ["PaperAdam", "PaperAMSGrad", "paper_adam_step", "init_state"]


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """num / den, with 0 wherever den == 0."""
    return torch.where(den > 0, num / torch.where(den > 0, den, torch.ones_like(den)), torch.zeros_like(num))


# ---------------------------------------------------------------------------
# Functional version (for the vectorised synthetic experiments)
# ---------------------------------------------------------------------------
def init_state(x: torch.Tensor) -> dict:
    """Fresh optimizer state for a tensor x of any shape."""
    return {
        "step": 0,
        "m": torch.zeros_like(x),
        "v": torch.zeros_like(x),
        "v_hat": torch.zeros_like(x),
    }


def paper_adam_step(
    x: torch.Tensor,
    g: torch.Tensor,
    state: dict,
    *,
    lr: Number,
    beta1: Number = 0.9,
    beta2: Number = 0.999,
    eps: float = 0.0,
    amsgrad: bool = False,
    bias_correction: bool = False,
    lr_schedule: str = "const",
    clamp: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """One Adam / AMSGrad step. Returns the new x and updates `state` in place.

    lr, beta1 and beta2 may be Python floats or tensors that broadcast against x.
    This lets one call run many configurations at once, e.g. x of shape
    (n_alpha, n_beta2) with lr of shape (n_alpha, 1) and beta2 of shape (1, n_beta2).
    """
    if lr_schedule not in ("const", "sqrt"):
        raise ValueError(f"lr_schedule must be 'const' or 'sqrt', got {lr_schedule!r}")

    state["step"] += 1
    t = state["step"]
    m, v, v_hat = state["m"], state["v"], state["v_hat"]

    m.mul_(beta1).add_((1 - beta1) * g)
    v.mul_(beta2).add_((1 - beta2) * g * g)
    if amsgrad:
        torch.maximum(v_hat, v, out=v_hat)
        second = v_hat
    else:
        second = v

    lr_t = lr / math.sqrt(t) if lr_schedule == "sqrt" else lr

    if bias_correction:
        bc1 = 1 - beta1 ** t
        bc2 = 1 - beta2 ** t
        if not torch.is_tensor(bc2):
            bc2 = torch.tensor(bc2, dtype=x.dtype, device=x.device)
        denom = second.sqrt() / bc2.sqrt() + eps
        update = (lr_t / bc1) * _safe_div(m, denom)
    else:
        update = lr_t * _safe_div(m, second.sqrt() + eps)

    x_new = x - update
    if clamp is not None:
        x_new = x_new.clamp(clamp[0], clamp[1])
    return x_new


# ---------------------------------------------------------------------------
# torch.optim.Optimizer versions (for MNIST / CIFAR models)
# ---------------------------------------------------------------------------
class _PaperAdamBase(Optimizer):
    _amsgrad = False

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 0.0,
        bias_correction: bool = False,
        lr_schedule: str = "const",
        clamp: Optional[Tuple[float, float]] = None,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if lr_schedule not in ("const", "sqrt"):
            raise ValueError(f"lr_schedule must be 'const' or 'sqrt', got {lr_schedule!r}")
        defaults = dict(lr=lr, betas=betas, eps=eps, bias_correction=bias_correction,
                        lr_schedule=lr_schedule, clamp=clamp)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported")
                state = self.state[p]
                if len(state) == 0:
                    state.update(init_state(p))
                new_p = paper_adam_step(
                    p, p.grad, state,
                    lr=group["lr"], beta1=beta1, beta2=beta2, eps=group["eps"],
                    amsgrad=self._amsgrad, bias_correction=group["bias_correction"],
                    lr_schedule=group["lr_schedule"], clamp=group["clamp"],
                )
                p.copy_(new_p)
        return loss


class PaperAdam(_PaperAdamBase):
    """Adam as analysed in the paper (Algorithm 1 with Adam's averaging functions)."""
    _amsgrad = False


class PaperAMSGrad(_PaperAdamBase):
    """AMSGrad (Algorithm 2 of the paper), with constant beta1 as used in practice."""
    _amsgrad = True
