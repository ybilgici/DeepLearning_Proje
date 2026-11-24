"""Unit tests for optim/optimizers.py.  Run with:  pytest -q tests/"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optim import PaperAdam, PaperAMSGrad, init_state, paper_adam_step  # noqa: E402


def _make_problem(seed=0):
    """Small MLP regression problem, deterministic."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(256, 10, generator=g, dtype=torch.float64)
    y = torch.randn(256, 1, generator=g, dtype=torch.float64)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(10, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)).double()
    return model, X, y


def _train(model, opt, X, y, steps=100):
    for i in range(steps):
        idx = slice((i * 32) % 256, (i * 32) % 256 + 32)
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(X[idx]), y[idx])
        loss.backward()
        opt.step()
    return torch.cat([p.detach().flatten() for p in model.parameters()])


@pytest.mark.parametrize("amsgrad", [False, True])
def test_matches_torch_adam_with_bias_correction(amsgrad):
    """(a) With bias_correction=True, eps=1e-8 we must equal torch.optim.Adam."""
    m1, X, y = _make_problem()
    m2, _, _ = _make_problem()
    m2.load_state_dict(m1.state_dict())
    cls = PaperAMSGrad if amsgrad else PaperAdam
    ours = cls(m1.parameters(), lr=1e-2, betas=(0.9, 0.99), eps=1e-8, bias_correction=True)
    ref = torch.optim.Adam(m2.parameters(), lr=1e-2, betas=(0.9, 0.99), eps=1e-8, amsgrad=amsgrad)
    p1 = _train(m1, ours, X, y)
    p2 = _train(m2, ref, X, y)
    assert torch.max(torch.abs(p1 - p2)).item() < 1e-6


def test_amsgrad_vhat_never_decreases():
    """(b) v_hat is monotone non-decreasing in AMSGrad."""
    torch.manual_seed(0)
    x = torch.zeros(50, dtype=torch.float64)
    state = init_state(x)
    prev = state["v_hat"].clone()
    for t in range(500):
        scale = 100.0 if t % 37 == 0 else 1.0          # rare large gradients
        g = scale * torch.randn(50, dtype=torch.float64)
        x = paper_adam_step(x, g, state, lr=1e-2, beta1=0.9, beta2=0.9, amsgrad=True)
        assert torch.all(state["v_hat"] >= prev)
        prev = state["v_hat"].clone()


def test_functional_matches_class():
    """(c) The functional step and the Optimizer class give identical results."""
    m1, X, y = _make_problem(seed=1)
    m2, _, _ = _make_problem(seed=1)
    m2.load_state_dict(m1.state_dict())
    opt = PaperAMSGrad(m1.parameters(), lr=1e-3, betas=(0.9, 0.999), lr_schedule="sqrt")
    p1 = _train(m1, opt, X, y, steps=50)

    params = list(m2.parameters())
    states = [init_state(p) for p in params]
    for i in range(50):
        idx = slice((i * 32) % 256, (i * 32) % 256 + 32)
        for p in params:
            p.grad = None
        torch.nn.functional.mse_loss(m2(X[idx]), y[idx]).backward()
        with torch.no_grad():
            for p, s in zip(params, states):
                p.copy_(paper_adam_step(p, p.grad, s, lr=1e-3, beta1=0.9, beta2=0.999,
                                        amsgrad=True, lr_schedule="sqrt"))
    p2 = torch.cat([p.detach().flatten() for p in params])
    assert torch.max(torch.abs(p1 - p2)).item() < 1e-12


def test_vectorised_configs_match_individual_runs():
    """Broadcasting lr / beta2 over a grid equals running each config separately."""
    lrs = torch.tensor([[0.1], [0.5]], dtype=torch.float64)          # (2, 1)
    b2s = torch.tensor([[0.99, 0.999]], dtype=torch.float64)         # (1, 2)
    x = torch.ones(2, 2, dtype=torch.float64)
    st = init_state(x)
    for t in range(1, 2001):
        g = torch.full_like(x, 1010.0 if t % 101 == 1 else -10.0)
        x = paper_adam_step(x, g, st, lr=lrs, beta1=0.9, beta2=b2s, amsgrad=True,
                            lr_schedule="sqrt", clamp=(-1, 1))
    for i in range(2):
        for j in range(2):
            xi = torch.ones(1, dtype=torch.float64)
            si = init_state(xi)
            for t in range(1, 2001):
                g = torch.full_like(xi, 1010.0 if t % 101 == 1 else -10.0)
                xi = paper_adam_step(xi, g, si, lr=lrs[i, 0].item(), beta1=0.9, beta2=b2s[0, j].item(),
                                     amsgrad=True, lr_schedule="sqrt", clamp=(-1, 1))
            assert abs(x[i, j].item() - xi.item()) < 1e-10


def test_zero_gradient_gives_no_nan_with_eps0():
    """A coordinate whose gradient is always 0 must not become NaN when eps=0."""
    x = torch.tensor([1.0, 1.0], dtype=torch.float64)
    st = init_state(x)
    for _ in range(10):
        x = paper_adam_step(x, torch.tensor([0.0, 1.0], dtype=torch.float64), st, lr=0.1)
    assert torch.isfinite(x).all()
    assert x[0].item() == 1.0


def test_synthetic_smoke():
    """Paper's synthetic problem: Adam goes to +1, AMSGrad moves toward -1."""
    x = torch.ones(2, dtype=torch.float64)              # [Adam, AMSGrad]
    s_adam, s_ams = init_state(x[:1]), init_state(x[1:])
    xa, xm = x[:1].clone(), x[1:].clone()
    for t in range(1, 30001):
        g = torch.tensor([1010.0 if t % 101 == 1 else -10.0], dtype=torch.float64)
        xa = paper_adam_step(xa, g, s_adam, lr=0.5, beta1=0.9, beta2=0.99, lr_schedule="sqrt", clamp=(-1, 1))
        xm = paper_adam_step(xm, g, s_ams, lr=0.5, beta1=0.9, beta2=0.99, lr_schedule="sqrt",
                             clamp=(-1, 1), amsgrad=True)
    assert xa.item() > 0.95, f"Adam should stick near +1, got {xa.item()}"
    assert xm.item() < 0.0, f"AMSGrad should move toward -1, got {xm.item()}"
