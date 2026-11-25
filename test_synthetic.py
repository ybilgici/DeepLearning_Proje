"""Tests for common/synthetic.py.  Run with:  pytest -q tests/"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.synthetic import PeriodicGrad, StochasticGrad, online_grad, simulate  # noqa: E402
from optim import init_state, paper_adam_step  # noqa: E402


def _cfg():
    return pd.DataFrame([dict(optimizer=o, lr=a, schedule=s, beta1=0.9, beta2=0.99, seed=0)
                         for o in ("adam", "amsgrad") for a in (0.1, 0.5) for s in ("sqrt", "const")])


def test_vectorised_simulate_matches_single_runs():
    """Grouped/vectorised simulate == running each config alone with paper_adam_step."""
    cfg = _cfg()
    out = simulate(cfg, 3000, online_grad, log_every=3000, verbose=False)
    for i, r in cfg.iterrows():
        x = torch.ones(1, dtype=torch.float64); st = init_state(x)
        for t in range(1, 3001):
            g = torch.tensor([online_grad(t)], dtype=torch.float64)
            x = paper_adam_step(x, g, st, lr=r.lr, beta1=0.9, beta2=0.99, amsgrad=r.optimizer == "amsgrad",
                                lr_schedule=r.schedule, clamp=(-1, 1))
        assert abs(out["x_final"][i] - x.item()) < 1e-12


def test_theorem1_invariant_short():
    """Theorem 1: Adam starts every cycle at x = 1 and never goes below 0."""
    Cs = [3.0, 5.0, 10.0]
    b2 = [1 / (1 + C * C) for C in Cs]
    cfg = pd.DataFrame(dict(optimizer="adam", lr=[0.5 * np.sqrt(1 - b) for b in b2], schedule="sqrt",
                            beta1=0.0, beta2=b2))
    out = simulate(cfg, 30_001, PeriodicGrad(Cs), log_every=30_001, track_cycle=(3, 1), verbose=False)
    assert np.all(out["min_x_hit"] >= 1 - 1e-12)
    assert np.all(out["min_x_all"] > 0)
    assert np.all(out["avg_regret_final"] >= (2 * np.array(Cs) - 4) / 3)


def test_stochastic_same_seed_same_gradients():
    """Adam and AMSGrad with the same seed see the same gradient sequence."""
    gen = StochasticGrad(seed_idx=[0, 1, 0, 1], n_seeds=2, chunk=1000)
    G = torch.stack([gen(t) for t in range(1, 2001)])
    assert torch.equal(G[:, 0], G[:, 2]) and torch.equal(G[:, 1], G[:, 3])
    assert not torch.equal(G[:, 0], G[:, 1])
