"""
Sentetik deneylerin (E1, E2, E3) ortak yardımcıları — Anıl.

* Optimizer adımı her zaman optim.paper_adam_step (Yiğit) ile atılır.
* Bütün konfigürasyonlar tek tensörde koşar: optimizer başına bir paper_adam_step çağrısı,
  grup içinde lr / beta1 / beta2 vektör olarak yayınlanır (broadcast). Aynı grupta hem sabit α
  hem α/√t varsa α/√t burada hesaplanıp 'const' olarak verilir (matematiksel olarak aynı).
* CSV ortak formattadır (plan Bölüm 2.2) + sentetiğe özel `x` ve `schedule` sütunları.
  Sentetikte train_loss = ortalama pişmanlık R_t / t; test_loss ve test_acc = NaN.
* Çizim yardımcıları GEÇİCİ: common/plots.py hazır olunca oradaki fonksiyonlara geçilecek.
  Stil aynı: Adam kırmızı kesikli, AMSGrad mavi düz, seed'ler için ortalama ± std bandı.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch

from optim import init_state, paper_adam_step

DTYPE = torch.float64           # milyonlarca adımda toplamlar için float64


# ---------------------------------------------------------------------------
# Gradyan üreticileri
# ---------------------------------------------------------------------------
def online_grad(t, big=1010.0, small=10.0, period=101):
    """E1 (makale Şekil 1): t mod period == 1 ise +big, değilse -small."""
    return big if t % period == 1 else -small


class PeriodicGrad:
    """E3: t mod period == 1 ise +C (koşu başına farklı olabilir), değilse -small."""
    def __init__(self, C, small=1.0, period=3):
        self.C = torch.tensor(np.array(C, dtype=float), dtype=DTYPE)
        self.neg = torch.full_like(self.C, -small)
        self.period = period

    def __call__(self, t):
        return self.C if t % self.period == 1 else self.neg


class StochasticGrad:
    """E2: her adımda p olasılıkla +big, değilse -small.
    Aynı seed'i paylaşan koşular (Adam ve AMSGrad) AYNI gradyan dizisini görür."""
    def __init__(self, seed_idx, n_seeds, p=0.01, big=1010.0, small=10.0,
                 chunk=100_000, base_seed=1234):
        self.seed_idx = torch.tensor(np.array(seed_idx), dtype=torch.long)
        self.gens = [torch.Generator().manual_seed(base_seed + s) for s in range(n_seeds)]
        self.p, self.chunk = p, chunk
        self.big = torch.tensor(big, dtype=DTYPE)
        self.small = torch.tensor(-small, dtype=DTYPE)
        self.buf, self.pos = None, chunk

    def __call__(self, t):
        if self.pos >= self.chunk:   # hız için 100k adımlık rastgele sayıyı önceden üret
            self.buf = torch.stack([torch.rand(self.chunk, generator=g) for g in self.gens], 1) < self.p
            self.pos = 0
        hit = self.buf[self.pos][self.seed_idx]
        self.pos += 1
        return torch.where(hit, self.big, self.small)


# ---------------------------------------------------------------------------
# Vektörize simülatör
# ---------------------------------------------------------------------------
def simulate(cfg: pd.DataFrame, T: int, grad_fn, x_star: float = -1.0, log_every: int = 1000,
             x0: float = 1.0, bias_correction: bool = False, eps: float = 0.0,
             clamp=(-1.0, 1.0), track_cycle=None, tail=None, verbose=True) -> dict:
    """
    cfg      : her satır bir koşu; sütunlar optimizer ('adam'/'amsgrad'), lr, beta1, beta2,
               schedule ('sqrt' → α/√t, 'const' → α)
    grad_fn  : t → gradyan (sayı ya da (B,) tensör, B = len(cfg))
    x_star   : optimum (doğrusal kayıpta pişmanlık = Σ g_t (x_t − x*))
    track_cycle : (period, hit_mod) → vuruş adımlarındaki ve tüm adımlardaki en küçük x (Teorem 1)
    tail     : son `tail` adımın x ortalaması (ısı haritası)
    """
    cfg = cfg.reset_index(drop=True)
    B = len(cfg)
    groups = []                                   # optimizer başına bir paper_adam_step çağrısı
    for opt, part in cfg.groupby("optimizer", sort=False):
        n = len(part)
        x = torch.full((n,), float(x0), dtype=DTYPE)
        scheds = part["schedule"].unique()
        # Takvimler grupta karışıksa α/√t'yi burada hesaplayıp 'const' olarak veriyoruz
        # (paper_adam_step'in 'sqrt' seçeneğiyle matematiksel olarak aynı: lr_t = lr/√t).
        mixed = len(scheds) > 1
        groups.append(dict(
            idx=torch.tensor(part.index.values, dtype=torch.long), amsgrad=(opt == "amsgrad"),
            schedule="const" if mixed else scheds[0], mixed=mixed,
            sqrt_mask=torch.tensor((part["schedule"] == "sqrt").values),
            lr=torch.tensor(part["lr"].values, dtype=DTYPE),
            beta1=torch.tensor(part["beta1"].values, dtype=DTYPE),
            beta2=torch.tensor(part["beta2"].values, dtype=DTYPE),
            x=x, state=init_state(x), reg=torch.zeros(n, dtype=DTYPE),
            min_hit=torch.full((n,), float("inf"), dtype=DTYPE),
            min_all=torch.full((n,), float("inf"), dtype=DTYPE),
            tail=torch.zeros(n, dtype=DTYPE), xs=[], rs=[]))

    steps = []
    t0 = time.time()
    report = max(1, T // 10)
    for t in range(1, T + 1):
        g = grad_fn(t)
        vec = torch.is_tensor(g)
        hit = track_cycle is not None and t % track_cycle[0] == track_cycle[1]
        for gr in groups:
            gg = g[gr["idx"]] if vec else g             # skaler gradyan: kopyalama yok
            x = gr["x"]
            gr["reg"] += gg * (x - x_star)               # f_t(x_t) - f_t(x*)  (doğrusal kayıp)
            if track_cycle is not None:
                if hit:
                    torch.minimum(gr["min_hit"], x, out=gr["min_hit"])
                torch.minimum(gr["min_all"], x, out=gr["min_all"])
            lr = torch.where(gr["sqrt_mask"], gr["lr"] / t ** 0.5, gr["lr"]) if gr["mixed"] else gr["lr"]
            x = paper_adam_step(x, gg, gr["state"], lr=lr, beta1=gr["beta1"], beta2=gr["beta2"],
                                eps=eps, amsgrad=gr["amsgrad"], bias_correction=bias_correction,
                                lr_schedule=gr["schedule"], clamp=clamp)
            gr["x"] = x
            if tail is not None and t > T - tail:
                gr["tail"] += x
            if t % log_every == 0:
                gr["xs"].append(x.clone()); gr["rs"].append(gr["reg"] / t)
        if t % log_every == 0:
            steps.append(t)
        if verbose and t % report == 0:
            print(f"  t={t:>11,}  ({time.time() - t0:6.1f} s)")

    def gather(key, per_step=False):
        if per_step:
            if not steps:
                return None
            M = np.zeros((len(steps), B))
            for gr in groups:
                M[:, gr["idx"].numpy()] = torch.stack(gr[key]).numpy()
            return M
        v = np.zeros(B)
        for gr in groups:
            v[gr["idx"].numpy()] = gr[key].numpy()
        return v

    for gr in groups:
        gr["x_final"] = gr["x"]; gr["reg_final"] = gr["reg"] / T; gr["tail"] = gr["tail"] / (tail or 1)
    return {"steps": np.array(steps), "x": gather("xs", True), "avg_regret": gather("rs", True),
            "min_x_hit": gather("min_hit"), "min_x_all": gather("min_all"),
            "x_tail": gather("tail") if tail else None,
            "x_final": gather("x_final"), "avg_regret_final": gather("reg_final"),
            "seconds": time.time() - t0}


# ---------------------------------------------------------------------------
# CSV (ortak format)
# ---------------------------------------------------------------------------
CSV_COLS = ["exp", "optimizer", "seed", "lr", "beta1", "beta2", "step",
            "train_loss", "test_loss", "test_acc"]


def write_csv(exp: str, cfg: pd.DataFrame, out: dict, folder: str, keep=None) -> pd.DataFrame:
    """folder/<optimizer>_seed<k>.csv yazar. keep: CSV'ye girecek log satırları (boolean maske)."""
    os.makedirs(folder, exist_ok=True)
    steps, X, R = out["steps"], out["x"], out["avg_regret"]
    if keep is not None:
        steps, X, R = steps[keep], X[keep], R[keep]
    frames = []
    for i, r in cfg.reset_index(drop=True).iterrows():
        frames.append(pd.DataFrame({
            "exp": exp, "optimizer": r["optimizer"], "seed": int(r.get("seed", 0)),
            "lr": r["lr"], "beta1": r["beta1"], "beta2": r["beta2"], "step": steps,
            "train_loss": R[:, i], "test_loss": np.nan, "test_acc": np.nan,
            "x": X[:, i], "schedule": r["schedule"]}))
    df = pd.concat(frames, ignore_index=True)
    for (opt, seed), part in df.groupby(["optimizer", "seed"]):
        part.to_csv(os.path.join(folder, f"{opt}_seed{seed}.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# Çizim (GEÇİCİ — common/plots.py gelince oraya geçilecek)
# ---------------------------------------------------------------------------
STYLE = {"adam": dict(color="#d62728", ls="--", label="Adam"),
         "amsgrad": dict(color="#1f77b4", ls="-", label="AMSGrad")}


def style(opt, **over):
    s = dict(STYLE[opt]); s.update(over)
    return s


def band(ax, steps, Y, opt, label=None):
    """Y: (adım, seed) → ortalama çizgisi + ± std bandı."""
    mu, sd = Y.mean(1), Y.std(1)
    s = style(opt, label=label or STYLE[opt]["label"])
    ax.plot(steps, mu, **s)
    ax.fill_between(steps, mu - sd, mu + sd, color=s["color"], alpha=0.2, lw=0)


def save_fig(fig, name, folder):
    os.makedirs(folder, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(folder, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    print("kaydedildi:", os.path.join(folder, name + ".png / .pdf"))
