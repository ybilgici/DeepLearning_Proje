"""
MNIST deney yardımcıları (E4 Lojistik Regresyon & E5 MLP) — Ahmet.

* Optimizer olarak yalnızca `optim.PaperAdam` ve `optim.PaperAMSGrad` (Yiğit) kullanılır
  (baseline kontrolü için opsiyonel `torch_adam` desteklenir).
* Veri yükleme ve minibatch örnekleme `common.data` üzerinden DataLoader olmadan yapılır.
* CSV şeması Plan Bölüm 2.2 ile birebir uyumludur:
  `exp, optimizer, seed, lr, beta1, beta2, step, train_loss, test_loss, test_acc`
* Seed, CSV ve çizim yardımcıları GEÇİCİDİR: Yiğit'in `common/utils.py` ve `common/plots.py`
  modülleri `main`'e gelince aynı arayüzle oraya bağlanabilir (merge çakışması olmaması için
  bu dosyalar ayrı tutulmuştur).
"""

from __future__ import annotations

import os
import random
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.data import make_generator, sample_batch
from optim import PaperAdam, PaperAMSGrad

CSV_COLS = [
    "exp",
    "optimizer",
    "seed",
    "lr",
    "beta1",
    "beta2",
    "step",
    "train_loss",
    "test_loss",
    "test_acc",
]

# Ortak figür stili (Anıl & Yiğit ile tutarlı: Adam kırmızı kesikli, AMSGrad mavi düz)
STYLE = {
    "adam": dict(color="#d62728", ls="--", lw=1.8, label="Adam"),
    "amsgrad": dict(color="#1f77b4", ls="-", lw=1.8, label="AMSGrad"),
    "torch_adam": dict(color="#2ca02c", ls=":", lw=1.8, label="torch.optim.Adam"),
}


def set_seed(seed: int) -> None:
    """Deterministik başlangıç için Python, NumPy ve PyTorch seed'lerini ayarlar."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_mnist_logreg(device: torch.device) -> nn.Module:
    """E4: Konveks çok sınıflı lojistik regresyon (784 -> 10)."""
    return nn.Linear(784, 10).to(device)


def build_mnist_mlp(device: torch.device) -> nn.Module:
    """E5: Tek gizli katmanlı MLP (784 -> 100 ReLU -> 10)."""
    return nn.Sequential(
        nn.Linear(784, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    ).to(device)


def make_optimizer(
    optimizer_name: str,
    params,
    *,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    bias_correction: bool = True,
    lr_schedule: str = "const",
):
    """Belirtilen isim için ortak optimizer örneğini oluşturur."""
    name = optimizer_name.lower()
    if name == "adam":
        return PaperAdam(
            params,
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            bias_correction=bias_correction,
            lr_schedule=lr_schedule,
        )
    if name == "amsgrad":
        return PaperAMSGrad(
            params,
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            bias_correction=bias_correction,
            lr_schedule=lr_schedule,
        )
    if name == "torch_adam":
        if lr_schedule != "const":
            raise ValueError("torch_adam baseline yalnızca sabit lr ile doğrudan kullanılır.")
        return torch.optim.Adam(params, lr=lr, betas=(beta1, beta2), eps=eps)
    raise ValueError(f"Bilinmeyen optimizer: {optimizer_name!r}")


@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    chunk_size: int = 10_000,
) -> Tuple[float, float]:
    """Verilen veri setinde ortalama CrossEntropy loss ve doğruluk (accuracy) hesaplar."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    n = X.shape[0]
    for start in range(0, n, chunk_size):
        xb = X[start : start + chunk_size]
        yb = y[start : start + chunk_size]
        logits = model(xb)
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
    model.train()
    return total_loss / n, total_correct / n


def run_single_mnist(
    exp: str,
    model_fn,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    optimizer_name: str,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    bias_correction: bool = True,
    lr_schedule: str = "const",
    steps: int = 5000,
    batch_size: int = 128,
    seed: int = 0,
    eval_every: int = 50,
    selection_window: int = 500,
) -> Dict[str, object]:
    """Tek bir (optimizer, lr, beta2, seed) koşusunu gerçekleştirir.

    Aynı seed için hem model başlangıç ağırlıkları hem de minibatch dizisi birebir aynıdır;
    böylece Adam ve AMSGrad tamamen adil (paired) karşılaştırılır.
    """
    t0 = time.perf_counter()
    device = X_train.device

    set_seed(seed)
    model = model_fn(device)
    opt = make_optimizer(
        optimizer_name,
        model.parameters(),
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        bias_correction=bias_correction,
        lr_schedule=lr_schedule,
    )
    batch_gen = make_generator(seed + 10_000, device=device)

    mb_losses = np.empty(steps, dtype=np.float64)
    rows: List[dict] = []

    model.train()
    for step in range(1, steps + 1):
        xb, yb = sample_batch(X_train, y_train, batch_size, generator=batch_gen)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
        mb_losses[step - 1] = loss.item()

        if step == 1 or step % eval_every == 0 or step == steps:
            tr_loss, _ = evaluate_dataset(model, X_train, y_train)
            te_loss, te_acc = evaluate_dataset(model, X_test, y_test)
            rows.append(
                {
                    "exp": exp,
                    "optimizer": optimizer_name.lower(),
                    "seed": int(seed),
                    "lr": float(lr),
                    "beta1": float(beta1),
                    "beta2": float(beta2),
                    "step": int(step),
                    "train_loss": tr_loss,
                    "test_loss": te_loss,
                    "test_acc": te_acc,
                }
            )

    history_df = pd.DataFrame(rows, columns=CSV_COLS)
    last_w = min(selection_window, steps)
    last_window_mb_loss = float(np.mean(mb_losses[-last_w:]))
    final_row = rows[-1]

    return {
        "history": history_df,
        "mb_losses": mb_losses,
        "last500_train_loss": last_window_mb_loss,
        "final_train_loss": float(final_row["train_loss"]),
        "final_test_loss": float(final_row["test_loss"]),
        "final_test_acc": float(final_row["test_acc"]),
        "seconds": time.perf_counter() - t0,
    }


def run_grid_search(
    exp: str,
    model_fn,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    lrs: Sequence[float],
    beta2s: Sequence[float],
    optimizers: Sequence[str] = ("adam", "amsgrad"),
    beta1: float = 0.9,
    eps: float = 1e-8,
    bias_correction: bool = True,
    lr_schedule: str = "const",
    steps: int = 5000,
    batch_size: int = 128,
    seed: int = 0,
    eval_every: int = 250,
    selection_window: int = 500,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """Seed 0 üzerinde grid search yapar ve her optimizer için en iyi (lr, beta2) çiftini seçer.

    Seçim kriteri: son `selection_window` (500) iterasyonun ortalama minibatch train loss'u.
    """
    records = []
    for opt_name in optimizers:
        for b2 in beta2s:
            for lr in lrs:
                res = run_single_mnist(
                    exp,
                    model_fn,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    optimizer_name=opt_name,
                    lr=lr,
                    beta1=beta1,
                    beta2=b2,
                    eps=eps,
                    bias_correction=bias_correction,
                    lr_schedule=lr_schedule,
                    steps=steps,
                    batch_size=batch_size,
                    seed=seed,
                    eval_every=eval_every,
                    selection_window=selection_window,
                )
                rec = {
                    "exp": exp,
                    "optimizer": opt_name,
                    "seed": seed,
                    "lr": lr,
                    "beta1": beta1,
                    "beta2": b2,
                    "last500_train_loss": res["last500_train_loss"],
                    "final_train_loss": res["final_train_loss"],
                    "final_test_loss": res["final_test_loss"],
                    "final_test_acc": res["final_test_acc"],
                    "seconds": round(float(res["seconds"]), 2),
                }
                records.append(rec)
                if verbose:
                    print(
                        f"[{opt_name:7s}] lr={lr:<7g} beta2={b2:<5g} | "
                        f"last500_tr={rec['last500_train_loss']:.4f} "
                        f"full_tr={rec['final_train_loss']:.4f} "
                        f"te_loss={rec['final_test_loss']:.4f} "
                        f"te_acc={rec['final_test_acc']*100:5.2f}% ({rec['seconds']:.1f}s)"
                    )

    grid_df = pd.DataFrame(records)
    best_cfgs: Dict[str, dict] = {}
    for opt_name in optimizers:
        sub = grid_df[grid_df["optimizer"] == opt_name].sort_values("last500_train_loss", ascending=True)
        top = sub.iloc[0]
        best_cfgs[opt_name] = {
            "lr": float(top["lr"]),
            "beta1": float(top["beta1"]),
            "beta2": float(top["beta2"]),
            "last500_train_loss": float(top["last500_train_loss"]),
            "final_train_loss": float(top["final_train_loss"]),
            "final_test_loss": float(top["final_test_loss"]),
            "final_test_acc": float(top["final_test_acc"]),
        }
    return grid_df, best_cfgs


def run_final_seeds(
    exp: str,
    model_fn,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    best_cfgs: Dict[str, dict],
    results_dir: str,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    eps: float = 1e-8,
    bias_correction: bool = True,
    lr_schedule: str = "const",
    steps: int = 5000,
    batch_size: int = 128,
    eval_every: int = 50,
    verbose: bool = True,
) -> pd.DataFrame:
    """En iyi config'lerle 5 seed final koşularını çalıştırır ve Bölüm 2.2 CSV'lerini yazar."""
    os.makedirs(results_dir, exist_ok=True)
    frames: List[pd.DataFrame] = []

    for opt_name, cfg in best_cfgs.items():
        for seed in seeds:
            res = run_single_mnist(
                exp,
                model_fn,
                X_train,
                y_train,
                X_test,
                y_test,
                optimizer_name=opt_name,
                lr=cfg["lr"],
                beta1=cfg["beta1"],
                beta2=cfg["beta2"],
                eps=eps,
                bias_correction=bias_correction,
                lr_schedule=lr_schedule,
                steps=steps,
                batch_size=batch_size,
                seed=seed,
                eval_every=eval_every,
            )
            df = res["history"]
            out_path = os.path.join(results_dir, f"{opt_name}_seed{seed}.csv")
            df.to_csv(out_path, index=False)
            frames.append(df)
            if verbose:
                print(
                    f"[FINAL {opt_name:7s}] seed={seed} lr={cfg['lr']:<7g} b2={cfg['beta2']:<5g} | "
                    f"tr_loss={res['final_train_loss']:.4f} te_loss={res['final_test_loss']:.4f} "
                    f"te_acc={res['final_test_acc']*100:5.2f}% -> {os.path.basename(out_path)}"
                )

    return pd.concat(frames, ignore_index=True)


def summarize_final_runs(all_df: pd.DataFrame) -> pd.DataFrame:
    """Son adımda (step = max) her optimizer için ortalama ± std özet tablosu döndürür."""
    max_step = all_df["step"].max()
    last = all_df[all_df["step"] == max_step]
    rows = []
    for opt_name, grp in last.groupby("optimizer"):
        rows.append(
            {
                "optimizer": opt_name,
                "lr": grp["lr"].iloc[0],
                "beta1": grp["beta1"].iloc[0],
                "beta2": grp["beta2"].iloc[0],
                "n_seeds": grp["seed"].nunique(),
                "train_loss_mean": grp["train_loss"].mean(),
                "train_loss_std": grp["train_loss"].std(ddof=1) if len(grp) > 1 else 0.0,
                "test_loss_mean": grp["test_loss"].mean(),
                "test_loss_std": grp["test_loss"].std(ddof=1) if len(grp) > 1 else 0.0,
                "test_acc_mean": grp["test_acc"].mean(),
                "test_acc_std": grp["test_acc"].std(ddof=1) if len(grp) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def plot_train_test_curves(
    all_df: pd.DataFrame,
    *,
    title: str,
    save_name: Optional[str] = None,
    figures_dir: Optional[str] = None,
    best_cfgs: Optional[Dict[str, dict]] = None,
    min_step: int = 50,
    zoom_from: Optional[int] = None,
):
    """Train loss ve Test loss için ortalama ± std bandı grafiği çizer (PNG + PDF kaydeder).

    `min_step` (varsayılan 50), ilk adımdaki (step=1) aşırı yüksek başlangıç kaybının
    y-eksenini ezmesini önler. `zoom_from` verilirse son iterasyonlar için sağ üstte
    yakınlaştırma (inset) eklenir.
    """
    plot_df = all_df[all_df["step"] >= min_step] if min_step else all_df
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True)
    metrics = [
        ("train_loss", "Train Loss (Cross-Entropy)"),
        ("test_loss", "Test Loss (Cross-Entropy)"),
    ]

    for ax, (col, ylabel) in zip(axes, metrics):
        curves = []
        for opt_name in ("adam", "amsgrad"):
            sub = plot_df[plot_df["optimizer"] == opt_name]
            if sub.empty:
                continue
            piv = sub.pivot(index="step", columns="seed", values=col).sort_index()
            steps = piv.index.to_numpy()
            mu = piv.mean(axis=1).to_numpy()
            sd = piv.std(axis=1, ddof=1).fillna(0.0).to_numpy() if piv.shape[1] > 1 else np.zeros_like(mu)

            st = dict(STYLE[opt_name])
            if best_cfgs and opt_name in best_cfgs:
                c = best_cfgs[opt_name]
                st["label"] = f"{st['label']} ($\\alpha$={c['lr']:g}, $\\beta_2$={c['beta2']:g})"
            ax.plot(steps, mu, **st)
            ax.fill_between(steps, mu - sd, mu + sd, color=st["color"], alpha=0.18, lw=0)
            curves.append((steps, mu, sd, st))

        ax.set_xlabel("İterasyon")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", frameon=True, fontsize=8.5)

        if zoom_from is not None and curves:
            axins = ax.inset_axes([0.42, 0.36, 0.54, 0.38])
            for steps, mu, sd, st in curves:
                mask = steps >= zoom_from
                if mask.any():
                    axins.plot(steps[mask], mu[mask], color=st["color"], ls=st["ls"], lw=1.4)
                    axins.fill_between(
                        steps[mask],
                        mu[mask] - sd[mask],
                        mu[mask] + sd[mask],
                        color=st["color"],
                        alpha=0.20,
                        lw=0,
                    )
            axins.set_title(f"Son kısım (adım ≥ {zoom_from})", fontsize=7.5, pad=2)
            axins.tick_params(labelsize=7.5)
            axins.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=11.5, fontweight="bold")
    fig.tight_layout()

    if save_name and figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
        for ext in ("png", "pdf"):
            out = os.path.join(figures_dir, f"{save_name}.{ext}")
            fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Kaydedildi: {os.path.join(figures_dir, save_name)}.png / .pdf")
    return fig
