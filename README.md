# DeepLearning_Proje
CMP4509 Term Projesi

## Sentetik deneyler (E1–E3) — Anıl

| Notebook | Ne yapar | Ürettiği dosyalar | Süre (CPU) |
|---|---|---|---|
| `notebooks/01_synthetic_online.ipynb` | **E1** — Şekil 1 online problemi (1010x / −10x). α ∈ {0.05, 0.1, 0.25, 0.5, 1} × {sabit, α/√t} grid (1M adım) → seçilenler 7M adım. Ablation: bias correction + ε ve `torch.optim.Adam(amsgrad=True)` | `results/e1_synthetic_online/`, `figures/e1.*`, `figures/e1_const.*`, `figures/e1_ablation.*` | ~20–25 dk |
| `notebooks/02_synthetic_stochastic.ipynb` | **E2** — stokastik sürüm (p = 0.01), 5 seed, 10M adım | `results/e2_synthetic_stochastic/`, `figures/e2.*` | ~25 dk |
| `notebooks/03_theorem1_heatmap.ipynb` | **E3** — Teorem 1'in birebir örneği (C ∈ {3, 5, 10}) + β₂ × C ısı haritası (β₁ = 0 ve 0.9) | `results/e3_theorem1/`, `figures/e3_theorem1.*`, `figures/e3_heatmap.*` | ~10 dk |

**Kaggle'da çalıştırma:** Accelerator = None (CPU yeter), Settings → Internet = On (ilk hücre repo'yu klonlar) → *Save Version → Save & Run All*.
Kısa test için: ilk hücredeki `QUICK = os.environ.get(...)` satırını `QUICK = True` yap (son sürümden önce geri al).

**Notlar**
- Optimizer adımı her zaman `optim.paper_adam_step` ile atılır. Bütün konfigürasyonlar tek tensörde koşar (`common/synthetic.py → simulate`). `tests/test_synthetic.py`, vektörize koşunun tek tek koşularla birebir aynı olduğunu ve Teorem 1 invaryantını kontrol eder.
- CSV ortak formattadır; sentetikte `train_loss` = ortalama pişmanlık R_t/t, ek sütunlar `x` ve `schedule`.
- `common/synthetic.py` içindeki çizim yardımcıları geçicidir; `common/plots.py` gelince oraya geçilecek.
