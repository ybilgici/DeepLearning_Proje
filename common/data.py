"""
MNIST ve CIFAR-10 veri yükleyicileri — Ahmet (H1.2).

* Verileri Kaggle Dataset'ten (/kaggle/input) veya lokalde torchvision önbelleğinden
  okuyup doğrudan normalize edilmiş GPU (veya belirtilen cihaz) tensörleri olarak döndürür.
* DataLoader kullanılmaz: eğitim döngüsünde rastgele indeksle hızlı minibatch seçimi
  için `sample_batch(X, y, batch_size, generator=...)` sağlanır.
* CIFAR-10 yükleyici Yiğit'in `06_cifarnet` notebook'u ile uyumludur.

Desteklenen Kaggle Dataset formatları:
- MNIST:
    1) IDX ubyte dosyaları (ör. `hojjatk/mnist-dataset` veya torchvision raw)
       `train-images-idx3-ubyte` / `train-images.idx3-ubyte` (veya `.gz`)
    2) CSV dosyaları (ör. `oddrationale/mnist-in-csv`: `mnist_train.csv`, `mnist_test.csv`)
    3) NPZ dosyası (`mnist.npz`: `x_train, y_train, x_test, y_test`)
    4) Bulunamazsa `torchvision.datasets.MNIST` fallback (download=True)
- CIFAR-10:
    1) Python pickle batch klasörü (`cifar-10-batches-py`: `data_batch_1..5` + `test_batch`)
    2) Sıkıştırılmış arşiv (`cifar-10-python.tar.gz`)
    3) NPZ dosyası (`cifar10.npz`: `x_train, y_train, x_test, y_test`)
    4) Bulunamazsa `torchvision.datasets.CIFAR10` fallback (download=True)
"""

from __future__ import annotations

import gzip
import os
import pickle
import struct
import tarfile
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "cmp4509_data")
KAGGLE_INPUT_DIR = "/kaggle/input"


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Varsayılan cihazı döndürür (CUDA varsa cuda:current, yoksa cpu)."""
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None and torch.cuda.is_available():
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def make_generator(seed: int, device: Optional[Union[str, torch.device]] = None) -> torch.Generator:
    """Belirtilen cihazla uyumlu deterministik torch.Generator üretir."""
    dev = get_device(device)
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    return gen


def sample_batch(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DataLoader olmadan rastgele indeksle minibatch çeker (iade ile örnekleme)."""
    n = X.shape[0]
    if generator is not None:
        idx = torch.randint(0, n, (batch_size,), device=generator.device, generator=generator)
        if idx.device != X.device:
            idx = idx.to(X.device)
    else:
        idx = torch.randint(0, n, (batch_size,), device=X.device)
    return X.index_select(0, idx), y.index_select(0, idx)


# ---------------------------------------------------------------------------
# İç yardımcılar: dosya arama & ikili/pickle okuyucular
# ---------------------------------------------------------------------------
def _search_roots(data_dir: Optional[Union[str, os.PathLike]]) -> Sequence[Path]:
    roots = []
    if data_dir is not None:
        p = Path(data_dir)
        if p.exists():
            roots.append(p)
    elif os.path.isdir(KAGGLE_INPUT_DIR):
        roots.append(Path(KAGGLE_INPUT_DIR))
    return roots


def _find_file(roots: Sequence[Path], candidate_names: Sequence[str]) -> Optional[Path]:
    """Verilen kök dizinlerde dosya adına göre ilk eşleşen gerçek dosyayı bulur."""
    wanted = {name.lower() for name in candidate_names}
    for root in roots:
        if root.is_file() and root.name.lower() in wanted:
            return root
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.lower() in wanted:
                    fpath = Path(dirpath) / fname
                    if fpath.is_file():
                        return fpath
    return None


def _read_idx_images(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Beklenmeyen IDX image magic={magic} ({path})")
        buf = f.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols).copy()


def _read_idx_labels(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Beklenmeyen IDX label magic={magic} ({path})")
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8).copy()


def _try_load_mnist_from_roots(roots: Sequence[Path]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if not roots:
        return None

    # 1) IDX ubyte formatı (Kaggle hojjatk/mnist-dataset veya torchvision raw)
    tr_img = _find_file(roots, [
        "train-images-idx3-ubyte", "train-images.idx3-ubyte",
        "train-images-idx3-ubyte.gz", "train-images.idx3-ubyte.gz",
    ])
    tr_lbl = _find_file(roots, [
        "train-labels-idx1-ubyte", "train-labels.idx1-ubyte",
        "train-labels-idx1-ubyte.gz", "train-labels.idx1-ubyte.gz",
    ])
    te_img = _find_file(roots, [
        "t10k-images-idx3-ubyte", "t10k-images.idx3-ubyte",
        "t10k-images-idx3-ubyte.gz", "t10k-images.idx3-ubyte.gz",
    ])
    te_lbl = _find_file(roots, [
        "t10k-labels-idx1-ubyte", "t10k-labels.idx1-ubyte",
        "t10k-labels-idx1-ubyte.gz", "t10k-labels.idx1-ubyte.gz",
    ])
    if tr_img and tr_lbl and te_img and te_lbl:
        return (
            _read_idx_images(tr_img),
            _read_idx_labels(tr_lbl),
            _read_idx_images(te_img),
            _read_idx_labels(te_lbl),
        )

    # 2) CSV formatı (oddrationale/mnist-in-csv)
    tr_csv = _find_file(roots, ["mnist_train.csv"])
    te_csv = _find_file(roots, ["mnist_test.csv"])
    if tr_csv and te_csv:
        def _load_mnist_csv(path: Path) -> np.ndarray:
            with open(path, "r", encoding="utf-8") as f:
                first_tok = f.readline().split(",", 1)[0].strip()
            skip = 0 if first_tok.isdigit() else 1
            return np.loadtxt(path, delimiter=",", skiprows=skip, dtype=np.uint8)

        tr_arr = _load_mnist_csv(tr_csv)
        te_arr = _load_mnist_csv(te_csv)
        return (
            tr_arr[:, 1:].reshape(-1, 28, 28),
            tr_arr[:, 0],
            te_arr[:, 1:].reshape(-1, 28, 28),
            te_arr[:, 0],
        )

    # 3) NPZ formatı (mnist.npz)
    npz_path = _find_file(roots, ["mnist.npz"])
    if npz_path:
        with np.load(npz_path) as d:
            return d["x_train"], d["y_train"], d["x_test"], d["y_test"]

    return None


def _parse_cifar_batch(buf: bytes) -> Tuple[np.ndarray, np.ndarray]:
    d = pickle.loads(buf, encoding="latin1")
    data = d["data"].reshape(-1, 3, 32, 32)
    labels = np.array(d.get("labels", d.get("fine_labels")), dtype=np.int64)
    return data, labels


def _try_load_cifar10_from_roots(roots: Sequence[Path]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if not roots:
        return None

    # 1) Açılmış cifar-10-batches-py klasörü
    batch1 = _find_file(roots, ["data_batch_1"])
    if batch1 is not None:
        bdir = batch1.parent
        train_files = [bdir / f"data_batch_{i}" for i in range(1, 6)]
        test_file = bdir / "test_batch"
        if all(p.is_file() for p in train_files) and test_file.is_file():
            xs, ys = [], []
            for p in train_files:
                x_b, y_b = _parse_cifar_batch(p.read_bytes())
                xs.append(x_b)
                ys.append(y_b)
            x_te, y_te = _parse_cifar_batch(test_file.read_bytes())
            return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), x_te, y_te

    # 2) Sıkıştırılmış cifar-10-python.tar.gz arşivi
    tar_path = _find_file(roots, ["cifar-10-python.tar.gz"])
    if tar_path is not None:
        with tarfile.open(tar_path, "r:gz") as tf:
            members = {Path(m.name).name: m for m in tf.getmembers() if m.isfile()}
            if all(f"data_batch_{i}" in members for i in range(1, 6)) and "test_batch" in members:
                xs, ys = [], []
                for i in range(1, 6):
                    x_b, y_b = _parse_cifar_batch(tf.extractfile(members[f"data_batch_{i}"]).read())
                    xs.append(x_b)
                    ys.append(y_b)
                x_te, y_te = _parse_cifar_batch(tf.extractfile(members["test_batch"]).read())
                return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), x_te, y_te

    # 3) NPZ formatı (cifar10.npz)
    npz_path = _find_file(roots, ["cifar10.npz", "cifar-10.npz"])
    if npz_path is not None:
        with np.load(npz_path) as d:
            x_tr, y_tr = d["x_train"], d["y_train"].reshape(-1)
            x_te, y_te = d["x_test"], d["y_test"].reshape(-1)
            if x_tr.ndim == 4 and x_tr.shape[-1] == 3:
                x_tr = np.transpose(x_tr, (0, 3, 1, 2))
                x_te = np.transpose(x_te, (0, 3, 1, 2))
            return x_tr, y_tr, x_te, y_te

    return None


def _normalize_images(
    x: torch.Tensor,
    normalize: Optional[str],
    mean: Union[float, Tuple[float, float, float]],
    std: Union[float, Tuple[float, float, float]],
    is_rgb: bool = False,
) -> torch.Tensor:
    if normalize in (None, "none"):
        return x
    if normalize == "unit":
        return x / 255.0
    if normalize == "centered":
        return (x / 127.5) - 1.0
    if normalize == "standard":
        x = x / 255.0
        if is_rgb:
            m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
            s = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
            return (x - m) / s
        return (x - float(mean)) / float(std)
    if normalize == "per_image":
        # TensorFlow CIFAR-10 tutorial: tf.image.per_image_standardization
        x = x / 255.0
        flat = x.reshape(x.shape[0], -1)
        img_mean = flat.mean(dim=1, keepdim=True)
        img_std = flat.std(dim=1, unbiased=False, keepdim=True)
        min_std = 1.0 / (flat.shape[1] ** 0.5)
        adj_std = torch.clamp(img_std, min=min_std)
        return ((flat - img_mean) / adj_std).reshape_as(x)
    raise ValueError(
        f"Desteklenmeyen normalize={normalize!r}. "
        "Beklenen: 'standard', 'unit', 'centered', 'per_image' veya None."
    )


# ---------------------------------------------------------------------------
# Genel API
# ---------------------------------------------------------------------------
def load_mnist(
    device: Optional[Union[str, torch.device]] = None,
    data_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    flatten: bool = True,
    normalize: Optional[str] = "standard",
    dtype: torch.dtype = torch.float32,
    download: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MNIST eğitim ve test verilerini normalize edilmiş cihaz tensörleri olarak döndürür.

    Returns
    -------
    X_train, y_train, X_test, y_test :
        flatten=True  -> X_train (60000, 784), X_test (10000, 784)
        flatten=False -> X_train (60000, 1, 28, 28), X_test (10000, 1, 28, 28)
        y_train (60000,), y_test (10000,) dtype=torch.long
    """
    dev = get_device(device)
    loaded = _try_load_mnist_from_roots(_search_roots(data_dir))

    if loaded is not None:
        x_tr_np, y_tr_np, x_te_np, y_te_np = loaded
        X_train = torch.as_tensor(np.asarray(x_tr_np), dtype=dtype, device=dev)
        y_train = torch.as_tensor(np.asarray(y_tr_np), dtype=torch.long, device=dev)
        X_test = torch.as_tensor(np.asarray(x_te_np), dtype=dtype, device=dev)
        y_test = torch.as_tensor(np.asarray(y_te_np), dtype=torch.long, device=dev)
    else:
        from torchvision.datasets import MNIST

        cache_root = str(data_dir) if data_dir is not None else DEFAULT_CACHE_DIR
        ds_tr = MNIST(root=cache_root, train=True, download=download)
        ds_te = MNIST(root=cache_root, train=False, download=download)
        X_train = ds_tr.data.to(device=dev, dtype=dtype)
        y_train = ds_tr.targets.to(device=dev, dtype=torch.long)
        X_test = ds_te.data.to(device=dev, dtype=dtype)
        y_test = ds_te.targets.to(device=dev, dtype=torch.long)

    X_train = _normalize_images(X_train, normalize, MNIST_MEAN, MNIST_STD, is_rgb=False)
    X_test = _normalize_images(X_test, normalize, MNIST_MEAN, MNIST_STD, is_rgb=False)

    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    else:
        if X_train.ndim == 3:
            X_train = X_train.unsqueeze(1)
            X_test = X_test.unsqueeze(1)

    return X_train.contiguous(), y_train.contiguous(), X_test.contiguous(), y_test.contiguous()


def load_cifar10(
    device: Optional[Union[str, torch.device]] = None,
    data_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    normalize: Optional[str] = "standard",
    dtype: torch.dtype = torch.float32,
    download: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CIFAR-10 eğitim ve test verilerini normalize edilmiş cihaz tensörleri olarak döndürür.

    Returns
    -------
    X_train, y_train, X_test, y_test :
        X_train (50000, 3, 32, 32), y_train (50000,)
        X_test  (10000, 3, 32, 32), y_test  (10000,)
    """
    dev = get_device(device)
    loaded = _try_load_cifar10_from_roots(_search_roots(data_dir))

    if loaded is not None:
        x_tr_np, y_tr_np, x_te_np, y_te_np = loaded
        X_train = torch.as_tensor(np.asarray(x_tr_np), dtype=dtype, device=dev)
        y_train = torch.as_tensor(np.asarray(y_tr_np), dtype=torch.long, device=dev)
        X_test = torch.as_tensor(np.asarray(x_te_np), dtype=dtype, device=dev)
        y_test = torch.as_tensor(np.asarray(y_te_np), dtype=torch.long, device=dev)
    else:
        from torchvision.datasets import CIFAR10

        cache_root = str(data_dir) if data_dir is not None else DEFAULT_CACHE_DIR
        ds_tr = CIFAR10(root=cache_root, train=True, download=download)
        ds_te = CIFAR10(root=cache_root, train=False, download=download)
        # torchvision CIFAR10.data is (N, 32, 32, 3) uint8 -> (N, 3, 32, 32)
        X_train = torch.from_numpy(ds_tr.data).permute(0, 3, 1, 2).to(device=dev, dtype=dtype)
        y_train = torch.tensor(ds_tr.targets, dtype=torch.long, device=dev)
        X_test = torch.from_numpy(ds_te.data).permute(0, 3, 1, 2).to(device=dev, dtype=dtype)
        y_test = torch.tensor(ds_te.targets, dtype=torch.long, device=dev)

    X_train = _normalize_images(X_train, normalize, CIFAR10_MEAN, CIFAR10_STD, is_rgb=True)
    X_test = _normalize_images(X_test, normalize, CIFAR10_MEAN, CIFAR10_STD, is_rgb=True)

    return X_train.contiguous(), y_train.contiguous(), X_test.contiguous(), y_test.contiguous()
