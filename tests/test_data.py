"""Unit tests for common/data.py.  Run with:  pytest -q tests/"""

import gzip
import os
import pickle
import struct
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import (  # noqa: E402
    CIFAR10_MEAN,
    CIFAR10_STD,
    MNIST_MEAN,
    MNIST_STD,
    load_cifar10,
    load_mnist,
    make_generator,
    sample_batch,
)


def _write_idx_images(path: Path, arr: np.ndarray, compress: bool = False):
    n, rows, cols = arr.shape
    header = struct.pack(">IIII", 2051, n, rows, cols)
    opener = gzip.open if compress else open
    with opener(path, "wb") as f:
        f.write(header + arr.astype(np.uint8).tobytes())


def _write_idx_labels(path: Path, arr: np.ndarray, compress: bool = False):
    header = struct.pack(">II", 2049, len(arr))
    opener = gzip.open if compress else open
    with opener(path, "wb") as f:
        f.write(header + arr.astype(np.uint8).tobytes())


def test_load_mnist_from_kaggle_idx(tmp_path):
    """MNIST IDX (Kaggle hojjatk/mnist-dataset formatı) doğru şekil, tip ve normalizasyonla okunur."""
    rng = np.random.default_rng(0)
    x_tr = rng.integers(0, 256, size=(20, 28, 28), dtype=np.uint8)
    y_tr = rng.integers(0, 10, size=(20,), dtype=np.uint8)
    x_te = rng.integers(0, 256, size=(8, 28, 28), dtype=np.uint8)
    y_te = rng.integers(0, 10, size=(8,), dtype=np.uint8)

    # Kaggle'da sık görülen iç içe klasör yapısı (train-images-idx3-ubyte/train-images-idx3-ubyte)
    nested = tmp_path / "train-images-idx3-ubyte"
    nested.mkdir()
    _write_idx_images(nested / "train-images-idx3-ubyte", x_tr)
    _write_idx_labels(tmp_path / "train-labels.idx1-ubyte", y_tr)
    _write_idx_images(tmp_path / "t10k-images-idx3-ubyte.gz", x_te, compress=True)
    _write_idx_labels(tmp_path / "t10k-labels-idx1-ubyte.gz", y_te, compress=True)

    Xtr, ytr, Xte, yte = load_mnist(device="cpu", data_dir=tmp_path, flatten=True, normalize="standard", download=False)
    assert Xtr.shape == (20, 784) and Xte.shape == (8, 784)
    assert ytr.shape == (20,) and yte.shape == (8,)
    assert Xtr.dtype == torch.float32 and ytr.dtype == torch.long

    expected0 = (x_tr[0].reshape(-1).astype(np.float32) / 255.0 - MNIST_MEAN) / MNIST_STD
    assert np.allclose(Xtr[0].numpy(), expected0, atol=1e-5)

    # flatten=False kontrolü
    Xtr4, _, Xte4, _ = load_mnist(device="cpu", data_dir=tmp_path, flatten=False, normalize="unit", download=False)
    assert Xtr4.shape == (20, 1, 28, 28) and Xte4.shape == (8, 1, 28, 28)
    assert 0.0 <= Xtr4.min().item() and Xtr4.max().item() <= 1.0


def test_load_mnist_from_csv(tmp_path):
    """MNIST CSV (oddrationale/mnist-in-csv) formatı okunur."""
    header = ",".join(["label"] + [f"p{i}" for i in range(784)])
    rng = np.random.default_rng(1)
    tr = np.hstack([rng.integers(0, 10, (12, 1)), rng.integers(0, 256, (12, 784))])
    te = np.hstack([rng.integers(0, 10, (5, 1)), rng.integers(0, 256, (5, 784))])
    np.savetxt(tmp_path / "mnist_train.csv", tr, fmt="%d", delimiter=",", header=header, comments="")
    np.savetxt(tmp_path / "mnist_test.csv", te, fmt="%d", delimiter=",", header=header, comments="")

    Xtr, ytr, Xte, yte = load_mnist(device="cpu", data_dir=tmp_path, flatten=True, normalize="unit", download=False)
    assert Xtr.shape == (12, 784) and Xte.shape == (5, 784)
    assert torch.equal(ytr, torch.tensor(tr[:, 0], dtype=torch.long))


def test_load_cifar10_from_pickle_batches(tmp_path):
    """CIFAR-10 Python pickle batch formatı doğru şekil ve kanal normalizasyonuyla okunur."""
    bdir = tmp_path / "cifar-10-batches-py"
    bdir.mkdir()
    rng = np.random.default_rng(2)
    first_batch_data = None
    for i in range(1, 6):
        data = rng.integers(0, 256, size=(6, 3072), dtype=np.uint8)
        labels = rng.integers(0, 10, size=(6,)).tolist()
        if i == 1:
            first_batch_data = data
        with open(bdir / f"data_batch_{i}", "wb") as f:
            pickle.dump({"data": data, "labels": labels}, f)

    te_data = rng.integers(0, 256, size=(4, 3072), dtype=np.uint8)
    te_labels = rng.integers(0, 10, size=(4,)).tolist()
    with open(bdir / "test_batch", "wb") as f:
        pickle.dump({"data": te_data, "labels": te_labels}, f)

    Xtr, ytr, Xte, yte = load_cifar10(device="cpu", data_dir=tmp_path, normalize="standard", download=False)
    assert Xtr.shape == (30, 3, 32, 32) and Xte.shape == (4, 3, 32, 32)
    assert ytr.shape == (30,) and yte.shape == (4,)
    assert Xtr.dtype == torch.float32 and ytr.dtype == torch.long

    img0 = first_batch_data[0].reshape(3, 32, 32).astype(np.float32) / 255.0
    for c in range(3):
        expected_c = (img0[c] - CIFAR10_MEAN[c]) / CIFAR10_STD[c]
        assert np.allclose(Xtr[0, c].numpy(), expected_c, atol=1e-5)

    # per_image normalizasyonu kontrolü (TF CIFAR-10 tutorial uyumlu)
    Xtr_pi, _, _, _ = load_cifar10(device="cpu", data_dir=tmp_path, normalize="per_image", download=False)
    assert torch.allclose(Xtr_pi.mean(dim=(1, 2, 3)), torch.zeros(30), atol=1e-5)


def test_load_cifar10_from_tar_gz(tmp_path):
    """CIFAR-10 sıkıştırılmış cifar-10-python.tar.gz arşivinden doğrudan okunur."""
    src = tmp_path / "src" / "cifar-10-batches-py"
    src.mkdir(parents=True)
    rng = np.random.default_rng(3)
    for i in range(1, 6):
        with open(src / f"data_batch_{i}", "wb") as f:
            pickle.dump({"data": rng.integers(0, 256, (4, 3072), dtype=np.uint8), "labels": [0, 1, 2, 3]}, f)
    with open(src / "test_batch", "wb") as f:
        pickle.dump({"data": rng.integers(0, 256, (3, 3072), dtype=np.uint8), "labels": [4, 5, 6]}, f)

    tar_dir = tmp_path / "kaggle_like"
    tar_dir.mkdir()
    with tarfile.open(tar_dir / "cifar-10-python.tar.gz", "w:gz") as tf:
        tf.add(src, arcname="cifar-10-batches-py")

    Xtr, ytr, Xte, yte = load_cifar10(device="cpu", data_dir=tar_dir, normalize="unit", download=False)
    assert Xtr.shape == (20, 3, 32, 32) and Xte.shape == (3, 3, 32, 32)
    assert 0.0 <= Xtr.min().item() and Xtr.max().item() <= 1.0


def test_sample_batch_deterministic():
    """sample_batch aynı seed ile aynı minibatch'i döndürür (DataLoader olmadan)."""
    X = torch.arange(100 * 4, dtype=torch.float32).reshape(100, 4)
    y = torch.arange(100, dtype=torch.long)

    xb1, yb1 = sample_batch(X, y, 16, generator=make_generator(42, device="cpu"))
    xb2, yb2 = sample_batch(X, y, 16, generator=make_generator(42, device="cpu"))
    assert xb1.shape == (16, 4) and yb1.shape == (16,)
    assert torch.equal(xb1, xb2) and torch.equal(yb1, yb2)
    assert torch.equal(xb1[:, 0].long(), yb1 * 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU yok")
def test_sample_batch_cuda_deterministic():
    """CUDA üzerinde make_generator('cuda') ve sample_batch hatasız ve deterministik çalışır."""
    X = torch.arange(100 * 4, dtype=torch.float32, device="cuda").reshape(100, 4)
    y = torch.arange(100, dtype=torch.long, device="cuda")

    xb1, yb1 = sample_batch(X, y, 16, generator=make_generator(7, device="cuda"))
    xb2, yb2 = sample_batch(X, y, 16, generator=make_generator(7, device="cuda"))
    assert xb1.device.type == "cuda" and yb1.device.type == "cuda"
    assert torch.equal(xb1, xb2) and torch.equal(yb1, yb2)

