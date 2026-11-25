"""Shared project utilities (data loaders by Ahmet)."""

from .data import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    MNIST_MEAN,
    MNIST_STD,
    get_device,
    load_cifar10,
    load_mnist,
    make_generator,
    sample_batch,
)

__all__ = [
    "MNIST_MEAN",
    "MNIST_STD",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "get_device",
    "make_generator",
    "sample_batch",
    "load_mnist",
    "load_cifar10",
]
