"""MNIST loading and batching, entirely on-device (no DataLoader)."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor
from torchvision import datasets


def to_minus_one_one(x: Tensor) -> Tensor:
    """uint8 [0,255] -> float32 [-1,1]."""
    return x.to(torch.float32) / 127.5 - 1.0


def load_mnist(
    root: Path | str = "data/", train: bool = True, device: torch.device | str = "cpu"
) -> tuple[Tensor, Tensor]:
    """Return (images (N,1,28,28) float32 in [-1,1], labels (N,) int64) on device."""
    ds = datasets.MNIST(str(root), train=train, download=True)
    images = to_minus_one_one(ds.data).unsqueeze(1).to(device)
    labels = ds.targets.to(torch.int64).to(device)
    return images, labels


def iterate_batches(
    x: Tensor, batch_size: int, generator: torch.Generator
) -> Iterator[Tensor]:
    """Yield shuffled full batches (last partial batch dropped)."""
    perm = torch.randperm(x.shape[0], generator=generator).to(x.device)
    for i in range(x.shape[0] // batch_size):
        yield x[perm[i * batch_size : (i + 1) * batch_size]]
