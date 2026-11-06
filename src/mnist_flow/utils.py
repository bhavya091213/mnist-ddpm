"""Small shared helpers."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn


def get_device(name: str = "auto") -> torch.device:
    """Resolve a device: 'auto' picks cuda > mps > cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
