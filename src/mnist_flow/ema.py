"""Exponential moving average of model weights."""
from __future__ import annotations

import copy

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.model = copy.deepcopy(model).requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p.detach().to(e.dtype), 1.0 - self.decay)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.model.state_dict())
