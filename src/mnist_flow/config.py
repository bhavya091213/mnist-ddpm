"""Frozen experiment configuration."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Config:
    image_size: int = 28
    channels: int = 1
    base_channels: int = 64
    channel_mults: tuple[int, ...] = (1, 2, 2)
    num_res_blocks: int = 2
    groupnorm_groups: int = 8
    time_embed_dim: int = 256
    lr: float = 2e-4
    batch_size: int = 128
    epochs: int = 20
    ema_decay: float = 0.999
    seed: int = 0
    ddpm_T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    rf_time_sampling: str = "uniform"

    def to_dict(self) -> dict[str, Any]:
        """Plain dict of builtins (tuples become lists) safe for weights_only load."""
        return {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in dataclasses.asdict(self).items()
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        names = {f.name for f in dataclasses.fields(cls)}
        kwargs = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in d.items()
            if k in names
        }
        return cls(**kwargs)

    @classmethod
    def tiny(cls) -> "Config":
        """Small config for fast tests."""
        return cls(base_channels=8, time_embed_dim=32, batch_size=8, epochs=1)
