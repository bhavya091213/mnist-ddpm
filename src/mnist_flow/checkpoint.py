"""Save/load training checkpoints (plain tensors + builtins; weights_only safe)."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from mnist_flow.config import Config
from mnist_flow.unet import UNet


def save_checkpoint(
    path: Path | str, *, model: nn.Module, ema, opt: torch.optim.Optimizer,
    cfg: Config, model_name: str, step: int, epoch: int,
) -> None:
    """Atomically write a checkpoint (tmp file then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "config": cfg.to_dict(),
            "model_name": model_name,
            "step": int(step),
            "epoch": int(epoch),
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(path: Path | str, device: torch.device | str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=True)


def build_model_from_checkpoint(
    ckpt: dict, device: torch.device | str, use_ema: bool = True
) -> tuple[UNet, Config]:
    """Rebuild a UNet in eval mode from a loaded checkpoint (EMA weights by default)."""
    cfg = Config.from_dict(ckpt["config"])
    model = UNet(cfg).to(device)
    model.load_state_dict(ckpt["ema" if use_ema else "model"])
    return model.eval(), cfg
