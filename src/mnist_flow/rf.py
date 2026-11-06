"""Rectified flow (Liu et al. 2022) for MNIST.

Convention: t=0 is DATA, t=1 is NOISE (as in minRF).
    x_t = (1 - t) * x0 + t * z,   z ~ N(0, I)
    target velocity v = dx_t/dt = z - x0
Sampling integrates dz/dt = v(z, t) from t=1 down to t=0, i.e. z <- z - dt * v.
The model always receives continuous t in [0, 1].

Sampler contract: fn(model, noise, *, steps, keep_every=None)
    -> (x0 clamped to [-1, 1], nfe, frames on CPU).
Heun NFE is exactly 2 * steps (the corrector is also applied on the last step,
evaluating the model at t=0), which keeps linear-in-t fields exact.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mnist_flow.config import Config

SampleResult = tuple[Tensor, int, list[Tensor]]


def _bcast(t: Tensor) -> Tensor:
    return t.view(-1, 1, 1, 1)


def interpolate(x0: Tensor, z: Tensor, t: Tensor) -> Tensor:
    tb = _bcast(t)
    return (1.0 - tb) * x0 + tb * z


def velocity_target(x0: Tensor, z: Tensor) -> Tensor:
    return z - x0


def sample_t(
    batch: int, mode: str = "uniform", generator: torch.Generator | None = None
) -> Tensor:
    """Sample training times in (0, 1), float32."""
    if mode == "uniform":
        t = torch.rand(batch, generator=generator)
        return t.clamp(min=1e-5, max=1.0 - 1e-5)
    if mode == "logit_normal":
        return torch.sigmoid(torch.randn(batch, generator=generator))
    raise ValueError(f"unknown rf_time_sampling mode: {mode!r}")


def rf_loss(
    model: nn.Module,
    x0: Tensor,
    cfg: Config,
    generator: torch.Generator | None = None,
) -> Tensor:
    b = x0.shape[0]
    dev = x0.device
    z = torch.randn(x0.shape, generator=generator).to(dev)
    t = sample_t(b, cfg.rf_time_sampling, generator).to(dev)
    pred = model(interpolate(x0, z, t), t)
    return F.mse_loss(pred, velocity_target(x0, z))


def _keep(frames: list[Tensor], z: Tensor, i: int, steps: int, keep_every: int | None) -> None:
    if keep_every and ((i + 1) % keep_every == 0 or i + 1 == steps):
        frames.append(z.detach().clamp(-1, 1).cpu())


def _t_batch(t: float, like: Tensor) -> Tensor:
    return torch.full((like.shape[0],), t, device=like.device, dtype=like.dtype)


@torch.no_grad()
def sample_euler(
    model: nn.Module, noise: Tensor, *, steps: int, keep_every: int | None = None
) -> SampleResult:
    model.eval()
    dt = 1.0 / steps
    z = noise
    frames: list[Tensor] = []
    nfe = 0
    for i in range(steps):
        v = model(z, _t_batch(1.0 - i * dt, z))
        nfe += 1
        z = z - dt * v
        _keep(frames, z, i, steps, keep_every)
    return z.clamp(-1, 1), nfe, frames


@torch.no_grad()
def sample_heun(
    model: nn.Module, noise: Tensor, *, steps: int, keep_every: int | None = None
) -> SampleResult:
    model.eval()
    dt = 1.0 / steps
    z = noise
    frames: list[Tensor] = []
    nfe = 0
    for i in range(steps):
        t_cur, t_next = 1.0 - i * dt, 1.0 - (i + 1) * dt
        v1 = model(z, _t_batch(t_cur, z))
        z_pred = z - dt * v1
        v2 = model(z_pred, _t_batch(max(t_next, 0.0), z))
        nfe += 2
        z = z - dt * 0.5 * (v1 + v2)
        _keep(frames, z, i, steps, keep_every)
    return z.clamp(-1, 1), nfe, frames
