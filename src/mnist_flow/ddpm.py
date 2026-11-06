"""DDPM: linear-beta schedule, eps-prediction loss, ancestral and DDIM samplers.

The shared UNet takes continuous time in [0, 1]; DDPM maps integer timesteps
via ``t_to_model_time`` (t_idx / T).

Sampler contract: ``fn(model, noise, *, steps, keep_every=None, **kw)
-> (x0 clamped to [-1, 1], nfe, frames)`` with frames on CPU.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from mnist_flow.config import Config


class DDPMSchedule(NamedTuple):
    betas: Tensor
    alphas: Tensor
    alpha_bar: Tensor
    T: int

    def to(self, device: torch.device | str) -> "DDPMSchedule":
        return DDPMSchedule(
            self.betas.to(device), self.alphas.to(device),
            self.alpha_bar.to(device), self.T,
        )


def make_schedule(cfg: Config) -> DDPMSchedule:
    betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.ddpm_T, dtype=torch.float64)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return DDPMSchedule(
        betas.float(), alphas.float(), alpha_bar.float(), cfg.ddpm_T
    )


def _bc(a: Tensor, idx: Tensor) -> Tensor:
    """Gather per-sample coefficients and broadcast to (B,1,1,1)."""
    return a.to(idx.device)[idx].view(-1, 1, 1, 1)


def q_sample(x0: Tensor, t_idx: Tensor, noise: Tensor, sched: DDPMSchedule) -> Tensor:
    ab = _bc(sched.alpha_bar, t_idx)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise


def t_to_model_time(t_idx: Tensor, T: int) -> Tensor:
    """Map integer timesteps {0..T-1} to the UNet's [0, 1) input: t_idx / T."""
    return t_idx.float() / T


def predict_x0(x_t: Tensor, eps: Tensor, t_idx: Tensor, sched: DDPMSchedule) -> Tensor:
    ab = _bc(sched.alpha_bar, t_idx)
    return ((x_t - (1.0 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1.0, 1.0)


def ddpm_loss(
    model: nn.Module, x0: Tensor, sched: DDPMSchedule,
    generator: torch.Generator | None = None,
) -> Tensor:
    b = x0.shape[0]
    t_idx = torch.randint(0, sched.T, (b,), generator=generator).to(x0.device)
    noise = torch.randn(x0.shape, generator=generator).to(x0.device)
    x_t = q_sample(x0, t_idx, noise, sched)
    eps_pred = model(x_t, t_to_model_time(t_idx, sched.T))
    return torch.mean((eps_pred - noise) ** 2)


def ddim_timesteps(T: int, steps: int) -> Tensor:
    """Evenly spaced, strictly decreasing timesteps starting at T-1."""
    if not 1 <= steps <= T:
        raise ValueError(f"steps must be in [1, {T}], got {steps}")
    ts = torch.linspace(T - 1, 0, steps, dtype=torch.float64).round().long()
    return ts


def _randn(shape, generator, device) -> Tensor:
    return torch.randn(shape, generator=generator).to(device)


def _eps(model: nn.Module, x: Tensor, t: int, T: int) -> Tensor:
    tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
    return model(x, t_to_model_time(tt, T))


def _run(model, noise, ts, step_fn, keep_every):
    """Shared loop: step_fn(x, eps, i) -> x_next; counts NFE and collects frames."""
    was_training = model.training
    model.eval()
    frames: list[Tensor] = []
    x = noise
    try:
        with torch.no_grad():
            for i in range(len(ts)):
                if keep_every and i % keep_every == 0:
                    frames.append(x.detach().cpu())
                x = step_fn(x, i)
    finally:
        model.train(was_training)
    x = x.clamp(-1.0, 1.0)
    if keep_every:
        frames.append(x.detach().cpu())
    return x, len(ts), frames


def sample_ancestral(
    model: nn.Module, noise: Tensor, *, steps: int | None = None,
    keep_every: int | None = None, sched: DDPMSchedule,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, int, list[Tensor]]:
    """Ho et al. ancestral sampling over all T steps (sigma^2 = beta)."""
    if steps is not None and steps != sched.T:
        raise ValueError(f"ancestral sampling uses all T={sched.T} steps, got {steps}")
    s = sched.to(noise.device)
    ts = list(range(s.T - 1, -1, -1))

    def step(x, i):
        t = ts[i]
        eps = _eps(model, x, t, s.T)
        mean = (x - s.betas[t] / (1.0 - s.alpha_bar[t]).sqrt() * eps) / s.alphas[t].sqrt()
        if t == 0:
            return mean
        return mean + s.betas[t].sqrt() * _randn(x.shape, generator, x.device)

    return _run(model, noise, ts, step, keep_every)


def sample_ddim(
    model: nn.Module, noise: Tensor, *, steps: int,
    keep_every: int | None = None, sched: DDPMSchedule, eta: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, int, list[Tensor]]:
    """DDIM (Song et al.) on a strided timestep subsequence; eta=0 deterministic."""
    s = sched.to(noise.device)
    ts = ddim_timesteps(s.T, steps).tolist()

    def step(x, i):
        t = ts[i]
        ab = s.alpha_bar[t]
        ab_prev = s.alpha_bar[ts[i + 1]] if i + 1 < len(ts) else x.new_tensor(1.0)
        eps = _eps(model, x, t, s.T)
        x0 = ((x - (1.0 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1.0, 1.0)
        eps = (x - ab.sqrt() * x0) / (1.0 - ab).sqrt()  # consistent with clamped x0
        sigma = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).clamp(min=0).sqrt()
        dir_xt = (1.0 - ab_prev - sigma**2).clamp(min=0).sqrt() * eps
        x_next = ab_prev.sqrt() * x0 + dir_xt
        if eta > 0:
            x_next = x_next + sigma * _randn(x.shape, generator, x.device)
        return x_next

    return _run(model, noise, ts, step, keep_every)
