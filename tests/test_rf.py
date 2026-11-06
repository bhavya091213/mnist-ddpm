import pytest
import torch
from torch import nn

from mnist_flow.config import Config
from mnist_flow.rf import (
    interpolate,
    rf_loss,
    sample_euler,
    sample_heun,
    sample_t,
    velocity_target,
)
from mnist_flow.unet import UNet


class ConstVel(nn.Module):
    """Model returning a fixed velocity regardless of input."""

    def __init__(self, v: torch.Tensor) -> None:
        super().__init__()
        self.v = v
        self.calls = 0

    def forward(self, x, t):
        self.calls += 1
        return self.v.expand_as(x)


class LinearTVel(nn.Module):
    """v(z, t) = a * t (exactly integrable by Heun, not by Euler)."""

    def __init__(self, a: float) -> None:
        super().__init__()
        self.a = a

    def forward(self, x, t):
        return self.a * t.view(-1, 1, 1, 1).expand_as(x)


def test_interpolate_endpoints_and_midpoint():
    x0, z = torch.randn(3, 1, 4, 4), torch.randn(3, 1, 4, 4)
    assert torch.allclose(interpolate(x0, z, torch.zeros(3)), x0)
    assert torch.allclose(interpolate(x0, z, torch.ones(3)), z)
    assert torch.allclose(interpolate(x0, z, torch.full((3,), 0.5)), 0.5 * (x0 + z))


def test_velocity_target_sign():
    x0, z = torch.zeros(2, 1, 2, 2), torch.ones(2, 1, 2, 2)
    assert torch.all(velocity_target(x0, z) == 1.0)


@pytest.mark.parametrize("mode", ["uniform", "logit_normal"])
def test_sample_t_range_dtype_shape(mode):
    t = sample_t(1000, mode)
    assert t.shape == (1000,) and t.dtype == torch.float32
    assert t.min() > 0 and t.max() < 1


def test_sample_t_reproducible_and_mean():
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    assert torch.equal(sample_t(16, "uniform", g1), sample_t(16, "uniform", g2))
    g = torch.Generator().manual_seed(0)
    assert abs(sample_t(10000, "logit_normal", g).mean().item() - 0.5) < 0.02


def test_sample_t_bad_mode():
    with pytest.raises(ValueError):
        sample_t(4, "bogus")


@pytest.mark.parametrize("mode", ["uniform", "logit_normal"])
def test_rf_loss_finite_and_grad(mode):
    cfg = Config(**{**Config.tiny().to_dict(), "channel_mults": (1, 2, 2), "rf_time_sampling": mode})
    model = UNet(cfg)
    x0 = torch.rand(4, 1, 28, 28) * 2 - 1
    loss = rf_loss(model, x0, cfg, generator=torch.Generator().manual_seed(0))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize("steps", [1, 7])
def test_euler_straight_path_exact(steps):
    x0, z = torch.randn(1, 1, 28, 28), torch.randn(1, 1, 28, 28)
    x0 = x0.clamp(-1, 1)
    model = ConstVel(z - x0)
    out, nfe, frames = sample_euler(model, z, steps=steps)
    assert torch.allclose(out, x0, atol=1e-5)
    assert nfe == steps == model.calls and frames == []


def test_heun_exact_where_euler_is_not():
    z = torch.zeros(2, 1, 28, 28)
    model = LinearTVel(0.8)  # z0 = z1 - a/2 = -0.4
    heun, _, _ = sample_heun(model, z, steps=2)
    euler, _, _ = sample_euler(model, z, steps=2)
    assert torch.allclose(heun, torch.full_like(z, -0.4), atol=1e-6)
    assert not torch.allclose(euler, torch.full_like(z, -0.4), atol=1e-3)


def test_heun_nfe_counts_true_calls():
    model = ConstVel(torch.zeros(1, 1, 28, 28))
    _, nfe, _ = sample_heun(model, torch.zeros(1, 1, 28, 28), steps=5)
    assert nfe == model.calls == 10


@pytest.mark.parametrize("fn", [sample_euler, sample_heun])
def test_frames_clamp_determinism_and_eval(fn):
    torch.manual_seed(0)
    model = UNet(Config.tiny())
    model.train()
    noise = torch.randn(2, 1, 28, 28)
    a, _, frames = fn(model, noise, steps=5, keep_every=2)
    b, _, _ = fn(model, noise, steps=5, keep_every=None)
    assert not model.training
    assert torch.allclose(a, b)
    assert a.min() >= -1 and a.max() <= 1
    assert len(frames) == 3  # after steps 2, 4 and final (5)
    assert all(f.device.type == "cpu" for f in frames)
    assert torch.allclose(frames[-1], a)


def test_frames_no_duplicate_final():
    model = ConstVel(torch.zeros(1, 1, 28, 28))
    _, _, frames = sample_euler(model, torch.zeros(1, 1, 28, 28), steps=4, keep_every=2)
    assert len(frames) == 2
