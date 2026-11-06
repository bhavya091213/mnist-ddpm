import dataclasses
import math

import pytest
import torch

from mnist_flow.config import Config
from mnist_flow.ddpm import (
    ddpm_loss,
    make_schedule,
    predict_x0,
    q_sample,
    sample_ancestral,
    sample_ddim,
    t_to_model_time,
)
from mnist_flow.unet import UNet

SHAPE = (2, 1, 28, 28)


@pytest.fixture
def small_cfg():
    return dataclasses.replace(Config.tiny(), ddpm_T=20)


@pytest.fixture
def tiny_model(small_cfg):
    torch.manual_seed(0)
    return UNet(small_cfg)


def test_schedule_properties():
    cfg = Config()
    s = make_schedule(cfg)
    assert s.T == cfg.ddpm_T
    for a in (s.betas, s.alphas, s.alpha_bar):
        assert a.dtype == torch.float32 and a.shape == (cfg.ddpm_T,)
    assert (s.alpha_bar > 0).all() and (s.alpha_bar < 1).all()
    assert (s.alpha_bar[1:] < s.alpha_bar[:-1]).all()
    assert s.to(torch.device("cpu")).T == cfg.ddpm_T


def test_q_sample_moments():
    s = make_schedule(Config())
    g = torch.Generator().manual_seed(0)
    x0 = torch.zeros(20000, 1, 2, 2)
    noise = torch.randn(x0.shape, generator=g)
    t = torch.full((20000,), 300, dtype=torch.long)
    xt = q_sample(x0, t, noise, s)
    assert xt.var().item() == pytest.approx(1 - s.alpha_bar[300].item(), rel=0.05)
    x0 = torch.rand(4, 1, 3, 3)
    t = torch.tensor([0, 10, 500, 999])
    xt = q_sample(x0, t, torch.zeros_like(x0), s)
    expect = s.alpha_bar[t].sqrt().view(-1, 1, 1, 1) * x0
    assert torch.allclose(xt, expect)


def test_t_to_model_time_range():
    t = t_to_model_time(torch.arange(1000), 1000)
    assert t.dtype == torch.float32
    assert t.min() == 0 and t.max() < 1


def test_predict_x0_inverts_q_sample():
    s = make_schedule(Config())
    x0 = torch.rand(4, 1, 3, 3) * 1.8 - 0.9
    noise = torch.randn(4, 1, 3, 3)
    t = torch.tensor([0, 10, 100, 500])
    xt = q_sample(x0, t, noise, s)
    assert torch.allclose(predict_x0(xt, noise, t, s), x0, atol=1e-4)
    assert predict_x0(xt * 100, noise, t, s).abs().max() <= 1.0


def test_loss_finite_grad_and_deterministic(tiny_model, small_cfg):
    s = make_schedule(small_cfg)
    x0 = torch.rand(SHAPE) * 2 - 1
    l1 = ddpm_loss(tiny_model, x0, s, generator=torch.Generator().manual_seed(1))
    l2 = ddpm_loss(tiny_model, x0, s, generator=torch.Generator().manual_seed(1))
    assert l1.ndim == 0 and torch.isfinite(l1)
    assert l1.item() == l2.item()
    l1.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in tiny_model.parameters())
    assert torch.isfinite(ddpm_loss(tiny_model, x0, s))


@pytest.mark.parametrize("steps", [1, 2, 5, 10, 20])
def test_ddim_nfe_shape_range(tiny_model, small_cfg, steps):
    s = make_schedule(small_cfg)
    noise = torch.randn(SHAPE, generator=torch.Generator().manual_seed(0))
    x, nfe, frames = sample_ddim(tiny_model, noise, steps=steps, sched=s)
    assert nfe == steps and frames == []
    assert x.shape == SHAPE and x.abs().max() <= 1.0


def test_ddim_deterministic_and_frames(tiny_model, small_cfg):
    s = make_schedule(small_cfg)
    noise = torch.randn(SHAPE, generator=torch.Generator().manual_seed(0))
    a, _, _ = sample_ddim(tiny_model, noise, steps=10, sched=s)
    b, _, fr = sample_ddim(tiny_model, noise, steps=10, keep_every=3, sched=s)
    assert torch.equal(a, b)
    assert len(fr) == math.ceil(10 / 3) + 1
    assert all(f.device.type == "cpu" for f in fr)
    assert torch.equal(fr[-1], b)


def test_ddim_eta1_stochastic_seeded(tiny_model, small_cfg):
    s = make_schedule(small_cfg)
    noise = torch.randn(SHAPE, generator=torch.Generator().manual_seed(0))
    def run(seed):
        return sample_ddim(tiny_model, noise, steps=5, eta=1.0, sched=s,
                           generator=torch.Generator().manual_seed(seed))[0]
    assert torch.equal(run(1), run(1))
    assert not torch.equal(run(1), run(2))


def test_ddim_bad_steps(tiny_model, small_cfg):
    s = make_schedule(small_cfg)
    noise = torch.randn(SHAPE)
    for bad in (0, 21):
        with pytest.raises(ValueError):
            sample_ddim(tiny_model, noise, steps=bad, sched=s)


def test_ddim_1000_steps_schedule_only():
    # timestep selection must work for the full 1000 schedule at 1000 steps
    from mnist_flow.ddpm import ddim_timesteps
    for n in (1, 2, 5, 10, 20, 50, 100, 1000):
        ts = ddim_timesteps(1000, n)
        assert len(ts) == n and ts[0] == 999 and len(set(ts.tolist())) == n
        assert (ts[1:] < ts[:-1]).all() and ts[-1] >= 0


def test_ancestral_nfe_and_contract(tiny_model, small_cfg):
    s = make_schedule(small_cfg)
    noise = torch.randn(SHAPE, generator=torch.Generator().manual_seed(0))
    g = lambda: torch.Generator().manual_seed(3)
    x, nfe, fr = sample_ancestral(tiny_model, noise, keep_every=5, sched=s, generator=g())
    assert nfe == small_cfg.ddpm_T and x.abs().max() <= 1.0
    assert len(fr) == 20 // 5 + 1
    x2, _, _ = sample_ancestral(tiny_model, noise, steps=20, sched=s, generator=g())
    assert torch.equal(x, x2)
    with pytest.raises(ValueError):
        sample_ancestral(tiny_model, noise, steps=10, sched=s)


class OracleModel(torch.nn.Module):
    """Returns the true eps for a known (x0, noise) pair along the path."""

    def __init__(self, x0, sched):
        super().__init__()
        self.x0, self.s = x0, sched

    def forward(self, x, t):
        idx = torch.round(t * self.s.T).long()
        ab = self.s.alpha_bar[idx].view(-1, 1, 1, 1)
        return (x - ab.sqrt() * self.x0) / (1 - ab).sqrt()


@pytest.mark.parametrize("steps", [20, 7, 1])
def test_ddim_oracle_recovers_x0(small_cfg, steps):
    s = make_schedule(small_cfg)
    g = torch.Generator().manual_seed(0)
    x0 = torch.rand(SHAPE, generator=g) * 1.8 - 0.9
    noise = torch.randn(SHAPE, generator=g)
    start = q_sample(x0, torch.full((2,), s.T - 1), noise, s)
    out, nfe, _ = sample_ddim(OracleModel(x0, s), start, steps=steps, sched=s)
    assert nfe == steps
    assert (out - x0).abs().max() < 1e-3
