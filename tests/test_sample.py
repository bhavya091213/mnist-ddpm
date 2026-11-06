import sys

import pytest
import torch

from mnist_flow import sample as S
from mnist_flow.checkpoint import save_checkpoint
from mnist_flow.config import Config
from mnist_flow.ddpm import make_schedule
from mnist_flow.ema import EMA
from mnist_flow.unet import UNet


@pytest.fixture(scope="module")
def tiny():
    cfg = Config.tiny()
    torch.manual_seed(0)
    return UNet(cfg), cfg


def test_noise_bank_deterministic():
    a = S.make_noise_bank(5, 1234, "cpu")
    b = S.make_noise_bank(5, 1234, "cpu")
    c = S.make_noise_bank(5, 1, "cpu")
    assert a.shape == (5, 1, 28, 28)
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert torch.equal(S.make_noise_bank(3, 1234, "cpu"), a[:3])


def test_to_uint8_range():
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0, -3.0]).view(5, 1, 1, 1)
    out = S.to_uint8(x)
    assert out.dtype == torch.uint8
    assert out.flatten().tolist() == [0, 128, 255, 255, 0]


@pytest.mark.parametrize("name,sampler,steps,nfe", [
    ("ddpm", "ancestral", 1000, 1000),
    ("ddpm", "ddim", 3, 3),
    ("rf", "euler", 3, 3),
    ("rf", "heun", 3, 6),
])
def test_generate_dispatch(tiny, name, sampler, steps, nfe):
    model, cfg = tiny
    if sampler == "ancestral":
        cfg = Config.tiny()
        from dataclasses import replace
        cfg = replace(cfg, ddpm_T=5)
        steps, nfe = 5, 5
    noise = S.make_noise_bank(2, 0, "cpu")
    sched = make_schedule(cfg) if name == "ddpm" else None
    kw = dict(sampler=sampler, steps=steps, noise=noise, sched=sched)
    x1, n1, _ = S.generate(model, name, generator=torch.Generator().manual_seed(1), **kw)
    x2, n2, _ = S.generate(model, name, generator=torch.Generator().manual_seed(1), **kw)
    assert x1.shape == (2, 1, 28, 28) and n1 == n2 == nfe
    assert torch.equal(x1, x2)


def test_generate_errors(tiny):
    model, cfg = tiny
    noise = S.make_noise_bank(1, 0, "cpu")
    with pytest.raises(ValueError):
        S.generate(model, "ddpm", sampler="euler", steps=2, noise=noise, sched=make_schedule(cfg))
    with pytest.raises(ValueError):
        S.generate(model, "rf", sampler="bogus", steps=2, noise=noise)
    with pytest.raises(ValueError):
        S.generate(model, "ddpm", sampler="ddim", steps=2, noise=noise, sched=None)


def test_save_grid(tmp_path):
    p = tmp_path / "g.png"
    S.save_grid(torch.rand(4, 1, 28, 28) * 2 - 1, p, nrow=2)
    assert p.exists() and p.stat().st_size > 0


def test_cli_main_writes_outputs(tmp_path, tiny, monkeypatch, capsys):
    model, cfg = tiny
    ckpt = tmp_path / "rf" / "final.pt"
    save_checkpoint(ckpt, model=model, ema=EMA(model, 0.9),
                    opt=torch.optim.Adam(model.parameters()), cfg=cfg,
                    model_name="rf", step=1, epoch=1)
    out = tmp_path / "res"
    monkeypatch.setattr(sys, "argv", [
        "sample", "--ckpt", str(ckpt), "--sampler", "heun", "--steps", "2",
        "--n", "5", "--noise-seed", "3", "--out", str(out),
        "--batch-size", "2", "--device", "cpu"])
    S.main()
    assert "nfe=4" in capsys.readouterr().out
    t = torch.load(out / "rf_heun_nfe4.pt", weights_only=True)
    assert t.dtype == torch.uint8 and t.shape == (5, 1, 28, 28)
    assert (out / "rf_heun_nfe4.png").exists()
