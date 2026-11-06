import pytest
import torch

from mnist_flow.checkpoint import build_model_from_checkpoint, load_checkpoint, save_checkpoint
from mnist_flow.config import Config
from mnist_flow.ema import EMA
from mnist_flow.unet import UNet


@pytest.fixture
def parts():
    cfg = Config.tiny()
    torch.manual_seed(0)
    model = UNet(cfg)
    ema = EMA(model, 0.9)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # make EMA differ from the live model
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    return cfg, model, ema, opt


def test_round_trip_weights_only(tmp_path, parts):
    cfg, model, ema, opt = parts
    path = tmp_path / "sub" / "last.pt"
    save_checkpoint(path, model=model, ema=ema, opt=opt, cfg=cfg,
                    model_name="ddpm", step=7, epoch=2)
    raw = torch.load(path, weights_only=True)
    assert set(raw) == {"model", "ema", "opt", "config", "model_name", "step", "epoch"}
    ckpt = load_checkpoint(path, "cpu")
    assert ckpt["step"] == 7 and ckpt["epoch"] == 2 and ckpt["model_name"] == "ddpm"
    assert Config.from_dict(ckpt["config"]) == cfg


def test_build_model_picks_ema_or_raw(tmp_path, parts):
    cfg, model, ema, opt = parts
    path = tmp_path / "c.pt"
    save_checkpoint(path, model=model, ema=ema, opt=opt, cfg=cfg,
                    model_name="rf", step=1, epoch=1)
    ckpt = load_checkpoint(path, "cpu")
    m_ema, cfg2 = build_model_from_checkpoint(ckpt, "cpu", use_ema=True)
    m_raw, _ = build_model_from_checkpoint(ckpt, "cpu", use_ema=False)
    assert cfg2 == cfg
    k = next(iter(ema.state_dict()))
    assert torch.equal(m_ema.state_dict()[k], ema.state_dict()[k])
    assert torch.equal(m_raw.state_dict()[k], model.state_dict()[k])
    assert not torch.equal(m_ema.state_dict()[k], m_raw.state_dict()[k])
    assert not m_ema.training


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt", "cpu")
