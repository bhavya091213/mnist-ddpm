import sys

import pytest
import torch
from PIL import Image

from mnist_flow import epoch_viz as V
from mnist_flow.config import Config
from mnist_flow.sample import make_noise_bank
from mnist_flow.unet import UNet


def _png(path, shade):
    Image.new("RGB", (32, 32), (shade, shade, shade)).save(path)


@pytest.mark.parametrize("name", ["ddpm", "rf"])
def test_save_epoch_grid_writes_png(tmp_path, name):
    cfg = Config.tiny()
    model = UNet(cfg)
    noise = make_noise_bank(64, 1234, "cpu")
    out = tmp_path / "sub" / f"{name}_epoch01.png"
    V.save_epoch_grid(model, name, cfg, noise, out, "cpu")
    assert out.is_file()
    assert Image.open(out).size[0] > 28


def test_make_epoch_gif_frames_sorted_and_looping(tmp_path):
    for e in (2, 1, 3):
        _png(tmp_path / f"rf_epoch{e:02d}.png", 40 * e)
    _png(tmp_path / "ddpm_epoch01.png", 0)  # other model must be ignored
    out = tmp_path / "rf_epochs.gif"
    V.make_epoch_gif(tmp_path, "rf", out)
    gif = Image.open(out)
    assert gif.n_frames == 3
    assert gif.info.get("loop") == 0


def test_make_epoch_gif_no_pngs_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        V.make_epoch_gif(tmp_path, "rf", tmp_path / "x.gif")


def test_side_by_side_pairs_only_common_epochs(tmp_path):
    for e in (1, 2):
        _png(tmp_path / f"ddpm_epoch{e:02d}.png", 10)
    _png(tmp_path / "rf_epoch01.png", 200)
    out = tmp_path / "side.gif"
    V.make_side_by_side_gif(tmp_path, out)
    gif = Image.open(out)
    assert gif.n_frames == 1
    assert gif.size[0] >= 64


def test_side_by_side_none_in_common_returns_none(tmp_path):
    _png(tmp_path / "ddpm_epoch01.png", 10)
    assert V.make_side_by_side_gif(tmp_path, tmp_path / "s.gif") is None
    assert not (tmp_path / "s.gif").exists()


def test_cli(tmp_path, monkeypatch):
    d = tmp_path / "epoch_samples"
    d.mkdir()
    for m in ("ddpm", "rf"):
        for e in (1, 2):
            _png(d / f"{m}_epoch{e:02d}.png", 50)
    monkeypatch.setattr(sys, "argv", ["epoch_viz", "--runs", str(tmp_path)])
    V.main()
    assert (d / "ddpm_epochs.gif").is_file()
    assert (d / "rf_epochs.gif").is_file()
    assert (d / "side_by_side.gif").is_file()
