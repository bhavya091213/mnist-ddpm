import dataclasses
from pathlib import Path

import pytest
import torch
from PIL import Image

from mnist_flow import trajectory as tr
from mnist_flow.config import Config
from mnist_flow.unet import UNet


def _ckpts(tmp_path) -> Path:
    cfg = dataclasses.replace(Config.tiny(), ddpm_T=50)
    d = tmp_path / "ck"
    for name in ("ddpm", "rf"):
        sd = UNet(cfg).state_dict()
        (d / name).mkdir(parents=True)
        torch.save({"model": sd, "ema": sd, "opt": {}, "config": cfg.to_dict(),
                    "model_name": name, "step": 1, "epoch": 1}, d / name / "final.pt")
    return d


@pytest.mark.parametrize("steps,frames", [(50, 10), (10, 4), (7, 3)])
def test_frame_schedule_matches_for_both_models(steps, frames):
    k = tr.keep_every_for(steps, frames)
    n_ddpm = -(-steps // k) + 1
    assert 2 <= n_ddpm <= frames


def test_keep_every_rejects_bad_frames():
    with pytest.raises(ValueError):
        tr.keep_every_for(10, 1)


def test_run_trajectory_outputs(tmp_path):
    out = tmp_path / "figs"
    res = tr.run_trajectory(_ckpts(tmp_path), out, seeds=2, steps=10, frames=4, device="cpu")
    strip, gif = Path(res["strip"]), Path(res["gif"])
    assert strip.exists() and gif.exists()
    n = res["n_frames"]
    assert n == len(res["ddpm_frames"]) == len(res["rf_frames"]) == len(res["ddpm_x0_frames"])
    assert n == 4  # k=ceil(10/3)=4 -> states after 0,4,8,10 steps
    # both trajectories start at the same initial noise
    assert torch.equal(res["ddpm_frames"][0], res["rf_frames"][0])
    assert res["ddpm_frames"][0].shape == (2, 1, 28, 28)
    with Image.open(gif) as im:
        assert im.n_frames == n
        assert im.info.get("loop") == 0
        assert im.info.get("duration") == 100
    assert res["ddpm_frames"][-1].abs().max() <= 1.0


def test_trajectory_cli(monkeypatch):
    captured = {}
    monkeypatch.setattr(tr, "run_trajectory", lambda *a, **k: captured.update(a=a, k=k) or {})
    tr.main(["--ckpt-dir", "c", "--out", "o", "--device", "cpu", "--steps", "20"])
    assert captured["a"] == (Path("c"), Path("o"))
    assert captured["k"]["steps"] == 20 and captured["k"]["device"] == "cpu"
