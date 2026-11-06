import dataclasses
import json
from pathlib import Path

import pytest
import torch

from mnist_flow import evaluate as ev
from mnist_flow import plots
from mnist_flow.config import Config
from mnist_flow.sample import make_noise_bank
from mnist_flow.unet import UNet

T_SMALL = 20


def _cfg(**kw) -> Config:
    return dataclasses.replace(Config.tiny(), ddpm_T=T_SMALL, **kw)


def make_ckpt(path: Path, cfg: Config, model_name: str, seed: int = 0) -> None:
    torch.manual_seed(seed)
    model = UNet(cfg)
    sd = model.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": sd, "ema": sd, "opt": {}, "config": cfg.to_dict(),
                "model_name": model_name, "step": 1, "epoch": 1}, path)


def fake_load_mnist(root="data/", train=True, device="cpu"):
    g = torch.Generator().manual_seed(0 if train else 1)
    n = 512 if train else 128
    x = torch.rand(n, 1, 28, 28, generator=g) * 2 - 1
    y = torch.randint(0, 10, (n,), generator=g)
    return x.to(device), y.to(device)


@pytest.fixture
def ckpt_dir(tmp_path):
    d = tmp_path / "checkpoints"
    make_ckpt(d / "ddpm" / "final.pt", _cfg(), "ddpm", seed=0)
    make_ckpt(d / "rf" / "last.pt", _cfg(rf_time_sampling="logit_normal"), "rf", seed=1)
    return d


@pytest.fixture(autouse=True)
def no_download(monkeypatch):
    monkeypatch.setattr(ev, "load_mnist", fake_load_mnist)


def _run(ckpt_dir, out, **kw):
    params = dict(n=16, nfe_grid=[1, 2], device="cpu", classifier_epochs=1, batch_size=8)
    params.update(kw)
    return ev.run_evaluation(ckpt_dir, out, "data", **params)


def test_run_evaluation_writes_rows_grids_samples(ckpt_dir, tmp_path):
    out = tmp_path / "results"
    seen = []
    result = _run(ckpt_dir, out, on_row=lambda row, path: seen.append(
        len(json.loads(Path(path).read_text())["rows"])))
    data = json.loads((out / "metrics.json").read_text())
    names = [r["name"] for r in data["rows"]]
    assert names == ["ddpm_ddim_nfe1", "ddpm_ancestral_strided_nfe1", "rf_euler_nfe1",
                     "ddpm_ddim_nfe2", "ddpm_ancestral_strided_nfe2", "rf_euler_nfe2",
                     f"ddpm_ancestral_nfe{T_SMALL}"]
    keys = {"model", "sampler", "requested_steps", "nfe", "n", "fid", "coverage_entropy",
            "mean_confidence", "class_histogram", "seconds"}
    for r in data["rows"]:
        assert keys <= r.keys()
        assert r["n"] == 16 and sum(r["class_histogram"]) == 16
        assert (out / "figures" / f"grid_{r['name']}.png").exists()
        s = torch.load(out / "samples" / f"{r['name']}.pt", weights_only=True)
        assert s.dtype == torch.uint8 and s.shape == (16, 1, 28, 28)
    assert [r["nfe"] for r in data["rows"]] == [1, 1, 1, 2, 2, 2, T_SMALL]
    # incremental: file had k rows after k-th row
    assert seen == list(range(1, 8))
    meta = data["meta"]
    assert meta["noise_seed"] == 1234 and meta["n"] == 16
    assert set(meta["configs"]) == {"ddpm", "rf"}
    assert 0.0 <= meta["classifier_test_accuracy"] <= 1.0
    assert "torch_version" in meta and meta["device"] == "cpu" and meta["ema"] is True
    assert data["floor"]["fid"] >= 0 and data["floor"]["n"] == 16
    assert (out / "classifier.pt").exists()
    assert (out / "figures" / "fid_vs_nfe.png").exists()
    assert (out / "metrics_table.md").exists()
    assert result["rows"] == data["rows"]


def test_same_noise_for_every_row(ckpt_dir, tmp_path):
    result = _run(ckpt_dir, tmp_path / "r", include_heun=True)
    hashes = {r["noise_sha256_first64"] for r in result["rows"]}
    assert len(hashes) == 1
    expected = ev._tensor_hash(make_noise_bank(16, 1234, "cpu")[:64])
    assert hashes == {expected}
    heun = [r for r in result["rows"] if r["sampler"] == "heun"]
    assert heun and all(r["nfe"] == 2 * r["requested_steps"] for r in heun)


def test_reproducible_and_classifier_reused(ckpt_dir, tmp_path):
    out = tmp_path / "r"
    a = _run(ckpt_dir, out)
    mtime = (out / "classifier.pt").stat().st_mtime_ns
    b = _run(ckpt_dir, out)
    assert (out / "classifier.pt").stat().st_mtime_ns == mtime
    assert [r["fid"] for r in a["rows"]] == pytest.approx([r["fid"] for r in b["rows"]])
    # previous metrics.json moved aside, not destroyed
    assert list(out.glob("metrics.json.bak-*"))


def test_fairness_assertion_raises_on_config_mismatch(tmp_path):
    d = tmp_path / "ck"
    make_ckpt(d / "ddpm" / "final.pt", _cfg(), "ddpm")
    make_ckpt(d / "rf" / "final.pt", _cfg(lr=1e-3), "rf")
    with pytest.raises(ValueError, match="lr"):
        _run(d, tmp_path / "out")


def test_check_fairness_ignores_rf_time_sampling():
    ev.check_fairness(_cfg().to_dict(), _cfg(rf_time_sampling="x").to_dict())
    with pytest.raises(ValueError):
        ev.check_fairness(_cfg().to_dict(), {**_cfg().to_dict(), "extra": 1})


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ev.find_checkpoint(tmp_path, "ddpm")


def test_wrong_model_name_raises(tmp_path):
    p = tmp_path / "ddpm" / "final.pt"
    make_ckpt(p, _cfg(), "rf")
    with pytest.raises(ValueError, match="model_name"):
        ev._load_model(p, "ddpm", torch.device("cpu"))


def test_parse_nfe_and_cli(ckpt_dir, tmp_path, monkeypatch):
    assert ev.parse_nfe("1, 2,5") == [1, 2, 5]
    with pytest.raises(ValueError):
        ev.parse_nfe("0,2")
    captured = {}
    monkeypatch.setattr(ev, "run_evaluation", lambda *a, **k: captured.update(args=a, kw=k) or {})
    ev.main(["--ckpt-dir", "c", "--out", "o", "--data", "d", "--n", "32",
             "--device", "cpu", "--nfe", "1,2", "--heun"])
    assert captured["args"] == (Path("c"), Path("o"), Path("d"))
    assert captured["kw"]["n"] == 32 and captured["kw"]["nfe_grid"] == [1, 2]
    assert captured["kw"]["include_heun"] is True


def _fake_metrics():
    rows = [
        {"name": f"{m}_{s}_nfe{k}", "model": m, "sampler": s, "requested_steps": k, "nfe": k,
         "n": 16, "fid": 10.0 / k, "coverage_entropy": 2.0, "mean_confidence": 0.9,
         "class_histogram": [1] * 10, "seconds": 0.1}
        for k in (1, 2, 5) for m, s in (("ddpm", "ddim"), ("rf", "euler"))
    ] + [{"name": "ddpm_ancestral_nfe1000", "model": "ddpm", "sampler": "ancestral",
          "requested_steps": None, "nfe": 1000, "n": 16, "fid": 1.0, "coverage_entropy": 2.0,
          "mean_confidence": 0.9, "class_histogram": [1] * 10, "seconds": 1.0}]
    floor = {"name": "floor_train_vs_test", "model": "real", "sampler": "train_subset",
             "requested_steps": None, "nfe": 0, "n": 16, "fid": 0.5, "coverage_entropy": 2.3,
             "mean_confidence": 0.99, "class_histogram": [1] * 10, "seconds": 0.0}
    return {"rows": rows, "floor": floor}


def test_plot_fid_vs_nfe_writes_file(tmp_path):
    p = tmp_path / "f" / "fid.png"
    plots.plot_fid_vs_nfe(_fake_metrics(), p)
    assert p.exists() and p.stat().st_size > 0


def test_plot_metric_table_one_line_per_row():
    m = _fake_metrics()
    lines = plots.plot_metric_table(m).strip().splitlines()
    # header + separator + rows + floor
    assert len(lines) == 2 + len(m["rows"]) + 1
    assert "ddpm_ancestral_nfe1000" in lines[-2] and "floor" in lines[-1]
    assert lines[0].startswith("|")


def test_plot_metric_table_without_floor():
    m = {"rows": _fake_metrics()["rows"]}
    assert len(plots.plot_metric_table(m).strip().splitlines()) == 2 + len(m["rows"])
