import json
import logging
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path

import pytest
from PIL import Image

from mnist_flow.build_presentation import GITHUB_URL, build


def _png(path: Path, shade: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), shade).save(path)


def _row(model, sampler, nfe, fid):
    return {"name": f"{model}_{sampler}_nfe{nfe}", "model": model, "sampler": sampler,
            "nfe": nfe, "fid": fid, "coverage_entropy": 2.1, "mean_confidence": 0.9}


FIDS = {"ddpm_ddim_nfe1": 1083.3, "rf_euler_nfe1": 2709.9,
        "ddpm_ancestral_strided_nfe1": 1083.3,
        "ddpm_ddim_nfe2": 1087.9, "rf_euler_nfe2": 599.4,
        "ddpm_ancestral_strided_nfe2": 1090.2}


@pytest.fixture
def fixture(tmp_path):
    res, runs, viz = tmp_path / "results", tmp_path / "runs", tmp_path / "runs_viz"
    rows = [_row(*n.replace("ancestral_strided", "ancestral-strided").split("_")[:2], int(n[-1]), f)
            for n, f in FIDS.items()]
    for r in rows:
        r["sampler"] = r["sampler"].replace("-", "_")
        r["name"] = [n for n in FIDS if n.endswith(f"nfe{r['nfe']}")
                     and n.startswith(r["model"]) and (r["sampler"] in n)][0]
    metrics = {"meta": {"nfe_grid": [1, 2], "torch_version": "x", "device": "cpu"},
               "floor": {"name": "floor", "nfe": 0, "fid": 2.9, "coverage_entropy": 2.3,
                         "mean_confidence": 0.98},
               "rows": rows}
    res.mkdir()
    (res / "metrics.json").write_text(json.dumps(metrics))
    imgs = [res / "figures" / f"grid_{r['name']}.png" for r in rows]
    imgs += [res / "figures" / "fid_vs_nfe.png", res / "figures" / "trajectory_strip.png"]
    for model in ("ddpm", "rf"):
        imgs += [viz / "epoch_samples" / f"{model}_epoch{e:02d}.png" for e in (1, 2)]
        runs.mkdir(exist_ok=True)
        (runs / f"train_{model}.json").write_text(json.dumps(
            [{"epoch": e, "step": e, "loss": 1.0 / e, "seconds": 11.0} for e in (1, 2)]))
    for i, p in enumerate(imgs):
        _png(p, 10 * i)
    return tmp_path, res, runs, viz, imgs


class _P(HTMLParser):
    pass


def test_build_ok(fixture):
    tmp, res, runs, viz, imgs = fixture
    out = build(res, runs, viz, tmp / "p" / "index.html", tmp / "docs" / "index.html")
    text = out.read_text()
    assert out.stat().st_size < 1_000_000
    assert (tmp / "docs" / "index.html").exists()
    assert set(re.findall(r"https?://[^\s\"'<>]+", text)) == {GITHUB_URL}
    assert text.count("data:image/png;base64") >= len(imgs)
    for fid in ("1083.3", "2709.9", "599.4", "2.9"):
        assert fid in text
    _P().feed(text)


def test_missing_runs_viz_skips_training(fixture, caplog):
    tmp, res, runs, viz, _ = fixture
    shutil.move(str(viz), str(tmp / "moved"))
    with caplog.at_level(logging.WARNING):
        text = build(res, runs, viz, tmp / "o.html").read_text()
    assert 'id="training"' not in text
    assert 'id="steps"' in text
    assert any("skipping training" in r.message for r in caplog.records)


def test_missing_metrics_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build(tmp_path, tmp_path, tmp_path, tmp_path / "o.html")
