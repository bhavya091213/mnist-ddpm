import dataclasses
import json
import sys

import pytest
import torch

from mnist_flow import data, train as T
from mnist_flow.checkpoint import load_checkpoint
from mnist_flow.config import Config


@pytest.fixture(autouse=True)
def synthetic_mnist(monkeypatch):
    g = torch.Generator().manual_seed(0)
    x = torch.rand(64, 1, 28, 28, generator=g) * 2 - 1
    y = torch.zeros(64, dtype=torch.int64)

    def fake(root="data/", train=True, device="cpu"):
        return x.to(device), y.to(device)

    monkeypatch.setattr(data, "load_mnist", fake)


def _run(tmp_path, model_name, epochs=2, **kw):
    cfg = Config.tiny()
    return T.train(cfg, model_name=model_name, device=torch.device("cpu"),
                   out_dir=tmp_path / "ck", data_root=tmp_path / "d", epochs=epochs,
                   log_dir=tmp_path / "runs", seed=0, **kw)


@pytest.mark.parametrize("name", ["ddpm", "rf"])
def test_train_end_to_end_and_resume(tmp_path, name):
    final = _run(tmp_path, name)
    assert final == tmp_path / "ck" / name / "final.pt"
    assert (tmp_path / "ck" / name / "last.pt").exists()
    ckpt = load_checkpoint(final, "cpu")
    assert ckpt["epoch"] == 2 and ckpt["step"] == 2 * (64 // 8)
    assert ckpt["model_name"] == name
    log = json.loads((tmp_path / "runs" / f"train_{name}.json").read_text())
    assert [e["epoch"] for e in log] == [1, 2]
    assert all(torch.isfinite(torch.tensor(e["loss"])) for e in log)
    assert {"epoch", "step", "loss", "seconds"} <= set(log[0])
    assert (tmp_path / "runs" / f"train_{name}.log").read_text().strip()

    # resume with nothing left to do
    _run(tmp_path, name)
    log2 = json.loads((tmp_path / "runs" / f"train_{name}.json").read_text())
    assert len(log2) == 2
    # extend to 3 epochs: only one more epoch runs
    _run(tmp_path, name, epochs=3)
    log3 = json.loads((tmp_path / "runs" / f"train_{name}.json").read_text())
    assert [e["epoch"] for e in log3] == [1, 2, 3]
    assert load_checkpoint(final, "cpu")["epoch"] == 3


def test_no_resume_restarts(tmp_path):
    _run(tmp_path, "rf", epochs=1)
    _run(tmp_path, "rf", epochs=1, resume=False)
    log = json.loads((tmp_path / "runs" / "train_rf.json").read_text())
    assert [e["epoch"] for e in log] == [1]


def test_resume_matches_uninterrupted(tmp_path):
    a = tmp_path / "a"; b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    fa = _run(a, "rf", epochs=2)
    _run(b, "rf", epochs=1)
    fb = _run(b, "rf", epochs=2)
    sa = load_checkpoint(fa, "cpu")["model"]
    sb = load_checkpoint(fb, "cpu")["model"]
    assert all(torch.allclose(sa[k], sb[k], atol=1e-6) for k in sa)


def test_same_seed_same_batch_order():
    for epoch in (0, 3):
        ga, gb = T.batch_generator(5, epoch), T.batch_generator(5, epoch)
        x = torch.arange(40).float().view(40, 1, 1, 1)
        a = next(data.iterate_batches(x, 8, ga))
        b = next(data.iterate_batches(x, 8, gb))
        assert torch.equal(a, b)
    assert not torch.equal(
        torch.randperm(40, generator=T.batch_generator(5, 0)),
        torch.randperm(40, generator=T.batch_generator(5, 1)))


def test_too_small_dataset_raises(tmp_path):
    cfg = dataclasses.replace(Config.tiny(), batch_size=1000)
    with pytest.raises(ValueError):
        T.train(cfg, model_name="rf", device=torch.device("cpu"), out_dir=tmp_path,
                data_root=tmp_path, epochs=1, log_dir=tmp_path, seed=0)


def test_subset_limits_dataset(tmp_path):
    cfg = Config.tiny()
    T.train(cfg, model_name="rf", device=torch.device("cpu"), out_dir=tmp_path / "ck",
            data_root=tmp_path, epochs=1, log_dir=tmp_path / "r", seed=0, subset=16)
    assert load_checkpoint(tmp_path / "ck" / "rf" / "final.pt", "cpu")["step"] == 2


def test_bad_model_name(tmp_path):
    with pytest.raises(ValueError):
        _run(tmp_path, "nope")


def test_cli_tiny(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "train", "--model", "rf", "--epochs", "1", "--device", "cpu", "--tiny",
        "--out", str(tmp_path / "ck"), "--data", str(tmp_path / "d"),
        "--runs", str(tmp_path / "runs")])
    T.main()
    assert (tmp_path / "ck" / "rf" / "final.pt").exists()
