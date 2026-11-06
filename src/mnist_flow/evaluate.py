"""Fair NFE-sweep evaluation of DDPM vs Rectified Flow.

Fairness invariants enforced here (see 01-plan.md "Shared config"):
EMA weights for both models, one shared noise bank (CPU generator, fixed seed)
reused by every row, one real reference set (MNIST test), one classifier, the
same N per row, and identical checkpoint configs except ``rf_time_sampling``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch import Tensor

from mnist_flow import ddpm
from mnist_flow.checkpoint import build_model_from_checkpoint, load_checkpoint
from mnist_flow.classifier import (
    MNISTClassifier, evaluate_accuracy, extract_features, load_classifier,
    predict_logits, save_classifier, train_classifier,
)
from mnist_flow.config import Config
from mnist_flow.data import load_mnist
from mnist_flow.metrics import (
    class_histogram, coverage_entropy, frechet_distance, mean_confidence,
)
from mnist_flow.sample import generate, make_noise_bank, save_grid, to_uint8
from mnist_flow.unet import UNet
from mnist_flow.utils import get_device

DEFAULT_NFE_GRID: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100)
NOISE_SEED = 1234
STOCHASTIC_SEED_OFFSET = 1  # per-step noise for stochastic samplers: seed + offset
FAIRNESS_EXEMPT_FIELDS = frozenset({"rf_time_sampling"})
MODEL_NAMES = ("ddpm", "rf")
GRID_N = 64
SAMPLES_SAVED = 1000
FLOOR_SUBSET_SEED = 0


@dataclass(frozen=True)
class RowSpec:
    model: str
    sampler: str
    requested_steps: int | None

    def name(self, nfe: int) -> str:
        return f"{self.model}_{self.sampler}_nfe{nfe}"


@dataclass(frozen=True)
class LoadedModel:
    model: UNet
    cfg: Config
    cfg_dict: dict[str, Any]
    path: Path
    step: int
    epoch: int


# ---------------------------------------------------------------- helpers
def _tensor_hash(x: Tensor) -> str:
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _generate(
    lm: LoadedModel, model_name: str, sampler: str, steps: int | None, noise: Tensor,
    *, generator: torch.Generator | None = None, keep_every: int | None = None,
) -> tuple[Tensor, int, list[Tensor]]:
    """Map evaluate's sampler names onto sample.generate; returns (x0, nfe, frames)."""
    sched = ddpm.make_schedule(lm.cfg) if model_name == "ddpm" else None
    eta = 0.0
    if sampler == "ancestral_strided":  # strided DDIM with eta=1 (stochastic)
        sampler, eta = "ddim", 1.0
    return generate(lm.model, model_name, sampler=sampler, steps=steps, noise=noise,
                    sched=sched, eta=eta, keep_every=keep_every, generator=generator)


def check_fairness(cfg_a: dict[str, Any], cfg_b: dict[str, Any]) -> None:
    """Raise ValueError if configs differ in any field except the exempt ones."""
    keys = (set(cfg_a) | set(cfg_b)) - FAIRNESS_EXEMPT_FIELDS
    diffs = sorted(k for k in keys if cfg_a.get(k, "<missing>") != cfg_b.get(k, "<missing>"))
    if diffs:
        detail = ", ".join(f"{k}: {cfg_a.get(k, '<missing>')!r} vs {cfg_b.get(k, '<missing>')!r}"
                           for k in diffs)
        raise ValueError(f"FAIRNESS VIOLATION: checkpoint configs differ in {detail}")


def find_checkpoint(ckpt_dir: Path, model_name: str) -> Path:
    for fname in ("final.pt", "last.pt"):
        p = Path(ckpt_dir) / model_name / fname
        if p.is_file():
            return p
    raise FileNotFoundError(f"no final.pt/last.pt for {model_name!r} under {ckpt_dir}")


def _load_model(path: Path, model_name: str, device: torch.device) -> LoadedModel:
    ckpt = load_checkpoint(path, "cpu")
    if ckpt.get("model_name", model_name) != model_name:
        raise ValueError(f"{path}: model_name={ckpt.get('model_name')!r}, expected {model_name!r}")
    model, cfg = build_model_from_checkpoint(ckpt, device, use_ema=True)  # fairness: EMA
    return LoadedModel(model, cfg, dict(ckpt["config"]), Path(path),
                       int(ckpt.get("step", -1)), int(ckpt.get("epoch", -1)))


def load_pair(ckpt_dir: Path, device: torch.device) -> dict[str, LoadedModel]:
    models = {n: _load_model(find_checkpoint(ckpt_dir, n), n, device) for n in MODEL_NAMES}
    check_fairness(models["ddpm"].cfg_dict, models["rf"].cfg_dict)
    return models


def parse_nfe(text: str) -> list[int]:
    vals = [int(v) for v in text.split(",") if v.strip()]
    if not vals or any(v < 1 for v in vals):
        raise ValueError(f"NFE grid must be positive integers, got {text!r}")
    return vals


def build_rows(nfe_grid: Sequence[int], include_heun: bool) -> list[RowSpec]:
    rows: list[RowSpec] = []
    heun_steps: set[int] = set()
    for nfe in nfe_grid:
        rows += [RowSpec("ddpm", "ddim", nfe), RowSpec("ddpm", "ancestral_strided", nfe),
                 RowSpec("rf", "euler", nfe)]
        steps = nfe // 2  # heun costs 2 NFE/step; recorded nfe is the true count
        if include_heun and steps >= 1 and steps not in heun_steps:
            heun_steps.add(steps)
            rows.append(RowSpec("rf", "heun", steps))
    rows.append(RowSpec("ddpm", "ancestral", None))  # headline: full T-step chain
    return rows


# ---------------------------------------------------------------- metrics
def _metrics_vs_real(clf: MNISTClassifier, real_feat: Tensor, x: Tensor, bs: int) -> dict:
    logits = predict_logits(clf, x, batch_size=bs)
    return {
        "fid": frechet_distance(real_feat, extract_features(clf, x, batch_size=bs)),
        "coverage_entropy": coverage_entropy(logits),
        "mean_confidence": mean_confidence(logits),
        "class_histogram": class_histogram(logits),
    }


def _get_classifier(path: Path, data_root: Path, device: torch.device, epochs: int):
    x_test, y_test = load_mnist(data_root, train=False, device="cpu")
    if path.is_file():
        clf = load_classifier(path, device)
    else:
        x_tr, y_tr = load_mnist(data_root, train=True, device="cpu")
        clf = train_classifier(x_tr, y_tr, device=device, epochs=epochs)
        save_classifier(clf, path)
    return clf, x_test, evaluate_accuracy(clf, x_test, y_test)


def _floor_row(clf, real_feat, data_root: Path, n: int, bs: int) -> dict:
    x_tr, _ = load_mnist(data_root, train=True, device="cpu")
    g = torch.Generator().manual_seed(FLOOR_SUBSET_SEED)
    subset = x_tr[torch.randperm(x_tr.shape[0], generator=g)[: min(n, x_tr.shape[0])]]
    t0 = time.perf_counter()
    m = _metrics_vs_real(clf, real_feat, subset, bs)
    return {"name": "floor_train_vs_test", "model": "real", "sampler": "train_subset",
            "requested_steps": None, "nfe": 0, "n": int(subset.shape[0]), **m,
            "seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------- driver
def _run_row(spec: RowSpec, lm: LoadedModel, noise_bank: Tensor, device, bs: int, seed: int):
    gen = torch.Generator().manual_seed(seed + STOCHASTIC_SEED_OFFSET)
    outs, nfes, seen = [], set(), []
    t0 = time.perf_counter()
    for i in range(0, noise_bank.shape[0], bs):
        chunk = noise_bank[i : i + bs]
        if sum(s.shape[0] for s in seen) < GRID_N:
            seen.append(chunk)
        x, nfe, _ = _generate(lm, spec.model, spec.sampler, spec.requested_steps,
                              chunk.to(device), generator=gen)
        outs.append(x.float().cpu())
        nfes.add(nfe)
    if len(nfes) != 1:
        raise RuntimeError(f"inconsistent NFE across batches for {spec}: {nfes}")
    noise_hash = _tensor_hash(torch.cat(seen)[:GRID_N])
    return torch.cat(outs), nfes.pop(), noise_hash, time.perf_counter() - t0


def _write_json(data: dict, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)  # atomic: a crash never leaves a half-written metrics.json


def _move_aside(path: Path) -> None:
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path.rename(path.with_name(f"{path.name}.bak-{stamp}"))


def run_evaluation(
    ckpt_dir: Path | str, out_dir: Path | str, data_root: Path | str, *,
    n: int = 10000, nfe_grid: Sequence[int] = DEFAULT_NFE_GRID, noise_seed: int = NOISE_SEED,
    device: torch.device | str = "auto", classifier_epochs: int = 2, batch_size: int = 500,
    include_heun: bool = False, on_row: Callable[[dict, Path], None] | None = None,
) -> dict:
    """Run the full sweep; metrics.json is rewritten after every row."""
    from mnist_flow.plots import plot_fid_vs_nfe, plot_metric_table

    if n < 1 or batch_size < 1:
        raise ValueError("n and batch_size must be >= 1")
    dev = get_device(device) if isinstance(device, str) else device
    out, data_root = Path(out_dir), Path(data_root)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    models = load_pair(Path(ckpt_dir), dev)  # fails loudly on config mismatch
    clf, x_test, acc = _get_classifier(out / "classifier.pt", data_root, dev, classifier_epochs)
    real_feat = extract_features(clf, x_test, batch_size=batch_size)  # once
    noise_bank = make_noise_bank(n, noise_seed, "cpu")  # shared by every row

    metrics_path = out / "metrics.json"
    _move_aside(metrics_path)
    metrics: dict[str, Any] = {
        "meta": {
            "n": n, "nfe_grid": list(nfe_grid), "noise_seed": noise_seed,
            "stochastic_sampler_seed": noise_seed + STOCHASTIC_SEED_OFFSET,
            "noise_sha256_first64": _tensor_hash(noise_bank[:GRID_N]),
            "torch_version": torch.__version__, "device": str(dev), "ema": True,
            "classifier_test_accuracy": acc, "n_real_reference": int(x_test.shape[0]),
            "configs": {k: v.cfg_dict for k, v in models.items()},
            "checkpoints": {k: {"path": str(v.path), "step": v.step, "epoch": v.epoch}
                            for k, v in models.items()},
        },
        "floor": _floor_row(clf, real_feat, data_root, n, batch_size),
        "rows": [],
    }
    _write_json(metrics, metrics_path)
    for spec in build_rows(nfe_grid, include_heun):
        x, nfe, noise_hash, secs = _run_row(spec, models[spec.model], noise_bank, dev,
                                            batch_size, noise_seed)
        name = spec.name(nfe)
        row = {"name": name, "model": spec.model, "sampler": spec.sampler,
               "requested_steps": spec.requested_steps, "nfe": nfe, "n": int(x.shape[0]),
               **_metrics_vs_real(clf, real_feat, x, batch_size), "seconds": secs,
               "noise_sha256_first64": noise_hash}
        torch.save(to_uint8(x[:SAMPLES_SAVED]).clone(), out / "samples" / f"{name}.pt")
        save_grid(x[:GRID_N], out / "figures" / f"grid_{name}.png")
        metrics["rows"].append(row)
        _write_json(metrics, metrics_path)
        print(f"[eval] {name}: fid={row['fid']:.3f} ({secs:.1f}s)", flush=True)
        if on_row is not None:
            on_row(row, metrics_path)
    plot_fid_vs_nfe(metrics, out / "figures" / "fid_vs_nfe.png")
    (out / "metrics_table.md").write_text(plot_metric_table(metrics))
    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Fair DDPM vs RF NFE-sweep evaluation")
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--device", default="auto")
    p.add_argument("--nfe", type=parse_nfe, default=list(DEFAULT_NFE_GRID))
    p.add_argument("--heun", action="store_true", help="also evaluate RF Heun (2 NFE/step)")
    p.add_argument("--noise-seed", type=int, default=NOISE_SEED)
    p.add_argument("--classifier-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=500)
    a = p.parse_args(argv)
    run_evaluation(a.ckpt_dir, a.out, a.data, n=a.n, nfe_grid=a.nfe, noise_seed=a.noise_seed,
                   device=a.device, classifier_epochs=a.classifier_epochs,
                   batch_size=a.batch_size, include_heun=a.heun)


if __name__ == "__main__":
    main()
