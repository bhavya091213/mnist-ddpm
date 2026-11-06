"""Train DDPM or rectified-flow UNets: python -m mnist_flow.train --model ddpm."""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path

import torch
from tqdm import tqdm

from mnist_flow import data
from mnist_flow.checkpoint import load_checkpoint, save_checkpoint
from mnist_flow.config import Config
from mnist_flow.ddpm import ddpm_loss, make_schedule
from mnist_flow.ema import EMA
from mnist_flow.rf import rf_loss
from mnist_flow.unet import UNet
from mnist_flow.utils import get_device, set_seed

MODELS = ("ddpm", "rf")
_SEED_STRIDE = 100_003


def batch_generator(seed: int, epoch: int) -> torch.Generator:
    """CPU generator for batch order; independent of model type so ddpm/rf match."""
    return torch.Generator().manual_seed(seed * _SEED_STRIDE + epoch)


def noise_generator(seed: int, epoch: int) -> torch.Generator:
    """CPU generator for t/noise draws; per-epoch seeding makes resume exact."""
    return torch.Generator().manual_seed(seed * _SEED_STRIDE + epoch + 10**9)


def _make_logger(path: Path, name: str) -> tuple[logging.Logger, logging.Handler]:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    return logger, handler


def _run_epoch(model, ema, opt, x, cfg, model_name, sched, epoch, seed, desc) -> tuple[float, int]:
    bgen, ngen = batch_generator(seed, epoch), noise_generator(seed, epoch)
    total, n = 0.0, 0
    model.train()
    bar = tqdm(data.iterate_batches(x, cfg.batch_size, bgen),
               total=x.shape[0] // cfg.batch_size, desc=desc, leave=False)
    for xb in bar:
        loss = (ddpm_loss(model, xb, sched, ngen) if model_name == "ddpm"
                else rf_loss(model, xb, cfg, ngen))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(model)
        total += loss.item()
        n += 1
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return total / n, n


def train(
    cfg: Config, *, model_name: str, device: torch.device | str, out_dir: Path | str,
    data_root: Path | str, epochs: int | None = None, resume: bool = True,
    log_dir: Path | str, seed: int, subset: int | None = None,
) -> Path:
    """Train and return the path to final.pt. Resumes from last.pt when present."""
    if model_name not in MODELS:
        raise ValueError(f"model_name must be one of {MODELS}, got {model_name!r}")
    epochs = cfg.epochs if epochs is None else epochs
    ckpt_dir = Path(out_dir) / model_name
    last, final = ckpt_dir / "last.pt", ckpt_dir / "final.pt"
    log_dir = Path(log_dir)
    json_path = log_dir / f"train_{model_name}.json"

    set_seed(seed)
    x, _ = data.load_mnist(data_root, train=True, device=device)
    if subset is not None and subset < x.shape[0]:
        idx = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(seed))[:subset]
        x = x[idx.to(x.device)]
    if x.shape[0] < cfg.batch_size:
        raise ValueError(f"dataset ({x.shape[0]}) smaller than batch_size ({cfg.batch_size})")

    model = UNet(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ema = EMA(model, cfg.ema_decay)
    sched = make_schedule(cfg).to(device) if model_name == "ddpm" else None

    step, start_epoch, history = 0, 0, []
    if resume and last.is_file():
        ck = load_checkpoint(last, device)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, start_epoch = ck["step"], ck["epoch"]
        if json_path.is_file():
            history = [e for e in json.loads(json_path.read_text()) if e["epoch"] <= start_epoch]

    logger, handler = _make_logger(log_dir / f"train_{model_name}.log", f"mnist_flow.train.{model_name}")
    try:
        logger.info("start model=%s epoch=%d/%d step=%d device=%s", model_name, start_epoch, epochs, step, device)
        for epoch in range(start_epoch, epochs):
            t0 = time.perf_counter()
            loss, n = _run_epoch(model, ema, opt, x, cfg, model_name, sched, epoch, seed,
                                 f"{model_name} {epoch + 1}/{epochs}")
            step += n
            secs = time.perf_counter() - t0
            history.append({"epoch": epoch + 1, "step": step, "loss": loss, "seconds": secs})
            json_path.write_text(json.dumps(history, indent=2))
            logger.info("epoch=%d step=%d loss=%.6f seconds=%.1f", epoch + 1, step, loss, secs)
            save_checkpoint(last, model=model, ema=ema, opt=opt, cfg=cfg,
                            model_name=model_name, step=step, epoch=epoch + 1)
        save_checkpoint(final, model=model, ema=ema, opt=opt, cfg=cfg,
                        model_name=model_name, step=step, epoch=max(epochs, start_epoch))
        logger.info("done final=%s", final)
    finally:
        logger.removeHandler(handler)
        handler.close()
    return final


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a DDPM or rectified-flow UNet on MNIST.")
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=Path("checkpoints"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="tiny model + 256-image subset (smoke test)")
    a = ap.parse_args()

    cfg = Config.tiny() if a.tiny else Config()
    overrides = {k: v for k, v in
                 {"batch_size": a.batch_size, "lr": a.lr, "seed": a.seed}.items() if v is not None}
    cfg = dataclasses.replace(cfg, **overrides)
    final = train(cfg, model_name=a.model, device=get_device(a.device), out_dir=a.out,
                  data_root=a.data, epochs=a.epochs, resume=not a.no_resume,
                  log_dir=a.runs, seed=cfg.seed, subset=256 if a.tiny else None)
    print(f"final checkpoint: {final}")


if __name__ == "__main__":
    main()
