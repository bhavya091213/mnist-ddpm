"""Sampling helpers and CLI: python -m mnist_flow.sample --ckpt ... --sampler ddim."""
from __future__ import annotations

import argparse
import functools
from pathlib import Path

import torch
from torch import Tensor, nn
from torchvision.utils import save_image

from mnist_flow.checkpoint import build_model_from_checkpoint, load_checkpoint
from mnist_flow.ddpm import DDPMSchedule, make_schedule, sample_ancestral, sample_ddim
from mnist_flow.rf import sample_euler, sample_heun
from mnist_flow.utils import get_device

SAMPLERS = {
    "ddpm": ("ancestral", "ddim"),
    "rf": ("euler", "heun"),
}


def generate(
    model: nn.Module, model_name: str, *, sampler: str, steps: int, noise: Tensor,
    sched: DDPMSchedule | None = None, eta: float = 0.0, keep_every: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, int, list[Tensor]]:
    """Dispatch to a sampler by name; returns (x0 in [-1,1], nfe, frames)."""
    if model_name not in SAMPLERS or sampler not in SAMPLERS[model_name]:
        raise ValueError(
            f"sampler {sampler!r} invalid for model {model_name!r}; "
            f"valid: {SAMPLERS.get(model_name)}"
        )
    if model_name == "ddpm":
        if sched is None:
            raise ValueError("ddpm samplers require sched")
        if sampler == "ancestral":
            fn = functools.partial(sample_ancestral, sched=sched, generator=generator)
            return fn(model, noise, steps=None, keep_every=keep_every)
        fn = functools.partial(sample_ddim, sched=sched, eta=eta, generator=generator)
        return fn(model, noise, steps=steps, keep_every=keep_every)
    fn = sample_euler if sampler == "euler" else sample_heun
    return fn(model, noise, steps=steps, keep_every=keep_every)


def make_noise_bank(n: int, seed: int, device: torch.device | str) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 1, 28, 28, generator=g).to(device)


def to_uint8(x: Tensor) -> Tensor:
    """[-1,1] float -> uint8 [0,255] (clamped)."""
    return ((x.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)


def save_grid(x: Tensor, path: Path | str, nrow: int = 8) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(x.float().cpu(), str(path), nrow=nrow, normalize=True, value_range=(-1, 1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample from a trained checkpoint.")
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--sampler", required=True, choices=["ancestral", "ddim", "euler", "heun"])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--noise-seed", type=int, default=1234)
    ap.add_argument("--out", type=Path, default=Path("results/samples"))
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    ckpt = load_checkpoint(args.ckpt, device)
    model_name = ckpt["model_name"]
    model, cfg = build_model_from_checkpoint(ckpt, device, use_ema=True)
    sched = make_schedule(cfg).to(device) if model_name == "ddpm" else None
    bank = make_noise_bank(args.n, args.noise_seed, device)
    gen = torch.Generator().manual_seed(args.noise_seed + 1)  # only used if eta > 0

    outs, nfe = [], 0
    for i in range(0, args.n, args.batch_size):
        x, nfe, _ = generate(
            model, model_name, sampler=args.sampler, steps=args.steps,
            noise=bank[i : i + args.batch_size], sched=sched, eta=args.eta, generator=gen,
        )
        outs.append(x.cpu())
    x_all = torch.cat(outs)
    stem = f"{model_name}_{args.sampler}_nfe{nfe}"
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(to_uint8(x_all), args.out / f"{stem}.pt")
    save_grid(x_all[:64], args.out / f"{stem}.png")
    print(f"{stem}: nfe={nfe} n={x_all.shape[0]}")


if __name__ == "__main__":
    main()
