"""Side-by-side denoising trajectories: DDPM (DDIM eta=0) vs RF (Euler).

Frame conventions differ between the samplers (DDPM frames include the initial
x and are taken *before* each kept step; RF frames are taken *after* each kept
step and exclude the initial noise). We normalise here: every returned frame
list starts with the shared initial noise and contains the state after
0, k, 2k, ..., steps steps for both models.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch import Tensor  # noqa: E402
from torchvision.utils import make_grid  # noqa: E402

from mnist_flow import ddpm  # noqa: E402
from mnist_flow.evaluate import (  # noqa: E402
    NOISE_SEED, LoadedModel, _generate, _make_noise_bank, load_pair,
)
from mnist_flow.utils import get_device  # noqa: E402

GIF_MS_PER_FRAME = 100
GIF_SCALE = 4
ROW_LABELS = ("DDPM x_t", "DDPM pred x0", "RF x_t")


def keep_every_for(steps: int, frames: int) -> int:
    """Stride so that start + strided states + end gives at most `frames` frames."""
    if frames < 2 or steps < 1:
        raise ValueError(f"need frames >= 2 and steps >= 1, got {frames=}, {steps=}")
    return max(1, math.ceil(steps / (frames - 1)))


def _ddpm_pred_x0(lm: LoadedModel, frames: list[Tensor], steps: int, k: int) -> list[Tensor]:
    """x0-prediction for each intermediate DDPM frame (1 extra NFE per frame; cheap)."""
    sched = ddpm.make_schedule(lm.cfg)
    ts = ddpm.ddim_timesteps(sched.T, steps).tolist()
    dev = next(lm.model.parameters()).device
    out = []
    with torch.no_grad():
        for j, x in enumerate(frames[:-1]):
            t_idx = torch.full((x.shape[0],), ts[j * k], dtype=torch.long, device=dev)
            xd = x.to(dev)
            eps = lm.model(xd, ddpm.t_to_model_time(t_idx, sched.T))
            out.append(ddpm.predict_x0(xd, eps, t_idx, sched.to(dev)).clamp(-1, 1).cpu())
    return out + [frames[-1]]


def _strip(rows: list[list[Tensor]], seeds: int, path: Path) -> None:
    n = len(rows[0])
    tiles = [rows[r][f][s] for s in range(seeds) for r in range(len(rows)) for f in range(n)]
    grid = make_grid((torch.stack(tiles).clamp(-1, 1) + 1) / 2, nrow=n, padding=2, pad_value=1)
    fig, ax = plt.subplots(figsize=(1.0 * n + 1.5, 1.0 * seeds * len(rows)))
    ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    cell = 30
    ax.set_yticks([cell * (i + 0.5) for i in range(seeds * len(rows))])
    ax.set_yticklabels([f"seed {s}: {lbl}" for s in range(seeds) for lbl in ROW_LABELS], fontsize=7)
    ax.set_xticks([cell * (i + 0.5) for i in range(n)])
    ax.set_xticklabels(["noise"] + [""] * (n - 2) + ["final"], fontsize=7)
    ax.set_title("Trajectories from shared noise: DDPM (DDIM, eta=0) vs RF (Euler)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _to_pil(x: Tensor) -> Image.Image:
    g = make_grid((x.clamp(-1, 1) + 1) / 2, nrow=x.shape[0], padding=2, pad_value=1)
    arr = (g[0] * 255).round().to(torch.uint8).numpy()
    img = Image.fromarray(arr, mode="L")
    return img.resize((img.width * GIF_SCALE, img.height * GIF_SCALE), Image.NEAREST)


def _gif(ddpm_frames: list[Tensor], rf_frames: list[Tensor], path: Path) -> None:
    frames = []
    for a, b in zip(ddpm_frames, rf_frames):
        left, right = _to_pil(a), _to_pil(b)
        gap = 8 * GIF_SCALE
        canvas = Image.new("L", (left.width + gap + right.width, left.height), color=128)
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width + gap, 0))
        frames.append(canvas)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=GIF_MS_PER_FRAME, loop=0)


def run_trajectory(
    ckpt_dir: Path | str, out_dir: Path | str, *, seeds: int = 4, steps: int = 50,
    frames: int = 10, device: torch.device | str = "auto", noise_seed: int = NOISE_SEED,
) -> dict:
    """Strip PNG + side-by-side GIF (DDPM | RF) from the same first `seeds` bank noises."""
    dev = get_device(device) if isinstance(device, str) else device
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = load_pair(Path(ckpt_dir), dev)  # EMA weights + fairness check
    noise = _make_noise_bank(seeds, noise_seed)  # same as first rows of eval bank
    k = keep_every_for(steps, frames)
    _, _, d_frames = _generate(models["ddpm"], "ddpm", "ddim", steps, noise.to(dev), keep_every=k)
    _, _, r_frames = _generate(models["rf"], "rf", "euler", steps, noise.to(dev), keep_every=k)
    d_frames = [f.cpu() for f in d_frames]
    r_frames = [noise.clone()] + [f.cpu() for f in r_frames]  # RF omits the start state
    if len(d_frames) != len(r_frames):
        raise RuntimeError(f"frame count mismatch: ddpm={len(d_frames)} rf={len(r_frames)}")
    x0_frames = _ddpm_pred_x0(models["ddpm"], d_frames, steps, k)  # needs raw x_t
    d_frames = [f.clamp(-1, 1) for f in d_frames]  # display range, same as RF frames
    r_frames = [f.clamp(-1, 1) for f in r_frames]
    strip_path, gif_path = out / "trajectory_strip.png", out / "trajectory.gif"
    _strip([d_frames, x0_frames, r_frames], seeds, strip_path)
    _gif(d_frames, r_frames, gif_path)
    print(f"[trajectory] {len(d_frames)} frames -> {strip_path}, {gif_path}", flush=True)
    return {"strip": str(strip_path), "gif": str(gif_path), "n_frames": len(d_frames),
            "keep_every": k, "ddpm_frames": d_frames,
            "ddpm_x0_frames": x0_frames, "rf_frames": r_frames}


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="DDPM vs RF trajectory strip + GIF")
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--out", type=Path, default=Path("results/figures"))
    p.add_argument("--device", default="auto")
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--frames", type=int, default=10)
    a = p.parse_args(argv)
    run_trajectory(a.ckpt_dir, a.out, seeds=a.seeds, steps=a.steps, frames=a.frames,
                   device=a.device)


if __name__ == "__main__":
    main()
