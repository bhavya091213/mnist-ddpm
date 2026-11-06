"""Per-epoch sample grids and GIFs: python -m mnist_flow.epoch_viz --runs runs."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import Tensor, nn

from mnist_flow.config import Config
from mnist_flow.ddpm import make_schedule
from mnist_flow.sample import generate, save_grid

SAMPLE_STEPS = 50
_EPOCH_RE = re.compile(r"^(?P<model>[a-z]+)_epoch(?P<epoch>\d+)\.png$")


def save_epoch_grid(ema_model: nn.Module, model_name: str, cfg: Config, noise: Tensor,
                    path: Path | str, device: torch.device | str) -> None:
    """Sample `noise` with the EMA model (ddim/euler, 50 steps) and save an 8x8 grid."""
    if model_name == "ddpm":
        sched, sampler = make_schedule(cfg).to(device), "ddim"
    else:
        sched, sampler = None, "euler"
    x, _, _ = generate(ema_model, model_name, sampler=sampler, steps=SAMPLE_STEPS,
                       noise=noise, sched=sched, eta=0.0)
    save_grid(x, path, nrow=8)


def _epoch_pngs(png_dir: Path | str, model_name: str) -> dict[int, Path]:
    found = {}
    for p in Path(png_dir).glob(f"{model_name}_epoch*.png"):
        m = _EPOCH_RE.match(p.name)
        if m and m["model"] == model_name:
            found[int(m["epoch"])] = p
    return dict(sorted(found.items()))


def _label(img: Image.Image, text: str) -> Image.Image:
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 8 * len(text) + 6, 14), fill=(0, 0, 0))
    draw.text((3, 2), text, fill=(255, 255, 255))
    return img


def _write_gif(frames: list[Image.Image], out_path: Path | str, duration_ms: int) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


def make_epoch_gif(png_dir: Path | str, model_name: str, out_path: Path | str,
                   duration_ms: int = 400) -> Path:
    """Stitch `{model}_epochXX.png` (sorted) into a looping GIF labelled with the epoch."""
    pngs = _epoch_pngs(png_dir, model_name)
    if not pngs:
        raise FileNotFoundError(f"no {model_name}_epoch*.png files in {png_dir}")
    frames = [_label(Image.open(p), f"epoch {e}") for e, p in pngs.items()]
    _write_gif(frames, out_path, duration_ms)
    return Path(out_path)


def make_side_by_side_gif(png_dir: Path | str, out_path: Path | str,
                          duration_ms: int = 400) -> Path | None:
    """Pair ddpm and rf grids per epoch; returns None when no epoch has both."""
    d, r = _epoch_pngs(png_dir, "ddpm"), _epoch_pngs(png_dir, "rf")
    frames = []
    for e in sorted(d.keys() & r.keys()):
        a = _label(Image.open(d[e]), f"DDPM epoch {e}")
        b = _label(Image.open(r[e]), f"RF epoch {e}")
        canvas = Image.new("RGB", (a.width + b.width, max(a.height, b.height)))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width, 0))
        frames.append(canvas)
    if not frames:
        return None
    _write_gif(frames, out_path, duration_ms)
    return Path(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-epoch sample GIFs from saved grids.")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    a = ap.parse_args()
    d = a.runs / "epoch_samples"
    for m in ("ddpm", "rf"):
        if _epoch_pngs(d, m):
            print(f"wrote {make_epoch_gif(d, m, d / f'{m}_epochs.gif')}")
    side = make_side_by_side_gif(d, d / "side_by_side.gif")
    print(f"wrote {side}" if side else "no epochs with both models; skipped side-by-side")


if __name__ == "__main__":
    main()
