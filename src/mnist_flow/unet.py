"""Small attention-free UNet taking (x, t in [0,1])."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mnist_flow.config import Config


def sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    """t: (B,) -> (B, dim) sin/cos features of t*1000."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = (t.to(torch.float32) * 1000.0)[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.avg_pool2d(x, 2)


class Upsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.base = cfg.base_channels
        g, ed = cfg.groupnorm_groups, cfg.time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(self.base, ed), nn.SiLU(), nn.Linear(ed, ed)
        )
        self.conv_in = nn.Conv2d(cfg.channels, self.base, 3, padding=1)

        self.down = nn.ModuleList()
        skip_chs = [self.base]
        ch = self.base
        n_levels = len(cfg.channel_mults)
        for lvl, mult in enumerate(cfg.channel_mults):
            out = self.base * mult
            for _ in range(cfg.num_res_blocks):
                self.down.append(ResBlock(ch, out, ed, g))
                ch = out
                skip_chs.append(ch)
            if lvl < n_levels - 1:
                self.down.append(Downsample())
                skip_chs.append(ch)

        self.mid1 = ResBlock(ch, ch, ed, g)
        self.mid2 = ResBlock(ch, ch, ed, g)

        self.up = nn.ModuleList()
        for lvl, mult in reversed(list(enumerate(cfg.channel_mults))):
            out = self.base * mult
            for _ in range(cfg.num_res_blocks + 1):
                self.up.append(ResBlock(ch + skip_chs.pop(), out, ed, g))
                ch = out
            if lvl > 0:
                self.up.append(Upsample(ch))

        self.norm_out = nn.GroupNorm(g, ch)
        self.conv_out = nn.Conv2d(ch, cfg.channels, 3, padding=1)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        emb = self.time_mlp(sinusoidal_embedding(t, self.base))
        h = self.conv_in(x)
        skips = [h]
        for layer in self.down:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
            skips.append(h)
        h = self.mid2(self.mid1(h, emb), emb)
        for layer in self.up:
            if isinstance(layer, ResBlock):
                h = layer(torch.cat([h, skips.pop()], dim=1), emb)
            else:
                h = layer(h)
        return self.conv_out(F.silu(self.norm_out(h)))
