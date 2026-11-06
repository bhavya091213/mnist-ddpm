"""FID-vs-NFE plot and markdown metric table."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TABLE_COLUMNS = ("name", "model", "sampler", "nfe", "n", "fid", "coverage_entropy",
                 "mean_confidence", "seconds")


def _family(row: dict) -> str:
    return f"{row['model']}/{row['sampler']}"


def plot_fid_vs_nfe(metrics: dict, path: Path | str) -> None:
    """Log-x FID vs NFE, one line per sampler family, floor as a dashed line."""
    families: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in metrics.get("rows", []):
        families[_family(row)].append((row["nfe"], row["fid"]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fam, pts in families.items():
        pts = sorted(pts)
        style = "*" if len(pts) == 1 else "o-"
        ax.plot([p[0] for p in pts], [p[1] for p in pts], style, label=fam, markersize=8)
    floor = metrics.get("floor")
    if floor is not None:
        ax.axhline(floor["fid"], linestyle="--", color="gray",
                   label=f"floor (train vs test) = {floor['fid']:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("NFE (network evaluations per sample)")
    ax.set_ylabel("FID (MNIST-classifier features, lower is better)")
    ax.set_title("DDPM vs Rectified Flow: quality vs sampling cost")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fmt(v) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def plot_metric_table(metrics: dict) -> str:
    """Markdown table: one line per row, floor (if present) last."""
    rows = list(metrics.get("rows", []))
    if metrics.get("floor") is not None:
        rows.append(metrics["floor"])
    lines = ["| " + " | ".join(TABLE_COLUMNS) + " |",
             "|" + "|".join("---" for _ in TABLE_COLUMNS) + "|"]
    lines += ["| " + " | ".join(_fmt(r.get(c)) for c in TABLE_COLUMNS) + " |" for r in rows]
    return "\n".join(lines) + "\n"
