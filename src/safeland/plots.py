"""Figures drawn only from logged results (never from synthetic data)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"unshielded": "#c0392b", "prob_shield": "#2e86c1", "sure_shield": "#1e8449"}


def _curves(root: Path, method: str) -> list[np.ndarray]:
    files = sorted((root / method).glob("seed_*/train_curve.csv"))
    return [np.loadtxt(f, delimiter=",", skiprows=1) for f in files]


def _band(ax, x, runs, color, label):
    arr = np.vstack(runs)
    mean = arr.mean(axis=0)
    if arr.shape[0] > 1:
        se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, color=color, alpha=0.2, lw=0)
    ax.plot(x, mean, color=color, lw=1.8, label=label)


def plot_training(root: str | Path, window: int = 500) -> Path:
    root = Path(root)
    methods = [p.name for p in sorted(root.iterdir()) if p.is_dir() and any(p.glob("seed_*"))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for m in methods:
        runs = _curves(root, m)
        if not runs:
            continue
        x = runs[0][:, 0]
        w = max(1, min(window, len(x) // 5))
        color = COLORS.get(m, "gray")
        _band(axes[0], x, [np.cumsum(r[:, 2]) for r in runs], color, m)
        k = np.ones(w) / w
        smooth = [np.convolve(r[:, 3], k, mode="valid") for r in runs]
        _band(axes[1], x[w - 1:], smooth, color, m)
    axes[0].set(xlabel="training episode", ylabel="cumulative safety violations",
                title="Safety during learning")
    axes[1].set(xlabel="training episode", ylabel="safe-landing rate (moving mean)",
                title="Task performance during learning", ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    fig.tight_layout()
    out = root / "training.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_reserve_map(root: str | Path) -> Path | None:
    from .config import Config

    root = Path(root)
    f = root / "reserve_map.npy"
    if not f.exists():
        return None
    r = np.load(f)
    cfg = Config.load(root / "config.yaml") if (root / "config.yaml").exists() else Config()
    w = cfg.world
    nfz = set(map(tuple, w.nfz))
    tall = {(x, y) for x, y, h in w.buildings if h >= w.start_alt}
    pads = set(map(tuple, w.pads))
    cmap = matplotlib.colormaps["viridis_r"].copy()
    cmap.set_bad("#d5d8dc")
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(np.ma.masked_invalid(r).T, origin="lower", cmap=cmap)
    mid = np.nanmean(r)
    for (x, y), v in np.ndenumerate(r):
        if np.isnan(v):
            txt = "NFZ" if (x, y) in nfz else ("bldg" if (x, y) in tall else "\u00d7")
            ax.text(x, y, txt, ha="center", va="center", color="#34495e", fontsize=7)
        else:
            ax.text(x, y, f"{int(v)}", ha="center", va="center",
                    color="white" if v > mid else "black", fontsize=8)
        if (x, y) in pads:
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, ec="#e74c3c", lw=2))
    wx, wy = w.wind_dir
    ax.set(xlabel="x", ylabel="y",
           title=f"Minimum battery certified safe (z={w.start_alt})\n"
                 f"red = pad, \u00d7 = unsafe at any battery, wind ({wx},{wy}) p={w.wind_prob}")
    fig.colorbar(im, ax=ax, label="battery units")
    fig.tight_layout()
    out = root / "reserve_map.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_all(root: str | Path) -> list[Path]:
    outs = [plot_training(root)]
    rm = plot_reserve_map(root)
    if rm:
        outs.append(rm)
    summ = Path(root) / "summary.json"
    if summ.exists():
        json.loads(summ.read_text())  # validates the file is complete
    return outs
