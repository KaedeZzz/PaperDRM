"""
Self-consistency contrast evaluator.

Given a detected grid of wire positions and a period, this evaluator
samples pixel intensities at the predicted wire columns vs. the
half-period-shifted "between-wires" columns. It then reports the
intensity gap as both an absolute value and a t-like z-score over grid
lines.

Independent of the detector itself (which is FFT-based) — this metric is
purely spatial-domain. A high z-score corroborates that the grid landed
on real wires; a near-zero score suggests the grid is off-phase, on the
wrong period, or that wires aren't darker than the substrate in this
image (`wire_is_darker` may be wrong).

Useful for cross-method comparison: applying this to outputs from both
the SIMPLE single-image detector and the MULTI-PHI detector on the same
image gives a method-agnostic "is the grid correct?" score.

Public API:
    self_consistency_contrast -- compute stats from image + grid + period
    print_self_contrast       -- console summary
    save_self_contrast        -- JSON dump (trimmed)
    plot_self_contrast        -- per-line on/off plot + difference histogram
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from paperdrm.stage3_detect.simple_detector import _rotate_to_vertical


def self_consistency_contrast(
    image: np.ndarray,
    grid_positions_x: np.ndarray,
    period_px: float,
    *,
    line_dir_deg: float = 90.0,
    band_half_width_px: int = 1,
    wire_is_darker: bool = True,
) -> dict:
    """
    Compare intensities at predicted wire positions vs. inter-wire
    positions (predicted wire position + period/2).

    Args:
        image: 2D grayscale image (the same image the grid was detected on).
        grid_positions_x: 1D array of predicted wire x positions in the
            rotated frame (where lines are vertical).
        period_px: detected period.
        line_dir_deg: laid-line direction (90 = vertical).
        band_half_width_px: include columns within +/- this many px of
            each grid position when computing the mean intensity. With
            =1 we sample a 3-pixel-wide band per line (more robust than
            a single column).
        wire_is_darker: when True, the metric is positive if on-wire
            columns are darker than off-wire columns; when False, signed
            the other way.

    Returns a JSON-able dict with mean intensities, signed contrast,
    standard error and z-score over grid lines.
    """
    grid_x = np.asarray(grid_positions_x, dtype=np.float64).ravel()
    if grid_x.size < 2:
        return _empty_result(grid_x.size, band_half_width_px, wire_is_darker,
                             warning="fewer than 2 grid lines")

    img = _rotate_to_vertical(image, line_dir_deg).astype(np.float64)
    h, w = img.shape
    half = max(0, int(band_half_width_px))
    half_period = float(period_px) / 2.0

    # Per-column row-mean precomputed once.
    col_mean = img.mean(axis=0)

    def _band_mean(xc: int) -> float:
        lo = max(0, xc - half)
        hi = min(w, xc + half + 1)
        if hi <= lo:
            return float("nan")
        return float(col_mean[lo:hi].mean())

    on_vals: list[float] = []
    off_vals: list[float] = []
    for x in grid_x:
        xo = int(round(float(x)))
        if xo < 0 or xo >= w:
            continue
        x_off = int(round(float(x) + half_period))
        if x_off >= w:
            x_off -= int(round(float(period_px)))
        if x_off < 0 or x_off >= w:
            continue
        on = _band_mean(xo)
        off = _band_mean(x_off)
        if np.isfinite(on) and np.isfinite(off):
            on_vals.append(on)
            off_vals.append(off)

    on_arr = np.asarray(on_vals, dtype=np.float64)
    off_arr = np.asarray(off_vals, dtype=np.float64)
    if on_arr.size == 0:
        return _empty_result(int(grid_x.size), band_half_width_px, wire_is_darker,
                             warning="no valid grid/off pairs in image bounds")

    on_mean = float(on_arr.mean())
    off_mean = float(off_arr.mean())

    diff = off_arr - on_arr
    if not wire_is_darker:
        diff = -diff
    contrast_abs = float(diff.mean())
    denom = float(0.5 * (abs(on_mean) + abs(off_mean))) or 1e-6
    contrast_rel = float(contrast_abs / denom)
    if diff.size > 1:
        se = float(diff.std(ddof=1) / np.sqrt(diff.size))
    else:
        se = float("nan")
    z = float(contrast_abs / se) if (se and np.isfinite(se) and se > 0) else float("nan")

    return {
        "n_lines": int(diff.size),
        "on_wire_mean": on_mean,
        "off_wire_mean": off_mean,
        "contrast_abs": contrast_abs,
        "contrast_rel": contrast_rel,
        "contrast_se": se,
        "contrast_z": z,
        "per_line_on": on_arr.tolist(),
        "per_line_off": off_arr.tolist(),
        "wire_is_darker": bool(wire_is_darker),
        "band_half_width_px": int(half),
        "period_px_used": float(period_px),
        "line_dir_deg": float(line_dir_deg),
        "warning": None,
    }


def _empty_result(
    n_lines: int,
    band_half_width_px: int,
    wire_is_darker: bool,
    *,
    warning: str,
) -> dict:
    return {
        "n_lines": int(n_lines),
        "on_wire_mean": float("nan"),
        "off_wire_mean": float("nan"),
        "contrast_abs": float("nan"),
        "contrast_rel": float("nan"),
        "contrast_se": float("nan"),
        "contrast_z": float("nan"),
        "per_line_on": [],
        "per_line_off": [],
        "wire_is_darker": bool(wire_is_darker),
        "band_half_width_px": int(band_half_width_px),
        "period_px_used": float("nan"),
        "line_dir_deg": float("nan"),
        "warning": warning,
    }


def print_self_contrast(stats: dict) -> None:
    print("[Eval] Self-consistency contrast")
    print(f"  n_lines={stats['n_lines']}"
          f"  band=±{stats['band_half_width_px']} px"
          f"  wire_is_darker={stats['wire_is_darker']}")
    if not np.isfinite(stats["contrast_abs"]):
        print(f"  (no valid pairs) warning: {stats.get('warning')}")
        return
    print(f"  on-wire mean={stats['on_wire_mean']:.2f}"
          f"  off-wire mean={stats['off_wire_mean']:.2f}")
    print(f"  contrast(off-on)={stats['contrast_abs']:+.3f}"
          f"  rel={stats['contrast_rel']*100:+.2f}%"
          f"  se={stats['contrast_se']:.3f}"
          f"  z={stats['contrast_z']:+.2f}")


def save_self_contrast(stats: dict, path: str | Path) -> None:
    keep = {k: v for k, v in stats.items() if k not in {"per_line_on", "per_line_off"}}
    keep["per_line_on_sample"] = stats["per_line_on"][:50]
    keep["per_line_off_sample"] = stats["per_line_off"][:50]
    Path(path).write_text(json.dumps(keep, indent=2))


def plot_self_contrast(stats: dict, save_path: str | Path | None = None) -> None:
    on = np.asarray(stats["per_line_on"], dtype=np.float64)
    off = np.asarray(stats["per_line_off"], dtype=np.float64)
    if on.size == 0:
        print("[Eval] plot_self_contrast: nothing to plot")
        return
    diff = off - on
    if not stats["wire_is_darker"]:
        diff = -diff

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(on, label="on-wire", color="darkred", linewidth=1.0)
    axes[0].plot(off, label="off-wire (+T/2)", color="steelblue", linewidth=1.0)
    axes[0].set_xlabel("Grid line index")
    axes[0].set_ylabel("Mean intensity")
    axes[0].set_title("Per-line on vs off-wire intensity")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(diff, bins=min(30, max(8, diff.size // 2)),
                 color="steelblue", edgecolor="white", alpha=0.85)
    axes[1].axvline(0.0, color="black", linewidth=1)
    axes[1].axvline(stats["contrast_abs"], color="red", linestyle="--",
                    linewidth=1.2, label=f"mean={stats['contrast_abs']:+.2f}")
    axes[1].set_xlabel("off − on (intensity)")
    axes[1].set_ylabel("# grid lines")
    axes[1].set_title(f"Contrast distribution  (z={stats['contrast_z']:+.2f})")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
