"""
Patch-level consistency diagnostics for laid-line detection.

Given the per-patch results from `estimate_laidline_frequency_gabor_patches`,
quantify how much the patches agree on period and orientation. Low agreement
means the global "dominant" estimate may be an outlier rather than a consensus.

Public API:
- patch_period_stats:        period agreement among patches
- patch_orientation_stats:   line-direction agreement among patches (circular)
- patch_consistency_report:  combined report dict (JSON-able)
- print_consistency_report:  pretty-print the report to stdout
- save_consistency_report:   write the report to JSON
- plot_patch_consistency:    three-panel figure (period histogram + maps)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def _valid_patches(out: dict, score_threshold: float) -> list[dict]:
    return [p for p in out.get("patch_results", []) if float(p.get("best_score", 0.0)) >= score_threshold]


def patch_period_stats(
    out: dict,
    *,
    score_threshold: float = 0.02,
    agreement_tol_px: float = 2.0,
) -> dict:
    """Agreement stats on per-patch laid-line period (pixels)."""
    valid = _valid_patches(out, score_threshold)
    if not valid:
        return {
            "n_valid": 0,
            "median_px": float("nan"),
            "mad_px": float("nan"),
            "iqr_px": [float("nan"), float("nan")],
            "mode_px": float("nan"),
            "score_weighted_mean_px": float("nan"),
            "agreement_tol_px": agreement_tol_px,
            "agreement_frac": float("nan"),
            "histogram": {},
        }

    periods = np.array([float(p["best_period_px"]) for p in valid], dtype=np.float64)
    scores = np.array([float(p["best_score"]) for p in valid], dtype=np.float64)

    median = float(np.median(periods))
    mad = float(np.median(np.abs(periods - median)))
    q25, q75 = (float(v) for v in np.percentile(periods, [25, 75]))
    weighted_mean = float(np.sum(periods * scores) / (np.sum(scores) + 1e-12))

    # Mode over the candidate period grid (periods are discrete in our scan).
    counts = Counter(periods.tolist())
    mode_px = float(max(counts.items(), key=lambda kv: kv[1])[0])

    dominant = float(out.get("dominant_period_px", float("nan")))
    if np.isnan(dominant):
        agreement_frac = float("nan")
    else:
        agreement_frac = float(np.mean(np.abs(periods - dominant) <= agreement_tol_px))

    return {
        "n_valid": int(len(valid)),
        "median_px": median,
        "mad_px": mad,
        "iqr_px": [q25, q75],
        "mode_px": mode_px,
        "score_weighted_mean_px": weighted_mean,
        "agreement_tol_px": agreement_tol_px,
        "agreement_frac": agreement_frac,
        "histogram": {f"{k:g}": int(v) for k, v in sorted(counts.items())},
    }


def _to_line_dir_deg(theta_deg: float) -> float:
    """Convert a Gabor normal angle to an inferred laid-line direction, mod 180."""
    return ((theta_deg + 90.0) % 180.0)


def _circular_stats_180(angles_deg: np.ndarray) -> tuple[float, float]:
    """
    Circular mean and resultant length for orientations mod 180 degrees.
    Returns (mean_deg in [0, 180), R in [0, 1]).
    """
    a_rad = 2.0 * np.pi * angles_deg / 180.0  # doubled-angle in radians
    c = float(np.cos(a_rad).mean())
    s = float(np.sin(a_rad).mean())
    resultant_length = float(np.sqrt(c * c + s * s))
    mean_doubled_deg = float(np.degrees(np.arctan2(s, c)))  # doubled-angle in degrees
    mean_deg = (mean_doubled_deg / 2.0) % 180.0
    return mean_deg, resultant_length


def _circular_dist_deg(a: np.ndarray, b: float, period_deg: float = 180.0) -> np.ndarray:
    """Smallest circular distance from each angle in `a` to `b`, in degrees."""
    half = period_deg / 2.0
    d = (a - b + half) % period_deg - half
    return np.abs(d)


def patch_orientation_stats(
    out: dict,
    *,
    score_threshold: float = 0.02,
    agreement_tol_deg: float = 5.0,
) -> dict:
    """Agreement stats on per-patch laid-line direction (degrees, mod 180)."""
    valid = _valid_patches(out, score_threshold)
    if not valid:
        return {
            "n_valid": 0,
            "circular_mean_deg": float("nan"),
            "resultant_length": float("nan"),
            "circular_mad_deg": float("nan"),
            "agreement_tol_deg": agreement_tol_deg,
            "agreement_frac": float("nan"),
        }

    line_dirs = np.array([_to_line_dir_deg(float(p["best_theta_deg"])) for p in valid], dtype=np.float64)

    circular_mean, resultant_length = _circular_stats_180(line_dirs)
    dists = _circular_dist_deg(line_dirs, circular_mean, period_deg=180.0)
    circular_mad = float(np.median(dists))

    dominant_line_dir = float(out.get("line_dir_deg", float("nan")))
    if np.isnan(dominant_line_dir):
        agreement_frac = float("nan")
    else:
        target_mod = dominant_line_dir % 180.0
        agreement_frac = float(np.mean(_circular_dist_deg(line_dirs, target_mod, 180.0) <= agreement_tol_deg))

    return {
        "n_valid": int(len(valid)),
        "circular_mean_deg": circular_mean,
        "resultant_length": resultant_length,
        "circular_mad_deg": circular_mad,
        "agreement_tol_deg": agreement_tol_deg,
        "agreement_frac": agreement_frac,
    }


def patch_consistency_report(
    out: dict,
    *,
    score_threshold: float = 0.02,
    agreement_tol_px: float = 2.0,
    agreement_tol_deg: float = 5.0,
) -> dict:
    """Combined per-patch consistency report (period + orientation)."""
    n_patches = len(out.get("patch_results", []))
    return {
        "n_patches": int(n_patches),
        "score_threshold": float(score_threshold),
        "dominant_period_px": float(out.get("dominant_period_px", float("nan"))),
        "dominant_line_dir_deg": float(out.get("line_dir_deg", float("nan"))),
        "period": patch_period_stats(
            out, score_threshold=score_threshold, agreement_tol_px=agreement_tol_px
        ),
        "orientation": patch_orientation_stats(
            out, score_threshold=score_threshold, agreement_tol_deg=agreement_tol_deg
        ),
    }


def print_consistency_report(report: dict) -> None:
    """One-pass readable summary."""
    p, o = report["period"], report["orientation"]
    print("[Eval] Patch consistency report")
    print(f"  patches: total={report['n_patches']}, valid(score>={report['score_threshold']:.2g})={p['n_valid']}")
    print(f"  period: dominant={report['dominant_period_px']:.1f} px"
          f" | median={p['median_px']:.1f} mad={p['mad_px']:.1f}"
          f" | IQR=[{p['iqr_px'][0]:.1f},{p['iqr_px'][1]:.1f}]"
          f" | mode={p['mode_px']:.1f}"
          f" | score-weighted-mean={p['score_weighted_mean_px']:.1f}")
    print(f"          agreement (|.-dominant|<={p['agreement_tol_px']:.1f} px): {p['agreement_frac']*100:.1f}%")
    print(f"  orient: dominant={report['dominant_line_dir_deg']:.1f} deg"
          f" | circ-mean={o['circular_mean_deg']:.1f} circ-mad={o['circular_mad_deg']:.2f}"
          f" | R={o['resultant_length']:.3f}")
    print(f"          agreement (|.-dominant|<={o['agreement_tol_deg']:.1f} deg): {o['agreement_frac']*100:.1f}%")


def save_consistency_report(report: dict, path: str | Path) -> None:
    """Write the report dict to JSON."""
    Path(path).write_text(json.dumps(report, indent=2))


def _patch_grid(patch_results: list[dict], key: str) -> tuple[np.ndarray, list[int], list[int]]:
    """Stitch per-patch scalar values onto a 2D grid indexed by patch (y, x)."""
    ys = sorted({p["y"] for p in patch_results})
    xs = sorted({p["x"] for p in patch_results})
    if not ys or not xs:
        return np.full((0, 0), np.nan, dtype=np.float32), ys, xs
    lookup = {(p["y"], p["x"]): float(p[key]) for p in patch_results}
    grid = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            grid[yi, xi] = lookup.get((y, x), np.nan)
    return grid, ys, xs


def plot_patch_consistency(
    out: dict,
    report: dict | None = None,
    *,
    score_threshold: float = 0.02,
) -> None:
    """
    Three-panel figure:
      1. histogram of per-patch best_period_px, with dominant and median marked
      2. spatial map of per-patch best_period_px
      3. spatial map of per-patch best_score (confidence)
    """
    patch_results = out.get("patch_results", [])
    if not patch_results:
        print("[Eval] plot_patch_consistency: no patches to plot.")
        return

    if report is None:
        report = patch_consistency_report(out, score_threshold=score_threshold)

    valid = _valid_patches(out, score_threshold)
    valid_periods = np.array([float(p["best_period_px"]) for p in valid], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # --- Panel 1: period histogram ---
    ax = axes[0]
    if valid_periods.size > 0:
        unique_periods = sorted({float(p["best_period_px"]) for p in patch_results})
        if len(unique_periods) >= 2:
            step = min(np.diff(unique_periods))
            bins = np.arange(min(unique_periods) - step / 2, max(unique_periods) + step, step)
        else:
            bins = 10
        ax.hist(valid_periods, bins=bins, color="steelblue", edgecolor="white")
    dominant = report["dominant_period_px"]
    median = report["period"]["median_px"]
    if not np.isnan(dominant):
        ax.axvline(dominant, color="red", linestyle="--", linewidth=1.5,
                   label=f"dominant={dominant:.1f}")
    if not np.isnan(median):
        ax.axvline(median, color="green", linestyle=":", linewidth=1.5,
                   label=f"median={median:.1f}")
    ax.set_xlabel("Period (px)")
    ax.set_ylabel("Patch count")
    n_valid = report["period"]["n_valid"]
    ax.set_title(f"Per-patch period (n_valid={n_valid}/{report['n_patches']})")
    ax.legend(loc="best", fontsize=9)

    # --- Panel 2: spatial period map ---
    period_grid, _, _ = _patch_grid(patch_results, "best_period_px")
    ax = axes[1]
    im = ax.imshow(period_grid, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="period (px)")
    ax.set_title("Spatial period map")
    ax.set_xlabel("Patch column")
    ax.set_ylabel("Patch row")

    # --- Panel 3: spatial score map ---
    score_grid, _, _ = _patch_grid(patch_results, "best_score")
    ax = axes[2]
    im = ax.imshow(score_grid, cmap="magma", aspect="auto")
    fig.colorbar(im, ax=ax, label="best score")
    ax.set_title("Spatial score map")
    ax.set_xlabel("Patch column")
    ax.set_ylabel("Patch row")

    plt.tight_layout()
    plt.show()
