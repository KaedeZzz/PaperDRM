"""
Wire-shadow width statistics across horizontal strips of the image.

Point-estimate strategy:
- Rotate the image so laid lines are vertical.
- Split into n_segments equal-height strips.
- For each strip, build the broadband 1D signal (column-mean + high-pass)
  and run estimate_wire_width.
- Aggregate per-segment sigma/FWHM values into descriptive statistics +
  confidence intervals.

Two CIs are reported:
  ci_t:   parametric mean +/- t_{alpha/2, n-1} * SE (uncertainty in MEAN).
  ci_pct: percentile [alpha/2, 1-alpha/2] of per-segment values (spread
          across the page; useful as a spatial heterogeneity range).

Public API:
- wire_width_statistics
- print_wire_width_statistics
- save_wire_width_statistics
- plot_wire_width_statistics
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as sp_stats

from paperdrm.stage3_detect.wire_width import estimate_wire_width


_FWHM_OVER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _rotate_to_vertical(image: np.ndarray, line_dir_deg: float) -> np.ndarray:
    rot_angle = 90.0 - float(line_dir_deg)
    if abs(rot_angle) < 1e-6:
        return image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rot_angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def _broadband_1d(strip: np.ndarray) -> np.ndarray:
    """Column-mean of a strip with moving-average high-pass detrend."""
    s = strip.astype(np.float64).mean(axis=0)
    win = min(301, max(31, (len(s) // 20) | 1))
    k = np.ones(win, dtype=np.float64) / win
    return s - np.convolve(s, k, mode="same")


def _aggregate(values_px: np.ndarray, alpha: float) -> dict:
    """Descriptive stats + parametric and percentile CIs."""
    vals = values_px[~np.isnan(values_px)]
    n = vals.size
    out: dict = {
        "n_valid": int(n),
        "median": float("nan"),
        "mean": float("nan"),
        "std": float("nan"),
        "sem": float("nan"),
        "ci_t": (float("nan"), float("nan")),
        "ci_pct": (float("nan"), float("nan")),
        "iqr": (float("nan"), float("nan")),
        "min": float("nan"),
        "max": float("nan"),
    }
    if n == 0:
        return out
    mean = float(np.mean(vals))
    median = float(np.median(vals))
    out["median"] = median
    out["mean"] = mean
    out["min"] = float(np.min(vals))
    out["max"] = float(np.max(vals))
    if n >= 2:
        std = float(np.std(vals, ddof=1))
        sem = std / np.sqrt(n)
        # parametric CI for the mean (two-sided)
        t_crit = float(sp_stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
        out["std"] = std
        out["sem"] = sem
        out["ci_t"] = (mean - t_crit * sem, mean + t_crit * sem)
    # percentile CI = spread across page
    lo_pct, hi_pct = np.percentile(vals, [100 * alpha / 2.0, 100 * (1 - alpha / 2.0)])
    q1, q3 = np.percentile(vals, [25.0, 75.0])
    out["ci_pct"] = (float(lo_pct), float(hi_pct))
    out["iqr"] = (float(q1), float(q3))
    return out


def wire_width_statistics(
    image: np.ndarray,
    period_px: float,
    *,
    line_dir_deg: float = 90.0,
    n_segments: int = 16,
    n_max: int = 4,
    alpha: float = 0.05,
    fov_width_cm: float | None = None,
) -> dict:
    """
    Compute wire-shadow width statistics over non-overlapping horizontal
    strips of the rotated image.

    Args:
        image:         2D grayscale image (background-subtracted recommended).
        period_px:     detected laid-line period in pixels.
        line_dir_deg:  laid-line direction (90 = vertical).
        n_segments:    number of horizontal strips. Each strip must have
                       enough rows for a clean column-mean (>= ~30 rows is
                       reasonable). Default 16.
        n_max:         highest harmonic order for estimate_wire_width.
        alpha:         significance level. CI level = 1 - alpha. Default 0.05.
        fov_width_cm:  optional, enables physical (cm/mm) outputs.

    Returns dict with:
        period_px, line_dir_deg, n_segments, n_max, alpha:    meta
        global: dict
            sigma_px, fwhm_px, model_ok, warning             (full-image fit)
        segments: dict
            sigma_px (np.ndarray), fwhm_px (np.ndarray)       (per strip)
            warnings (list[str | None])
        aggregate: dict with keys sigma_px, fwhm_px each holding
            n_valid, median, mean, std, sem, ci_t, ci_pct, iqr, min, max
        physical: dict (only if fov_width_cm given)
            cm_per_px,
            sigma_cm: aggregate-shaped dict
            fwhm_cm:  aggregate-shaped dict
            fwhm_mm:  aggregate-shaped dict
            global:   {sigma_cm, fwhm_cm, fwhm_mm}
    """
    if image.ndim != 2:
        raise ValueError("wire_width_statistics expects a 2D grayscale image.")
    if n_segments < 2:
        raise ValueError("n_segments must be >= 2 to produce statistics.")

    rot = _rotate_to_vertical(image, line_dir_deg)
    H, W = rot.shape

    # Global estimate
    global_signal = _broadband_1d(rot)
    g = estimate_wire_width(global_signal, period_px, n_max=n_max)
    global_out = {
        "sigma_px": g["sigma_px"],
        "fwhm_px": g["fwhm_px"],
        "model_ok": g["model_ok"],
        "warning": g["warning"],
    }

    # Per-segment estimates
    strips = np.array_split(rot, n_segments, axis=0)
    sigma_seg = np.full(n_segments, np.nan, dtype=np.float64)
    fwhm_seg = np.full(n_segments, np.nan, dtype=np.float64)
    warnings: list[str | None] = []
    for i, strip in enumerate(strips):
        if strip.shape[0] < 10:
            warnings.append(f"strip too thin ({strip.shape[0]} rows)")
            continue
        sig = _broadband_1d(strip)
        est = estimate_wire_width(sig, period_px, n_max=n_max)
        warnings.append(est["warning"])
        if est["model_ok"]:
            sigma_seg[i] = est["sigma_px"]
            fwhm_seg[i] = est["fwhm_px"]

    agg_sigma = _aggregate(sigma_seg, alpha)
    agg_fwhm = _aggregate(fwhm_seg, alpha)

    result: dict = {
        "period_px": float(period_px),
        "line_dir_deg": float(line_dir_deg),
        "n_segments": int(n_segments),
        "n_max": int(n_max),
        "alpha": float(alpha),
        "global": global_out,
        "segments": {
            "sigma_px": sigma_seg,
            "fwhm_px": fwhm_seg,
            "warnings": warnings,
        },
        "aggregate": {
            "sigma_px": agg_sigma,
            "fwhm_px": agg_fwhm,
        },
    }

    if fov_width_cm is not None:
        cm_per_px = float(fov_width_cm) / float(image.shape[1])
        result["physical"] = {
            "cm_per_px": cm_per_px,
            "sigma_cm": _scale_aggregate(agg_sigma, cm_per_px),
            "fwhm_cm": _scale_aggregate(agg_fwhm, cm_per_px),
            "fwhm_mm": _scale_aggregate(agg_fwhm, cm_per_px * 10.0),
            "global": {
                "sigma_cm": (g["sigma_px"] * cm_per_px) if g["model_ok"] else float("nan"),
                "fwhm_cm": (g["fwhm_px"] * cm_per_px) if g["model_ok"] else float("nan"),
                "fwhm_mm": (g["fwhm_px"] * cm_per_px * 10.0) if g["model_ok"] else float("nan"),
            },
        }

    return result


def _scale_aggregate(agg: dict, scale: float) -> dict:
    """Multiply scalar / pair stats by `scale` (NaN preserved)."""
    scaled: dict = {"n_valid": agg["n_valid"]}
    for key in ("median", "mean", "std", "sem", "min", "max"):
        scaled[key] = agg[key] * scale
    for key in ("ci_t", "ci_pct", "iqr"):
        lo, hi = agg[key]
        scaled[key] = (lo * scale, hi * scale)
    return scaled


def print_wire_width_statistics(stats: dict) -> None:
    """Pretty-print to stdout."""
    g = stats["global"]
    agg = stats["aggregate"]
    sigma_agg = agg["sigma_px"]
    fwhm_agg = agg["fwhm_px"]
    ci_level = (1.0 - stats["alpha"]) * 100.0

    print("[Eval] Wire-shadow width statistics (Gaussian-comb model)")
    if g["model_ok"]:
        print(f"  global (full image) : sigma = {g['sigma_px']:.3f} px"
              f" | FWHM = {g['fwhm_px']:.3f} px")
    else:
        print(f"  global (full image) : FAILED ({g['warning']})")

    n = sigma_agg["n_valid"]
    print(f"  segments            : {n}/{stats['n_segments']} valid")
    if n >= 1:
        print(f"  sigma_px            : median={sigma_agg['median']:.3f} "
              f"mean={sigma_agg['mean']:.3f} "
              f"std={sigma_agg['std']:.3f} "
              f"sem={sigma_agg['sem']:.3f}")
        lo, hi = sigma_agg["ci_t"]
        print(f"    {ci_level:.0f}% CI (mean)   : [{lo:.3f}, {hi:.3f}] px")
        plo, phi = sigma_agg["ci_pct"]
        print(f"    {ci_level:.0f}% spread      : [{plo:.3f}, {phi:.3f}] px (per-segment percentile)")
        print(f"  fwhm_px             : median={fwhm_agg['median']:.3f} "
              f"mean={fwhm_agg['mean']:.3f} "
              f"std={fwhm_agg['std']:.3f}")
        flo, fhi = fwhm_agg["ci_t"]
        print(f"    {ci_level:.0f}% CI (mean)   : [{flo:.3f}, {fhi:.3f}] px")
        if "physical" in stats:
            mm = stats["physical"]["fwhm_mm"]
            ml, mh = mm["ci_t"]
            print(f"  fwhm_mm             : median={mm['median']:.4f} "
                  f"mean={mm['mean']:.4f} "
                  f"{ci_level:.0f}% CI=[{ml:.4f}, {mh:.4f}] mm")


def save_wire_width_statistics(stats: dict, path: str | Path) -> None:
    """Serialize to JSON (arrays -> lists, tuples -> lists)."""
    def _norm(obj):
        if isinstance(obj, np.ndarray):
            return [None if (isinstance(x, float) and np.isnan(x)) else float(x)
                    for x in obj.tolist()]
        if isinstance(obj, tuple):
            return [_norm(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _norm(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_norm(x) for x in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    Path(path).write_text(json.dumps(_norm(stats), indent=2))


def plot_wire_width_statistics(
    stats: dict,
    *,
    save_path: str | Path | None = None,
) -> "plt.Figure":
    """Plot per-segment FWHM with median + CI band."""
    fwhm_seg = stats["segments"]["fwhm_px"]
    n = len(fwhm_seg)
    idx = np.arange(n)
    valid = ~np.isnan(fwhm_seg)

    agg = stats["aggregate"]["fwhm_px"]
    g_fwhm = stats["global"]["fwhm_px"]
    g_ok = stats["global"]["model_ok"]
    ci_lo, ci_hi = agg["ci_t"]
    ci_level = (1.0 - stats["alpha"]) * 100.0

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(idx[valid], fwhm_seg[valid], s=55, color="C0",
               label="segment estimate", zorder=3)
    if (~valid).any():
        y_fail = agg["median"] if not np.isnan(agg["median"]) else 0.0
        ax.scatter(idx[~valid], np.full((~valid).sum(), y_fail),
                   marker="x", color="red", s=80, label="failed", zorder=3)

    if g_ok:
        ax.axhline(g_fwhm, color="black", lw=1.4,
                   label=f"global = {g_fwhm:.2f} px")
    if not np.isnan(agg["median"]):
        ax.axhline(agg["median"], color="C0", lw=1.0, linestyle="--",
                   label=f"median = {agg['median']:.2f} px")
    if not np.isnan(ci_lo):
        ax.axhspan(ci_lo, ci_hi, color="C0", alpha=0.15,
                   label=f"{ci_level:.0f}% CI for mean = [{ci_lo:.2f}, {ci_hi:.2f}]")

    ax.set_xlabel("segment index (top -> bottom)")
    ax.set_ylabel("FWHM (px)")
    ax.set_title("Wire-shadow FWHM across image segments")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=120)
        plt.close(fig)
    return fig
