"""
Laid-line interval (gap) distribution analysis.

Catalogers traditionally describe a paper sample by the distribution of
*intervals between adjacent laid lines*, not by a single global period.
This module extracts those intervals from the 1D projected signal produced
by the Gabor stage and reports a standard set of descriptive parameters.

Public API:
- gap_distribution_from_signal: compute stats dict from a 1D signal + period
- print_gap_distribution:       pretty-print the stats to stdout
- save_gap_distribution:        write the stats to JSON
- plot_gap_distribution:        histogram of gaps with central-tendency marks
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as sp_stats

from paperdrm.stage3_detect.gabor import peaks_from_signal


def gap_distribution_from_signal(
    signal_1d: np.ndarray,
    period_px: float,
    *,
    fov_width_cm: float | None = None,
    image_width_px: int | None = None,
) -> dict:
    """
    Detect laid-line peaks in the projected 1D signal and return descriptive
    statistics for the distribution of intervals between adjacent peaks.

    Required:
        signal_1d:  the 1D projection along the laid-line direction (rotated
                    so lines are vertical and the signal varies across rows
                    after row-averaging -- i.e. what gabor returns as
                    dominant_signal_1d).
        period_px:  the consensus period (used by peaks_from_signal for its
                    minimum-distance and smoothing windows).

    Optional (for physical units in cm and lines/cm):
        fov_width_cm:    width of the imaged field of view in cm.
        image_width_px:  pixel width of the image; required if fov is given.

    Returns a JSON-able dict with the gap distribution parameters.
    """
    peaks_x = peaks_from_signal(np.asarray(signal_1d, dtype=np.float32), float(period_px))
    n_peaks = int(peaks_x.size)

    base = {
        "n_peaks": n_peaks,
        "n_gaps": max(0, n_peaks - 1),
        "period_px_used": float(period_px),
    }
    if n_peaks < 2:
        return {**base, "gaps_px": [], "px": _nan_stats(), "physical": None}

    gaps_px = np.diff(peaks_x).astype(np.float64)
    px = _describe(gaps_px)

    physical: dict | None = None
    if fov_width_cm is not None and image_width_px is not None and image_width_px > 0:
        cm_per_px = float(fov_width_cm) / float(image_width_px)
        gaps_cm = gaps_px * cm_per_px
        spectral_interval_cm = float(period_px) * cm_per_px
        gap_iqr_cm = [float(px["iqr"][0] * cm_per_px), float(px["iqr"][1] * cm_per_px)]
        mean_relative_error = float(abs(px["mean"] - period_px) / period_px)
        median_relative_error = float(abs(px["median"] - period_px) / period_px)
        physical = {
            "fov_width_cm": float(fov_width_cm),
            "image_width_px": int(image_width_px),
            "cm_per_px": cm_per_px,
            "spectral_interval_cm": spectral_interval_cm,
            "spectral_lines_per_cm": (
                float(1.0 / spectral_interval_cm)
                if spectral_interval_cm > 0 else float("inf")
            ),
            "mean_interval_cm": float(np.mean(gaps_cm)),
            "median_interval_cm": float(np.median(gaps_cm)),
            "gap_iqr_cm": gap_iqr_cm,
            "std_interval_cm": float(np.std(gaps_cm, ddof=1)) if gaps_cm.size > 1 else 0.0,
            "lines_per_cm_mean": float(1.0 / np.mean(gaps_cm)) if np.mean(gaps_cm) > 0 else float("inf"),
            "lines_per_cm_median": float(1.0 / np.median(gaps_cm)) if np.median(gaps_cm) > 0 else float("inf"),
            "gap_mean_relative_error_vs_spectral": mean_relative_error,
            "gap_median_relative_error_vs_spectral": median_relative_error,
            "gap_median_agrees_with_spectral": bool(median_relative_error <= 0.15),
        }

    return {
        **base,
        "gaps_px": [float(g) for g in gaps_px],
        "px": px,
        "physical": physical,
    }


def _nan_stats() -> dict:
    """Stats template when there aren't enough gaps to compute anything."""
    return {
        "mean": float("nan"),
        "median": float("nan"),
        "mode": float("nan"),
        "std": float("nan"),
        "mad": float("nan"),
        "iqr": [float("nan"), float("nan")],
        "cv": float("nan"),
        "skewness": float("nan"),
        "kurtosis_excess": float("nan"),
        "min": float("nan"),
        "max": float("nan"),
        "range": float("nan"),
    }


def _describe(arr: np.ndarray) -> dict:
    """Standard parameters describing the distribution of `arr` (in pixels)."""
    n = arr.size
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    mad = float(np.median(np.abs(arr - median)))
    q25, q75 = (float(v) for v in np.percentile(arr, [25, 75]))
    cv = float(std / mean) if mean != 0 else float("nan")

    # Mode: peaks_from_signal returns integer x positions so gaps are integers.
    counts = Counter(int(round(v)) for v in arr)
    mode_val = float(max(counts.items(), key=lambda kv: kv[1])[0])

    # Higher moments are undefined for a constant distribution. Handle that
    # explicitly to avoid SciPy precision-loss warnings while preserving NaN.
    is_constant = bool(np.all(arr == arr[0]))
    skewness = (
        float(sp_stats.skew(arr, bias=False))
        if n >= 3 and not is_constant else float("nan")
    )
    kurtosis_excess = (
        float(sp_stats.kurtosis(arr, fisher=True, bias=False))
        if n >= 4 and not is_constant else float("nan")
    )

    return {
        "mean": mean,
        "median": median,
        "mode": mode_val,
        "std": std,
        "mad": mad,
        "iqr": [q25, q75],
        "cv": cv,
        "skewness": skewness,
        "kurtosis_excess": kurtosis_excess,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.max(arr) - np.min(arr)),
    }


def print_gap_distribution(stats: dict) -> None:
    """One-pass readable summary."""
    px = stats["px"]
    print("[Eval] Laid-line interval (gap) distribution")
    print(f"  peaks={stats['n_peaks']}, gaps={stats['n_gaps']}, period_used={stats['period_px_used']:.1f} px")
    if stats["n_gaps"] == 0:
        print("  (no gaps -- need >=2 detected peaks)")
        return
    print(f"  gap_px: mean={px['mean']:.2f} median={px['median']:.1f} mode={px['mode']:.0f}"
          f" | std={px['std']:.2f} mad={px['mad']:.1f}"
          f" | IQR=[{px['iqr'][0]:.1f},{px['iqr'][1]:.1f}]"
          f" | CV={px['cv']:.3f}")
    print(f"  shape : skew={px['skewness']:+.2f} kurt_excess={px['kurtosis_excess']:+.2f}"
          f" | min={px['min']:.0f} max={px['max']:.0f} range={px['range']:.0f}")
    if stats["physical"] is not None:
        ph = stats["physical"]
        print(f"  cm    : spectral_interval={ph['spectral_interval_cm']:.4f} cm"
              f" | spectral_density={ph['spectral_lines_per_cm']:.2f} lines/cm")
        print(f"          local median={ph['median_interval_cm']:.4f} cm"
              f" | IQR=[{ph['gap_iqr_cm'][0]:.4f},{ph['gap_iqr_cm'][1]:.4f}] cm"
              f" | median error vs spectral={ph['gap_median_relative_error_vs_spectral'] * 100:.1f}%")


def save_gap_distribution(stats: dict, path: str | Path) -> None:
    """Write stats to JSON (omits the full gaps_px array for compactness)."""
    payload = {k: v for k, v in stats.items() if k != "gaps_px"}
    payload["gap_histogram"] = _hist_dict(stats.get("gaps_px", []))
    Path(path).write_text(json.dumps(payload, indent=2))


def _hist_dict(gaps: list[float]) -> dict[str, int]:
    """Histogram with integer keys for compactness in JSON."""
    if not gaps:
        return {}
    rounded = [int(round(g)) for g in gaps]
    return {str(k): v for k, v in sorted(Counter(rounded).items())}


def plot_gap_distribution(stats: dict) -> None:
    """Histogram of gaps with mean/median/mode marked; Gaussian fit overlay."""
    gaps = stats.get("gaps_px", [])
    if len(gaps) < 2:
        print("[Eval] plot_gap_distribution: need >=2 gaps to plot.")
        return

    px = stats["px"]
    gaps_arr = np.asarray(gaps, dtype=np.float64)

    # Integer-bin histogram (gaps are integer px from peaks_from_signal).
    int_min = int(np.floor(gaps_arr.min()))
    int_max = int(np.ceil(gaps_arr.max()))
    bins = np.arange(int_min - 0.5, int_max + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(gaps_arr, bins=bins, color="steelblue", edgecolor="white", alpha=0.85,
            density=True, label=f"{stats['n_gaps']} gaps")

    # Gaussian fit overlay
    xs = np.linspace(int_min, int_max, 400)
    g = np.exp(-0.5 * ((xs - px["mean"]) / max(px["std"], 1e-6)) ** 2) / (max(px["std"], 1e-6) * np.sqrt(2 * np.pi))
    ax.plot(xs, g, color="black", linewidth=1.2, label=f"N({px['mean']:.1f}, {px['std']:.2f}²)")

    ax.axvline(px["mean"], color="red", linestyle="--", linewidth=1.2, label=f"mean={px['mean']:.2f}")
    ax.axvline(px["median"], color="green", linestyle=":", linewidth=1.2, label=f"median={px['median']:.1f}")
    ax.axvline(px["mode"], color="orange", linestyle="-.", linewidth=1.2, label=f"mode={px['mode']:.0f}")

    ax.set_xlabel("Laid-line interval (px)")
    ax.set_ylabel("Density")
    cv_text = f"CV={px['cv']:.3f}"
    skew_text = f"skew={px['skewness']:+.2f}"
    ax.set_title(f"Interval distribution  ({cv_text}, {skew_text})")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()
