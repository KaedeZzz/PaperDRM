"""
Fit-quality diagnostics: how well does the detected period explain the data?

This module produces an ML-style "loss" for laid-line detection without
requiring ground-truth annotations. The primary metric is the coefficient
of determination R^2 of a low-order sinusoidal fit to the 1D projected
signal at the detected period.

Public API:
- sinusoidal_fit_r2:   R^2 at a given period (the "1-loss")
- frequency_concentration: fraction of power in a narrow band around 1/period
- fit_quality_report:  R^2 + FC at the consensus period (JSON-able dict)
- fit_quality_curve:   R^2 as a function of candidate period
- print_fit_quality, save_fit_quality, plot_fit_quality_curve
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def _design_matrix(x: np.ndarray, period_px: float, n_harmonics: int) -> np.ndarray:
    """Stack cos+sin pairs at the fundamental and (n_harmonics-1) higher harmonics."""
    cols = []
    for k in range(1, n_harmonics + 1):
        w = 2.0 * np.pi * k / float(period_px)
        cols.append(np.cos(w * x))
        cols.append(np.sin(w * x))
    return np.stack(cols, axis=1)


def sinusoidal_fit_r2(signal_1d: np.ndarray, period_px: float, n_harmonics: int = 2) -> float:
    """
    R^2 of a least-squares sinusoidal fit at `period_px` (with `n_harmonics`).

    R^2 in [0, 1]: 1.0 means the period perfectly explains the signal, 0 means
    no better than the mean. The "loss" is (1 - R^2).
    """
    s = np.asarray(signal_1d, dtype=np.float64)
    s = s - s.mean()
    n = s.size
    if n < 2 * n_harmonics + 1 or period_px <= 0:
        return float("nan")
    x = np.arange(n, dtype=np.float64)
    X = _design_matrix(x, period_px, n_harmonics)
    coef, *_ = np.linalg.lstsq(X, s, rcond=None)
    s_fit = X @ coef
    ss_res = float(np.sum((s - s_fit) ** 2))
    ss_tot = float(np.sum(s ** 2))
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def frequency_concentration(signal_1d: np.ndarray, period_px: float, band_frac: float = 0.2) -> float:
    """
    Fraction of 1D signal power inside [(1-band_frac)*f0, (1+band_frac)*f0]
    where f0 = 1 / period_px. Mirrors gabor._score_period but used as a
    standalone fit-quality measure.
    """
    s = np.asarray(signal_1d, dtype=np.float64) - np.mean(signal_1d)
    n = s.size
    if n < 8 or period_px <= 0:
        return float("nan")
    spectrum = np.fft.rfft(s)
    freqs = np.fft.rfftfreq(n, d=1.0)
    f0 = 1.0 / float(period_px)
    band = (freqs >= (1.0 - band_frac) * f0) & (freqs <= (1.0 + band_frac) * f0)
    band_energy = float(np.sum(np.abs(spectrum[band]) ** 2))
    total_energy = float(np.sum(np.abs(spectrum) ** 2)) + 1e-12
    return band_energy / total_energy


def fit_quality_curve(
    signal_1d: np.ndarray,
    periods_px: list[float] | np.ndarray,
    n_harmonics: int = 2,
) -> np.ndarray:
    """R^2 for each candidate period in `periods_px`. The 'loss curve'."""
    return np.array(
        [sinusoidal_fit_r2(signal_1d, float(p), n_harmonics) for p in periods_px],
        dtype=np.float64,
    )


def fit_quality_report(
    detect_out: dict,
    *,
    n_harmonics: int = 2,
    band_frac: float = 0.2,
    curve_periods: list[float] | np.ndarray | None = None,
) -> dict:
    """
    Combined fit-quality report at the dominant period, plus optional
    R^2 curve over `curve_periods` (default: 4..80 px integer grid).

    Returns a JSON-able dict.
    """
    signal = np.asarray(detect_out["dominant_signal_1d"], dtype=np.float64)
    period = float(detect_out["dominant_period_px"])

    r2_h1 = sinusoidal_fit_r2(signal, period, n_harmonics=1)
    r2_h2 = sinusoidal_fit_r2(signal, period, n_harmonics=n_harmonics)
    fc = frequency_concentration(signal, period, band_frac=band_frac)

    if curve_periods is None:
        curve_periods = list(range(4, 81))
    curve_periods = np.asarray(curve_periods, dtype=np.float64)
    curve_r2 = fit_quality_curve(signal, curve_periods, n_harmonics=n_harmonics)

    # Did the global best R^2 land at the detected period?
    best_idx = int(np.argmax(curve_r2))
    best_period_by_r2 = float(curve_periods[best_idx])
    best_r2 = float(curve_r2[best_idx])

    return {
        "period_px_used": period,
        "n_harmonics": int(n_harmonics),
        "fc_band_frac": float(band_frac),
        "r2_fundamental_only": r2_h1,
        "r2_with_harmonics": r2_h2,
        "loss": 1.0 - r2_h2,
        "frequency_concentration": fc,
        "curve_periods_px": [float(p) for p in curve_periods],
        "curve_r2": [float(v) for v in curve_r2],
        "best_period_by_r2": best_period_by_r2,
        "best_r2": best_r2,
        "agrees_with_dominant": bool(abs(best_period_by_r2 - period) <= 1.0),
    }


def print_fit_quality(report: dict) -> None:
    """Readable summary line."""
    print("[Eval] Fit quality (sinusoidal-model R^2 / 'loss')")
    print(f"  at period={report['period_px_used']:.1f} px"
          f" | R^2 (k=1)={report['r2_fundamental_only']:.4f}"
          f" | R^2 (k={report['n_harmonics']})={report['r2_with_harmonics']:.4f}"
          f" | loss=1-R^2={report['loss']:.4f}")
    print(f"  FC (band ±{report['fc_band_frac']*100:.0f}%)={report['frequency_concentration']:.4f}")
    print(f"  argmax R^2 over [{int(min(report['curve_periods_px']))}, "
          f"{int(max(report['curve_periods_px']))}] px = {report['best_period_by_r2']:.1f} px"
          f" (R^2={report['best_r2']:.4f})"
          f" {'<- matches detected' if report['agrees_with_dominant'] else '<- DISAGREES with detected'}")


def save_fit_quality(report: dict, path: str | Path) -> None:
    """Write the report dict to JSON."""
    Path(path).write_text(json.dumps(report, indent=2))


def plot_fit_quality_curve(report: dict) -> None:
    """Plot R^2 vs candidate period; the 'loss curve' for laid-line detection."""
    periods = np.asarray(report["curve_periods_px"], dtype=np.float64)
    r2 = np.asarray(report["curve_r2"], dtype=np.float64)
    detected = report["period_px_used"]
    best = report["best_period_by_r2"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(periods, r2, color="steelblue", linewidth=1.5)
    ax.axvline(detected, color="red", linestyle="--", linewidth=1.2,
               label=f"detected = {detected:.1f} px")
    ax.axvline(best, color="green", linestyle=":", linewidth=1.2,
               label=f"argmax R^2 = {best:.1f} px")
    ax.set_xlabel("Candidate period (px)")
    ax.set_ylabel(f"R^2 (sinusoidal fit, k=1..{report['n_harmonics']})")
    ax.set_title(f"Fit quality curve  (loss at detected = {report['loss']:.4f})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
