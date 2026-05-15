"""
Fit-quality diagnostics: how well does the detected period explain the data?

Three R^2 metrics are reported at the detected period:

  Fourier R^2 (k=1):        free cosine amplitude at the fundamental only.
  Fourier R^2 (k=n_harm):   free cos+sin amplitudes at k=1..n_harmonics.
                            Captures arbitrary periodic shape via a
                            truncated Fourier series.
  Gaussian-comb R^2:        parametric template Σ exp(-(x-x_k)^2/2σ^2),
                            with x_k, T, σ all *fixed* by the detection
                            (radial FFT + phase_fit + estimate_wire_width).
                            Only amplitude and baseline are fit by LS.
                            This is an end-to-end consistency check on
                            (period, phase, width) jointly.

Reads `broadband_signal_1d` from detect_out (NOT `dominant_signal_1d`,
which is Gabor-band-passed and contains no harmonics by construction).

Public API:
- sinusoidal_fit_r2:        free Fourier R^2 at a given period.
- gaussian_comb_r2:         Gaussian-comb R^2 given T, phase, sigma.
- frequency_concentration:  fraction of power in a narrow band around 1/period.
- fit_quality_report:       all three R^2 + FC + curve, JSON-able dict.
- fit_quality_curve:        Fourier R^2 as a function of candidate period.
- print_fit_quality, save_fit_quality, plot_fit_quality_curve.
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


def _gaussian_comb_template(
    length: int,
    period_px: float,
    phase: float,
    sigma_px: float,
    *,
    truncate_sigmas: float = 5.0,
) -> np.ndarray:
    """Unit-amplitude Gaussian comb centered on cos(ωx + phase) maxima.

    Each Gaussian has peak 1.0; the caller fits amplitude + baseline.
    """
    omega = 2.0 * np.pi / float(period_px)
    x0 = -float(phase) / omega
    T = float(period_px)
    pad = truncate_sigmas * float(sigma_px)
    k_min = int(np.floor((-pad - x0) / T))
    k_max = int(np.ceil((length + pad - x0) / T))
    x = np.arange(length, dtype=np.float64)
    centers = x0 + np.arange(k_min, k_max + 1, dtype=np.float64) * T
    diffs = x[:, None] - centers[None, :]
    return np.sum(np.exp(-(diffs ** 2) / (2.0 * sigma_px ** 2)), axis=1)


def gaussian_comb_r2(
    signal_1d: np.ndarray,
    period_px: float,
    phase: float,
    sigma_px: float,
) -> float:
    """R^2 of a Gaussian-comb template fit:  ŝ(x) = A · template + b.

    The template's positions x_k, period T and width σ are all *fixed*
    by the detection. Only amplitude A and baseline b are LS-fit. Low
    R^2 means at least one of (T, phase, σ) is inconsistent with the
    signal, or the wire shadow isn't actually Gaussian-shaped.
    """
    s = np.asarray(signal_1d, dtype=np.float64)
    n = s.size
    if n < 4 or period_px <= 0 or not np.isfinite(sigma_px) or sigma_px <= 0:
        return float("nan")
    template = _gaussian_comb_template(n, period_px, phase, sigma_px)
    X = np.stack([template, np.ones(n)], axis=1)
    coef, *_ = np.linalg.lstsq(X, s, rcond=None)
    s_fit = X @ coef
    ss_res = float(np.sum((s - s_fit) ** 2))
    ss_tot = float(np.sum((s - s.mean()) ** 2))
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
    n_harmonics: int = 4,
    band_frac: float = 0.2,
    curve_periods: list[float] | np.ndarray | None = None,
) -> dict:
    """
    Combined fit-quality report at the dominant period, plus optional
    R^2 curve over `curve_periods` (default: 4..80 px integer grid).

    Reads `broadband_signal_1d` from detect_out (the Gabor-cleaned
    `dominant_signal_1d` is narrow-band and has no harmonic content).

    Returns a JSON-able dict including:
        r2_fundamental_only:   Fourier R^2 with k=1.
        r2_with_harmonics:     Fourier R^2 with k=n_harmonics.
        r2_gaussian_comb:      parametric Gaussian-comb R^2 (uses detected
                               phase + wire_sigma_px). NaN if detect_out
                               lacks wire-width info.
        loss:                  1 - r2_with_harmonics.

    The gap (r2_with_harmonics - r2_gaussian_comb) measures how much the
    actual wire-shadow shape deviates from a Gaussian — small gap means
    Gaussian assumption is good.
    """
    if "broadband_signal_1d" not in detect_out:
        raise KeyError(
            "detect_out missing 'broadband_signal_1d'. "
            "Re-run detect_laid_lines_simple to populate it."
        )
    signal = np.asarray(detect_out["broadband_signal_1d"], dtype=np.float64)
    period = float(detect_out["dominant_period_px"])

    r2_h1 = sinusoidal_fit_r2(signal, period, n_harmonics=1)
    r2_hk = sinusoidal_fit_r2(signal, period, n_harmonics=n_harmonics)
    fc = frequency_concentration(signal, period, band_frac=band_frac)

    phase = detect_out.get("phase")
    sigma = detect_out.get("wire_sigma_px")
    wire_ok = bool(detect_out.get("wire_model_ok", False))
    if phase is not None and sigma is not None and wire_ok and np.isfinite(sigma):
        r2_gauss = gaussian_comb_r2(signal, period, float(phase), float(sigma))
    else:
        r2_gauss = float("nan")

    if curve_periods is None:
        curve_periods = list(range(4, 81))
    curve_periods = np.asarray(curve_periods, dtype=np.float64)
    curve_r2 = fit_quality_curve(signal, curve_periods, n_harmonics=n_harmonics)

    best_idx = int(np.argmax(curve_r2))
    best_period_by_r2 = float(curve_periods[best_idx])
    best_r2 = float(curve_r2[best_idx])

    return {
        "period_px_used": period,
        "n_harmonics": int(n_harmonics),
        "fc_band_frac": float(band_frac),
        "r2_fundamental_only": r2_h1,
        "r2_with_harmonics": r2_hk,
        "r2_gaussian_comb": r2_gauss,
        "loss": 1.0 - r2_hk,
        "frequency_concentration": fc,
        "curve_periods_px": [float(p) for p in curve_periods],
        "curve_r2": [float(v) for v in curve_r2],
        "best_period_by_r2": best_period_by_r2,
        "best_r2": best_r2,
        "agrees_with_dominant": bool(abs(best_period_by_r2 - period) <= 1.0),
    }


def print_fit_quality(report: dict) -> None:
    """Readable summary lines."""
    print("[Eval] Fit quality (R^2 against broadband 1D signal)")
    print(f"  at period={report['period_px_used']:.2f} px"
          f" | Fourier R^2 (k=1)={report['r2_fundamental_only']:.4f}"
          f" | Fourier R^2 (k={report['n_harmonics']})={report['r2_with_harmonics']:.4f}"
          f" | Gauss-comb R^2={report['r2_gaussian_comb']:.4f}")
    gap_h = report['r2_with_harmonics'] - report['r2_fundamental_only']
    gap_g = report['r2_with_harmonics'] - report['r2_gaussian_comb']
    print(f"  harmonics contribute +{gap_h:+.4f}; Gaussian-fit gap = {gap_g:+.4f}"
          f" (positive => signal has non-Gaussian shape)")
    print(f"  FC (band ±{report['fc_band_frac']*100:.0f}%)={report['frequency_concentration']:.4f}")
    print(f"  argmax Fourier R^2 over [{int(min(report['curve_periods_px']))}, "
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
    r2_gauss = report.get("r2_gaussian_comb", float("nan"))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(periods, r2, color="steelblue", linewidth=1.5,
            label=f"Fourier R^2 (k=1..{report['n_harmonics']})")
    ax.axvline(detected, color="red", linestyle="--", linewidth=1.2,
               label=f"detected = {detected:.1f} px")
    ax.axvline(best, color="green", linestyle=":", linewidth=1.2,
               label=f"argmax R^2 = {best:.1f} px")
    if np.isfinite(r2_gauss):
        ax.axhline(r2_gauss, color="orange", linewidth=1.5,
                   label=f"Gauss-comb R^2 = {r2_gauss:.3f}")
    ax.set_xlabel("Candidate period (px)")
    ax.set_ylabel("R^2")
    ax.set_title(f"Fit quality curve  (loss at detected = {report['loss']:.4f})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
