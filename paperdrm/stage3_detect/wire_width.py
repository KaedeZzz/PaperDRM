"""
Wire-shadow width from harmonic amplitudes of a periodic signal.

Models each laid-line shadow as a Gaussian dip of stddev sigma. A periodic
comb of such Gaussians at spacing T has Fourier coefficients

    |c_n| ~ exp(-2 pi^2 sigma^2 n^2 / T^2)

so log-amplitudes decay linearly in n^2 and sigma is recovered from a
linear fit of ln|c_n| against (n^2 - 1). Slope is -2 pi^2 sigma^2 / T^2.

The DTFT is sampled directly at harmonic frequencies n/T to avoid spectral
leakage from reading FFT bins at non-integer 1/T.

NOTE: input signal must be broadband (column-mean of the rotated image with
high-pass detrend). Do NOT pass a Gabor-filtered signal; the Gabor pass-band
is narrow around the fundamental and would attenuate the harmonics that
carry the width information.
"""

from __future__ import annotations

import numpy as np


_FWHM_OVER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ~ 2.3548


def _dtft_at(signal: np.ndarray, frequencies_cpp: np.ndarray) -> np.ndarray:
    """Sample DTFT of `signal` at given normalized frequencies (cycles/pixel)."""
    x = np.arange(signal.size, dtype=np.float64)
    phases = 2.0 * np.pi * frequencies_cpp[:, None] * x[None, :]
    basis = np.exp(-1j * phases)
    return basis @ signal


def estimate_wire_width(
    signal_1d: np.ndarray,
    period_px: float,
    *,
    n_max: int = 4,
) -> dict:
    """
    Estimate Gaussian wire-shadow width from harmonic amplitude decay.

    Args:
        signal_1d: 1D periodic signal (broadband, detrended). Length N.
        period_px: detected period T in pixels.
        n_max: highest harmonic order to use; must be >= 2. Harmonics with
            n/T >= 0.5 (above Nyquist) are dropped.

    Returns dict with:
        sigma_px:              Gaussian stddev (pixels), or NaN on failure.
        fwhm_px:               FWHM = 2 sqrt(2 ln 2) sigma ~ 2.355 sigma.
        harmonic_orders:       [1, 2, ..., n_max].
        harmonic_amplitudes:   |c_n| at each harmonic, NaN above Nyquist.
        regression_slope:      -2 pi^2 sigma^2 / T^2 estimate; NaN on failure.
        regression_residuals:  residuals of ln(|c_n|/|c_1|) vs (n^2 - 1) fit;
                               NaN for n=1 (reference) and above Nyquist.
        model_ok:              False if Gaussian model is inconsistent.
        warning:               str describing soft/hard issues, or None.
    """
    if n_max < 2:
        raise ValueError("n_max must be >= 2.")
    s = np.asarray(signal_1d, dtype=np.float64)
    s = s - s.mean()
    T = float(period_px)
    if not (np.isfinite(T) and T > 0):
        raise ValueError(f"period_px must be positive and finite, got {period_px}.")

    orders = np.arange(1, n_max + 1, dtype=np.int64)
    freqs = orders.astype(np.float64) / T
    above_nyquist = freqs >= 0.5

    if above_nyquist[0] or above_nyquist[1]:
        return _failed_result(
            orders, n_max,
            "first two harmonics above Nyquist; period_px too small",
        )

    used = ~above_nyquist
    amps = np.full(n_max, np.nan, dtype=np.float64)
    amps[used] = np.abs(_dtft_at(s, freqs[used]))

    c1 = amps[0]
    if not np.isfinite(c1) or c1 <= 0:
        return _failed_result(orders, n_max, "fundamental amplitude is zero", amps=amps)

    rho_used = amps[used] / c1
    if np.any(rho_used[1:] >= 1.0):
        return _failed_result(
            orders, n_max,
            "non-monotonic harmonics (|c_n| >= |c_1| for some n>1); Gaussian model fails",
            amps=amps,
        )

    y = np.log(rho_used)                                 # 0 at n=1
    x = orders[used].astype(np.float64) ** 2 - 1.0       # 0 at n=1
    pos = x > 0
    if not np.any(pos):
        return _failed_result(orders, n_max, "no harmonics above fundamental", amps=amps)
    # Weighted through-origin LS. Weight w_n = rho_n^2 is a proxy for
    # SNR^2: in log-amplitude space, var(ln|c_n|) scales like 1/|c_n|^2,
    # so inverse-variance weighting is proportional to |c_n|^2. This down-
    # weights high-n bins where the harmonic has already decayed into noise.
    w = rho_used[pos] ** 2
    slope = float(np.sum(w * x[pos] * y[pos]) / np.sum(w * x[pos] ** 2))
    if slope >= 0:
        return _failed_result(
            orders, n_max,
            "regression slope non-negative; Gaussian model fails",
            amps=amps,
        )

    sigma_sq = -slope * (T ** 2) / (2.0 * np.pi ** 2)
    sigma_px = float(np.sqrt(sigma_sq))
    fwhm_px = float(_FWHM_OVER_SIGMA * sigma_px)

    residuals = np.full(n_max, np.nan, dtype=np.float64)
    residuals_used = y - slope * x
    residuals_used[~pos] = np.nan
    residuals[used] = residuals_used

    warning: str | None = None
    if sigma_px >= T / 2.0:
        warning = f"sigma ({sigma_px:.2f} px) >= T/2 ({T/2:.2f} px); physically implausible"
    elif sigma_px < 1.0:
        warning = f"sigma ({sigma_px:.2f} px) < 1 px; below image resolution"

    return {
        "sigma_px": sigma_px,
        "fwhm_px": fwhm_px,
        "harmonic_orders": orders.tolist(),
        "harmonic_amplitudes": amps,
        "regression_slope": slope,
        "regression_residuals": residuals,
        "model_ok": True,
        "warning": warning,
    }


def _failed_result(
    orders: np.ndarray,
    n_max: int,
    reason: str,
    *,
    amps: np.ndarray | None = None,
) -> dict:
    if amps is None:
        amps = np.full(n_max, np.nan, dtype=np.float64)
    return {
        "sigma_px": float("nan"),
        "fwhm_px": float("nan"),
        "harmonic_orders": orders.tolist(),
        "harmonic_amplitudes": amps,
        "regression_slope": float("nan"),
        "regression_residuals": np.full(n_max, np.nan, dtype=np.float64),
        "model_ok": False,
        "warning": reason,
    }
