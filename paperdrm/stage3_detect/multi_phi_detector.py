"""
Multi-phi laid-line detector.

Uses N grazing images (one per phi-azimuth) from the DRP stack instead of
a single image. Each per-image radial power spectrum is normalised then
summed: the laid-line peak sits at the same frequency in every image and
adds coherently, while phi-random structure (ink, paper texture noise)
averages down. Period is read off the aggregated spectrum; phase is the
amplitude-weighted circular mean of per-image phases at the locked period.

Compared to the single-image SIMPLE track in `simple_detector.py`, this
keeps the same downstream contract (returns the same keys + extras) but
trades single-frame brittleness for an SNR boost that scales with the
number of independent phi observations.

Public API:
    collect_grazing_per_phi    -- pick one grazing image per phi from a Pack
    aggregate_radial_power     -- per-image FFT -> normalised power sum
    detect_laid_lines_multi_phi -- end-to-end
"""

from __future__ import annotations

import numpy as np

from paperdrm.stage3_detect.simple_detector import (
    _broadband_signal_1d,
    _rotate_to_vertical,
    gabor_clean_signal,
    grid_positions,
)
from paperdrm.stage3_detect.wire_width import estimate_wire_width


def collect_grazing_per_phi(pack) -> tuple[list[np.ndarray], np.ndarray]:
    """
    For each phi index, return the bg-subtracted image at the steepest
    theta available in the filtered stack.

    Returns:
        images: list of N 2D arrays where N = pack.param.ph_num.
        phi_deg: 1D array of phi values in degrees, length N.
    """
    th_num = int(pack.param.th_num)
    ph_num = int(pack.param.ph_num)
    images: list[np.ndarray] = []
    for phi_idx in range(ph_num):
        idx = phi_idx * th_num + (th_num - 1)
        images.append(pack.images[idx])
    phi_deg = np.linspace(
        float(pack.param.ph_min),
        float(pack.param.ph_max),
        ph_num,
        endpoint=True,
    )
    return images, phi_deg


def _power_spectrum_along_normal(
    image: np.ndarray,
    line_dir_deg: float,
    period_range_px: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Single-image radial power profile restricted to the search band."""
    img = _rotate_to_vertical(image, line_dir_deg).astype(np.float64)
    img = img - img.mean()
    F = np.fft.fftshift(np.fft.fft2(img))
    P = np.abs(F) ** 2
    prof = P.sum(axis=0)
    W = img.shape[1]
    freqs = np.fft.fftshift(np.fft.fftfreq(W, d=1.0))
    pos = freqs > 0
    freqs_pos = freqs[pos]
    prof_pos = prof[pos]
    f_min = 1.0 / float(period_range_px[1])
    f_max = 1.0 / float(period_range_px[0])
    mask = (freqs_pos >= f_min) & (freqs_pos <= f_max)
    return freqs_pos[mask], prof_pos[mask]


def aggregate_radial_power(
    images: list[np.ndarray],
    *,
    line_dir_deg: float = 90.0,
    period_range_px: tuple[float, float] = (8.0, 80.0),
    normalize: str = "sum",
) -> dict:
    """
    Sum normalised radial power spectra across a list of images.

    Per-image normalisation modes:
      - "sum" : divide by total in-band power (default; equalises images).
      - "max" : divide by peak power (each image contributes a unit peak).
      - "none": no normalisation (bright images dominate).
    """
    per_image: list[np.ndarray] = []
    band_freqs: np.ndarray | None = None
    for img in images:
        f, P = _power_spectrum_along_normal(img, line_dir_deg, period_range_px)
        if band_freqs is None:
            band_freqs = f
        elif f.shape != band_freqs.shape:
            raise ValueError("images differ in width; cannot aggregate spectra")
        if normalize == "sum":
            denom = float(P.sum()) + 1e-30
        elif normalize == "max":
            denom = float(P.max()) + 1e-30
        elif normalize == "none":
            denom = 1.0
        else:
            raise ValueError(f"unknown normalize={normalize!r}")
        per_image.append(P / denom)
    per_arr = np.stack(per_image, axis=0)  # (N, K)
    return {
        "freqs": band_freqs,
        "power_agg": per_arr.sum(axis=0),
        "power_per_image": per_arr,
    }


def detect_laid_lines_multi_phi(
    images: list[np.ndarray],
    *,
    line_dir_deg: float = 90.0,
    period_range_px: tuple[float, float] = (8.0, 80.0),
    wire_is_darker: bool = True,
    representative_index: int | None = None,
    use_gabor_refinement: bool = True,
    gabor_ksize: int | None = None,
    normalize: str = "sum",
) -> dict:
    """
    End-to-end multi-phi laid-line detection.

    Args:
        images: list of grayscale images, one per phi (grazing theta).
        line_dir_deg: laid-line direction (90 = vertical).
        period_range_px: (low, high) period search band in pixels.
        wire_is_darker: True => phase fit lands grid on signal minima.
        representative_index: image index used for the returned
            broadband / Gabor-cleaned signals and wire-width estimate.
            Default = the image with maximum power at the detected peak
            (highest individual SNR contributor).
        use_gabor_refinement: same meaning as in `detect_laid_lines_simple`.
        normalize: per-image power normalisation mode.

    Returns a dict with the same keys as `detect_laid_lines_simple` plus:
        n_images, representative_index,
        per_image_phase_rad, per_image_weight,
        phase_circular_var, phase_resultant_length,
        power_per_image, normalize.
    """
    if len(images) < 2:
        raise ValueError(f"need >= 2 images for multi-phi detection; got {len(images)}.")

    agg = aggregate_radial_power(
        images,
        line_dir_deg=line_dir_deg,
        period_range_px=period_range_px,
        normalize=normalize,
    )
    freqs = agg["freqs"]
    P_agg = agg["power_agg"]
    P_per = agg["power_per_image"]

    peak_idx = int(np.argmax(P_agg))
    peak_f = float(freqs[peak_idx])
    period_px = 1.0 / peak_f
    omega = 2.0 * np.pi / period_px

    phases = np.zeros(len(images), dtype=np.float64)
    weights = np.maximum(P_per[:, peak_idx].astype(np.float64), 0.0)
    for i, img in enumerate(images):
        s = _broadband_signal_1d(img, line_dir_deg)
        s = s - s.mean()
        if wire_is_darker:
            s = -s
        x = np.arange(s.size, dtype=np.float64)
        c = float(np.sum(s * np.cos(omega * x)))
        si = float(np.sum(s * np.sin(omega * x)))
        phases[i] = float(np.arctan2(si, c))

    wsum = float(weights.sum()) + 1e-30

    # Coherence before polarity correction (diagnostic).
    cmean_raw = float(np.sum(weights * np.cos(phases)) / wsum)
    smean_raw = float(np.sum(weights * np.sin(phases)) / wsum)
    R_raw = float(np.sqrt(cmean_raw ** 2 + smean_raw ** 2))

    # As phi rotates, the wire shadow can swap which side is darker, so
    # per-image phases can differ by ~pi without indicating a wrong period.
    # Anchor on the highest-weight image and flip any phase whose anchor-
    # relative difference exceeds pi/2 (i.e. closer to anchor+pi than to
    # anchor itself).
    anchor_idx = int(np.argmax(weights))
    anchor_phi = float(phases[anchor_idx])
    delta = ((phases - anchor_phi) + np.pi) % (2.0 * np.pi) - np.pi
    flipped = np.abs(delta) > (0.5 * np.pi)
    phases_aligned = np.where(flipped, phases + np.pi, phases)
    phases_aligned = ((phases_aligned + np.pi) % (2.0 * np.pi)) - np.pi

    cmean = float(np.sum(weights * np.cos(phases_aligned)) / wsum)
    smean = float(np.sum(weights * np.sin(phases_aligned)) / wsum)
    phi_mean = float(np.arctan2(smean, cmean))
    R = float(np.sqrt(cmean ** 2 + smean ** 2))
    circ_var = 1.0 - R
    n_flipped = int(np.sum(flipped))

    if representative_index is None:
        representative_index = int(np.argmax(weights))
    rep_image = images[representative_index]
    broadband_1d = _broadband_signal_1d(rep_image, line_dir_deg)

    if use_gabor_refinement:
        gabor = gabor_clean_signal(
            rep_image,
            period_px=period_px,
            line_dir_deg=line_dir_deg,
            ksize=gabor_ksize,
        )
        signal_1d = np.asarray(gabor["best_signal_1d"])
        gabor_score: float | None = float(gabor["score"])
        gabor_theta: float | None = float(gabor["best_theta_deg"])
    else:
        signal_1d = broadband_1d.astype(np.float32)
        gabor_score = None
        gabor_theta = None

    length = rep_image.shape[1]
    grid_x = grid_positions(phi_mean, period_px, length)

    # Auto-correct half-period phase ambiguity.  The FFT peak determines the
    # period reliably, but the absolute phase can land on either the wire side
    # or the inter-wire side depending on which phi images dominate the weighted
    # mean.  We settle this by sampling the representative image: if the grid
    # positions have higher column-mean intensity than the half-period-shifted
    # positions, the grid is on the bright side — shift by T/2 so it always
    # marks the darker feature (more physically meaningful; independent of
    # lighting azimuth).
    _col = _rotate_to_vertical(rep_image, line_dir_deg).mean(axis=0).astype(np.float64)
    _W = _col.size
    _hw = 1  # ±1 px sampling band around each grid position

    def _band_mean(xs: np.ndarray) -> float:
        vals = [_col[max(0, x - _hw): min(_W, x + _hw + 1)].mean()
                for x in xs if max(0, x - _hw) < min(_W, x + _hw + 1)]
        return float(np.mean(vals)) if vals else float("nan")

    _on_mean = _band_mean(grid_x)
    _off_x = np.clip(np.round(grid_x + period_px / 2.0).astype(int), 0, _W - 1)
    _off_mean = _band_mean(_off_x)

    phase_auto_corrected = False
    if not (np.isnan(_on_mean) or np.isnan(_off_mean)) and _on_mean > _off_mean:
        phi_mean = ((phi_mean + np.pi) + np.pi) % (2.0 * np.pi) - np.pi
        grid_x = grid_positions(phi_mean, period_px, length)
        phase_auto_corrected = True

    width = estimate_wire_width(broadband_1d, period_px)

    return {
        "dominant_period_px": period_px,
        "dominant_freq_cpp": peak_f,
        "dominant_signal_1d": signal_1d,
        "broadband_signal_1d": broadband_1d,
        "grid_positions_x": grid_x,
        "phase": phi_mean,
        "phase_auto_corrected": phase_auto_corrected,
        "line_dir_deg": float(line_dir_deg),
        "wire_is_darker": bool(wire_is_darker),
        "radial_freqs": freqs,
        "radial_power": P_agg,
        "gabor_score": gabor_score,
        "gabor_theta_deg": gabor_theta,
        "wire_sigma_px": width["sigma_px"],
        "wire_fwhm_px": width["fwhm_px"],
        "wire_harmonic_orders": width["harmonic_orders"],
        "wire_harmonic_amplitudes": width["harmonic_amplitudes"],
        "wire_regression_slope": width["regression_slope"],
        "wire_regression_residuals": width["regression_residuals"],
        "wire_model_ok": width["model_ok"],
        "wire_warning": width["warning"],
        "n_images": int(len(images)),
        "representative_index": int(representative_index),
        "per_image_phase_rad": phases,
        "per_image_phase_aligned_rad": phases_aligned,
        "per_image_weight": weights,
        "per_image_polarity_flipped": flipped.astype(bool),
        "anchor_index": int(anchor_idx),
        "n_polarity_flipped": int(n_flipped),
        "phase_circular_var": circ_var,
        "phase_resultant_length": R,
        "phase_resultant_length_raw": R_raw,
        "power_per_image": P_per,
        "normalize": str(normalize),
    }
