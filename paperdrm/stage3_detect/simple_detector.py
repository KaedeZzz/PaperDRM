"""
Minimal laid-line detector based on radial FFT period estimation.

Replaces the broken patch-wise Gabor scan in stage3_detect.gabor by using
the maximum-likelihood frequency estimator (the periodogram peak), then
optionally uses a single Gabor pass at the detected period (with
`use_abs_response=False`) to produce a clean 1D signal for phase fitting
and grid placement.

Pipeline:
    radial FFT  ->  dominant period
    Gabor at known period (use_abs=False) -> clean 1D signal
    phase fit (polarity-aware) -> phase
    grid x positions -> overlay

Public API:
    detect_laid_lines_simple    -- end-to-end
    radial_fft_period           -- period only
    gabor_clean_signal          -- clean 1D signal at known period
    phase_fit                   -- polarity-aware phase estimate
    grid_positions              -- grid x coordinates from phase
    overlay_grid                -- draw grid lines on an image
"""

from __future__ import annotations

import cv2
import numpy as np

from paperdrm.stage3_detect.gabor import _best_for_period, _normalize01
from paperdrm.stage3_detect.wire_width import estimate_wire_width


def _rotate_to_vertical(image: np.ndarray, line_dir_deg: float) -> np.ndarray:
    """Rotate so laid lines run vertically (line_dir_deg -> 90deg)."""
    rot_angle = 90.0 - float(line_dir_deg)
    if abs(rot_angle) < 1e-6:
        return image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rot_angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _broadband_signal_1d(image: np.ndarray, line_dir_deg: float) -> np.ndarray:
    """Column-mean of the rotated image with moving-average high-pass detrend.

    This is broadband (contains harmonics 2/T, 3/T, ...) and is required for
    wire-width estimation. The Gabor-cleaned signal is narrow-band and
    unsuitable for that purpose.
    """
    img_rot = _rotate_to_vertical(image, line_dir_deg)
    s = img_rot.astype(np.float64).mean(axis=0)
    win = min(301, max(31, (len(s) // 20) | 1))
    k = np.ones(win, dtype=np.float64) / win
    return s - np.convolve(s, k, mode="same")


def radial_fft_period(
    image: np.ndarray,
    line_dir_deg: float = 90.0,
    *,
    period_range_px: tuple[float, float] = (8.0, 80.0),
) -> dict:
    """
    Find the dominant period by integrating the 2D power spectrum along the
    laid-line direction and locating the peak in the perpendicular axis.

    Returns dict with:
        dominant_period_px
        dominant_freq_cpp
        radial_freqs: 1D array, positive freqs inside the search range
        radial_power: 1D array, P_rad(u) at those freqs
    """
    img = _rotate_to_vertical(image, line_dir_deg).astype(np.float64)
    img = img - img.mean()

    # 2D power spectrum, DC-centred
    F = np.fft.fftshift(np.fft.fft2(img))
    P = np.abs(F) ** 2

    # Lines are vertical (along v); integrate over v to find peak along u.
    prof = P.sum(axis=0)
    W = img.shape[1]
    freqs = np.fft.fftshift(np.fft.fftfreq(W, d=1.0))

    pos = freqs > 0
    freqs_pos = freqs[pos]
    prof_pos = prof[pos]
    f_min = 1.0 / float(period_range_px[1])
    f_max = 1.0 / float(period_range_px[0])
    mask = (freqs_pos >= f_min) & (freqs_pos <= f_max)
    band_freqs = freqs_pos[mask]
    band_prof = prof_pos[mask]

    if band_freqs.size == 0:
        raise ValueError(f"No frequency bins in range {period_range_px}; image too small?")

    peak_idx = int(np.argmax(band_prof))
    peak_f = float(band_freqs[peak_idx])

    return {
        "dominant_period_px": 1.0 / peak_f,
        "dominant_freq_cpp": peak_f,
        "radial_freqs": band_freqs,
        "radial_power": band_prof,
    }


def gabor_clean_signal(
    image: np.ndarray,
    period_px: float,
    line_dir_deg: float = 90.0,
    *,
    ksize: int | None = None,
    angle_jitter_deg: float = 6.0,
    angle_step_deg: float = 2.0,
    sigma_factor: float = 0.6,
    gamma: float = 0.4,
) -> dict:
    """
    Single Gabor filter pass at known period with use_abs_response=False
    (no harmonic doubling). Returns a clean 1D signal suitable for phase
    fitting.

    Returns dict with: score, best_theta_deg, best_signal_1d, best_response.
    """
    if ksize is None:
        # Kernel large enough to span ~1.5 periods; force odd.
        ksize = int(np.ceil(1.5 * period_px))
        if ksize % 2 == 0:
            ksize += 1
    img01 = _normalize01(image)
    return _best_for_period(
        img01,
        line_dir_deg=line_dir_deg,
        period_px=float(period_px),
        angle_jitter_deg=angle_jitter_deg,
        angle_step_deg=angle_step_deg,
        sigma_factor=sigma_factor,
        gamma=gamma,
        ksize=ksize,
        use_abs_response=False,  # KEY: linear, no 2x harmonic
    )


def phase_fit(signal_1d: np.ndarray, period_px: float, *, wire_is_darker: bool = True) -> float:
    """
    Estimate the phase of a cosine of given period in the signal.

    If wire_is_darker=True, the fit is done on -signal so that the cosine
    maxima of the model correspond to signal *minima* (= wire positions in
    a raw bg-subtracted image where wires cast shadows).
    """
    s = np.asarray(signal_1d, dtype=np.float64)
    s = s - s.mean()
    if wire_is_darker:
        s = -s
    n = s.size
    x = np.arange(n, dtype=np.float64)
    omega = 2.0 * np.pi / float(period_px)
    c = float(np.sum(s * np.cos(omega * x)))
    si = float(np.sum(s * np.sin(omega * x)))
    return float(np.arctan2(si, c))


def grid_positions(phase: float, period_px: float, length: int) -> np.ndarray:
    """
    x positions where cos(omega*x + phase) attains +1.

    With phase from phase_fit(wire_is_darker=True), these are wire positions
    in the original-image coordinate frame.
    """
    omega = 2.0 * np.pi / float(period_px)
    x0 = -phase / omega
    k_start = int(np.floor((0.0 - x0) / period_px))
    k_end = int(np.ceil((length - x0) / period_px))
    xs = x0 + period_px * np.arange(k_start, k_end + 1)
    xs = xs[(xs >= 0.0) & (xs < float(length))]
    return np.round(xs).astype(int)


def overlay_grid(
    image: np.ndarray,
    grid_x: np.ndarray,
    *,
    line_dir_deg: float = 90.0,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 1,
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Draw lines along the laid-line direction at grid_x positions and alpha-
    blend onto the image. Returns a 3-channel BGR overlay.

    grid_x are positions in the rotated frame (where lines are vertical).
    Lines are drawn directly in the original frame via the inverse rotation
    matrix — avoids BORDER_REFLECT seam artifacts from double warpAffine.
    """
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    h, w = base.shape[:2]
    overlay = base.copy()

    rot_angle = 90.0 - float(line_dir_deg)
    if abs(rot_angle) > 1e-6:
        # M_inv maps rotated-frame coordinates → original-frame coordinates.
        M_inv = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -rot_angle, 1.0)
        A, t = M_inv[:, :2], M_inv[:, 2]
        extend = float(max(h, w))
        for x_pos in grid_x:
            pts_rot = np.array([[float(x_pos), -extend],
                                [float(x_pos), h - 1.0 + extend]])
            pts_orig = (A @ pts_rot.T).T + t
            p1 = (int(round(pts_orig[0, 0])), int(round(pts_orig[0, 1])))
            p2 = (int(round(pts_orig[1, 0])), int(round(pts_orig[1, 1])))
            cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
    else:
        for x_pos in grid_x:
            cv2.line(overlay, (int(x_pos), 0), (int(x_pos), h - 1), color, thickness, cv2.LINE_AA)

    return cv2.addWeighted(base, 1.0 - alpha, overlay, alpha, 0.0)


def overlay_grid_bands(
    image: np.ndarray,
    grid_x: np.ndarray,
    band_width_px: float,
    *,
    line_dir_deg: float = 90.0,
    color: tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.40,
) -> np.ndarray:
    """
    Draw filled bands of total width `band_width_px` along the laid-line
    direction at each grid_x position, alpha-blended onto the image.

    Unlike `overlay_grid` (1-px lines at wire centers), this renders the
    full estimated wire-shadow extent. Pass FWHM (= 2.355 sigma) for a
    visual match with the perceived dark band.

    grid_x are positions in the rotated frame (lines vertical). Bands are
    drawn as rotated quadrilaterals directly in the original frame via the
    inverse rotation matrix — no double warpAffine, no BORDER_REFLECT seams.
    """
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    h, w = base.shape[:2]
    half = float(band_width_px) / 2.0
    fill = base.copy()

    rot_angle = 90.0 - float(line_dir_deg)
    if abs(rot_angle) > 1e-6:
        M_inv = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -rot_angle, 1.0)
        A, t = M_inv[:, :2], M_inv[:, 2]
        extend = float(max(h, w))
        for x_pos in grid_x:
            # 4 corners of the band in the rotated frame (TL, TR, BR, BL)
            corners_rot = np.array([
                [float(x_pos) - half, -extend],
                [float(x_pos) + half, -extend],
                [float(x_pos) + half, h - 1.0 + extend],
                [float(x_pos) - half, h - 1.0 + extend],
            ])
            corners_orig = (A @ corners_rot.T).T + t
            poly = corners_orig.round().astype(np.int32)
            cv2.fillPoly(fill, [poly], color)
    else:
        for x_pos in grid_x:
            x0 = int(round(float(x_pos) - half))
            x1 = int(round(float(x_pos) + half))
            cv2.rectangle(fill, (x0, 0), (x1, h - 1), color, thickness=cv2.FILLED)

    return cv2.addWeighted(base, 1.0 - alpha, fill, alpha, 0.0)


def auto_detect_line_dir(
    image: np.ndarray,
    *,
    period_range_px: tuple[float, float] = (8.0, 80.0),
    n_angles: int = 180,
    downsample: int = 4,
) -> float:
    """
    Estimate the laid-line direction by scanning 1D projection spectral power.

    For each candidate direction φ the image is rotated so that lines at φ
    become vertical, then the column mean (1D projection perpendicular to φ)
    is computed.  The direction that maximises the peak FFT power in the
    target period range is returned as ``line_dir_deg``.

    Spectral power (not autocorrelation) is used because autocorrelation
    normalised by total power is dominated by broad low-frequency trends and
    confuses second harmonics with the fundamental.  Absolute spectral power
    at the target frequency is ~100-1000× higher at the correct angle than
    at any incorrect angle for typical manuscript images.

    The image is downsampled (default 4×) before processing; period limits
    are scaled accordingly.  Returns ``line_dir_deg`` folded into (−90, 90].
    """
    H, W = image.shape[:2]
    s = downsample

    small = cv2.resize(image, (W // s, H // s), interpolation=cv2.INTER_AREA)
    small = small.astype(np.float32) - float(small.mean())
    Hs, Ws = small.shape

    # Frequency bounds corresponding to the target period range (downsampled px)
    lag_min = max(1, round(period_range_px[0] / s))
    lag_max = min(Ws // 2, round(period_range_px[1] / s))
    freq_lo = 1.0 / lag_max
    freq_hi = 1.0 / lag_min

    cand   = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    scores = np.zeros(n_angles)
    cx, cy = Ws / 2.0, Hs / 2.0

    for i, phi in enumerate(cand):
        # Rotate so lines at φ become vertical; column mean = projection.
        rot_angle = 90.0 - phi
        M = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
        rotated = cv2.warpAffine(small, M, (Ws, Hs),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
        proj = rotated.mean(axis=0).astype(np.float64)
        proj -= proj.mean()

        # Peak spectral power in target frequency band
        F     = np.fft.rfft(proj)
        power = np.abs(F) ** 2
        freqs = np.fft.rfftfreq(len(proj))
        band  = (freqs >= freq_lo) & (freqs <= freq_hi)
        scores[i] = float(power[band].max()) if band.any() else 0.0

    best = float(cand[int(np.argmax(scores))])
    if best > 90.0:
        best -= 180.0
    return best


def detect_laid_lines_simple(
    image: np.ndarray,
    line_dir_deg: float = 90.0,
    *,
    period_range_px: tuple[float, float] = (8.0, 80.0),
    wire_is_darker: bool = True,
    use_gabor_refinement: bool = True,
    gabor_ksize: int | None = None,
) -> dict:
    """
    End-to-end simple detector. Single-image, no patches.

    Args:
        image: 2D grayscale, ideally bg-subtracted. wire = darker than
            surroundings if wire_is_darker=True (true for raw reflective
            grazing images).
        line_dir_deg: laid-line direction (90 = vertical).
        period_range_px: plausible period search range (low, high).
        wire_is_darker: True => phase fit lands grid on signal minima
            (wire). False => grid on signal maxima.
        use_gabor_refinement: True => single Gabor pass for clean 1D signal.
            False => use plain column-mean (less clean but faster).
        gabor_ksize: kernel size override. Default = ceil(1.5 * period).

    Returns a dict with:
        dominant_period_px, dominant_freq_cpp     -- from radial FFT
        dominant_signal_1d                         -- clean 1D signal (Gabor
                                                      if refinement, else
                                                      broadband)
        broadband_signal_1d                        -- column-mean + high-pass,
                                                      used for wire width
        grid_positions_x                            -- wire x positions
        phase                                       -- fitted cosine phase
        line_dir_deg, wire_is_darker               -- echoed
        radial_freqs, radial_power                  -- for diagnostic plots
        gabor_score, gabor_theta_deg                -- if refinement used
        wire_sigma_px, wire_fwhm_px                 -- Gaussian wire width
        wire_harmonic_orders, wire_harmonic_amplitudes
        wire_regression_slope, wire_regression_residuals
        wire_model_ok, wire_warning                 -- Gaussian-fit diagnostics
    """
    fft_result = radial_fft_period(image, line_dir_deg=line_dir_deg,
                                   period_range_px=period_range_px)
    period_px = fft_result["dominant_period_px"]

    broadband_1d = _broadband_signal_1d(image, line_dir_deg)

    if use_gabor_refinement:
        gabor_result = gabor_clean_signal(
            image, period_px=period_px, line_dir_deg=line_dir_deg,
            ksize=gabor_ksize,
        )
        signal_1d = np.asarray(gabor_result["best_signal_1d"])
        gabor_score = float(gabor_result["score"])
        gabor_theta = float(gabor_result["best_theta_deg"])
    else:
        signal_1d = broadband_1d.astype(np.float32)
        gabor_score = None
        gabor_theta = None

    phi = phase_fit(signal_1d, period_px, wire_is_darker=wire_is_darker)

    # Grid is generated in the rotated frame where lines are vertical.
    # image.shape[1] is the width of the original; after rotation the
    # vertical-line frame has the same width up to BORDER_REFLECT effects.
    length = image.shape[1]
    grid_x = grid_positions(phi, period_px, length)

    # Wire width from harmonic amplitudes of the broadband signal.
    # Gabor signal is narrow-band -> harmonics suppressed -> unusable here.
    width = estimate_wire_width(broadband_1d, period_px)

    return {
        "dominant_period_px": period_px,
        "dominant_freq_cpp": fft_result["dominant_freq_cpp"],
        "dominant_signal_1d": signal_1d,
        "broadband_signal_1d": broadband_1d,
        "grid_positions_x": grid_x,
        "phase": phi,
        "line_dir_deg": float(line_dir_deg),
        "wire_is_darker": bool(wire_is_darker),
        "radial_freqs": fft_result["radial_freqs"],
        "radial_power": fft_result["radial_power"],
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
    }
