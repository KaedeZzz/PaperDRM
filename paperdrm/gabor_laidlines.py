from __future__ import annotations

"""
Gabor-based laid-line frequency estimation.

High-level idea:
- Laid lines are approximately parallel, faint ridges in paper.
- A Gabor filter is a sinusoid modulated by a Gaussian envelope; it is a
  band-pass filter tuned to a specific orientation and spatial frequency.
- By scanning candidate line spacings (periods) and a small range of
  orientations, we can measure which period best matches the paper texture.
- After filtering, we project along the line direction to get a 1D signal
  in the normal direction, then score how much energy sits near the target
  frequency in the 1D spectrum.
"""

import numpy as np
import cv2
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


def _normalize01(img: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1] so the filter responses are comparable across inputs.
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    return (img - mn) / (mx - mn + 1e-8)


def _rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate keeping shape."""
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _project_1d(resp: np.ndarray, line_dir_deg: float) -> np.ndarray:
    """
    Average along the line direction to get a 1D signal in the normal direction.

    Method: rotate lines to vertical (90 deg) and average across rows.
    """
    # If lines are at line_dir_deg, then the normal direction is 90° away.
    # Rotate so lines become vertical: then averaging across rows collapses
    # along the line direction and preserves variation across the normal.
    rot = _rotate(resp, angle_deg=(90.0 - line_dir_deg))
    s = rot.mean(axis=0)
    # Detrend via high-pass: remove slow illumination/fold variations.
    # This keeps periodic structure while discarding very low-frequency drift.
    win = min(301, max(31, (len(s) // 20) | 1))
    k = np.ones(win, np.float32) / win
    s_hp = s - np.convolve(s, k, mode="same")
    return s_hp.astype(np.float32)


def _score_period(signal_1d: np.ndarray, period_px: float) -> float:
    # Score via spectral concentration:
    # 1) Compute the 1D FFT of the projected signal.
    # 2) Find the target frequency f0 = 1 / period (cycles per pixel).
    # 3) Measure the fraction of total spectral energy inside a narrow
    #    band around f0 (here ±20%).
    # This ratio is high when the signal is strongly periodic at the
    # expected spacing, and low for non-periodic/noisy signals.
    s = signal_1d - signal_1d.mean()
    n = len(s)
    S = np.fft.rfft(s)
    freqs = np.fft.rfftfreq(n, d=1.0)
    f0 = 1.0 / float(period_px)
    band = (freqs >= 0.8 * f0) & (freqs <= 1.2 * f0)
    band_energy = float(np.sum(np.abs(S[band]) ** 2))
    total_energy = float(np.sum(np.abs(S) ** 2)) + 1e-12
    return band_energy / total_energy


def _best_for_period(
    img01: np.ndarray,
    *,
    line_dir_deg: float,
    period_px: float,
    angle_jitter_deg: float,
    angle_step_deg: float,
    sigma_factor: float,
    gamma: float,
    ksize: int,
    use_abs_response: bool,
) -> dict:
    # Gabor theta is the normal direction (oscillation direction).
    # If lines are at line_dir_deg, then the filter's oscillation should be
    # perpendicular to the lines to capture the repeating bright/dark pattern.
    normal_dir = (line_dir_deg + 90.0) % 180.0
    thetas = np.arange(normal_dir - angle_jitter_deg, normal_dir + angle_jitter_deg + 1e-6, angle_step_deg)
    thetas = (thetas + 180.0) % 180.0

    best = {"score": -1.0}
    lambd = float(period_px)
    sigma = max(1.0, sigma_factor * lambd)
    for th in thetas:
        theta = np.deg2rad(float(th))
        # A real-valued, even-symmetric Gabor acts as a band-pass
        # filter around the target orientation and frequency.
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0.0, ktype=cv2.CV_32F)
        # Remove the DC component so slow illumination changes are suppressed.
        kernel -= kernel.mean()
        # Convolve image with the Gabor kernel.
        r = cv2.filter2D(img01, cv2.CV_32F, kernel)
        resp = np.abs(r) if use_abs_response else r
        # Collapse to a 1D signal along the normal direction.
        s1d = _project_1d(resp, line_dir_deg=line_dir_deg)
        sc = _score_period(s1d, period_px=period_px)
        if sc > float(best["score"]):
            best.update(
                score=sc,
                best_theta_deg=float(th),
                best_signal_1d=s1d,
                best_response=resp,
            )
    return best


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    # Weighted mean: average with weights normalized to sum to 1.
    # Use this when you want strong patches to dominate the consensus.
    if len(values) == 0:
        return float("nan")
    wsum = float(np.sum(weights)) + 1e-12
    return float(np.sum(values * weights) / wsum)


def estimate_laidline_frequency_gabor(
    img: np.ndarray,
    *,
    line_dir_deg: float,
    periods_px: list[float] | tuple[float, ...] = (8, 10, 12, 14, 16, 18, 22, 26, 32),
    angle_jitter_deg: float = 6.0,
    angle_step_deg: float = 2.0,
    sigma_factor: float = 0.6,
    gamma: float = 0.4,
    ksize: int = 41,
    use_abs_response: bool = True,
) -> dict:
    """
    Scan candidate periods (pixel spacing) using oriented Gabor filters.

    Returns a dict with:
    best_period_px, best_freq_cpp, best_theta_deg, scores{period:score},
    best_signal_1d, best_response, line_dir_deg
    """
    # Normalize once to make responses comparable across inputs.
    img01 = _normalize01(img)

    best: dict[str, object] = {"score": -1.0}
    scores: dict[float, float] = {}

    for period in periods_px:
        best_for_period = _best_for_period(
            img01,
            line_dir_deg=line_dir_deg,
            period_px=float(period),
            angle_jitter_deg=angle_jitter_deg,
            angle_step_deg=angle_step_deg,
            sigma_factor=sigma_factor,
            gamma=gamma,
            ksize=ksize,
            use_abs_response=use_abs_response,
        )
        scores[float(period)] = float(best_for_period["score"])
        if float(best_for_period["score"]) > float(best["score"]):
            best.update(
                score=best_for_period["score"],
                best_period_px=float(period),
                best_freq_cpp=1.0 / float(period),
                best_theta_deg=best_for_period["best_theta_deg"],
                best_signal_1d=best_for_period["best_signal_1d"],
                best_response=best_for_period["best_response"],
            )

    out = {
        "best_period_px": float(best["best_period_px"]),
        "best_freq_cpp": float(best["best_freq_cpp"]),
        "best_theta_deg": float(best["best_theta_deg"]),
        "scores": scores,
        "best_signal_1d": best["best_signal_1d"],
        "best_response": best["best_response"],
        "line_dir_deg": float(line_dir_deg),
    }
    return out


def estimate_laidline_frequency_gabor_patches(
    img: np.ndarray,
    *,
    line_dir_deg: float,
    patch_size: tuple[int, int] = (256, 256),
    stride: tuple[int, int] = (128, 128),
    periods_px: list[float] | tuple[float, ...] = (8, 10, 12, 14, 16, 18, 22, 26, 32),
    angle_jitter_deg: float = 6.0,
    angle_step_deg: float = 2.0,
    sigma_factor: float = 0.6,
    gamma: float = 0.4,
    ksize: int = 41,
    use_abs_response: bool = True,
    min_score: float = 0.02,
    show_progress: bool = True,
    weight_scale: float = 1.0,
) -> dict:
    """
    Estimate laid-line frequency per patch and aggregate to a dominant period.

    Returns a dict with:
    dominant_period_px, dominant_freq_cpp, dominant_theta_deg, dominant_signal_1d,
    dominant_response, patch_results, line_dir_deg
    """
    # Patch strategy motivation:
    # - Laid lines can be locally clearer in some regions than others.
    # - Estimating per patch reduces the impact of local stains/ink/defects.
    # - Aggregating patch proposals yields a more robust global period.
    if img.ndim != 2:
        raise ValueError("estimate_laidline_frequency_gabor_patches expects a 2D image.")

    h, w = img.shape
    ph, pw = patch_size
    sy, sx = stride
    if ph <= 0 or pw <= 0 or sy <= 0 or sx <= 0:
        raise ValueError("patch_size and stride must be positive.")

    patch_results: list[dict] = []
    ys = list(range(0, h - ph + 1, sy))
    xs = list(range(0, w - pw + 1, sx))
    positions = [(y, x) for y in ys for x in xs]
    total_patches = len(positions)
    iterator = positions
    if show_progress and tqdm is not None:
        iterator = tqdm(positions, total=total_patches, desc="Gabor patches", leave=False)
    for y, x in iterator:
            patch = img[y : y + ph, x : x + pw]
            res = estimate_laidline_frequency_gabor(
                patch,
                line_dir_deg=line_dir_deg,
                periods_px=periods_px,
                angle_jitter_deg=angle_jitter_deg,
                angle_step_deg=angle_step_deg,
                sigma_factor=sigma_factor,
                gamma=gamma,
                ksize=ksize,
                use_abs_response=use_abs_response,
            )
            patch_results.append(
                {
                    "x": x,
                    "y": y,
                    "w": pw,
                    "h": ph,
                    "best_period_px": float(res["best_period_px"]),
                    "best_theta_deg": float(res["best_theta_deg"]),
                    # Use the winning period's score as a patch confidence.
                    "best_score": float(res["scores"][float(res["best_period_px"])]),
                }
            )

    # Filter out weak/noisy patches by score threshold.
    valid = [r for r in patch_results if r["best_score"] >= min_score]
    if len(valid) == 0:
        # Fall back to full-image estimate.
        full = estimate_laidline_frequency_gabor(
            img,
            line_dir_deg=line_dir_deg,
            periods_px=periods_px,
            angle_jitter_deg=angle_jitter_deg,
            angle_step_deg=angle_step_deg,
            sigma_factor=sigma_factor,
            gamma=gamma,
            ksize=ksize,
            use_abs_response=use_abs_response,
        )
        return {
            "dominant_period_px": float(full["best_period_px"]),
            "dominant_freq_cpp": float(full["best_freq_cpp"]),
            "dominant_theta_deg": float(full["best_theta_deg"]),
            "dominant_signal_1d": full["best_signal_1d"],
            "dominant_response": full["best_response"],
            "patch_results": patch_results,
            "line_dir_deg": float(line_dir_deg),
        }

    periods = np.array([r["best_period_px"] for r in valid], dtype=np.float32)
    scores = np.array([r["best_score"] for r in valid], dtype=np.float32)
    thetas = np.array([r["best_theta_deg"] for r in valid], dtype=np.float32)

    # Weighted mean of periods using exp(score * weight_scale) as weights.
    # Use a max-shift for numerical stability (softmax-style).
    if not np.isfinite(weight_scale):
        raise ValueError("weight_scale must be finite.")
    scaled = scores * float(weight_scale)
    scaled = scaled - float(np.max(scaled))
    weights = np.exp(scaled)
    dominant_period = _weighted_mean(periods, weights)
    # Median theta is a simple robust summary of patch orientations.
    dominant_theta = float(np.median(thetas))
    # Recompute best response on the full image at the dominant period.
    img01 = _normalize01(img)
    dominant_best = _best_for_period(
        img01,
        line_dir_deg=line_dir_deg,
        period_px=dominant_period,
        angle_jitter_deg=angle_jitter_deg,
        angle_step_deg=angle_step_deg,
        sigma_factor=sigma_factor,
        gamma=gamma,
        ksize=ksize,
        use_abs_response=use_abs_response,
    )

    return {
        "dominant_period_px": float(dominant_period),
        "dominant_freq_cpp": float(1.0 / dominant_period),
        "dominant_theta_deg": dominant_theta,
        "dominant_signal_1d": dominant_best["best_signal_1d"],
        "dominant_response": dominant_best["best_response"],
        "patch_results": patch_results,
        "line_dir_deg": float(line_dir_deg),
    }


def rotate_keep_shape(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate keeping original shape."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def smooth1d(x: np.ndarray, win: int = 31) -> np.ndarray:
    win = int(win)
    if win < 3:
        return x
    if win % 2 == 0:
        win += 1
    # Moving average smoothing: reduces noise before peak picking.
    k = np.ones(win, np.float32) / win
    return np.convolve(x, k, mode="same")


def peaks_from_signal(s: np.ndarray, period_px: float) -> np.ndarray:
    """Find peak x-positions in the rotated image coordinate frame."""
    s = np.asarray(s, np.float32)
    # Center to reduce bias from global brightness shifts.
    s = s - np.median(s)
    # Smooth at a scale tied to the expected period to stabilize peaks.
    s = smooth1d(s, win=max(9, int(period_px * 0.7) | 1))

    # Expected spacing between adjacent laid lines.
    min_dist = max(3, int(period_px * 0.7))

    # Simple local maxima with minimum distance filtering.
    idx = np.where((s[1:-1] > s[:-2]) & (s[1:-1] > s[2:]))[0] + 1
    if len(idx) == 0:
        return idx.astype(int)
    keep = [int(idx[0])]
    for p in idx[1:]:
        if int(p) - keep[-1] >= min_dist:
            keep.append(int(p))
    return np.array(keep, dtype=int)


def _estimate_phase(signal_1d: np.ndarray, period_px: float) -> float:
    """
    Estimate phase of a sinusoid at the target period using dot products.

    We fit s(x) ~ A cos(ωx + φ) by projecting onto cos and sin at ω = 2π/period.
    The phase φ = atan2(sin_proj, cos_proj) gives the horizontal offset of peaks.
    """
    s = np.asarray(signal_1d, np.float32)
    n = len(s)
    x = np.arange(n, dtype=np.float32)
    w = 2.0 * np.pi / float(period_px)
    c = float(np.sum(s * np.cos(w * x)))
    si = float(np.sum(s * np.sin(w * x)))
    return float(np.arctan2(si, c))


def grid_positions_from_signal(
    signal_1d: np.ndarray,
    period_px: float,
    length: int,
) -> np.ndarray:
    """
    Construct an evenly spaced grid of line centers from the signal phase.

    The grid positions are the cosine maxima of the fitted sinusoid:
    ωx + φ = 2πk -> x = (-φ + 2πk) / ω
    """
    phase = _estimate_phase(signal_1d, period_px)
    w = 2.0 * np.pi / float(period_px)
    x0 = (-phase) / w
    # Generate k so x is in [0, length)
    k_start = int(np.floor((0.0 - x0) / period_px))
    k_end = int(np.ceil((length - x0) / period_px))
    xs = x0 + period_px * np.arange(k_start, k_end + 1)
    xs = xs[(xs >= 0.0) & (xs < float(length))]
    return np.round(xs).astype(int)


def filter_peaks_to_grid(
    peaks_x: np.ndarray,
    grid_x: np.ndarray,
    tol_frac: float = 0.15,
) -> np.ndarray:
    """
    Keep only peaks that are close to the grid, within tol_frac * period.

    This enforces regularity without forcing every grid line to be used.
    """
    if len(peaks_x) == 0 or len(grid_x) == 0:
        return np.asarray([], dtype=int)
    peaks_x = np.asarray(peaks_x, dtype=int)
    grid_x = np.asarray(grid_x, dtype=int)
    period_px = float(np.median(np.diff(grid_x))) if len(grid_x) > 1 else 0.0
    tol = max(1.0, tol_frac * period_px) if period_px > 0 else 1.0
    kept = []
    for p in peaks_x:
        idx = int(np.argmin(np.abs(grid_x - p)))
        if abs(int(grid_x[idx]) - int(p)) <= tol:
            kept.append(int(p))
    return np.array(kept, dtype=int)


def overlay_laid_lines(
    img: np.ndarray,
    *,
    line_dir_deg: float,
    best_signal_1d: np.ndarray,
    best_period_px: float,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 1,
    alpha: float = 0.6,
    mode: str = "peaks",
    grid_tol_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    overlay: original image with laid lines overlaid
    peaks_x: peak positions in the rotated image coordinate frame
    """
    if img.ndim == 2:
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        base = img.copy()

    h, w = base.shape[:2]

    # Rotate so laid lines are vertical, making peaks correspond to x positions.
    rot_angle = 90.0 - float(line_dir_deg)
    base_rot = rotate_keep_shape(base, rot_angle)

    # Peaks correspond to line centers in the rotated frame.
    peaks_x = peaks_from_signal(best_signal_1d, best_period_px)
    grid_x = grid_positions_from_signal(best_signal_1d, best_period_px, w)
    if mode == "grid":
        line_x = grid_x
    elif mode == "grid_tolerant":
        line_x = filter_peaks_to_grid(peaks_x, grid_x, tol_frac=grid_tol_frac)
    else:
        line_x = peaks_x

    # Draw vertical lines at peak x positions.
    overlay_rot = base_rot.copy()
    for x in line_x:
        x = int(np.clip(x, 0, w - 1))
        cv2.line(overlay_rot, (x, 0), (x, h - 1), color, thickness, lineType=cv2.LINE_AA)

    # Rotate back and alpha-blend with the original.
    overlay_back = rotate_keep_shape(overlay_rot, -rot_angle)
    overlay = cv2.addWeighted(base, 1.0 - alpha, overlay_back, alpha, 0.0)
    return overlay, line_x
