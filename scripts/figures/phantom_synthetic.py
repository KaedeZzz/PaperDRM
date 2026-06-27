"""
Synthetic phantom validation for the laid-line detector.

Generates images with exactly known period, angle, and wire width (Gaussian
comb model), runs detect_laid_lines_simple, and measures detection error.

Four sweeps:
  1. Period accuracy    — period 10..60 px,  SNR=15, angle=90°, sigma=2px
  2. SNR sensitivity    — SNR 2..50,         period=25px, angle=90°, sigma=2px
  3. Angle accuracy     — angle -30..+30°,   period=25px, SNR=15, sigma=2px
  4. Wire-width accuracy— sigma 1..8 px,     period=25px, SNR=15, angle=90°

Outputs:
  phantom_synthetic_results.json   — raw numbers
  phantom_synthetic.png            — 4-panel error figure
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from paperdrm.stage3_detect.simple_detector import (
    detect_laid_lines_simple,
    auto_detect_line_dir,
)


# ---------------------------------------------------------------------------
# Image generator
# ---------------------------------------------------------------------------

def make_synthetic_image(
    period_px: float,
    line_dir_deg: float,
    sigma_px: float,
    snr: float,
    shape: tuple[int, int] = (512, 512),
    seed: int = 0,
) -> np.ndarray:
    """
    Gaussian comb at known period/angle/width, plus white Gaussian noise.

    Signal amplitude is normalised to 1 so SNR = peak / noise_std.
    Returns uint8 image in [0, 255].
    """
    H, W = shape
    rng = np.random.default_rng(seed)
    cy, cx = H / 2.0, W / 2.0

    # Row / col grids (image coords: y increases downward)
    rows, cols = np.mgrid[0:H, 0:W].astype(np.float64)

    # Projection onto the axis perpendicular to the line direction.
    # For line_dir_deg θ (display/math coords, y-up):
    #   proj = sin(θ)*(col - cx) + cos(θ)*(row - cy)
    theta = np.radians(float(line_dir_deg))
    proj = np.sin(theta) * (cols - cx) + np.cos(theta) * (rows - cy)

    # Gaussian comb: sum over enough harmonics to cover the image
    max_proj = float(np.hypot(H, W)) / 2.0
    n = int(np.ceil(max_proj / period_px)) + 2
    signal = sum(
        np.exp(-0.5 * ((proj - k * period_px) / sigma_px) ** 2)
        for k in range(-n, n + 1)
    )

    # Invert: bright background, dark wire dips (matches real grazing/MSI data)
    signal /= signal.max()
    signal = 1.0 - signal
    noise = rng.standard_normal((H, W)) * (1.0 / snr)
    img_f = np.clip(signal + noise, 0.0, 1.0)
    return (img_f * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Single-trial runner
# ---------------------------------------------------------------------------

def run_trial(
    period_px: float,
    line_dir_deg: float,
    sigma_px: float,
    snr: float,
    *,
    use_auto_dir: bool = False,
    period_range_px: tuple[float, float] = (8.0, 80.0),
    seed: int = 0,
) -> dict:
    img = make_synthetic_image(period_px, line_dir_deg, sigma_px, snr, seed=seed)

    # Optionally test direction auto-detection
    detected_dir = line_dir_deg
    dir_error = 0.0
    if use_auto_dir:
        detected_dir = auto_detect_line_dir(img, period_range_px=period_range_px)
        # Wrap error into (-90, 90]
        dir_error = ((detected_dir - line_dir_deg) + 90.0) % 180.0 - 90.0

    try:
        result = detect_laid_lines_simple(
            img,
            line_dir_deg=detected_dir,
            period_range_px=period_range_px,
            wire_is_darker=True,   # dark dips on bright background
        )
        return {
            "ok": True,
            "period_true": period_px,
            "period_detected": result["dominant_period_px"],
            "period_error_pct": (result["dominant_period_px"] - period_px) / period_px * 100.0,
            "sigma_true": sigma_px,
            "sigma_detected": result["wire_sigma_px"],
            "sigma_error_pct": (result["wire_sigma_px"] - sigma_px) / sigma_px * 100.0,
            "wire_model_ok": result["wire_model_ok"],
            "angle_true": line_dir_deg,
            "angle_detected": detected_dir,
            "angle_error_deg": dir_error,
            "snr": snr,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "period_true": period_px,
            "angle_true": line_dir_deg,
            "sigma_true": sigma_px,
            "snr": snr,
        }


# ---------------------------------------------------------------------------
# Four parameter sweeps
# ---------------------------------------------------------------------------

PERIODS = [10, 12, 15, 18, 20, 24, 28, 32, 40, 50, 60]
SNRS    = [2, 3, 5, 8, 10, 15, 20, 30, 50]
# Near-vertical angles (manuscript data is ~90°). Values >90 fold to negative
# via auto_detect_line_dir, so test both sides: 60-90 and their folded mirrors.
ANGLES  = list(range(60, 91, 5)) + [-85, -80, -75, -70, -65, -60]
SIGMAS  = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]

N_SEEDS = 5   # average over seeds to reduce noise in the curves


def _mean_trials(trials: list[dict], key: str, *, require_wire_model: bool = False) -> float:
    vals = [t[key] for t in trials
            if t.get("ok") and (not require_wire_model or t.get("wire_model_ok", False))]
    return float(np.mean(vals)) if vals else float("nan")


def sweep_period(snr=15.0, angle=90.0, sigma=2.0) -> list[dict]:
    print("  sweep: period")
    rows = []
    for p in PERIODS:
        rng = (max(8.0, p * 0.4), min(80.0, p * 2.5))
        trials = [run_trial(p, angle, sigma, snr, period_range_px=rng, seed=s)
                  for s in range(N_SEEDS)]
        rows.append({
            "period_true": p,
            "period_error_pct_mean": _mean_trials(trials, "period_error_pct"),
            "sigma_error_pct_mean": _mean_trials(trials, "sigma_error_pct", require_wire_model=True),
            "n_ok": sum(t["ok"] for t in trials),
            "n_wire_ok": sum(t.get("wire_model_ok", False) for t in trials if t.get("ok")),
        })
    return rows


def sweep_snr(period=24.0, angle=90.0, sigma=2.0) -> list[dict]:
    print("  sweep: SNR")
    rows = []
    for s in SNRS:
        trials = [run_trial(period, angle, sigma, s,
                            period_range_px=(period * 0.4, period * 2.5), seed=sd)
                  for sd in range(N_SEEDS)]
        rows.append({
            "snr": s,
            "period_error_pct_mean": _mean_trials(trials, "period_error_pct"),
            "sigma_error_pct_mean": _mean_trials(trials, "sigma_error_pct", require_wire_model=True),
            "n_ok": sum(t["ok"] for t in trials),
        })
    return rows


def sweep_angle(period=24.0, snr=15.0, sigma=2.0) -> list[dict]:
    print("  sweep: angle")
    rows = []
    for a in ANGLES:
        trials = [run_trial(period, a, sigma, snr,
                            use_auto_dir=True,
                            period_range_px=(period * 0.4, period * 2.5), seed=sd)
                  for sd in range(N_SEEDS)]
        rows.append({
            "angle_true": a,
            "angle_error_deg_mean": _mean_trials(trials, "angle_error_deg"),
            "period_error_pct_mean": _mean_trials(trials, "period_error_pct"),
            "n_ok": sum(t["ok"] for t in trials),
        })
    return rows


def sweep_sigma(period=25.0, snr=15.0, angle=90.0) -> list[dict]:
    print("  sweep: wire width")
    rows = []
    for sig in SIGMAS:
        trials = [run_trial(period, angle, sig, snr,
                            period_range_px=(period * 0.4, period * 2.5), seed=sd)
                  for sd in range(N_SEEDS)]
        rows.append({
            "sigma_true": sig,
            "sigma_error_pct_mean": _mean_trials(trials, "sigma_error_pct", require_wire_model=True),
            "n_ok": sum(t["ok"] for t in trials),
            "n_wire_ok": sum(t.get("wire_model_ok", False) for t in trials if t.get("ok")),
        })
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    period_rows, snr_rows, angle_rows, sigma_rows,
    *, save_path: str = "phantom_synthetic.png",
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Synthetic Phantom: Detector Accuracy", fontsize=14)

    # 1. Period sweep — period error
    ax = axes[0, 0]
    xs = [r["period_true"] for r in period_rows]
    ys = [r["period_error_pct_mean"] for r in period_rows]
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.plot(xs, ys, "o-", color="steelblue")
    ax.set_xlabel("True period (px)")
    ax.set_ylabel("Period error (%)")
    ax.set_title("Period accuracy vs. true period\n(SNR=15, angle=90°, σ=2px)")
    ax.grid(True, alpha=0.3)

    # 2. SNR sweep — period error
    ax = axes[0, 1]
    xs = [r["snr"] for r in snr_rows]
    ys = [r["period_error_pct_mean"] for r in snr_rows]
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.plot(xs, ys, "o-", color="darkorange")
    ax.set_xlabel("SNR (peak / noise σ)")
    ax.set_ylabel("Period error (%)")
    ax.set_title("Period accuracy vs. SNR\n(period=24px, angle=90°, σ=2px)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # 3. Angle sweep — angle detection error
    ax = axes[1, 0]
    xs = [r["angle_true"] for r in angle_rows]
    ys = [abs(r["angle_error_deg_mean"]) for r in angle_rows]
    ax.plot(xs, ys, "o-", color="seagreen")
    ax.set_xlabel("True line direction (°, 90=vertical)")
    ax.set_ylabel("|Angle error| (°)")
    ax.set_title("auto_detect_line_dir accuracy (near-vertical)\n(period=24px, SNR=15, σ=2px)")
    ax.grid(True, alpha=0.3)

    # 4. Wire-width sweep — sigma error
    ax = axes[1, 1]
    xs = [r["sigma_true"] for r in sigma_rows]
    ys = [r["sigma_error_pct_mean"] for r in sigma_rows]
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.plot(xs, ys, "o-", color="crimson")
    ax.set_xlabel("True wire σ (px)")
    ax.set_ylabel("σ error (%)")
    ax.set_title("Wire-width accuracy vs. true σ\n(period=25px, SNR=15, angle=90°)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Phantom] Figure saved -> {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[Phantom] Running synthetic phantom sweeps ...")

    print("[Phantom] (1/4) Period sweep")
    period_rows = sweep_period()

    print("[Phantom] (2/4) SNR sweep")
    snr_rows = sweep_snr()

    print("[Phantom] (3/4) Angle sweep")
    angle_rows = sweep_angle()

    print("[Phantom] (4/4) Wire-width sweep")
    sigma_rows = sweep_sigma()

    results = {
        "period_sweep": period_rows,
        "snr_sweep": snr_rows,
        "angle_sweep": angle_rows,
        "sigma_sweep": sigma_rows,
    }

    out_json = Path("phantom_synthetic_results.json")
    out_json.write_text(json.dumps(results, indent=2))
    print(f"[Phantom] Results saved -> {out_json}")

    plot_results(period_rows, snr_rows, angle_rows, sigma_rows)
    print("[Phantom] Done.")
