"""
Head-to-head comparison of five laid-line detectors on the same data.

Detectors:
  1. radial_fft     -- 2D FFT, integrate along line direction, peak pick
                       (period only; grid built from broadband phase fit).
  2. gabor_full     -- single full-image Gabor scan over candidate periods.
  3. gabor_patches  -- patchwise Gabor scan + score-weighted mode (LEGACY).
  4. simple         -- radial FFT period + Gabor cleanup at known period
                       (current default single-image track).
  5. multi_phi      -- per-phi normalised power-spectrum sum across N
                       grazing images (new DRP-aware track).

Metrics (all computed against the *same* reference image's broadband
signal so methods are directly comparable):
  - dominant_period_px / lines_per_cm
  - r2_with_harmonics:    Fourier R^2 at the method's detected period
  - self_contrast_z:      method-agnostic "is the grid on real wires?"
  - self_contrast_rel:    relative intensity gap, signed by wire_is_darker
  - n_grid_lines:         number of (on, off) sample pairs that landed in bounds
  - wall_time_s:          end-to-end detector time (no I/O)
  - split_half_diff_std:  multi_phi only -- A/B partition period spread

Output: prints a fixed-width table; writes detector_comparison.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from paperdrm import ImagePack, Settings
from paperdrm.stage3_detect.gabor import (
    estimate_laidline_frequency_gabor,
    estimate_laidline_frequency_gabor_patches,
    grid_positions_from_signal,
)
from paperdrm.stage3_detect.multi_phi_detector import (
    collect_grazing_per_phi,
    detect_laid_lines_multi_phi,
)
from paperdrm.stage3_detect.simple_detector import (
    _broadband_signal_1d,
    detect_laid_lines_simple,
    grid_positions,
    phase_fit,
    radial_fft_period,
)
from paperdrm.stage5_evaluation.fit_quality import sinusoidal_fit_r2
from paperdrm.stage5_evaluation.self_contrast import self_consistency_contrast
from paperdrm.stage5_evaluation.split_half import split_half_period_stability


PERIOD_RANGE_PX = (8.0, 80.0)
LINE_DIR_DEG = 90.0
WIRE_IS_DARKER = True
N_HARMONICS = 4
N_SPLITS = 200


def _eval(
    name: str,
    period_px: float,
    grid_x: np.ndarray,
    ref_image: np.ndarray,
    all_images: list[np.ndarray],
    broadband: np.ndarray,
    wall_time_s: float,
    cm_per_px: float,
) -> dict:
    """Apply the common evaluation suite to a (period, grid) prediction.

    Self-contrast is sampled on every phi image because the on-/off-wire
    polarity can flip across phi (specular highlight vs. shadow). We
    report median |z| (polarity-robust magnitude), median signed z, and
    the fraction of phi images where the grid lands with the assumed
    wire_is_darker polarity.
    """
    z_signed = []
    rel_signed = []
    for img in all_images:
        sc_i = self_consistency_contrast(
            img,
            grid_x,
            period_px,
            line_dir_deg=LINE_DIR_DEG,
            band_half_width_px=1,
            wire_is_darker=WIRE_IS_DARKER,
        )
        if np.isfinite(sc_i["contrast_z"]):
            z_signed.append(float(sc_i["contrast_z"]))
            rel_signed.append(float(sc_i["contrast_rel"]))
    z_arr = np.asarray(z_signed, dtype=np.float64)
    rel_arr = np.asarray(rel_signed, dtype=np.float64)

    sc_ref = self_consistency_contrast(
        ref_image,
        grid_x,
        period_px,
        line_dir_deg=LINE_DIR_DEG,
        band_half_width_px=1,
        wire_is_darker=WIRE_IS_DARKER,
    )

    r2 = sinusoidal_fit_r2(broadband, period_px, n_harmonics=N_HARMONICS)
    return {
        "name": name,
        "period_px": float(period_px),
        "period_cm": float(period_px * cm_per_px),
        "lines_per_cm": float(1.0 / (period_px * cm_per_px)) if period_px > 0 else float("nan"),
        "r2_with_harmonics": float(r2),
        "sc_ref_z": float(sc_ref["contrast_z"]),
        "sc_median_abs_z": float(np.median(np.abs(z_arr))) if z_arr.size else float("nan"),
        "sc_median_signed_z": float(np.median(z_arr)) if z_arr.size else float("nan"),
        "sc_median_abs_rel": float(np.median(np.abs(rel_arr))) if rel_arr.size else float("nan"),
        "sc_frac_positive": float(np.mean(z_arr > 0)) if z_arr.size else float("nan"),
        "n_grid_lines": int(sc_ref["n_lines"]),
        "wall_time_s": float(wall_time_s),
    }


def run_radial_fft(ref_image, all_images, broadband, cm_per_px) -> dict:
    t0 = time.perf_counter()
    res = radial_fft_period(
        ref_image,
        line_dir_deg=LINE_DIR_DEG,
        period_range_px=PERIOD_RANGE_PX,
    )
    period = float(res["dominant_period_px"])
    phase = phase_fit(broadband, period, wire_is_darker=WIRE_IS_DARKER)
    grid_x = grid_positions(phase, period, ref_image.shape[1])
    elapsed = time.perf_counter() - t0
    return _eval("radial_fft", period, grid_x, ref_image, all_images, broadband, elapsed, cm_per_px)


def run_gabor_full(ref_image, all_images, broadband, cm_per_px) -> dict:
    t0 = time.perf_counter()
    res = estimate_laidline_frequency_gabor(
        ref_image,
        line_dir_deg=LINE_DIR_DEG,
        periods_px=list(range(int(PERIOD_RANGE_PX[0]), int(PERIOD_RANGE_PX[1]) + 1)),
    )
    period = float(res["best_period_px"])
    grid_x = grid_positions_from_signal(res["best_signal_1d"], period, ref_image.shape[1])
    elapsed = time.perf_counter() - t0
    return _eval("gabor_full", period, grid_x, ref_image, all_images, broadband, elapsed, cm_per_px)


def run_gabor_patches(ref_image, all_images, broadband, cm_per_px) -> dict:
    t0 = time.perf_counter()
    res = estimate_laidline_frequency_gabor_patches(
        ref_image,
        line_dir_deg=LINE_DIR_DEG,
        patch_size=(512, 512),
        stride=(256, 256),
        periods_px=list(range(int(PERIOD_RANGE_PX[0]), int(PERIOD_RANGE_PX[1]) + 1)),
        min_score=0.02,
        weight_scale=3.0,
        show_progress=False,
    )
    period = float(res["dominant_period_px"])
    grid_x = grid_positions_from_signal(res["dominant_signal_1d"], period, ref_image.shape[1])
    elapsed = time.perf_counter() - t0
    return _eval("gabor_patches", period, grid_x, ref_image, all_images, broadband, elapsed, cm_per_px)


def run_simple(ref_image, all_images, broadband, cm_per_px) -> dict:
    t0 = time.perf_counter()
    res = detect_laid_lines_simple(
        ref_image,
        line_dir_deg=LINE_DIR_DEG,
        period_range_px=PERIOD_RANGE_PX,
        wire_is_darker=WIRE_IS_DARKER,
        use_gabor_refinement=True,
    )
    period = float(res["dominant_period_px"])
    grid_x = np.asarray(res["grid_positions_x"])
    elapsed = time.perf_counter() - t0
    return _eval("simple", period, grid_x, ref_image, all_images, broadband, elapsed, cm_per_px)


def run_multi_phi(images, ref_image, broadband, cm_per_px) -> dict:
    t0 = time.perf_counter()
    res = detect_laid_lines_multi_phi(
        images,
        line_dir_deg=LINE_DIR_DEG,
        period_range_px=PERIOD_RANGE_PX,
        wire_is_darker=WIRE_IS_DARKER,
    )
    period = float(res["dominant_period_px"])
    grid_x = np.asarray(res["grid_positions_x"])
    elapsed = time.perf_counter() - t0
    out = _eval("multi_phi", period, grid_x, ref_image, images, broadband, elapsed, cm_per_px)
    out["phase_circular_var"] = float(res["phase_circular_var"])
    out["phase_resultant_length"] = float(res["phase_resultant_length"])
    return out


def main(yaml_path: str = "exp_param.yaml") -> dict:
    settings = Settings.from_yaml(yaml_path).with_overrides(angle_slice=(2, 2), verbose=False)
    pack = ImagePack(settings=settings)

    images, phi_deg = collect_grazing_per_phi(pack)
    ref_image = images[0]
    W = int(ref_image.shape[1])
    fov_cm = float(pack.settings.fov_width_cm)
    cm_per_px = fov_cm / float(W)
    broadband = _broadband_signal_1d(ref_image, LINE_DIR_DEG)

    print(f"[Compare] dataset: data_serial={pack.settings.drp.data_serial}"
          f"  phi_n={len(images)}  ref_image.shape={ref_image.shape}"
          f"  fov={fov_cm:.3f} cm  cm/px={cm_per_px:.6e}")
    print()

    rows: list[dict] = []
    rows.append(run_radial_fft(ref_image, images, broadband, cm_per_px))
    rows.append(run_gabor_full(ref_image, images, broadband, cm_per_px))
    rows.append(run_gabor_patches(ref_image, images, broadband, cm_per_px))
    rows.append(run_simple(ref_image, images, broadband, cm_per_px))
    rows.append(run_multi_phi(images, ref_image, broadband, cm_per_px))

    print("[Compare] Split-half stability (multi_phi only)")
    sh = split_half_period_stability(
        images,
        line_dir_deg=LINE_DIR_DEG,
        period_range_px=PERIOD_RANGE_PX,
        n_splits=N_SPLITS,
        fov_width_cm=fov_cm,
    )
    for r in rows:
        if r["name"] == "multi_phi":
            r["split_half_diff_std_px"] = float(sh["period_diff_std"])
            r["split_half_agree_within_1px"] = float(sh["agree_rate_within_1px"])

    _print_table(rows)

    out_path = Path("detector_comparison.json")
    payload = {
        "data_serial": int(pack.settings.drp.data_serial),
        "n_phi": int(len(images)),
        "image_shape": list(map(int, ref_image.shape)),
        "fov_width_cm": fov_cm,
        "cm_per_px": cm_per_px,
        "line_dir_deg": LINE_DIR_DEG,
        "wire_is_darker": WIRE_IS_DARKER,
        "period_range_px": list(PERIOD_RANGE_PX),
        "n_harmonics_for_r2": N_HARMONICS,
        "n_splits_for_split_half": N_SPLITS,
        "detectors": rows,
        "split_half_full": {
            "period_pooled_mean_px": float(sh["period_pooled_mean"]),
            "period_pooled_std_px": float(sh["period_pooled_std"]),
            "period_diff_std_px": float(sh["period_diff_std"]),
            "agree_rate_within_1px": float(sh["agree_rate_within_1px"]),
            "agree_rate_within_0p5px": float(sh["agree_rate_within_0p5px"]),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[Compare] wrote {out_path}")
    return payload


def _print_table(rows: list[dict]) -> None:
    header = (
        f"{'detector':<14}"
        f"  {'period_px':>9}"
        f"  {'lines/cm':>8}"
        f"  {'R^2(k=4)':>9}"
        f"  {'med|z|':>7}"
        f"  {'med z':>7}"
        f"  {'frac+':>6}"
        f"  {'sc_ref':>7}"
        f"  {'time_s':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<14}"
            f"  {r['period_px']:>9.3f}"
            f"  {r['lines_per_cm']:>8.3f}"
            f"  {r['r2_with_harmonics']:>9.4f}"
            f"  {r['sc_median_abs_z']:>7.2f}"
            f"  {r['sc_median_signed_z']:>+7.2f}"
            f"  {r['sc_frac_positive']*100:>5.0f}%"
            f"  {r['sc_ref_z']:>+7.2f}"
            f"  {r['wall_time_s']:>7.2f}"
        )
    multi = next((r for r in rows if r["name"] == "multi_phi"), None)
    if multi and "split_half_diff_std_px" in multi:
        print()
        print(f"multi_phi split-half: diff_std={multi['split_half_diff_std_px']:.3f} px"
              f"  agree<=1px={multi['split_half_agree_within_1px']*100:.1f}%"
              f"  phase_circ_var={multi['phase_circular_var']:.3f}")
    print()
    print("Columns: med|z| = polarity-robust contrast magnitude (median across 20 phi);"
          " frac+ = % of phi where grid lands with assumed dark-on-wire polarity;"
          " sc_ref = signed z on phi=0 only (legacy, polarity-sensitive)")


if __name__ == "__main__":
    main()
