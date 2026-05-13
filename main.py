"""
End-to-end laid-line detection pipeline (dual-track).

Two detection paths are supported via USE_SIMPLE_DETECTOR:

SIMPLE TRACK (default, recommended)
  Single grazing-light image -> radial FFT -> Gabor cleanup -> grid.
  Bypasses DRP direction map and trig-mask preprocessing.
  Fast (~1 sec for period), ML-optimal, correct on this data.

LEGACY TRACK (USE_SIMPLE_DETECTOR = False)
  DRP stack -> direction map -> trig mask -> patchwise Gabor -> grid.
  Slow (~4.5 min), KNOWN BIASED: detects half the true period due to
  abs-response harmonic doubling + proportional band-width bias in the
  Gabor scoring. Kept for reference / ablation.

Stages used by each track:
  0. stage0_loader     -- DRP stack loading             (both)
  1. stage1_features   -- DRP direction map             (legacy only)
  2. stage2_enhance    -- trig mask                     (legacy only)
  3. stage3_detect     -- period + grid                 (both, different)
  4. (drawing)         -- overlay onto image            (both)
  5. stage5_evaluation -- gap distribution + fit-quality (both)
                          + patch consistency           (legacy only)
"""

import cv2

from paperdrm import ImagePack, Settings
from paperdrm.stage1_features.direction import drp_direction_map
from paperdrm.stage2_enhance.trig_mask import (
    azimuth_to_laidline_gray,
    orientation_comparison_maps,
    patchwise_trigonometric_mask,
)
from paperdrm.stage3_detect.gabor import (
    estimate_laidline_frequency_gabor_patches,
    overlay_laid_lines,
)
from paperdrm.stage3_detect.simple_detector import (
    detect_laid_lines_simple,
    overlay_grid,
)
from paperdrm.stage4_viz.comparison import (
    plot_orientation_comparison,
    plot_patch_best_score_map,
    plot_trig_mask_comparison,
)
from paperdrm.stage5_evaluation.consistency import (
    patch_consistency_report,
    plot_patch_consistency,
    print_consistency_report,
    save_consistency_report,
)
from paperdrm.stage5_evaluation.interval_distribution import (
    gap_distribution_from_signal,
    plot_gap_distribution,
    print_gap_distribution,
    save_gap_distribution,
)
from paperdrm.stage5_evaluation.fit_quality import (
    fit_quality_report,
    plot_fit_quality_curve,
    print_fit_quality,
    save_fit_quality,
)


# ---------------------------------------------------------------------------
# Track selector
# ---------------------------------------------------------------------------
USE_SIMPLE_DETECTOR = True


# ---------------------------------------------------------------------------
# Stage 0 (shared by both tracks)
# ---------------------------------------------------------------------------
def stage_load(yaml_path: str) -> ImagePack:
    print("[Stage 0] Loading settings + DRP stack")
    settings = Settings.from_yaml(yaml_path).with_overrides(angle_slice=(2, 2), verbose=True)
    return ImagePack(settings=settings)


def pick_grazing_image(pack: ImagePack, *, phi_index: int = 0) -> "tuple[np.ndarray, int]":
    """
    Pick a single bg-subtracted image at the steepest available grazing angle
    for the given phi index. Images in `pack.images` are ordered (phi major,
    theta minor) after slicing + theta_min filtering.

    Returns (image, chosen_index).
    """
    th_num = pack.param.th_num
    idx = phi_index * th_num + (th_num - 1)
    if idx >= len(pack.images):
        idx = len(pack.images) - 1
    return pack.images[idx], idx


# ---------------------------------------------------------------------------
# Stages 1-3 LEGACY TRACK
# ---------------------------------------------------------------------------
def stage_direction(pack: ImagePack):
    print("[Stage 1] Computing DRP direction map")
    mag_map, deg_map = drp_direction_map(pack, verbose=pack.verbose)
    return mag_map, deg_map


def stage_enhance(deg_map, *, target_angle=90.0, sigma_deg=10.0, patch_size=(512, 512), stride=(256, 256)):
    print("[Stage 2] Building trigonometric masks (baseline + patchwise)")
    prev_raw_img, prev_img = azimuth_to_laidline_gray(
        deg_map, target_deg=target_angle, sigma_deg=sigma_deg, enhance=True
    )
    patch_raw_img, patch_img, patch_target_deg_map, target_img = patchwise_trigonometric_mask(
        deg_map, patch_size=patch_size, stride=stride, sigma_deg=sigma_deg, enhance=True
    )

    diff_img = cv2.absdiff(prev_img, patch_img)
    raw_azimuth_8u, orientation_diff_deg = orientation_comparison_maps(deg_map, patch_target_deg_map)
    plot_orientation_comparison(raw_azimuth_8u, target_img, orientation_diff_deg)
    plot_trig_mask_comparison(
        prev_raw_img, patch_raw_img, target_img, prev_img, patch_img, diff_img, target_angle
    )
    return patch_img


def stage_detect_legacy(gabor_input_gray, *, patch_size, stride, fov_width_cm=None):
    print("[Stage 3 LEGACY] Patchwise Gabor scan (known biased)")
    out = estimate_laidline_frequency_gabor_patches(
        gabor_input_gray,
        line_dir_deg=90.0,
        patch_size=patch_size,
        stride=stride,
        periods_px=list(range(4, 81)),
        min_score=0.02,
        weight_scale=3.0,
    )
    plot_patch_best_score_map(out["patch_results"])
    period_px = out["dominant_period_px"]
    print(f"dominant period={period_px:.1f} px  freq={out['dominant_freq_cpp']:.4f} cpp")
    if fov_width_cm is not None:
        cm_per_px = float(fov_width_cm) / float(gabor_input_gray.shape[1])
        interval_cm = period_px * cm_per_px
        print(f"laid line interval = {interval_cm:.4f} cm | density = {1.0/interval_cm:.4f} lines/cm")
    return out


def stage_overlay_legacy(gabor_input_gray, detect_out, *, out_path="laid_lines_overlay_legacy.png"):
    print(f"[Stage 4 LEGACY] Overlay -> {out_path}")
    overlay, _ = overlay_laid_lines(
        gabor_input_gray,
        line_dir_deg=detect_out["line_dir_deg"],
        best_signal_1d=detect_out["dominant_signal_1d"],
        best_period_px=detect_out["dominant_period_px"],
        color=(0, 0, 255), thickness=1, alpha=0.65, mode="grid",
    )
    cv2.imwrite(out_path, overlay)


# ---------------------------------------------------------------------------
# Stages 3-4 SIMPLE TRACK
# ---------------------------------------------------------------------------
def stage_detect_simple(image, *, line_dir_deg=90.0, fov_width_cm=None,
                        period_range_px=(8.0, 80.0), wire_is_darker=True):
    print("[Stage 3 SIMPLE] Radial FFT + Gabor cleanup")
    result = detect_laid_lines_simple(
        image,
        line_dir_deg=line_dir_deg,
        period_range_px=period_range_px,
        wire_is_darker=wire_is_darker,
        use_gabor_refinement=True,
    )
    period_px = result["dominant_period_px"]
    print(f"period={period_px:.2f} px  freq={result['dominant_freq_cpp']:.5f} cpp"
          f"  gabor_score={result['gabor_score']:.3f}")
    if fov_width_cm is not None:
        cm_per_px = float(fov_width_cm) / float(image.shape[1])
        interval_cm = period_px * cm_per_px
        print(f"laid line interval = {interval_cm:.4f} cm | density = {1.0/interval_cm:.4f} lines/cm")
    return result


def stage_overlay_simple(image, detect_out, *, out_path="laid_lines_overlay.png"):
    print(f"[Stage 4 SIMPLE] Overlay -> {out_path}")
    overlay = overlay_grid(
        image,
        detect_out["grid_positions_x"],
        line_dir_deg=detect_out["line_dir_deg"],
        color=(0, 0, 255), thickness=1, alpha=0.55,
    )
    cv2.imwrite(out_path, overlay)


# ---------------------------------------------------------------------------
# Stage 5 (shared, gracefully skips patch consistency when not applicable)
# ---------------------------------------------------------------------------
def stage_evaluate(
    detect_out,
    *,
    score_threshold=0.02,
    fov_width_cm=None,
    image_width_px=None,
    consistency_path="evaluation_report.json",
    intervals_path="interval_distribution.json",
    fit_quality_path="fit_quality.json",
):
    has_patches = bool(detect_out.get("patch_results"))
    if has_patches:
        print("[Stage 5] Evaluating patch consistency")
        report = patch_consistency_report(detect_out, score_threshold=score_threshold)
        print_consistency_report(report)
        save_consistency_report(report, consistency_path)
        plot_patch_consistency(detect_out, report, score_threshold=score_threshold)
    else:
        print("[Stage 5] No patch_results -> skipping patch consistency")
        report = None

    print("[Stage 5] Computing laid-line interval distribution")
    gap_stats = gap_distribution_from_signal(
        detect_out["dominant_signal_1d"],
        detect_out["dominant_period_px"],
        fov_width_cm=fov_width_cm,
        image_width_px=image_width_px,
    )
    print_gap_distribution(gap_stats)
    save_gap_distribution(gap_stats, intervals_path)
    plot_gap_distribution(gap_stats)

    print("[Stage 5] Computing fit-quality (R^2 / loss)")
    fq = fit_quality_report(detect_out)
    print_fit_quality(fq)
    save_fit_quality(fq, fit_quality_path)
    plot_fit_quality_curve(fq)
    return report, gap_stats, fq


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pack = stage_load("exp_param.yaml")

    if USE_SIMPLE_DETECTOR:
        image, idx = pick_grazing_image(pack, phi_index=0)
        print(f"[Main] Using image index {idx} (phi=0 column, steepest theta)")
        detect_out = stage_detect_simple(
            image, line_dir_deg=90.0, fov_width_cm=pack.settings.fov_width_cm,
        )
        stage_evaluate(
            detect_out,
            fov_width_cm=pack.settings.fov_width_cm,
            image_width_px=image.shape[1],
        )
        stage_overlay_simple(image, detect_out)
    else:
        _mag, deg_map = stage_direction(pack)
        patch_size, stride = (512, 512), (256, 256)
        gabor_input = stage_enhance(deg_map, patch_size=patch_size, stride=stride)
        detect_out = stage_detect_legacy(
            gabor_input, patch_size=patch_size, stride=stride,
            fov_width_cm=pack.settings.fov_width_cm,
        )
        stage_evaluate(
            detect_out,
            fov_width_cm=pack.settings.fov_width_cm,
            image_width_px=gabor_input.shape[1],
        )
        stage_overlay_legacy(gabor_input, detect_out)
