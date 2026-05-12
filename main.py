"""
End-to-end laid-line detection pipeline.

Stages (each maps to a paperdrm.stageN_* subpackage):
  0. stage0_loader     -- load DRP stack from disk
  1. stage1_features   -- per-pixel DRP direction map
  2. stage2_enhance    -- orientation -> laid-line-likelihood grayscale
  3. stage3_detect     -- patchwise Gabor period estimation + grid overlay
  4. stage4_viz        -- plotting helpers (called from inside the stages)
  5. stage5_evaluation -- evaluation tooling (not used in this end-to-end script)
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


def stage_load(yaml_path: str) -> ImagePack:
    print("[Stage 0] Loading settings + DRP stack")
    settings = Settings.from_yaml(yaml_path).with_overrides(angle_slice=(2, 2), verbose=True)
    return ImagePack(settings=settings)


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

    print(
        f"Patchwise trig: patch_size={patch_size} stride={stride} sigma_deg={sigma_deg}"
        f" | diff mean={diff_img.mean():.2f} max={diff_img.max()}"
    )
    return patch_img


def stage_detect(gabor_input_gray, *, patch_size, stride, fov_width_cm=None):
    print("[Stage 3] Patchwise Gabor laid-line frequency estimation")
    out = estimate_laidline_frequency_gabor_patches(
        gabor_input_gray,
        line_dir_deg=90.0,
        patch_size=patch_size,
        stride=stride,
        periods_px=list(range(6, 41, 2)),
        min_score=0.02,
        weight_scale=3.0,
    )
    plot_patch_best_score_map(out["patch_results"])

    period_px = out["dominant_period_px"]
    print(f"dominant orientation={out['line_dir_deg']:.1f} deg | period={period_px:.1f} px | freq={out['dominant_freq_cpp']:.4f} cpp")

    if fov_width_cm is not None:
        cm_per_px = float(fov_width_cm) / float(gabor_input_gray.shape[1])
        interval_cm = period_px * cm_per_px
        lines_per_cm = (1.0 / interval_cm) if interval_cm > 0 else float("inf")
        print(f"laid line interval = {interval_cm:.4f} cm | density = {lines_per_cm:.4f} lines/cm")

    return out


def stage_evaluate(detect_out, *, score_threshold=0.02, report_path="evaluation_report.json"):
    print("[Stage 5] Evaluating patch consistency")
    report = patch_consistency_report(detect_out, score_threshold=score_threshold)
    print_consistency_report(report)
    save_consistency_report(report, report_path)
    plot_patch_consistency(detect_out, report, score_threshold=score_threshold)
    return report


def stage_overlay(gabor_input_gray, detect_out, *, out_path="laid_lines_overlay_grid.png"):
    print(f"[Stage 4] Overlay laid-line grid -> {out_path}")
    overlay, _ = overlay_laid_lines(
        gabor_input_gray,
        line_dir_deg=detect_out["line_dir_deg"],
        best_signal_1d=detect_out["dominant_signal_1d"],
        best_period_px=detect_out["dominant_period_px"],
        color=(0, 0, 255),
        thickness=1,
        alpha=0.65,
        mode="grid",
    )
    cv2.imwrite(out_path, overlay)
    win = "Laid Lines Overlay (Grid)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1200, 800)
    cv2.imshow(win, overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    pack = stage_load("exp_param.yaml")
    _mag_map, deg_map = stage_direction(pack)
    patch_size, stride = (512, 512), (256, 256)
    gabor_input = stage_enhance(deg_map, patch_size=patch_size, stride=stride)
    detect_out = stage_detect(
        gabor_input,
        patch_size=patch_size,
        stride=stride,
        fov_width_cm=pack.settings.fov_width_cm,
    )
    stage_evaluate(detect_out)
    stage_overlay(gabor_input, detect_out)
