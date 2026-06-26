"""
End-to-end laid-line detection pipeline (three tracks).

Tracks are selected by the DETECTOR_TRACK constant below:

MULTI_PHI TRACK (default, recommended when a DRP stack is available)
  N grazing images (one per phi) -> per-image radial FFT ->
  normalise + sum spectra -> dominant period -> per-image phase fit ->
  weighted circular mean phase -> grid. Boosts period SNR by averaging
  phi-stationary signal across phi-random noise.

SIMPLE TRACK
  Single grazing-light image -> radial FFT -> Gabor cleanup -> grid.
  Bypasses DRP direction map and trig-mask preprocessing. Kept as an
  ablation baseline against MULTI_PHI.

LEGACY TRACK
  DRP stack -> direction map -> trig mask -> patchwise Gabor -> grid.
  Slow (~4.5 min), KNOWN BIASED: detects half the true period due to
  abs-response harmonic doubling + proportional band-width bias in the
  Gabor scoring. Kept for reference / ablation only.

Stages used by each track:
  0. stage0_loader     -- DRP stack loading             (all)
  1. stage1_features   -- DRP direction map             (legacy only)
  2. stage2_enhance    -- trig mask                     (legacy only)
  3. stage3_detect     -- period + grid                 (all, different)
  4. (drawing)         -- overlay onto image            (all)
  5. stage5_evaluation -- gap distribution + fit-quality (all)
                          + split-half + self-contrast  (multi_phi + simple)
                          + patch consistency           (legacy only)
"""

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from detect_paper_roi import detect_paper_roi_texture

from paperdrm import ImagePack, Settings
from paperdrm.stage0_drp.slicing import apply_angle_slice, apply_theta_min_filter
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
    auto_detect_line_dir as _auto_detect_line_dir,
    overlay_grid,
    overlay_grid_bands,
)
from paperdrm.stage3_detect.multi_phi_detector import (
    collect_grazing_per_phi,
    detect_laid_lines_multi_phi,
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
from paperdrm.stage5_evaluation.wire_width_stats import (
    wire_width_statistics,
    print_wire_width_statistics,
    save_wire_width_statistics,
    plot_wire_width_statistics,
)
from paperdrm.stage5_evaluation.split_half import (
    plot_split_half,
    print_split_half,
    save_split_half,
    split_half_period_stability,
)
from paperdrm.stage5_evaluation.self_contrast import (
    plot_self_contrast,
    print_self_contrast,
    save_self_contrast,
    self_consistency_contrast,
)


# ---------------------------------------------------------------------------
# Track selector: "multi_phi" | "simple" | "legacy" | "single_image"
# ---------------------------------------------------------------------------
DETECTOR_TRACK = "multi_phi"


# ---------------------------------------------------------------------------
# Result archiving
# ---------------------------------------------------------------------------
def archive_results(pack: "ImagePack | None", config_path: str, *, serial: str | None = None) -> Path:
    """
    Copy all pipeline output files (JSON + PNG) from the repo root into
    results/<serial>/ and save a snapshot of the config yaml.
    Returns the archive directory.
    """
    if serial is None:
        serial = str(pack.data_serial) if pack is not None and pack.data_serial is not None else "unknown"
    root = Path(__file__).parent
    archive_dir = root / "results" / serial
    archive_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for src in sorted(root.glob("*.json")) + sorted(root.glob("*.png")):
        dst = archive_dir / src.name
        shutil.copy2(src, dst)
        copied.append(src.name)

    # Config snapshot. Skip if --config already points inside archive_dir
    # (e.g. results/<serial>/exp_param.yaml): copying a file onto itself
    # raises PermissionError on Windows.
    cfg_src = Path(config_path)
    if cfg_src.exists():
        cfg_dst = archive_dir / cfg_src.name
        if cfg_src.resolve() != cfg_dst.resolve():
            shutil.copy2(cfg_src, cfg_dst)
            copied.append(cfg_src.name)

    print(f"[Archive] {len(copied)} files -> results/{serial}/  ({', '.join(copied)})")

    # Generate plain-language reports
    try:
        import subprocess, sys
        report_script = root / "scripts" / "generate_report.py"
        subprocess.run(
            [sys.executable, str(report_script), "--serial", serial,
             "--results-dir", str(root / "results")],
            check=True,
        )
    except Exception as exc:
        print(f"[Archive] Report generation failed: {exc}")

    return archive_dir


# ---------------------------------------------------------------------------
# Stage 0 (shared by both tracks)
# ---------------------------------------------------------------------------
def stage_load(yaml_path: str) -> ImagePack:
    print("[Stage 0] Loading settings + DRP stack")
    settings = Settings.from_yaml(yaml_path)
    # Default angle_slice to (2, 2) unless the yaml supplied one explicitly.
    if settings.angle_slice == (1, 1):
        settings = settings.with_overrides(angle_slice=(2, 2))
    settings = settings.with_overrides(verbose=True)
    return ImagePack(settings=settings)


def stage_load_single_image(
    image_path: Path,
    *,
    subtract_background: bool = True,
    subtraction_scale_percentile: float = 99.5,
    crop_roi: "tuple[int,int,int,int] | None" = None,
    fov_width_cm: float | None = None,
) -> "tuple[np.ndarray, np.ndarray, float | None]":
    """
    Load one image for the single-image track.
    Returns (processed_image, raw_image, effective_fov_width_cm).
    processed_image has background subtracted (if enabled) and ROI cropped.
    raw_image is the original pixels, ROI cropped only (no bg subtraction),
    used for the visual overlay.
    """
    print(f"[Stage 0 SINGLE] Loading {image_path}")
    raw = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise IOError(f"Could not open image: {image_path}")
    print(f"[Stage 0 SINGLE] Image shape: {raw.shape}")

    image = raw.copy()
    if subtract_background:
        bg = cv2.GaussianBlur(image, (0, 0), sigmaX=100, borderType=cv2.BORDER_REFLECT_101)
        diff = image.astype(np.float32) - bg.astype(np.float32)
        diff = np.clip(diff, 0, None)
        ref = float(np.percentile(diff, subtraction_scale_percentile))
        scale = 255.0 / max(ref, 1.0)
        image = np.clip(diff * scale, 0, 255).astype(np.uint8)
        print("[Stage 0 SINGLE] Gaussian blur background subtracted (sigma=100)")

    effective_fov = fov_width_cm
    if crop_roi is not None:
        x, y, w, h = crop_roi
        orig_w = image.shape[1]
        raw = raw[y:y + h, x:x + w]
        image = image[y:y + h, x:x + w]
        if fov_width_cm is not None:
            effective_fov = fov_width_cm * w / orig_w
        print(f"[Stage 0 SINGLE] ROI crop [x={x},y={y},w={w},h={h}] -> {image.shape}, "
              f"fov_width_cm -> {effective_fov}")

    return image, raw, effective_fov


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


def pick_grazing_image_raw(pack: ImagePack, *, phi_index: int = 0) -> "tuple[np.ndarray, int]":
    """
    Same selection as `pick_grazing_image`, but loads the **raw** image from
    disk (pre background-subtraction). Used for visual overlays so the
    underlying paper texture is preserved.
    """
    paths = sorted(pack.folder.glob(f"*.{pack.settings.img_format}"))
    paths_sliced, cfg = apply_angle_slice(paths, pack.base_config, pack.angle_slice)
    paths_kept, _ = apply_theta_min_filter(paths_sliced, cfg, pack.settings.theta_min_deg)
    th_num = pack.param.th_num
    idx = phi_index * th_num + (th_num - 1)
    if idx >= len(paths_kept):
        idx = len(paths_kept) - 1
    raw = cv2.imread(str(paths_kept[idx]), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise IOError(f"Could not load raw image at {paths_kept[idx]}")
    return raw, idx


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
    cm_per_px = (float(fov_width_cm) / float(image.shape[1])) if fov_width_cm else None
    if cm_per_px is not None:
        interval_cm = period_px * cm_per_px
        print(f"laid line interval = {interval_cm:.4f} cm | density = {1.0/interval_cm:.4f} lines/cm")
    if result["wire_model_ok"]:
        fwhm_px = result["wire_fwhm_px"]
        if cm_per_px is not None:
            print(f"wire FWHM = {fwhm_px:.2f} px = {fwhm_px * cm_per_px * 10.0:.3f} mm")
        else:
            print(f"wire FWHM = {fwhm_px:.2f} px")
        if result["wire_warning"]:
            print(f"  [wire-width warning] {result['wire_warning']}")
    else:
        print(f"wire width: model failed ({result['wire_warning']})")
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


def stage_overlay_simple_bands(
    image, detect_out, ww_stats=None,
    *, out_path="laid_lines_overlay_bands.png", alpha=0.4,
):
    """Filled band overlay; band width = FWHM (segment-median if available)."""
    if ww_stats and ww_stats["aggregate"]["fwhm_px"]["n_valid"] >= 1:
        fwhm = ww_stats["aggregate"]["fwhm_px"]["median"]
        source = "segment-median"
    else:
        fwhm = detect_out["wire_fwhm_px"]
        source = "global"
    print(f"[Stage 4 SIMPLE] Band overlay (FWHM={fwhm:.2f} px from {source}) -> {out_path}")
    overlay = overlay_grid_bands(
        image,
        detect_out["grid_positions_x"],
        fwhm,
        line_dir_deg=detect_out["line_dir_deg"],
        color=(0, 0, 255),
        alpha=alpha,
    )
    cv2.imwrite(out_path, overlay)


# ---------------------------------------------------------------------------
# Stage 5 (shared, gracefully skips patch consistency when not applicable)
# ---------------------------------------------------------------------------
def stage_evaluate(
    detect_out,
    *,
    image=None,
    score_threshold=0.02,
    fov_width_cm=None,
    image_width_px=None,
    consistency_path="evaluation_report.json",
    intervals_path="interval_distribution.json",
    fit_quality_path="fit_quality.json",
    wire_width_path="wire_width_stats.json",
    n_segments=16,
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

    ww = None
    if image is not None:
        print(f"[Stage 5] Wire-width statistics ({n_segments} segments)")
        ww = wire_width_statistics(
            image,
            detect_out["dominant_period_px"],
            line_dir_deg=detect_out["line_dir_deg"],
            n_segments=n_segments,
            fov_width_cm=fov_width_cm,
        )
        print_wire_width_statistics(ww)
        save_wire_width_statistics(ww, wire_width_path)
        plot_wire_width_statistics(ww, save_path="wire_width_segments.png")
    return report, gap_stats, fq, ww


# ---------------------------------------------------------------------------
# Stage 3 MULTI_PHI TRACK
# ---------------------------------------------------------------------------
def stage_detect_multi_phi(
    images,
    *,
    line_dir_deg: float = 90.0,
    period_range_px: tuple[float, float] = (8.0, 80.0),
    wire_is_darker: bool = True,
    fov_width_cm: float | None = None,
):
    print(f"[Stage 3 MULTI_PHI] Aggregating power spectra across {len(images)} phi images")
    result = detect_laid_lines_multi_phi(
        images,
        line_dir_deg=line_dir_deg,
        period_range_px=period_range_px,
        wire_is_darker=wire_is_darker,
        use_gabor_refinement=True,
    )
    period_px = result["dominant_period_px"]
    rep = result["representative_index"]
    R_raw = result["phase_resultant_length_raw"]
    R = result["phase_resultant_length"]
    n_flipped = result["n_polarity_flipped"]
    n = result["n_images"]
    corrected = result["phase_auto_corrected"]
    print(f"period={period_px:.2f} px  freq={result['dominant_freq_cpp']:.5f} cpp"
          f"  representative_phi_idx={rep}  anchor={result['anchor_index']}"
          + ("  [phase +T/2 auto-corrected]" if corrected else ""))
    print(f"  phase coherence: R_raw={R_raw:.3f} -> R_aligned={R:.3f}"
          f"  (circ_var={result['phase_circular_var']:.4f},"
          f"  polarity-flipped {n_flipped}/{n} phi)")
    image_width = images[rep].shape[1]
    cm_per_px = (float(fov_width_cm) / float(image_width)) if fov_width_cm else None
    if cm_per_px is not None:
        interval_cm = period_px * cm_per_px
        print(f"laid line interval = {interval_cm:.4f} cm | density = {1.0/interval_cm:.4f} lines/cm")
    if result["wire_model_ok"]:
        fwhm_px = result["wire_fwhm_px"]
        if cm_per_px is not None:
            print(f"wire FWHM = {fwhm_px:.2f} px = {fwhm_px * cm_per_px * 10.0:.3f} mm")
        else:
            print(f"wire FWHM = {fwhm_px:.2f} px")
        if result["wire_warning"]:
            print(f"  [wire-width warning] {result['wire_warning']}")
    else:
        print(f"wire width: model failed ({result['wire_warning']})")
    return result


def stage_split_half(
    images,
    *,
    line_dir_deg: float = 90.0,
    period_range_px: tuple[float, float] = (8.0, 80.0),
    fov_width_cm: float | None = None,
    n_splits: int = 100,
    seed: int = 0,
    out_path: str = "split_half_stability.json",
    plot_path: str | None = "split_half_stability.png",
):
    print(f"[Stage 5] Split-half period stability ({n_splits} splits)")
    stats = split_half_period_stability(
        images,
        line_dir_deg=line_dir_deg,
        period_range_px=period_range_px,
        n_splits=n_splits,
        seed=seed,
        fov_width_cm=fov_width_cm,
    )
    print_split_half(stats)
    save_split_half(stats, out_path)
    plot_split_half(stats, save_path=plot_path)
    return stats


def stage_self_contrast(
    image,
    detect_out,
    *,
    band_half_width_px: int = 1,
    out_path: str = "self_contrast.json",
    plot_path: str | None = "self_contrast.png",
):
    print("[Stage 5] Self-consistency contrast")
    stats = self_consistency_contrast(
        image,
        detect_out["grid_positions_x"],
        detect_out["dominant_period_px"],
        line_dir_deg=detect_out["line_dir_deg"],
        band_half_width_px=band_half_width_px,
        wire_is_darker=detect_out["wire_is_darker"],
    )
    print_self_contrast(stats)
    save_self_contrast(stats, out_path)
    plot_self_contrast(stats, save_path=plot_path)
    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(description="PaperDRM pipeline entry point.")
    _parser.add_argument(
        "--config",
        default="exp_param.yaml",
        help="Path to the settings yaml (default: exp_param.yaml).",
    )
    _parser.add_argument(
        "--image",
        default=None,
        help="Path to a single image for the single_image track. "
             "Overrides image_path in the yaml.",
    )
    _args = _parser.parse_args()

    # Single-image track: bypass ImagePack entirely.
    # Triggered by --image CLI arg or by image_path being set in the config yaml.
    _single_image_path: Path | None = None
    if _args.image:
        _single_image_path = Path(_args.image)
    else:
        _settings_peek = Settings.from_yaml(_args.config)
        _single_image_path = _settings_peek.image_path

    if _single_image_path is not None:
        _cfg = Settings.from_yaml(_args.config)
        _serial = str(_cfg.data_serial) if _cfg.data_serial is not None else _single_image_path.stem
        print(f"[Main] SINGLE_IMAGE track: {_single_image_path}  serial={_serial}")

        _crop_roi = _cfg.crop_roi
        if _crop_roi is None:
            _raw_for_roi = cv2.imread(str(_single_image_path), cv2.IMREAD_GRAYSCALE)
            _crop_roi = detect_paper_roi_texture(_raw_for_roi)
            print(f"[Main] auto crop_roi -> {list(_crop_roi)}")
            del _raw_for_roi

        _image, _raw_image, _eff_fov = stage_load_single_image(
            _single_image_path,
            subtract_background=_cfg.subtract_background,
            subtraction_scale_percentile=_cfg.subtraction_scale_percentile,
            crop_roi=_crop_roi,
            fov_width_cm=_cfg.fov_width_cm,
        )
        _w_px = _image.shape[1]
        if _cfg.period_range_cm is not None and _eff_fov is not None:
            _cm_per_px = _eff_fov / _w_px
            _period_range_px = (
                _cfg.period_range_cm[0] / _cm_per_px,
                _cfg.period_range_cm[1] / _cm_per_px,
            )
            print(f"[Main] period_range_cm={_cfg.period_range_cm} -> "
                  f"period_range_px=({_period_range_px[0]:.1f}, {_period_range_px[1]:.1f})")
        else:
            _period_range_px = (8.0, 80.0)

        if _cfg.auto_line_dir:
            # centre search on direction perpendicular to the long side of the crop
            _h_img, _w_img = _image.shape[:2]
            _center_deg = 0.0 if _h_img >= _w_img else 90.0
            _line_dir = _auto_detect_line_dir(
                _image,
                period_range_px=_period_range_px,
                center_deg=_center_deg,
            )
            print(f"[Main] auto_line_dir (center={_center_deg:.0f}°±20°) -> {_line_dir:.1f} deg")
            _cfg = _cfg.with_overrides(line_dir_deg=_line_dir)

        _detect_out = stage_detect_simple(
            _image,
            line_dir_deg=_cfg.line_dir_deg,
            period_range_px=_period_range_px,
            wire_is_darker=_cfg.wire_is_darker,
            fov_width_cm=_eff_fov,
        )
        _, _, _, _ww = stage_evaluate(
            _detect_out,
            image=_image,
            fov_width_cm=_eff_fov,
            image_width_px=_w_px,
        )
        stage_self_contrast(_image, _detect_out)
        stage_overlay_simple(_raw_image, _detect_out)
        stage_overlay_simple_bands(_raw_image, _detect_out, _ww)
        archive_results(None, _args.config, serial=_serial)
        raise SystemExit(0)

    pack = stage_load(_args.config)

    # Convert period_range_cm -> period_range_px using the (post-crop) fov.
    # fov_width_cm is already scaled by ImagePack when crop_roi is applied.
    if pack.settings.period_range_cm is not None:
        if pack.settings.fov_width_cm is None:
            raise ValueError("period_range_cm requires fov_width_cm to be set in the config.")
        _cm_per_px = pack.settings.fov_width_cm / pack.w
        _period_range_px: tuple[float, float] = (
            pack.settings.period_range_cm[0] / _cm_per_px,
            pack.settings.period_range_cm[1] / _cm_per_px,
        )
        print(f"[Main] period_range_cm={pack.settings.period_range_cm} -> "
              f"period_range_px=({_period_range_px[0]:.1f}, {_period_range_px[1]:.1f})")
    else:
        _period_range_px = (8.0, 80.0)

    if DETECTOR_TRACK == "multi_phi":
        images, phi_deg = collect_grazing_per_phi(pack)
        rep_overlay_idx = 0  # raw image used for the visual overlay
        raw_image, _ = pick_grazing_image_raw(pack, phi_index=rep_overlay_idx)
        print(f"[Main] MULTI_PHI track: {len(images)} grazing phi images "
              f"(phi range {phi_deg[0]:.1f}..{phi_deg[-1]:.1f} deg)")
        detect_out = stage_detect_multi_phi(
            images,
            line_dir_deg=90.0,
            period_range_px=_period_range_px,
            fov_width_cm=pack.settings.fov_width_cm,
        )
        ref_image = images[detect_out["representative_index"]]
        _, _, _, ww = stage_evaluate(
            detect_out,
            image=ref_image,
            fov_width_cm=pack.settings.fov_width_cm,
            image_width_px=ref_image.shape[1],
        )
        stage_split_half(
            images,
            period_range_px=_period_range_px,
            fov_width_cm=pack.settings.fov_width_cm,
            n_splits=200,
        )
        stage_self_contrast(ref_image, detect_out)
        stage_overlay_simple(raw_image, detect_out)
        stage_overlay_simple_bands(raw_image, detect_out, ww)
        archive_results(pack, _args.config)
    elif DETECTOR_TRACK == "simple":
        image, idx = pick_grazing_image(pack, phi_index=0)
        raw_image, _ = pick_grazing_image_raw(pack, phi_index=0)
        print(f"[Main] SIMPLE track: image index {idx} (phi=0 column, steepest theta)")
        detect_out = stage_detect_simple(
            image, line_dir_deg=90.0,
            period_range_px=_period_range_px,
            fov_width_cm=pack.settings.fov_width_cm,
        )
        _, _, _, ww = stage_evaluate(
            detect_out,
            image=image,
            fov_width_cm=pack.settings.fov_width_cm,
            image_width_px=image.shape[1],
        )
        stage_self_contrast(
            image, detect_out,
            out_path="self_contrast.simple.json",
            plot_path="self_contrast.simple.png",
        )
        stage_overlay_simple(raw_image, detect_out)
        stage_overlay_simple_bands(raw_image, detect_out, ww)
        archive_results(pack, _args.config)
    elif DETECTOR_TRACK == "legacy":
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
        archive_results(pack, _args.config)
    else:
        raise ValueError(f"Unknown DETECTOR_TRACK={DETECTOR_TRACK!r}; "
                         "expected 'multi_phi', 'simple', or 'legacy'.")
