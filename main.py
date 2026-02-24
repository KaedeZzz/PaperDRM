import cv2

from paperdrm import ImagePack, Settings
from paperdrm.drp_direction import drp_direction_map
from paperdrm.gabor_laidlines import (
        estimate_laidline_frequency_gabor_patches,
        overlay_laid_lines,
    )
from paperdrm.trig_mask import (
    azimuth_to_laidline_gray,
    orientation_comparison_maps,
    patchwise_trigonometric_mask,
)
from paperdrm.visualization import (
    plot_orientation_comparison,
    plot_patch_best_score_map,
    plot_trig_mask_comparison,
)


if __name__ == "__main__":
    def log_stage(msg: str) -> None:
        print(f"[Stage] {msg}")

    log_stage("Loading settings")
    # Centralised settings loaded from YAML with a small override for angle slicing
    settings = Settings.from_yaml("exp_param.yaml").with_overrides(angle_slice=(2, 2), verbose=True)

    log_stage("Loading images + DRP stack")
    # Load images + DRP (2x2 angular slice)
    images = ImagePack(settings=settings)

    log_stage("Computing direction map + mask")
    # DRP direction output:
    # - mag_map: confidence/anisotropy-like strength per pixel
    # - deg_map: dominant orientation angle (degrees) per pixel
    mag_map, deg_map = drp_direction_map(images, verbose=settings.verbose)

    log_stage("Building trigonometric mask")
    # Build both:
    # 1) previous single-target trig mask (baseline),
    # 2) patch-adaptive trig mask (current approach).
    sigma_deg = 10.0
    target_angle = 90.0
    prev_raw_img, prev_img = azimuth_to_laidline_gray(
        deg_map,
        target_deg=target_angle,
        sigma_deg=sigma_deg,
        enhance=True,
    )

    patch_size = (512, 512)
    stride = (256, 256)
    patch_raw_img, patch_img, patch_target_deg_map, target_img = patchwise_trigonometric_mask(
        deg_map,
        patch_size=patch_size,
        stride=stride,
        sigma_deg=sigma_deg,
        enhance=True,
    )

    # Absolute difference map to highlight where patchwise diverges from baseline.
    diff_img = cv2.absdiff(prev_img, patch_img)

    # Compare patch dominant orientation with raw azimuthal map from earlier method.
    raw_azimuth_orientation_8u, orientation_diff_deg = orientation_comparison_maps(
        deg_map, patch_target_deg_map
    )
    plot_orientation_comparison(raw_azimuth_orientation_8u, target_img, orientation_diff_deg)
    plot_trig_mask_comparison(
        prev_raw_img,
        patch_raw_img,
        target_img,
        prev_img,
        patch_img,
        diff_img,
        target_angle,
    )

    print(
        "Patchwise trig params:",
        f"patch_size={patch_size}",
        f"stride={stride}",
        f"sigma_deg={sigma_deg}",
    )
    print("Difference stats:", f"mean_abs_diff={diff_img.mean():.2f}", f"max_abs_diff={diff_img.max()}")

    # Use patchwise trig enhanced grayscale map for downstream Gabor estimation.
    gabor_input_gray = patch_img
    # log_stage("Running spectral TV decomposition")
    # # Spectral TV decomposition to split text vs paper background
    # img_float = img.astype(np.float32)
    # img_float = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-8)
    # text_layer, paper_layer, spectral_result = split_text_background(
    #     img_float,
    #     text_max_time=30.0,
    #     num_steps=600,
    #     dt=0.1,
    #     eps=2e-6,
    #     verbose=settings.verbose,
    #     progress_every=20,
    #     text_is_coarse=True,
    # )

    # corr = img_float / (paper_layer + 1e-6)     
    # ink = np.maximum(1.0 - corr, 0)
    # ink = ink / (np.percentile(ink, 99.5) + 1e-8)
    # ink = np.clip(ink, 0, 1)

    # log_stage("Plotting spectral TV layers")
    # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # axes[0].imshow(img_float, cmap="gray")
    # axes[0].set_title("Input")
    # axes[1].imshow(text_layer, cmap="gray")
    # axes[1].set_title("Ink (small scale)")
    # axes[2].imshow(paper_layer, cmap="gray")
    # axes[2].set_title("Paper background (large scale)")
    # for ax in axes:
    #     ax.axis("off")
    # plt.tight_layout()
    # plt.show()

    log_stage("Gabor filter for laid line frequency estimation")
    # Patch-based Gabor scan:
    # estimate local dominant spacing (period in pixels), then aggregate.
    out = estimate_laidline_frequency_gabor_patches(
        gabor_input_gray,
        line_dir_deg=90.0,
        patch_size=patch_size,
        stride=stride,
        periods_px=list(range(6, 41, 2)),
        min_score=0.02,
        weight_scale=3.0
    )
    # Visualize per-patch best score (higher means stronger periodic evidence).
    plot_patch_best_score_map(out["patch_results"])
    print(
        "dominant orientation: ",
        out["line_dir_deg"],
        " dominant period(px) =",
        out["dominant_period_px"],
        " best freq =",
        out["dominant_freq_cpp"],
        "cpp",
    )
    if settings.fov_width_cm is not None:
        image_width_px = gabor_input_gray.shape[1]
        cm_per_px = float(settings.fov_width_cm) / float(image_width_px)
        laid_interval_cm = float(out["dominant_period_px"]) * cm_per_px
        lines_per_cm = (1.0 / laid_interval_cm) if laid_interval_cm > 0 else float("inf")
        print(
            "estimated laid line interval =",
            f"{laid_interval_cm:.4f}",
            "cm",
            f"(using fov_width_cm={float(settings.fov_width_cm):.4f}, image_width_px={image_width_px})",
        )
        print("estimated laid line density =", f"{lines_per_cm:.4f}", "lines/cm")
    else:
        print("No fov_width_cm set in exp_param.yaml; skipping laid line interval and density in centimeters.")

    # Overlay the inferred laid-line grid on the grayscale likelihood image.
    overlay, peaks_x = overlay_laid_lines(
        gabor_input_gray,
        line_dir_deg=out["line_dir_deg"],
        best_signal_1d=out["dominant_signal_1d"],
        best_period_px=out["dominant_period_px"],
        color=(0, 0, 255),
        thickness=1,
        alpha=0.65,
        mode="grid",
    )
    cv2.imwrite("laid_lines_overlay_grid.png", overlay)
    win_grid = "Laid Lines Overlay (Grid)"
    cv2.namedWindow(win_grid, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_grid, 1200, 800)
    cv2.imshow(win_grid, overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # log_stage("Computing column intensity profile")
    # # Column intensity profile
    # img_stacked = img.mean(axis=0)
    # plt.plot(img_stacked)
    # plt.title("Stacked Image Intensity Profile")
    # plt.xlabel("Pixel Position")
    # plt.ylabel("Average Intensity")
    # plt.show()

    # log_stage("Peak detection and overlay")
    # # Peak detection (top 13 by height) and overlay
    # peaks, _ = scipy.signal.find_peaks(img_stacked)
    # peaks_by_height = peaks[np.argsort(img_stacked[peaks])[::-1]]
    # keep = 50
    # plt.imshow(img, cmap="gray")
    # for peak in peaks_by_height[:keep]:
    #     plt.plot([peak, peak], [0, img.shape[0]], color="red", linewidth=1)
    # plt.show()
