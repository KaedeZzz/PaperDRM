import sys
import numpy as np
import scipy
from matplotlib import pyplot as plt
import cv2

from paperdrm import ImagePack, Settings
from paperdrm.drp_direction import drp_direction_map, drp_mask_angle
from paperdrm.spectral_tv import split_text_background
from paperdrm.line_detection import (
        hough_transform,
        find_hough_peaks,
        dominant_orientation_from_accumulator,
        rotate_image_to_orientation,
        overlay_hough_lines,
    )
from paperdrm.gabor_laidlines import (
        estimate_laidline_frequency_gabor,
        estimate_laidline_frequency_gabor_patches,
        overlay_laid_lines,
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
    # Direction map + mask
    mag_map, deg_map = drp_direction_map(images, verbose=settings.verbose)

    log_stage("Building trigonometric mask")
    # Normalize orientation to 0–255 and show
    target_angle = -90.0  # target direction for DRP
    sigma_deg = 12.0
    # shortest signed angular difference in [-180, 180]
    ang = (deg_map - target_angle + 180) % 360 - 180
    resp = np.exp(-(ang ** 2) / (2 * sigma_deg ** 2))
    img = (resp * 255).astype(np.uint8)
    plt.imshow(img, cmap="gray")
    plt.title("Direction Map (0–255 normalized)")
    plt.show()

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
    
    out = estimate_laidline_frequency_gabor_patches(
        img,
        line_dir_deg=90.0,
        patch_size=(1024, 1024),
        stride=(512, 512),
        periods_px=list(range(6, 41, 2)),
        min_score=0.02,
        weight_scale=3.0
    )
    # Visualize per-patch best score on a grid.
    patch_map = {}
    for p in out["patch_results"]:
        patch_map[(p["y"], p["x"])] = p["best_score"]
    ys = sorted(set([p["y"] for p in out["patch_results"]]))
    xs = sorted(set([p["x"] for p in out["patch_results"]]))
    if len(ys) > 0 and len(xs) > 0:
        period_grid = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                period_grid[yi, xi] = patch_map.get((y, x), np.nan)
        plt.figure(figsize=(6, 5))
        plt.imshow(period_grid, cmap="magma", aspect="auto")
        plt.colorbar(label="Best score")
        plt.title("Patch Best-Score Map")
        plt.xlabel("Patch column")
        plt.ylabel("Patch row")
        plt.tight_layout()
        plt.show()
    print(
        "dominant orientation: ",
        out["line_dir_deg"],
        " dominant period(px) =",
        out["dominant_period_px"],
        " best freq =",
        out["dominant_freq_cpp"],
        "cpp",
    )
    overlay, peaks_x = overlay_laid_lines(
        img,
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






