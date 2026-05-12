"""
Comparison plots for Stage 2 (trig masks) and Stage 3 (Gabor patches).
"""

import numpy as np
from matplotlib import pyplot as plt


def plot_orientation_comparison(
    raw_azimuth_orientation_8u: np.ndarray,
    patch_target_8u: np.ndarray,
    orientation_diff_deg: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw_azimuth_orientation_8u, cmap="hsv", vmin=0, vmax=255)
    axes[0].set_title("Raw Azimuthal Orientation")
    axes[0].axis("off")

    axes[1].imshow(patch_target_8u, cmap="hsv", vmin=0, vmax=255)
    axes[1].set_title("Patch Dominant Orientation")
    axes[1].axis("off")

    im_diff = axes[2].imshow(orientation_diff_deg, cmap="inferno", vmin=0, vmax=180)
    axes[2].set_title("Absolute Angular Difference (deg)")
    axes[2].axis("off")

    cbar = fig.colorbar(im_diff, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("deg")
    plt.tight_layout()
    plt.show()


def plot_trig_mask_comparison(
    prev_raw_img: np.ndarray,
    patch_raw_img: np.ndarray,
    target_img: np.ndarray,
    prev_img: np.ndarray,
    patch_img: np.ndarray,
    diff_img: np.ndarray,
    target_angle: float,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes[0, 0].imshow(prev_raw_img, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(f"Previous Trig Raw (target={target_angle:.0f} deg)")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(patch_raw_img, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Patchwise Trig Raw")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(target_img, cmap="hsv", vmin=0, vmax=255)
    axes[0, 2].set_title("Patch Dominant Orientation")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(prev_img, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("Previous Trig Enhanced")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(patch_img, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("Patchwise Trig Enhanced")
    axes[1, 1].axis("off")
    axes[1, 2].imshow(diff_img, cmap="inferno", vmin=0, vmax=255)
    axes[1, 2].set_title("Absolute Difference (Enhanced)")
    axes[1, 2].axis("off")
    plt.tight_layout()
    plt.show()


def plot_patch_best_score_map(patch_results: list[dict]) -> None:
    """Per-patch confidence (best Gabor score) shown as a 2D heatmap."""
    patch_map = {(p["y"], p["x"]): p["best_score"] for p in patch_results}

    ys = sorted({p["y"] for p in patch_results})
    xs = sorted({p["x"] for p in patch_results})
    if not ys or not xs:
        return

    score_grid = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            score_grid[yi, xi] = patch_map.get((y, x), np.nan)

    plt.figure(figsize=(6, 5))
    plt.imshow(score_grid, cmap="magma", aspect="auto")
    plt.colorbar(label="Best score")
    plt.title("Patch Best-Score Map")
    plt.xlabel("Patch column")
    plt.ylabel("Patch row")
    plt.tight_layout()
    plt.show()
