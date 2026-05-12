"""Visualizations for the DRP direction map (Stage 1)."""

import numpy as np
from matplotlib import pyplot as plt


def plot_direction_map(
    sample_image: np.ndarray,
    mag_map: np.ndarray,
    deg_map: np.ndarray,
) -> None:
    """
    Three-panel summary: sample image, normalised magnitude, angle map.
    """
    fig, axes = plt.subplots(figsize=(13, 4), ncols=3)
    im1 = axes[0].imshow(sample_image, cmap="gray")
    fig.colorbar(im1, ax=axes[0])
    axes[0].set_title("ROI image")
    im2 = axes[1].imshow(mag_map, cmap="afmhot")
    fig.colorbar(im2, ax=axes[1])
    axes[1].set_title("Normalised DRP magnitudes")
    im3 = axes[2].imshow(deg_map, cmap="hsv")
    axes[2].set_title("DRP angles")
    fig.colorbar(im3, ax=axes[2])
    plt.tight_layout()
    plt.show()
