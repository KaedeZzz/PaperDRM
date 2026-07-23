"""Plot a DRP array in stereo or direct projection on the unit hemisphere."""

import numpy as np
from matplotlib import pyplot as plt

from paperdrm.stage0_loader.settings import DRPConfig


def plot_drp(drp_array: np.ndarray, config: DRPConfig, cmap: str = "jet", project: str = "stereo", ax=None):
    """
    Plot a DRP array using stereo or direct projection.
    """
    if drp_array.shape != (config.ph_num, config.th_num):
        raise ValueError("Input DRP array size does not match image pack parameters.")

    if np.issubdtype(drp_array.dtype, np.floating) and drp_array.max() <= 1.0:
        drp_array = (drp_array * 255).astype(np.uint8)

    phi, theta = np.meshgrid(
        np.linspace(0, 360 + config.ph_step, config.ph_num + 1),
        np.linspace(config.th_min, config.th_max + config.th_step, config.th_num + 1),
    )

    if project == "stereo":
        xx = np.cos(np.radians(theta)) * np.cos(np.radians(phi)) / (1 + np.sin(np.radians(theta)))
        yy = np.cos(np.radians(theta)) * np.sin(np.radians(phi)) / (1 + np.sin(np.radians(theta)))
    elif project == "direct":
        xx = np.cos(np.radians(theta)) * np.cos(np.radians(phi))
        yy = np.cos(np.radians(theta)) * np.sin(np.radians(phi))
    else:
        raise ValueError("Unknown project type. Use 'stereo' or 'direct'.")

    if ax is None:
        _, ax = plt.subplots()
    h = ax.pcolormesh(xx, yy, drp_array.T, cmap=cmap, shading="auto")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.colorbar(h, ax=ax)
    return ax
