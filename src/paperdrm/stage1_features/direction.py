"""
Per-pixel azimuthal direction from a DRP stack.

The DRP stack `[h, w, phi, theta]` is reduced to two maps:
- mag_map: anisotropy strength (how directional the reflectance is)
- deg_map: dominant azimuth angle (degrees)

Computed as the first-order Fourier coefficient of the phi-marginalized
reflectance profile.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from paperdrm.stage0_loader.imagepack import ImagePack


def get_drp_direction(drp_mat: np.ndarray, ph_num: int, attenuation: float = 1.0) -> np.ndarray:
    """
    Calculate overall azimuthal direction of a DRP array.
    Returns a 2D vector representing the weighted average direction.
    """
    mat_mean = np.mean(drp_mat)
    phi_angles = np.linspace(0, 2 * np.pi, ph_num, endpoint=False)
    mag = (drp_mat.mean(axis=1) - mat_mean) * attenuation
    x = np.sum(mag * np.cos(phi_angles))
    y = np.sum(mag * np.sin(phi_angles))
    return np.array([x, y])


def drp_direction_map(imp: "ImagePack", verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-pixel azimuthal direction map from the DRP stack.

    Returns (mag_map, deg_map). Use paperdrm.stage4_viz.direction.plot_direction_map
    to visualize.
    """
    if verbose:
        print("[DRP] computing DRP direction map")
    h, w = imp.h, imp.w
    phi_vec = np.mean(imp.drp_stack, axis=3)  # [h, w, ph_num]
    mean_mat = np.mean(phi_vec, axis=2, keepdims=True)
    phi_angles = np.linspace(0, 2 * np.pi, imp.param.ph_num, endpoint=False)[:, None]
    phi_cos = np.cos(phi_angles)
    phi_sin = np.sin(phi_angles)

    X = (phi_vec - mean_mat) @ phi_cos
    Y = (phi_vec - mean_mat) @ phi_sin
    X = X.reshape(h, w)
    Y = Y.reshape(h, w)
    mag_map = np.sqrt(X**2 + Y**2)
    deg_map = np.degrees(np.arctan2(Y, X))

    mat_mean = np.mean(mag_map)
    mag_map = np.clip(mag_map, None, 2 * mat_mean)
    norm_mag_map = (mag_map - mag_map.min()) / (mag_map.max() - mag_map.min() + 1e-9)

    if verbose:
        print(f"[DRP] direction map computed; mag_map range=({norm_mag_map.min():.3f}, {norm_mag_map.max():.3f})")
    return norm_mag_map, deg_map


def drp_mask_angle(
    mag_map: np.ndarray,
    deg_map: np.ndarray,
    orientation: float,
    threshold: float,
    *,
    verbose: bool = False,
) -> np.ndarray:
    """
    Magnitude-weighted mask of pixels whose orientation is within `threshold`
    of the target `orientation` (degrees).
    """
    if mag_map.shape != deg_map.shape:
        raise ValueError("Magnitude and orientation dimensions do not match.")

    norm_mag_map = (mag_map - mag_map.min()) / (mag_map.max() - mag_map.min() + 1e-9)

    angle_diff = np.abs(((deg_map - orientation + 180) % 360) - 180)
    mask = angle_diff <= threshold
    if verbose:
        keep_pct = 100 * np.mean(mask)
        print(f"[DRP] mask around {orientation}±{threshold} keeps {keep_pct:.1f}% of pixels")
    return norm_mag_map * mask
