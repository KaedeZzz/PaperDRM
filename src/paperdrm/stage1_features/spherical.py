"""
Per-pixel spherical moments and descriptors from the DRP stack.

The DRP stack is treated as samples on a unit hemisphere indexed by (phi, theta).
First-order (vector) and second-order (covariance) moments give the dominant
3D direction, its spread, and anisotropy.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from paperdrm.stage0_loader.imagepack import ImagePack


def spherical_descriptor(
    imp: "ImagePack",
    subtract_mean: bool = True,
    include_sin_theta: bool = False,
    eps: float = 1e-9,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-pixel first-order (vector) and second-order (covariance) spherical moments.

    Returns:
        m1_map: [h, w, 3] first-order moment (dominant 3D direction, not normalised).
        cov_map: [h, w, 3, 3] second-order spread around m1.
        weight_sum: [h, w] sum of absolute weights used for normalisation.
    """
    if verbose:
        print("spherical_descriptor: preparing stack")
    stack = imp.drp_stack.astype(np.float64)
    if subtract_mean:
        stack = stack - stack.mean(axis=(2, 3), keepdims=True)

    phi = np.deg2rad(np.linspace(imp.param.ph_min, imp.param.ph_max, imp.param.ph_num, endpoint=True))
    theta = np.deg2rad(np.linspace(imp.param.th_min, imp.param.th_max, imp.param.th_num, endpoint=True))
    phi_grid = phi[:, None]
    theta_grid = theta[None, :]

    sin_theta = np.sin(theta_grid)
    sin_phi = np.sin(phi_grid)
    cos_theta = np.cos(theta_grid)
    cos_phi = np.cos(phi_grid)
    dir_grid = np.stack(
        [
            cos_theta * cos_phi,
            cos_theta * sin_phi,
            np.broadcast_to(sin_theta, (imp.param.ph_num, imp.param.th_num)),
        ],
        axis=-1,
    )

    if verbose:
        print("spherical_descriptor: computing weights")
    weight = stack
    if include_sin_theta:
        weight = weight * np.sin(theta_grid)

    if verbose:
        print("spherical_descriptor: normalising weights")
    weight_sum = np.sum(np.abs(weight), axis=(2, 3), keepdims=True)
    norm_weight = weight / (weight_sum + eps)

    if verbose:
        print("spherical_descriptor: computing first-order moment")
    m1_map = np.sum(norm_weight[..., None] * dir_grid[None, None, ...], axis=(2, 3))

    if verbose:
        print("spherical_descriptor: computing covariance")
    centered = dir_grid[None, None, ...] - m1_map[..., None, None, :]
    cov_map = np.einsum(
        "hwpt,hwpti,hwptj->hwij",
        norm_weight,
        centered,
        centered,
    )

    if verbose:
        print("spherical_descriptor: done")
    return m1_map, cov_map, weight_sum[..., 0, 0]


def spherical_descriptor_maps(
    imp: "ImagePack",
    subtract_mean: bool = True,
    include_sin_theta: bool = False,
    eps: float = 1e-9,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper to compute direction, projection, and strength maps
    from per-pixel spherical moments.

    Returns:
        dir_map: [h, w, 3] unit vectors (normalised first-order moment).
        proj_map: [h, w, 2] azimuth/elevation in degrees.
        strength_map: [h, w] magnitude of the first-order moment.
        m1_map: [h, w, 3] raw first-order moment.
        cov_map: [h, w, 3, 3] covariance.
    """
    m1_map, cov_map, _ = spherical_descriptor(
        imp,
        subtract_mean=subtract_mean,
        include_sin_theta=include_sin_theta,
        eps=eps,
        verbose=verbose,
    )

    strength_map = np.linalg.norm(m1_map, axis=2, keepdims=True)
    dir_map = m1_map / (strength_map + eps)

    az_map = np.degrees(np.arctan2(dir_map[..., 1], dir_map[..., 0]))
    el_map = np.degrees(np.arcsin(np.clip(dir_map[..., 2], -1.0, 1.0)))
    proj_map = np.stack([az_map, el_map], axis=-1)

    strength_map = strength_map[..., 0]
    return dir_map, proj_map, strength_map, m1_map, cov_map


def _sorted_eigh_3x3(cov_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetrise and eigen-decompose a per-pixel 3x3 covariance."""
    cov_sym = 0.5 * (cov_map + np.swapaxes(cov_map, -1, -2))
    evals, evecs = np.linalg.eigh(cov_sym)
    return evals, evecs


def anisotropy_map_from_cov(cov_map: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Per-pixel anisotropy score from covariance eigenvalues.
    High when one principal axis dominates (line-like), low when isotropic.
    """
    evals, _ = _sorted_eigh_3x3(cov_map)
    lam0, lam1, lam2 = evals[..., 0], evals[..., 1], evals[..., 2]
    lam_sum = lam0 + lam1 + lam2
    score = (lam2 - lam1) / (lam_sum + eps)
    return score


def plane_orientation_map_from_cov(cov_map: np.ndarray, eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """
    Dominant in-plane orientation and tilt from the leading eigenvector of covariance.

    Returns (az_map, el_map) in degrees.
    """
    _, evecs = _sorted_eigh_3x3(cov_map)
    v = evecs[..., :, -1]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]

    xy_norm = np.sqrt(vx**2 + vy**2)
    az_map = np.degrees(np.arctan2(vy, vx))
    az_map = np.where(xy_norm > eps, az_map, np.nan)

    el_map = np.degrees(np.arcsin(np.clip(vz, -1.0, 1.0)))
    return az_map, el_map
