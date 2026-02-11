from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from .imagepack import ImagePack


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


def drp_direction_map(imp: "ImagePack", display: bool = True, verbose: bool = False):
    """
    Calculate (and optionally display) the direction map of a DRP over all pixels.
    Returns magnitude and angle maps.
    """
    if verbose:
        print("[DRP] computing DRP direction map")
    h, w = imp.h, imp.w
    phi_vec = np.mean(imp.drp_stack, axis=3)  # [h, w, ph_num]
    mean_mat = np.mean(phi_vec, axis=2, keepdims=True)
    phi_angles = np.linspace(0, 2 * np.pi, imp.param.ph_num, endpoint=False)[:, None]
    phi_cos = np.cos(phi_angles)
    phi_sin = np.sin(phi_angles)

    # Project onto unit circle basis for each pixel
    X = (phi_vec - mean_mat) @ phi_cos
    Y = (phi_vec - mean_mat) @ phi_sin
    X = X.reshape(h, w)
    Y = Y.reshape(h, w)
    mag_map = np.sqrt(X**2 + Y**2)
    deg_map = np.degrees(np.arctan2(Y, X))

    # Clip extreme values and normalise magnitude
    mat_mean = np.mean(mag_map)
    mag_map = np.clip(mag_map, None, 2 * mat_mean)
    norm_mag_map = (mag_map - mag_map.min()) / (mag_map.max() - mag_map.min() + 1e-9)

    if display:
        fig, axes = plt.subplots(figsize=(13, 4), ncols=3)
        im1 = axes[0].imshow(imp.images[0], cmap="gray")
        fig.colorbar(im1, ax=axes[0])
        axes[0].set_title("ROI image")
        im2 = axes[1].imshow(norm_mag_map, cmap="afmhot")
        fig.colorbar(im2, ax=axes[1])
        axes[1].set_title("Normalised DRP magnitudes")
        im3 = axes[2].imshow(deg_map, cmap="hsv")
        axes[2].set_title("DRP angles")
        fig.colorbar(im3, ax=axes[2])
        plt.show()

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
    Create a mask highlighting pixels with DRP orientations around a designated direction.
    Mask values are magnitude-weighted in [0,1].
    """
    if mag_map.shape != deg_map.shape:
        raise ValueError("Magnitude and orientation dimensions do not match.")

    norm_mag_map = (mag_map - mag_map.min()) / (mag_map.max() - mag_map.min() + 1e-9)

    # Smallest angular difference to target orientation in degrees
    angle_diff = np.abs(((deg_map - orientation + 180) % 360) - 180)
    mask = angle_diff <= threshold
    if verbose:
        keep_pct = 100 * np.mean(mask)
        print(f"[DRP] mask around {orientation}±{threshold} keeps {keep_pct:.1f}% of pixels")
    return norm_mag_map * mask


def spherical_descriptor(
    imp: "ImagePack",
    subtract_mean: bool = True,
    include_sin_theta: bool = False,
    eps: float = 1e-9,
    verbose: bool = False,
    *,
    chunk_phi: int | None = None,
    chunk_theta: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-pixel first-order (vector) and second-order (covariance) spherical moments.

    Memory-efficient implementation that avoids materializing the full [h, w, phi, theta]
    float64 stack by streaming across phi/theta blocks.

    Returns:
        m1_map: [h, w, 3] first-order moment (dominant 3D direction, not normalised).
        cov_map: [h, w, 3, 3] second-order spread around m1.
        weight_sum: [h, w] sum of absolute weights used for normalisation.
    """
    if verbose:
        print("spherical_descriptor: streaming over DRP stack")

    h, w = imp.h, imp.w
    ph_num = imp.param.ph_num
    th_num = imp.param.th_num

    # Angle grids in radians (inclusive endpoints to match ph/th counts)
    phi = np.deg2rad(np.linspace(imp.param.ph_min, imp.param.ph_max, ph_num, endpoint=True))
    theta = np.deg2rad(np.linspace(imp.param.th_min, imp.param.th_max, th_num, endpoint=True))

    # Chunk sizes (default: full)
    if chunk_phi is None:
        chunk_phi = ph_num
    if chunk_theta is None:
        chunk_theta = th_num

    # Precompute per-pixel mean if requested (stream to reduce peak memory)
    if subtract_mean:
        if verbose:
            print("spherical_descriptor: computing per-pixel mean")
        sum_map = np.zeros((h, w), dtype=np.float32)
        count = 0
        for p0 in range(0, ph_num, chunk_phi):
            p1 = min(p0 + chunk_phi, ph_num)
            for t0 in range(0, th_num, chunk_theta):
                t1 = min(t0 + chunk_theta, th_num)
                block = imp.drp_stack[:, :, p0:p1, t0:t1].astype(np.float32, copy=False)
                sum_map += block.sum(axis=(2, 3))
                count += (p1 - p0) * (t1 - t0)
        mean_map = sum_map / max(count, 1)
    else:
        mean_map = None

    # Accumulators
    if verbose:
        print("spherical_descriptor: accumulating first-order moment")
    weight_sum = np.zeros((h, w), dtype=np.float32)
    m1_map = np.zeros((h, w, 3), dtype=np.float32)

    for p0 in range(0, ph_num, chunk_phi):
        p1 = min(p0 + chunk_phi, ph_num)
        phi_block = phi[p0:p1]
        cos_phi = np.cos(phi_block)[:, None]
        sin_phi = np.sin(phi_block)[:, None]
        for t0 in range(0, th_num, chunk_theta):
            t1 = min(t0 + chunk_theta, th_num)
            theta_block = theta[t0:t1]
            cos_theta = np.cos(theta_block)[None, :]
            sin_theta = np.sin(theta_block)[None, :]

            dir_x = cos_theta * cos_phi
            dir_y = cos_theta * sin_phi
            dir_z = np.broadcast_to(sin_theta, dir_x.shape)

            weight = imp.drp_stack[:, :, p0:p1, t0:t1].astype(np.float32, copy=False)
            if subtract_mean:
                weight = weight - mean_map[:, :, None, None]
            if include_sin_theta:
                weight = weight * sin_theta

            weight_sum += np.sum(np.abs(weight), axis=(2, 3))
            m1_map[..., 0] += np.sum(weight * dir_x[None, None, ...], axis=(2, 3))
            m1_map[..., 1] += np.sum(weight * dir_y[None, None, ...], axis=(2, 3))
            m1_map[..., 2] += np.sum(weight * dir_z[None, None, ...], axis=(2, 3))

    # Normalize first-order moment by per-pixel weight sum
    norm = weight_sum + eps
    m1_map = m1_map / norm[..., None]

    # Covariance accumulation
    if verbose:
        print("spherical_descriptor: accumulating covariance")
    cov_map = np.zeros((h, w, 3, 3), dtype=np.float32)
    for p0 in range(0, ph_num, chunk_phi):
        p1 = min(p0 + chunk_phi, ph_num)
        phi_block = phi[p0:p1]
        cos_phi = np.cos(phi_block)[:, None]
        sin_phi = np.sin(phi_block)[:, None]
        for t0 in range(0, th_num, chunk_theta):
            t1 = min(t0 + chunk_theta, th_num)
            theta_block = theta[t0:t1]
            cos_theta = np.cos(theta_block)[None, :]
            sin_theta = np.sin(theta_block)[None, :]

            dir_x = cos_theta * cos_phi
            dir_y = cos_theta * sin_phi
            dir_z = np.broadcast_to(sin_theta, dir_x.shape)

            weight = imp.drp_stack[:, :, p0:p1, t0:t1].astype(np.float32, copy=False)
            if subtract_mean:
                weight = weight - mean_map[:, :, None, None]
            if include_sin_theta:
                weight = weight * sin_theta

            # Normalized weights for covariance
            w_norm = weight / (norm[..., None, None])

            cx = dir_x[None, None, ...] - m1_map[..., 0][..., None, None]
            cy = dir_y[None, None, ...] - m1_map[..., 1][..., None, None]
            cz = dir_z[None, None, ...] - m1_map[..., 2][..., None, None]

            cov_map[..., 0, 0] += np.sum(w_norm * cx * cx, axis=(2, 3))
            cov_map[..., 0, 1] += np.sum(w_norm * cx * cy, axis=(2, 3))
            cov_map[..., 0, 2] += np.sum(w_norm * cx * cz, axis=(2, 3))
            cov_map[..., 1, 0] += np.sum(w_norm * cy * cx, axis=(2, 3))
            cov_map[..., 1, 1] += np.sum(w_norm * cy * cy, axis=(2, 3))
            cov_map[..., 1, 2] += np.sum(w_norm * cy * cz, axis=(2, 3))
            cov_map[..., 2, 0] += np.sum(w_norm * cz * cx, axis=(2, 3))
            cov_map[..., 2, 1] += np.sum(w_norm * cz * cy, axis=(2, 3))
            cov_map[..., 2, 2] += np.sum(w_norm * cz * cz, axis=(2, 3))

    if verbose:
        print("spherical_descriptor: done")
    return m1_map, cov_map, weight_sum


def spherical_descriptor_maps(
    imp: "ImagePack",
    subtract_mean: bool = True,
    include_sin_theta: bool = False,
    eps: float = 1e-9,
    verbose: bool = False,
    *,
    chunk_phi: int | None = None,
    chunk_theta: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper to compute direction, projection, and strength maps
    from per-pixel spherical moments.

    Returns:
        dir_map: [h, w, 3] unit vectors (normalised first-order moment).
        proj_map: [h, w, 2] azimuth/elevation in degrees (azimuth from +x, elevation from xy-plane).
        strength_map: [h, w] magnitude of the first-order moment (0→isotropic).
        m1_map: [h, w, 3] raw first-order moment before normalisation.
        cov_map: [h, w, 3, 3] covariance (second-order spread).
    """
    m1_map, cov_map, _ = spherical_descriptor(
        imp,
        subtract_mean=subtract_mean,
        include_sin_theta=include_sin_theta,
        eps=eps,
        verbose=verbose,
        chunk_phi=chunk_phi,
        chunk_theta=chunk_theta,
    )

    # Direction (unit vector) and strength (norm)
    strength_map = np.linalg.norm(m1_map, axis=2, keepdims=True)
    dir_map = m1_map / (strength_map + eps)

    # Project to azimuth/elevation in degrees
    az_map = np.degrees(np.arctan2(dir_map[..., 1], dir_map[..., 0]))
    el_map = np.degrees(np.arcsin(np.clip(dir_map[..., 2], -1.0, 1.0)))
    proj_map = np.stack([az_map, el_map], axis=-1)

    # strength_map back to [h, w]
    strength_map = strength_map[..., 0]
    return dir_map, proj_map, strength_map, m1_map, cov_map


def _sorted_eigh_3x3(cov_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Symmetrise and eigen-decompose a per-pixel 3x3 covariance.
    Eigenvalues are ascending; eigenvectors are column-wise.
    """
    cov_sym = 0.5 * (cov_map + np.swapaxes(cov_map, -1, -2))
    evals, evecs = np.linalg.eigh(cov_sym)
    return evals, evecs


def anisotropy_map_from_cov(cov_map: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Per-pixel anisotropy score from covariance eigenvalues.
    High when one principal axis dominates (line-like), low when isotropic.
    Uses (lambda_max - lambda_mid) / (lambda_sum + eps). Shape: [h, w].
    """
    evals, _ = _sorted_eigh_3x3(cov_map)
    lam0, lam1, lam2 = evals[..., 0], evals[..., 1], evals[..., 2]
    lam_sum = lam0 + lam1 + lam2
    score = (lam2 - lam1) / (lam_sum + eps)
    return score


def plane_orientation_map_from_cov(cov_map: np.ndarray, eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """
    Dominant in-plane orientation and tilt from the leading eigenvector of covariance.

    Returns:
        az_map: [h, w] azimuth of dominant axis projected to xy-plane (degrees).
        el_map: [h, w] elevation of dominant axis (degrees).
    """
    _, evecs = _sorted_eigh_3x3(cov_map)
    v = evecs[..., :, -1]  # dominant eigenvector (largest eigenvalue)
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]

    xy_norm = np.sqrt(vx**2 + vy**2)
    az_map = np.degrees(np.arctan2(vy, vx))
    az_map = np.where(xy_norm > eps, az_map, np.nan)  # undefined if projection is degenerate

    el_map = np.degrees(np.arcsin(np.clip(vz, -1.0, 1.0)))
    return az_map, el_map
