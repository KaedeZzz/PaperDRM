from __future__ import annotations

import cv2
import numpy as np


def azimuth_to_laidline_gray(
    deg_map: np.ndarray,
    target_deg: float = 90.0,
    sigma_deg: float = 10.0,
    *,
    enhance: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert azimuth angles to a grayscale likelihood map for laid-line detection.
    Uses 360-degree periodic distance (full-direction azimuth).
    Returns (raw_uint8, enhanced_uint8).
    """
    d = ((deg_map - target_deg + 180.0) % 360.0) - 180.0  # [-180, 180]
    resp = np.exp(-0.5 * (d / sigma_deg) ** 2)
    raw = (resp - resp.min()) / (resp.max() - resp.min() + 1e-8)
    raw_8u = (255.0 * raw).astype(np.uint8)

    if not enhance:
        return raw_8u, raw_8u

    enhanced = cv2.GaussianBlur(raw_8u, (0, 0), 1.0)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(enhanced)
    return raw_8u, enhanced


def patchwise_trigonometric_mask(
    deg_map: np.ndarray,
    patch_size: tuple[int, int] = (512, 512),
    stride: tuple[int, int] = (256, 256),
    sigma_deg: float = 10.0,
    *,
    enhance: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a grayscale mask by:
    1) estimating one dominant orientation per patch,
    2) applying a per-patch trigonometric likelihood,
    3) stitching all patch responses with overlap-weighted blending.

    Returns (raw_uint8, enhanced_uint8, patch_target_map_deg, patch_target_map_uint8).
    """
    h, w = deg_map.shape
    ph = min(max(1, int(patch_size[0])), h)
    pw = min(max(1, int(patch_size[1])), w)
    sy = max(1, int(stride[0]))
    sx = max(1, int(stride[1]))

    def _starts(length: int, patch_len: int, step: int) -> list[int]:
        out = list(range(0, max(1, length - patch_len + 1), step))
        last = length - patch_len
        if out[-1] != last:
            out.append(last)
        return out

    ys = _starts(h, ph, sy)
    xs = _starts(w, pw, sx)

    wy = np.hanning(ph).astype(np.float32) if ph > 1 else np.ones((1,), dtype=np.float32)
    wx = np.hanning(pw).astype(np.float32) if pw > 1 else np.ones((1,), dtype=np.float32)
    win = np.outer(wy, wx)
    win += 1e-6

    acc = np.zeros((h, w), dtype=np.float32)
    wsum = np.zeros((h, w), dtype=np.float32)
    target_sin = np.zeros((h, w), dtype=np.float32)
    target_cos = np.zeros((h, w), dtype=np.float32)

    for y in ys:
        for x in xs:
            patch = deg_map[y : y + ph, x : x + pw]

            theta = np.deg2rad(patch)
            s = np.sum(np.sin(theta))
            c = np.sum(np.cos(theta))
            target_deg = np.rad2deg(np.arctan2(s, c))

            d = ((patch - target_deg + 180.0) % 360.0) - 180.0
            resp = np.exp(-0.5 * (d / sigma_deg) ** 2).astype(np.float32)

            acc[y : y + ph, x : x + pw] += resp * win
            wsum[y : y + ph, x : x + pw] += win

            t = np.deg2rad(target_deg)
            target_sin[y : y + ph, x : x + pw] += np.sin(t) * win
            target_cos[y : y + ph, x : x + pw] += np.cos(t) * win

    stitched = acc / (wsum + 1e-8)
    targets = np.rad2deg(np.arctan2(target_sin, target_cos))

    raw = (stitched - stitched.min()) / (stitched.max() - stitched.min() + 1e-8)
    raw_8u = (255.0 * raw).astype(np.uint8)

    targets01 = (targets + 180.0) / 360.0
    target_8u = np.clip(255.0 * targets01, 0.0, 255.0).astype(np.uint8)

    if not enhance:
        return raw_8u, raw_8u, targets.astype(np.float32), target_8u

    enhanced = cv2.GaussianBlur(raw_8u, (0, 0), 1.0)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(enhanced)
    return raw_8u, enhanced, targets.astype(np.float32), target_8u


def orientation_comparison_maps(
    deg_map: np.ndarray,
    patch_target_deg_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build display maps to compare raw azimuth against patch dominant orientation.

    Returns (raw_azimuth_orientation_8u, orientation_diff_deg).
    """
    raw_azimuth_orientation_deg = ((deg_map + 180.0) % 360.0) - 180.0
    raw_azimuth_orientation_8u = np.clip(
        255.0 * ((raw_azimuth_orientation_deg + 180.0) / 360.0), 0.0, 255.0
    ).astype(np.uint8)
    orientation_diff_deg = np.abs(
        ((patch_target_deg_map - raw_azimuth_orientation_deg + 180.0) % 360.0) - 180.0
    )
    return raw_azimuth_orientation_8u, orientation_diff_deg
