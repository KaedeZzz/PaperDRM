"""
Angular slicing of the DRP image list.

These operate on the *list of images* (one per (phi, theta) sample) and on the
companion DRPConfig describing the angular grid. They produce a reduced image
list and an updated config that downstream stack-building consumes.
"""

from dataclasses import replace

import numpy as np

from paperdrm.loader.settings import DRPConfig


def slice_indices(config: DRPConfig, angle_slice: tuple[int, int]) -> np.ndarray:
    ph_slice, th_slice = angle_slice
    if ph_slice <= 0 or th_slice <= 0:
        raise ValueError("phi_step and theta_step must be positive.")
    if config.ph_num % ph_slice != 0:
        raise ValueError("ph_num must be divisible by phi_step.")
    if config.th_num % th_slice != 0:
        raise ValueError("theta_num must be divisible by theta_step.")
    indices = np.arange(config.ph_num * config.th_num).reshape(config.ph_num, config.th_num)
    return indices[0::ph_slice, 0::th_slice].ravel()


def apply_angle_slice(
    images: list[np.ndarray],
    config: DRPConfig,
    angle_slice: tuple[int, int],
) -> tuple[list[np.ndarray], DRPConfig]:
    ph_slice, th_slice = angle_slice
    if (ph_slice, th_slice) == (1, 1):
        return images, replace(config, phi_slice=1, theta_slice=1)

    expected = config.ph_num * config.th_num
    if len(images) != expected:
        raise ValueError(f"Number of images {len(images)} does not match number of angles {expected}.")

    indices = slice_indices(config, angle_slice)
    sliced_images = [images[i] for i in indices]

    ph_step = config.ph_step
    th_step = config.th_step
    new_cfg = replace(
        config,
        ph_num=config.ph_num // ph_slice,
        th_num=config.th_num // th_slice,
        ph_max=int(config.ph_max - (ph_slice - 1) * ph_step),
        th_max=int(config.th_max - (th_slice - 1) * th_step),
        phi_slice=ph_slice,
        theta_slice=th_slice,
    )
    new_cfg.validate()
    return sliced_images, new_cfg


def apply_theta_min_filter(
    images: list[np.ndarray],
    config: DRPConfig,
    theta_min_deg: float | None,
) -> tuple[list[np.ndarray], DRPConfig]:
    """
    Keep only theta samples >= theta_min_deg (after any angle slicing).
    """
    if theta_min_deg is None:
        return images, config

    expected = config.ph_num * config.th_num
    if len(images) != expected:
        raise ValueError(f"Number of images {len(images)} does not match number of angles {expected}.")

    theta_vals = np.linspace(float(config.th_min), float(config.th_max), int(config.th_num), endpoint=True)
    keep_theta_idx = np.where(theta_vals >= float(theta_min_deg))[0]
    if keep_theta_idx.size < 2:
        raise ValueError(
            f"theta_min_deg={theta_min_deg} leaves fewer than 2 theta samples "
            f"(th range {config.th_min}..{config.th_max}, th_num={config.th_num})."
        )
    if keep_theta_idx.size == config.th_num:
        return images, config

    indices = np.arange(expected).reshape(config.ph_num, config.th_num)
    kept_indices = indices[:, keep_theta_idx].ravel()
    filtered_images = [images[int(i)] for i in kept_indices]

    th_min_new = float(theta_vals[int(keep_theta_idx[0])])
    th_max_new = float(theta_vals[int(keep_theta_idx[-1])])
    if float(th_min_new).is_integer():
        th_min_new = int(th_min_new)
    if float(th_max_new).is_integer():
        th_max_new = int(th_max_new)

    new_cfg = replace(
        config,
        th_min=th_min_new,
        th_max=th_max_new,
        th_num=int(keep_theta_idx.size),
    )
    new_cfg.validate()
    return filtered_images, new_cfg
