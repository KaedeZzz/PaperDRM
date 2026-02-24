from dataclasses import replace

import numpy as np
from tqdm import tqdm

from .settings import DRPConfig


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

    # Select one image per slice step to reduce angular resolution
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

    # Image order is [phi major, theta minor], so filter theta columns in a [phi, theta] grid.
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


def mask_images(images: list[np.ndarray], mask: np.ndarray, normalize: bool = False) -> list[np.ndarray]:
    if not images:
        return []
    if mask.shape != images[0].shape:
        raise ValueError(f"Mask shape {mask.shape} must match image shape {images[0].shape}.")
    res_list: list[np.ndarray] = []
    for image in tqdm(images, desc="masking images"):
        arr = image.astype(np.float64)
        arr *= mask
        if normalize:
            denom = arr.max() - arr.min()
            arr = 255 * (arr - arr.min()) / denom if denom != 0 else arr
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        res_list.append(arr)
    return res_list


def drp_from_images(images: list[np.ndarray], config: DRPConfig, loc: tuple[int, int]) -> np.ndarray:
    """
    Vectorized DRP extraction at a single pixel across the image stack.
    """
    y, x = loc
    # Stack to [N, h, w] then reshape to [phi, theta, h, w] and slice pixel
    arr = np.stack(images, axis=0).reshape(config.ph_num, config.th_num, images[0].shape[0], images[0].shape[1])
    return arr[:, :, y, x]


def drp_from_stack(stack: np.ndarray, loc: tuple[int, int]) -> np.ndarray:
    y, x = loc
    return stack[y, x, :, :]


def build_drp_stack(
    images: list[np.ndarray],
    config: DRPConfig,
    memmap: np.memmap,
    *,
    verbose: bool = False,
) -> np.memmap:
    ph, th = config.ph_num, config.th_num
    if len(images) != ph * th:
        raise ValueError(f"Number of images {len(images)} does not match number of angles {ph * th}.")
    if verbose:
        print(f"[DRP] building DRP stack from {len(images)} images -> shape ({ph}, {th})")
    arr = np.stack(images, axis=0, dtype=np.uint8)
    h, w = arr.shape[1:]
    # Build a contiguous phi/theta-first view then move axes once when copying to the memmap.
    phi_theta_view = arr.reshape((ph, th, h, w))
    np.copyto(memmap, np.moveaxis(phi_theta_view, (0, 1), (2, 3)))
    memmap.flush()
    if verbose:
        print("[DRP] DRP stack build complete and flushed to disk")
    return memmap


def mean_drp_from_stack(stack: np.ndarray) -> np.ndarray:
    return np.mean(stack, axis=(0, 1))
