"""
DRP stack construction, indexing, and aggregation.

The DRP stack is a 4D array of shape [h, w, phi, theta] storing the per-pixel
reflectance profile as a function of light direction.
"""

import numpy as np

from paperdrm.loader.settings import DRPConfig


def drp_from_images(images: list[np.ndarray], config: DRPConfig, loc: tuple[int, int]) -> np.ndarray:
    """
    Vectorized DRP extraction at a single pixel across the image stack.
    """
    y, x = loc
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
    phi_theta_view = arr.reshape((ph, th, h, w))
    np.copyto(memmap, np.moveaxis(phi_theta_view, (0, 1), (2, 3)))
    memmap.flush()
    if verbose:
        print("[DRP] DRP stack build complete and flushed to disk")
    return memmap


def mean_drp_from_stack(stack: np.ndarray) -> np.ndarray:
    return np.mean(stack, axis=(0, 1))
