"""
Spatial masking of the image list (e.g. to focus DRP analysis on an ROI).
"""

import numpy as np
from tqdm import tqdm


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
