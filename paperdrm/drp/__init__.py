"""
Operations on the DRP image stack.

Submodules:
- slicing: angular slicing and theta_min filtering of the image stack
- stack:   building, indexing, and aggregating the [h, w, phi, theta] DRP stack
- mask:    applying spatial masks to the image stack
"""

from paperdrm.drp.slicing import (
    apply_angle_slice,
    apply_theta_min_filter,
    slice_indices,
)
from paperdrm.drp.stack import (
    build_drp_stack,
    drp_from_images,
    drp_from_stack,
    mean_drp_from_stack,
)
from paperdrm.drp.mask import mask_images

__all__ = [
    "apply_angle_slice",
    "apply_theta_min_filter",
    "slice_indices",
    "build_drp_stack",
    "drp_from_images",
    "drp_from_stack",
    "mean_drp_from_stack",
    "mask_images",
]
