"""
Image enhancement: turn DRP orientation maps into laid-line-likelihood grayscale.

Submodules:
- trig_mask: Gaussian-on-angular-distance masks (single target and patchwise)
"""

from paperdrm.enhance.trig_mask import (
    azimuth_to_laidline_gray,
    patchwise_trigonometric_mask,
    orientation_comparison_maps,
)

__all__ = [
    "azimuth_to_laidline_gray",
    "patchwise_trigonometric_mask",
    "orientation_comparison_maps",
]
