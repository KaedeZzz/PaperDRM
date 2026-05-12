"""
Per-pixel features extracted from the DRP stack.

Submodules:
- direction: first-order azimuthal direction map (drp_direction_map, drp_mask_angle)
- spherical: per-pixel spherical moments and anisotropy descriptors
"""

from paperdrm.features.direction import (
    drp_direction_map,
    drp_mask_angle,
    get_drp_direction,
)
from paperdrm.features.spherical import (
    spherical_descriptor,
    spherical_descriptor_maps,
    anisotropy_map_from_cov,
    plane_orientation_map_from_cov,
)

__all__ = [
    "drp_direction_map",
    "drp_mask_angle",
    "get_drp_direction",
    "spherical_descriptor",
    "spherical_descriptor_maps",
    "anisotropy_map_from_cov",
    "plane_orientation_map_from_cov",
]
