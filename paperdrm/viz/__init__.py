"""
Plotting helpers for each pipeline stage.

Submodules:
- drp:        plot a DRP array in stereo/direct projection
- direction:  Stage 1 direction-map summary plots
- comparison: Stage 2/3 mask + Gabor patch score comparisons
"""

from paperdrm.viz.drp import plot_drp
from paperdrm.viz.direction import plot_direction_map
from paperdrm.viz.comparison import (
    plot_orientation_comparison,
    plot_trig_mask_comparison,
    plot_patch_best_score_map,
)

__all__ = [
    "plot_drp",
    "plot_direction_map",
    "plot_orientation_comparison",
    "plot_trig_mask_comparison",
    "plot_patch_best_score_map",
]
