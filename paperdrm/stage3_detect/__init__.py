"""
Laid-line detection: estimate period, orientation, and grid positions.

Submodules:
- gabor: Gabor filter bank + 1D FFT scoring, patchwise period estimation,
         phase-fit grid construction, and overlay generation.
"""

from paperdrm.stage3_detect.gabor import (
    estimate_laidline_frequency_gabor,
    estimate_laidline_frequency_gabor_patches,
    overlay_laid_lines,
    grid_positions_from_signal,
    filter_peaks_to_grid,
    peaks_from_signal,
)

__all__ = [
    "estimate_laidline_frequency_gabor",
    "estimate_laidline_frequency_gabor_patches",
    "overlay_laid_lines",
    "grid_positions_from_signal",
    "filter_peaks_to_grid",
    "peaks_from_signal",
]
