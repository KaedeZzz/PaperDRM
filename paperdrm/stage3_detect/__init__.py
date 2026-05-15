"""
Laid-line detection: estimate period, orientation, and grid positions.

Submodules:
- gabor:           Gabor filter bank + 1D FFT scoring, patchwise period
                   estimation, phase-fit grid construction, and overlay
                   generation. NOTE: known to be biased toward
                   period/2 on this data (use_abs + score band-width).
                   Kept for reference and as fallback.
- simple_detector: Radial-FFT-based period detection, ML-optimal,
                   parameter-free. Use this by default.
"""

from paperdrm.stage3_detect.gabor import (
    estimate_laidline_frequency_gabor,
    estimate_laidline_frequency_gabor_patches,
    overlay_laid_lines,
    grid_positions_from_signal,
    filter_peaks_to_grid,
    peaks_from_signal,
)
from paperdrm.stage3_detect.simple_detector import (
    detect_laid_lines_simple,
    radial_fft_period,
    gabor_clean_signal,
    phase_fit,
    grid_positions,
    overlay_grid,
    overlay_grid_bands,
)
from paperdrm.stage3_detect.wire_width import estimate_wire_width

__all__ = [
    "estimate_laidline_frequency_gabor",
    "estimate_laidline_frequency_gabor_patches",
    "overlay_laid_lines",
    "grid_positions_from_signal",
    "filter_peaks_to_grid",
    "peaks_from_signal",
    "detect_laid_lines_simple",
    "radial_fft_period",
    "gabor_clean_signal",
    "phase_fit",
    "grid_positions",
    "overlay_grid",
    "overlay_grid_bands",
    "estimate_wire_width",
]
