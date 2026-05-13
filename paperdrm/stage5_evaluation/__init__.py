"""
Evaluating laid-line detection results.

Submodules:
- consistency:           per-patch agreement statistics for period and orientation
- interval_distribution: gap-between-lines distribution from the projected signal

Planned (not yet implemented):
- phantom:      synthetic test images with known laid-line period
- ground_truth: comparison against manual line annotations
- baselines:    alternative methods (radial FFT, autocorrelation) for cross-validation
"""

from paperdrm.stage5_evaluation.consistency import (
    patch_period_stats,
    patch_orientation_stats,
    patch_consistency_report,
    print_consistency_report,
    save_consistency_report,
    plot_patch_consistency,
)
from paperdrm.stage5_evaluation.interval_distribution import (
    gap_distribution_from_signal,
    print_gap_distribution,
    save_gap_distribution,
    plot_gap_distribution,
)
from paperdrm.stage5_evaluation.fit_quality import (
    sinusoidal_fit_r2,
    frequency_concentration,
    fit_quality_curve,
    fit_quality_report,
    print_fit_quality,
    save_fit_quality,
    plot_fit_quality_curve,
)

__all__ = [
    "patch_period_stats",
    "patch_orientation_stats",
    "patch_consistency_report",
    "print_consistency_report",
    "save_consistency_report",
    "plot_patch_consistency",
    "gap_distribution_from_signal",
    "print_gap_distribution",
    "save_gap_distribution",
    "plot_gap_distribution",
    "sinusoidal_fit_r2",
    "frequency_concentration",
    "fit_quality_curve",
    "fit_quality_report",
    "print_fit_quality",
    "save_fit_quality",
    "plot_fit_quality_curve",
]
