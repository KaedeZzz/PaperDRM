"""Detector diagnostics that do not depend on image-processing libraries."""

from __future__ import annotations


def period_boundary_diagnostic(
    peak_index: int,
    n_bins: int,
    period_range_px: tuple[float, float],
    detected_period_px: float,
    *,
    edge_bins: int = 1,
) -> dict:
    """Describe whether a spectral maximum is pinned to the search boundary."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if peak_index < 0 or peak_index >= n_bins:
        raise ValueError("peak_index is outside the search band")

    distance = min(peak_index, n_bins - 1 - peak_index)
    at_boundary = distance <= edge_bins
    side = None
    if at_boundary:
        # Frequency bins ascend, so index 0 corresponds to the largest period.
        side = "upper" if peak_index <= edge_bins else "lower"

    warning = None
    if side is not None:
        bound = period_range_px[1] if side == "upper" else period_range_px[0]
        warning = (
            f"Detected period {detected_period_px:.3f} px is pinned to the "
            f"{side} search boundary ({bound:.3f} px); expand or correct "
            "the configured period range before interpreting physical outputs."
        )

    return {
        "period_range_px": [float(period_range_px[0]), float(period_range_px[1])],
        "period_peak_index": int(peak_index),
        "period_search_bins": int(n_bins),
        "period_boundary_distance_bins": int(distance),
        "period_at_search_boundary": bool(at_boundary),
        "period_boundary_side": side,
        "period_warning": warning,
    }
