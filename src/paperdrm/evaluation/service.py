"""Map unchanged Stage 5 metric kernels into typed, file-free V2 evidence."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from paperdrm.models import (
    ContrastEvaluation,
    EvaluationSummary,
    FitEvaluation,
    IntervalEvaluation,
    SplitHalfEvaluation,
    WireWidthEstimate,
)
from paperdrm.stage5_evaluation.fit_quality import fit_quality_report
from paperdrm.stage5_evaluation.interval_distribution import (
    gap_distribution_from_signal,
)
from paperdrm.stage5_evaluation.self_contrast import self_consistency_contrast
from paperdrm.stage5_evaluation.split_half import split_half_period_stability
from paperdrm.stage5_evaluation.wire_width_stats import wire_width_statistics


@dataclass(frozen=True)
class EvaluationOptions:
    wire_segments: int = 16
    split_half_splits: int = 200
    split_half_seed: int = 0

    def __post_init__(self) -> None:
        if self.wire_segments < 2:
            raise ValueError("wire_segments must be at least 2")
        if self.split_half_splits < 1:
            raise ValueError("split_half_splits must be positive")


@dataclass(frozen=True)
class EvaluationOutcome:
    summary: EvaluationSummary
    wire_width: WireWidthEstimate


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None


class EvaluationService:
    """Compute Stage 5 evidence without printing, plotting, or writing files."""

    def __init__(self, options: EvaluationOptions | None = None) -> None:
        self.options = options or EvaluationOptions()

    def evaluate(
        self,
        detector_output: dict[str, Any],
        *,
        representative_image: np.ndarray,
        fov_width_cm: float | None,
        multi_phi_images: tuple[np.ndarray, ...] | None = None,
    ) -> EvaluationOutcome:
        period_px = float(detector_output["dominant_period_px"])
        line_direction = float(detector_output["line_dir_deg"])
        grid_positions = np.asarray(detector_output["grid_positions_x"])

        interval_raw = gap_distribution_from_signal(
            detector_output["dominant_signal_1d"],
            period_px,
            fov_width_cm=fov_width_cm,
            image_width_px=representative_image.shape[1],
        )
        px = interval_raw["px"]
        physical = interval_raw.get("physical") or {}
        iqr_values = px.get("iqr")
        interval = IntervalEvaluation(
            n_peaks=int(interval_raw["n_peaks"]),
            n_gaps=int(interval_raw["n_gaps"]),
            median_gap_px=_finite_or_none(px.get("median")),
            gap_iqr_px=(
                (float(iqr_values[0]), float(iqr_values[1]))
                if iqr_values is not None
                and len(iqr_values) == 2
                and all(isfinite(float(value)) for value in iqr_values)
                else None
            ),
            gap_median_relative_error_vs_spectral=_finite_or_none(
                physical.get("gap_median_relative_error_vs_spectral")
            ),
        )

        fit_raw = fit_quality_report(detector_output)
        fit = FitEvaluation(
            r2_fundamental=_finite_or_none(fit_raw["r2_fundamental_only"]),
            r2_with_harmonics=_finite_or_none(fit_raw["r2_with_harmonics"]),
            r2_gaussian_comb=_finite_or_none(fit_raw.get("r2_gaussian_comb")),
            frequency_concentration=_finite_or_none(
                fit_raw["frequency_concentration"]
            ),
            best_period_by_r2_px=_finite_or_none(fit_raw["best_period_by_r2"]),
            best_r2=_finite_or_none(fit_raw["best_r2"]),
            agrees_with_dominant=bool(fit_raw["agrees_with_dominant"]),
        )

        contrast_raw = self_consistency_contrast(
            representative_image,
            grid_positions,
            period_px,
            line_dir_deg=line_direction,
            wire_is_darker=bool(detector_output["wire_is_darker"]),
        )
        contrast = ContrastEvaluation(
            n_lines=int(contrast_raw["n_lines"]),
            contrast_relative=_finite_or_none(contrast_raw.get("contrast_rel")),
            contrast_z=_finite_or_none(contrast_raw.get("contrast_z")),
            warning=contrast_raw.get("warning"),
        )

        split_half = None
        if multi_phi_images is not None and len(multi_phi_images) >= 4:
            split_raw = split_half_period_stability(
                list(multi_phi_images),
                line_dir_deg=line_direction,
                period_range_px=tuple(detector_output["period_range_px"]),
                n_splits=self.options.split_half_splits,
                seed=self.options.split_half_seed,
                fov_width_cm=fov_width_cm,
            )
            split_half = SplitHalfEvaluation(
                n_images=int(split_raw["n_images"]),
                n_splits=int(split_raw["n_splits"]),
                period_difference_std_px=float(split_raw["period_diff_std"]),
                agree_rate_within_1px=float(split_raw["agree_rate_within_1px"]),
                agree_rate_within_half_px=float(
                    split_raw["agree_rate_within_0p5px"]
                ),
            )

        wire_raw = wire_width_statistics(
            representative_image,
            period_px,
            line_dir_deg=line_direction,
            n_segments=self.options.wire_segments,
            fov_width_cm=fov_width_cm,
        )
        aggregate = wire_raw["aggregate"]["fwhm_px"]
        physical_wire = wire_raw.get("physical", {}).get("fwhm_mm", {})
        global_wire = wire_raw["global"]
        wire_width = WireWidthEstimate(
            fwhm_px=(
                _finite_or_none(global_wire.get("fwhm_px"))
                if global_wire.get("model_ok")
                else None
            ),
            model_ok=bool(global_wire.get("model_ok")),
            warning=global_wire.get("warning"),
            segment_median_fwhm_px=_finite_or_none(aggregate.get("median")),
            segment_valid_count=int(aggregate["n_valid"]),
            segment_count=int(wire_raw["n_segments"]),
            median_fwhm_mm=_finite_or_none(physical_wire.get("median")),
        )

        return EvaluationOutcome(
            summary=EvaluationSummary(
                interval=interval,
                fit=fit,
                contrast=contrast,
                split_half=split_half,
            ),
            wire_width=wire_width,
        )
