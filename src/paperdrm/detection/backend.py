"""Native V2 adapter for the active single-image and multi-phi detectors."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from paperdrm.confidence import ConfidencePolicyV1
from paperdrm.evaluation import EvaluationService
from paperdrm.models import (
    DetectionDiagnostics,
    DetectorTrack,
    GridEstimate,
    PipelineResult,
    SpacingMeasurement,
    WireWidthEstimate,
)
from paperdrm.pipeline import MultiPhiInput, PipelineRequest, SingleImageInput
from paperdrm.stage3_detect.multi_phi_detector import detect_laid_lines_multi_phi
from paperdrm.stage3_detect.simple_detector import (
    auto_detect_line_dir,
    detect_laid_lines_simple,
)


def _period_range_px(request: PipelineRequest, image_width: int) -> tuple[float, float]:
    config = request.config
    if config.period_range_cm is None:
        return (8.0, 80.0)
    if config.fov_width_cm is None:
        raise ValueError("period_range_cm requires fov_width_cm")
    cm_per_px = config.fov_width_cm / float(image_width)
    return (
        config.period_range_cm[0] / cm_per_px,
        config.period_range_cm[1] / cm_per_px,
    )


def _line_direction(request: PipelineRequest, image) -> float:
    config = request.config
    if not config.auto_line_direction:
        return config.line_direction_deg
    height, width = image.shape[:2]
    center_deg = 0.0 if height >= width else 90.0
    return auto_detect_line_dir(
        image,
        period_range_px=_period_range_px(request, width),
        center_deg=center_deg,
    )


def _to_result(
    request: PipelineRequest,
    detector_output: dict[str, Any],
    *,
    image_width: int,
    evaluation,
) -> PipelineResult:
    config = request.config
    cm_per_px = (
        config.fov_width_cm / float(image_width)
        if config.fov_width_cm is not None
        else None
    )
    measurement = SpacingMeasurement.from_period(
        float(detector_output["dominant_period_px"]),
        cm_per_px=cm_per_px,
    )
    boundary = bool(detector_output.get("period_at_search_boundary", False))
    diagnostics = DetectionDiagnostics(
        period_at_search_boundary=boundary,
        period_boundary_side=(
            detector_output.get("period_boundary_side") if boundary else None
        ),
        period_warning=detector_output.get("period_warning"),
        gabor_score=detector_output.get("gabor_score"),
        phase_resultant_length=detector_output.get("phase_resultant_length"),
        n_images=detector_output.get("n_images"),
        n_polarity_flipped=detector_output.get("n_polarity_flipped"),
        self_contrast_z=evaluation.summary.contrast.contrast_z,
        polarity_contradiction=(
            evaluation.summary.contrast.contrast_z is not None
            and evaluation.summary.contrast.contrast_z < 0.0
        ),
        split_half_period_diff_std_px=(
            evaluation.summary.split_half.period_difference_std_px
            if evaluation.summary.split_half is not None
            else None
        ),
        split_half_agree_rate_within_1px=(
            evaluation.summary.split_half.agree_rate_within_1px
            if evaluation.summary.split_half is not None
            else None
        ),
    )
    grid = GridEstimate(
        line_direction_deg=float(detector_output["line_dir_deg"]),
        phase_rad=float(detector_output["phase"]),
        positions_px=tuple(
            int(value) for value in detector_output["grid_positions_x"]
        ),
    )
    provenance = {
        "backend": "paperdrm.detection.NativeDetectorBackend",
        "detector": config.track.value,
        "line_direction_deg": grid.line_direction_deg,
        "period_range_px": list(_period_range_px(request, image_width)),
        "representative_index": int(
            detector_output.get("representative_index", 0)
        ),
    }
    return PipelineResult(
        dataset_id=config.dataset_id,
        track=config.track,
        measurement=measurement,
        diagnostics=diagnostics,
        grid=grid,
        wire_width=evaluation.wire_width,
        evaluation=evaluation.summary,
        provenance=provenance,
    )


class NativeDetectorBackend:
    """Execute unchanged detector kernels through the V2 pipeline contract."""

    def __init__(
        self,
        evaluator: EvaluationService | None = None,
        confidence_policy: ConfidencePolicyV1 | None = None,
    ) -> None:
        self._evaluator = evaluator or EvaluationService()
        self._confidence_policy = confidence_policy or ConfidencePolicyV1()

    def _assess(self, result: PipelineResult) -> PipelineResult:
        confidence = self._confidence_policy.assess(
            result.diagnostics,
            result.evaluation,
        )
        return replace(result, confidence=confidence)

    def execute(self, request: PipelineRequest) -> PipelineResult:
        config = request.config
        data = request.input_data

        if config.track in (DetectorTrack.SINGLE_IMAGE, DetectorTrack.SIMPLE):
            if not isinstance(data, SingleImageInput):
                raise ValueError(f"{config.track.value} requires SingleImageInput")
            image = data.image
            line_direction = _line_direction(request, image)
            output = detect_laid_lines_simple(
                image,
                line_dir_deg=line_direction,
                period_range_px=_period_range_px(request, image.shape[1]),
                wire_is_darker=config.wire_is_darker,
                use_gabor_refinement=True,
            )
            evaluation = self._evaluator.evaluate(
                output,
                representative_image=image,
                fov_width_cm=config.fov_width_cm,
            )
            return self._assess(_to_result(
                request,
                output,
                image_width=image.shape[1],
                evaluation=evaluation,
            ))

        if config.track is DetectorTrack.MULTI_PHI:
            if not isinstance(data, MultiPhiInput):
                raise ValueError("multi_phi requires MultiPhiInput")
            images = list(data.images)
            line_direction = _line_direction(request, images[0])
            output = detect_laid_lines_multi_phi(
                images,
                line_dir_deg=line_direction,
                period_range_px=_period_range_px(request, images[0].shape[1]),
                wire_is_darker=config.wire_is_darker,
                use_gabor_refinement=True,
            )
            representative = images[int(output["representative_index"])]
            evaluation = self._evaluator.evaluate(
                output,
                representative_image=representative,
                fov_width_cm=config.fov_width_cm,
                multi_phi_images=data.images,
            )
            return self._assess(_to_result(
                request,
                output,
                image_width=images[0].shape[1],
                evaluation=evaluation,
            ))

        raise NotImplementedError(
            "legacy execution remains behind the V1 compatibility entry point"
        )
