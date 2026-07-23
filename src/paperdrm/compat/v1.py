"""Translate V1 settings and result files into V2 boundary models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paperdrm.config import AcquisitionConfig, PipelineConfig
from paperdrm.models import (
    DetectionDiagnostics,
    DetectorTrack,
    PipelineResult,
    SpacingMeasurement,
)
from paperdrm.pipeline import PipelineRequest
from paperdrm.stage0_loader.settings import Settings


def config_from_settings(
    settings: Settings,
    *,
    requested_track: str | DetectorTrack = DetectorTrack.MULTI_PHI,
    image_override: str | Path | None = None,
) -> PipelineConfig:
    """Preserve V1 routing while producing an immutable V2 config."""

    requested = DetectorTrack(requested_track)
    if requested is DetectorTrack.SINGLE_IMAGE:
        raise ValueError("single_image is selected by image input, not --track")

    image_path = Path(image_override) if image_override is not None else settings.image_path
    track = DetectorTrack.SINGLE_IMAGE if image_path is not None else requested
    serial = settings.data_serial
    if serial is not None:
        dataset_id = str(serial)
    elif image_path is not None:
        dataset_id = image_path.stem
    else:
        dataset_id = "unknown"

    acquisition = None
    if settings.drp is not None:
        acquisition = AcquisitionConfig(
            theta_min=settings.drp.th_min,
            theta_max=settings.drp.th_max,
            theta_count=settings.drp.th_num,
            phi_min=settings.drp.ph_min,
            phi_max=settings.drp.ph_max,
            phi_count=settings.drp.ph_num,
        )

    return PipelineConfig(
        dataset_id=dataset_id,
        track=track,
        data_root=Path(settings.data_root),
        folder=settings.folder,
        image_path=image_path,
        image_format=settings.img_format,
        angle_slice=settings.angle_slice,
        use_cached_stack=settings.use_cached_stack,
        subtract_background=settings.subtract_background,
        subtraction_scale_percentile=settings.subtraction_scale_percentile,
        load_workers=settings.load_workers,
        config_path=settings.config_path,
        acquisition=acquisition,
        square_crop=settings.square_crop,
        theta_min_deg=settings.theta_min_deg,
        fov_width_cm=settings.fov_width_cm,
        crop_roi=settings.crop_roi,
        period_range_cm=settings.period_range_cm,
        line_direction_deg=settings.line_dir_deg,
        auto_line_direction=settings.auto_line_dir,
        wire_is_darker=settings.wire_is_darker,
    )


def load_v1_config(
    path: str | Path,
    *,
    requested_track: str | DetectorTrack = DetectorTrack.MULTI_PHI,
    image_override: str | Path | None = None,
) -> PipelineConfig:
    settings = Settings.from_yaml(path)
    return config_from_settings(
        settings,
        requested_track=requested_track,
        image_override=image_override,
    )


def _load_json(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def result_from_directory(
    result_directory: str | Path,
    config: PipelineConfig,
) -> PipelineResult:
    """Aggregate V1's multi-file output into one versioned V2 result."""

    result_dir = Path(result_directory)
    interval = _load_json(result_dir / "interval_distribution.json", required=True)
    fit = _load_json(result_dir / "fit_quality.json")
    contrast = _load_json(result_dir / "self_contrast.json")
    split_half = _load_json(result_dir / "split_half_stability.json")

    physical = interval.get("physical", {})
    period_px = float(interval["period_px_used"])
    cm_per_px_value = physical.get("cm_per_px")
    cm_per_px = float(cm_per_px_value) if cm_per_px_value is not None else None
    local_gap = interval.get("px", {}).get("median")
    measurement = SpacingMeasurement.from_period(
        period_px,
        cm_per_px=cm_per_px,
        local_gap_median_px=float(local_gap) if local_gap is not None else None,
    )

    contrast_z_value = contrast.get("contrast_z")
    contrast_z = float(contrast_z_value) if contrast_z_value is not None else None
    boundary = bool(fit.get("period_at_search_boundary", False))
    boundary_side = fit.get("period_boundary_side") if boundary else None
    diagnostics = DetectionDiagnostics(
        period_at_search_boundary=boundary,
        period_boundary_side=boundary_side,
        period_warning=fit.get("period_warning"),
        self_contrast_z=contrast_z,
        polarity_contradiction=contrast_z is not None and contrast_z < 0.0,
        split_half_period_diff_std_px=split_half.get("period_diff_std"),
        split_half_agree_rate_within_1px=split_half.get(
            "agree_rate_within_1px"
        ),
    )

    artifacts = {
        path.name: str(path)
        for path in sorted(result_dir.iterdir())
        if path.is_file()
    }
    provenance = {
        "adapter": "paperdrm.compat.v1",
        "source_format": "v1-multi-file-results",
        "result_directory": str(result_dir),
    }
    if config.config_path is not None:
        provenance["config_path"] = str(config.config_path)

    return PipelineResult(
        dataset_id=config.dataset_id,
        track=config.track,
        measurement=measurement,
        diagnostics=diagnostics,
        artifacts=artifacts,
        provenance=provenance,
    )


class V1ResultDirectoryBackend:
    """Read an already-produced V1 result directory through the V2 pipeline."""

    def execute(self, request: PipelineRequest) -> PipelineResult:
        if request.result_directory is None:
            raise ValueError("V1 result backend requires result_directory")
        return result_from_directory(request.result_directory, request.config)
