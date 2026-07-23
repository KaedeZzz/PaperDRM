"""Versioned domain and result models for PaperDRM V2."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


RESULT_SCHEMA_VERSION = 2
MANIFEST_SCHEMA_VERSION = 1


class DetectorTrack(str, Enum):
    MULTI_PHI = "multi_phi"
    SIMPLE = "simple"
    LEGACY = "legacy"
    SINGLE_IMAGE = "single_image"


class InputMode(str, Enum):
    DRP_STACK = "drp_stack"
    SINGLE_IMAGE = "single_image"


class MeasurementSource(str, Enum):
    GLOBAL_SPECTRAL_PERIOD = "global_spectral_period"


class ResultDisposition(str, Enum):
    ACCEPTED = "accepted"
    REVIEW_REQUIRED = "review_required"
    REJECTED = "rejected"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNKNOWN = "unknown"


class ConfidenceReason(str, Enum):
    PERIOD_SEARCH_BOUNDARY = "period_search_boundary"
    POLARITY_CONTRADICTION = "polarity_contradiction"
    MISSING_SELF_CONTRAST = "missing_self_contrast"
    STRONG_SELF_CONTRAST = "strong_self_contrast"
    MODERATE_SELF_CONTRAST = "moderate_self_contrast"
    WEAK_SELF_CONTRAST = "weak_self_contrast"
    FIT_PERIOD_DISAGREEMENT = "fit_period_disagreement"
    LOCAL_GAP_DISAGREEMENT = "local_gap_disagreement"
    SPLIT_HALF_INSTABILITY = "split_half_instability"


@dataclass(frozen=True)
class SpacingMeasurement:
    """Primary spectral spacing, kept separate from local gap diagnostics."""

    period_px: float
    source: MeasurementSource = MeasurementSource.GLOBAL_SPECTRAL_PERIOD
    cm_per_px: float | None = None
    interval_cm: float | None = None
    lines_per_cm: float | None = None
    local_gap_median_px: float | None = None

    def __post_init__(self) -> None:
        if self.period_px <= 0:
            raise ValueError("period_px must be positive")
        if self.cm_per_px is not None and self.cm_per_px <= 0:
            raise ValueError("cm_per_px must be positive")
        if self.interval_cm is not None and self.interval_cm <= 0:
            raise ValueError("interval_cm must be positive")
        if self.lines_per_cm is not None and self.lines_per_cm <= 0:
            raise ValueError("lines_per_cm must be positive")

    @classmethod
    def from_period(
        cls,
        period_px: float,
        *,
        cm_per_px: float | None = None,
        local_gap_median_px: float | None = None,
    ) -> "SpacingMeasurement":
        interval_cm = period_px * cm_per_px if cm_per_px is not None else None
        lines_per_cm = 1.0 / interval_cm if interval_cm is not None else None
        return cls(
            period_px=float(period_px),
            cm_per_px=float(cm_per_px) if cm_per_px is not None else None,
            interval_cm=interval_cm,
            lines_per_cm=lines_per_cm,
            local_gap_median_px=local_gap_median_px,
        )


@dataclass(frozen=True)
class DetectionDiagnostics:
    """Evidence used by the future confidence/rejection policy.

    Phase 1 stores evidence but intentionally does not assign a confidence
    label; central policy is introduced in Phase 3.
    """

    period_at_search_boundary: bool = False
    period_boundary_side: str | None = None
    period_warning: str | None = None
    self_contrast_z: float | None = None
    polarity_contradiction: bool = False
    split_half_period_diff_std_px: float | None = None
    split_half_agree_rate_within_1px: float | None = None
    gabor_score: float | None = None
    phase_resultant_length: float | None = None
    n_images: int | None = None
    n_polarity_flipped: int | None = None
    wire_width_experimental: bool = True

    def __post_init__(self) -> None:
        if self.period_boundary_side not in (None, "lower", "upper"):
            raise ValueError("period_boundary_side must be lower, upper, or None")
        if self.period_boundary_side is not None and not self.period_at_search_boundary:
            raise ValueError("period_boundary_side requires a boundary hit")
        if self.split_half_agree_rate_within_1px is not None:
            value = self.split_half_agree_rate_within_1px
            if not 0.0 <= value <= 1.0:
                raise ValueError("split-half agreement rate must be between 0 and 1")
        if self.phase_resultant_length is not None:
            if not 0.0 <= self.phase_resultant_length <= 1.0:
                raise ValueError("phase resultant length must be between 0 and 1")
        if self.n_images is not None and self.n_images < 1:
            raise ValueError("n_images must be positive")
        if self.n_polarity_flipped is not None and self.n_polarity_flipped < 0:
            raise ValueError("n_polarity_flipped must be non-negative")


@dataclass(frozen=True)
class GridEstimate:
    line_direction_deg: float
    phase_rad: float
    positions_px: tuple[int, ...]


@dataclass(frozen=True)
class WireWidthEstimate:
    fwhm_px: float | None
    model_ok: bool
    warning: str | None = None
    experimental: bool = True
    segment_median_fwhm_px: float | None = None
    segment_valid_count: int | None = None
    segment_count: int | None = None
    median_fwhm_mm: float | None = None

    def __post_init__(self) -> None:
        if self.fwhm_px is not None and self.fwhm_px <= 0:
            raise ValueError("fwhm_px must be positive")


@dataclass(frozen=True)
class IntervalEvaluation:
    n_peaks: int
    n_gaps: int
    median_gap_px: float | None
    gap_iqr_px: tuple[float, float] | None
    gap_median_relative_error_vs_spectral: float | None = None


@dataclass(frozen=True)
class FitEvaluation:
    r2_fundamental: float | None
    r2_with_harmonics: float | None
    r2_gaussian_comb: float | None
    frequency_concentration: float | None
    best_period_by_r2_px: float | None
    best_r2: float | None
    agrees_with_dominant: bool


@dataclass(frozen=True)
class ContrastEvaluation:
    n_lines: int
    contrast_relative: float | None
    contrast_z: float | None
    warning: str | None = None


@dataclass(frozen=True)
class SplitHalfEvaluation:
    n_images: int
    n_splits: int
    period_difference_std_px: float
    agree_rate_within_1px: float
    agree_rate_within_half_px: float


@dataclass(frozen=True)
class EvaluationSummary:
    interval: IntervalEvaluation
    fit: FitEvaluation
    contrast: ContrastEvaluation
    split_half: SplitHalfEvaluation | None = None


@dataclass(frozen=True)
class ConfidenceAssessment:
    disposition: ResultDisposition
    level: ConfidenceLevel
    primary_reason: ConfidenceReason
    warnings: tuple[ConfidenceReason, ...] = ()
    policy_version: str = "v1"

    def __post_init__(self) -> None:
        expected_levels = {
            ResultDisposition.ACCEPTED: {
                ConfidenceLevel.HIGH,
                ConfidenceLevel.MODERATE,
            },
            ResultDisposition.REVIEW_REQUIRED: {ConfidenceLevel.LOW},
            ResultDisposition.REJECTED: {ConfidenceLevel.UNKNOWN},
            ResultDisposition.INSUFFICIENT_EVIDENCE: {ConfidenceLevel.UNKNOWN},
        }
        if self.level not in expected_levels[self.disposition]:
            raise ValueError(
                f"confidence level {self.level.value} is inconsistent with "
                f"disposition {self.disposition.value}"
            )
        if not self.policy_version:
            raise ValueError("policy_version must not be empty")


@dataclass(frozen=True)
class PipelineResult:
    dataset_id: str
    track: DetectorTrack
    measurement: SpacingMeasurement
    diagnostics: DetectionDiagnostics = field(default_factory=DetectionDiagnostics)
    grid: GridEstimate | None = None
    wire_width: WireWidthEstimate | None = None
    evaluation: EvaluationSummary | None = None
    confidence: ConfidenceAssessment | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: int = field(default=RESULT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must not be empty")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["track"] = self.track.value
        value["measurement"]["source"] = self.measurement.source.value
        if self.confidence is not None:
            value["confidence"]["disposition"] = self.confidence.disposition.value
            value["confidence"]["level"] = self.confidence.level.value
            value["confidence"]["primary_reason"] = (
                self.confidence.primary_reason.value
            )
            value["confidence"]["warnings"] = [
                reason.value for reason in self.confidence.warnings
            ]
        return value

    def to_json(self, path: str | Path | None = None) -> str:
        payload = json.dumps(
            self.to_dict(), indent=2, sort_keys=True, allow_nan=False
        )
        if path is not None:
            Path(path).write_text(payload + "\n", encoding="utf-8")
        return payload


@dataclass(frozen=True)
class ArtifactManifestEntry:
    """Integrity metadata for one artifact inside an immutable run."""

    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("artifact path must not be empty")
        if self.size_bytes < 0:
            raise ValueError("artifact size must be non-negative")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("artifact sha256 must be 64 lowercase hex characters")


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    dataset_id: str
    track: DetectorTrack
    config: dict[str, Any]
    inputs: tuple[str, ...] = ()
    result_file: str = "result.json"
    created_at_utc: str | None = None
    result_schema_version: int = field(default=RESULT_SCHEMA_VERSION, init=False)
    policy_version: str | None = None
    artifacts: tuple[ArtifactManifestEntry, ...] = ()
    manifest_schema_version: int = field(default=MANIFEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if not self.dataset_id:
            raise ValueError("dataset_id must not be empty")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["track"] = self.track.value
        value["inputs"] = list(self.inputs)
        value["artifacts"] = [asdict(artifact) for artifact in self.artifacts]
        return value
