"""Typed, side-effect-free configuration models for the V2 boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from paperdrm.models import DetectorTrack, InputMode


@dataclass(frozen=True)
class AcquisitionConfig:
    """Explicit DRP acquisition geometry when it is not filename-inferred."""

    theta_min: int
    theta_max: int
    theta_count: int
    phi_min: int
    phi_max: int
    phi_count: int

    def __post_init__(self) -> None:
        if self.theta_min >= self.theta_max:
            raise ValueError("theta_min must be less than theta_max")
        if self.phi_min >= self.phi_max:
            raise ValueError("phi_min must be less than phi_max")
        if self.theta_count < 2 or self.phi_count < 2:
            raise ValueError("theta_count and phi_count must be at least 2")


@dataclass(frozen=True)
class PipelineConfig:
    """Normalised configuration consumed by the V2 pipeline boundary.

    The compatibility adapter converts mutable V1 ``Settings`` into this model.
    Paths are kept as paths internally and serialized as strings.
    """

    dataset_id: str
    track: DetectorTrack
    data_root: Path = Path("data")
    folder: Path | None = None
    image_path: Path | None = None
    image_format: str = "jpg"
    angle_slice: tuple[int, int] = (1, 1)
    use_cached_stack: bool = True
    subtract_background: bool = True
    subtraction_scale_percentile: float = 99.5
    load_workers: int | None = None
    config_path: Path | None = None
    acquisition: AcquisitionConfig | None = None
    square_crop: bool = False
    theta_min_deg: float | None = None
    fov_width_cm: float | None = None
    crop_roi: tuple[int, int, int, int] | None = None
    period_range_cm: tuple[float, float] | None = None
    line_direction_deg: float = 90.0
    auto_line_direction: bool = False
    wire_is_darker: bool = True

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must not be empty")
        if len(self.angle_slice) != 2 or any(value <= 0 for value in self.angle_slice):
            raise ValueError("angle_slice must contain two positive integers")
        if self.fov_width_cm is not None and self.fov_width_cm <= 0:
            raise ValueError("fov_width_cm must be positive")
        if self.period_range_cm is not None:
            low, high = self.period_range_cm
            if low <= 0 or low >= high:
                raise ValueError("period_range_cm must be positive with low < high")
        if self.crop_roi is not None:
            _, _, width, height = self.crop_roi
            if width <= 0 or height <= 0:
                raise ValueError("crop_roi width and height must be positive")
        if self.track is DetectorTrack.SINGLE_IMAGE and self.image_path is None:
            raise ValueError("single_image track requires image_path")
        if self.track is not DetectorTrack.SINGLE_IMAGE and self.image_path is not None:
            raise ValueError("a configured image_path must use the single_image track")

    @property
    def input_mode(self) -> InputMode:
        if self.track is DetectorTrack.SINGLE_IMAGE:
            return InputMode.SINGLE_IMAGE
        return InputMode.DRP_STACK

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for manifests."""

        value = asdict(self)
        value["track"] = self.track.value
        for key in ("data_root", "folder", "image_path", "config_path"):
            if value[key] is not None:
                value[key] = str(value[key])
        return value
