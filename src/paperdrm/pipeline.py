"""Execution boundary for V2 pipeline implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from paperdrm.config import PipelineConfig
from paperdrm.models import PipelineResult, RESULT_SCHEMA_VERSION


@dataclass(frozen=True)
class SingleImageInput:
    image: np.ndarray

    def __post_init__(self) -> None:
        if self.image.ndim != 2 or self.image.size == 0:
            raise ValueError("single-image input must be a non-empty 2D array")


@dataclass(frozen=True)
class MultiPhiInput:
    images: tuple[np.ndarray, ...]
    phi_deg: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.images) < 2:
            raise ValueError("multi-phi input requires at least two images")
        shape = self.images[0].shape
        if len(shape) != 2 or any(image.ndim != 2 for image in self.images):
            raise ValueError("multi-phi images must be 2D arrays")
        if any(image.shape != shape for image in self.images):
            raise ValueError("multi-phi images must have identical shapes")
        if self.phi_deg is not None and len(self.phi_deg) != len(self.images):
            raise ValueError("phi_deg length must match the image count")


PipelineInput = SingleImageInput | MultiPhiInput


@dataclass(frozen=True)
class PipelineRequest:
    config: PipelineConfig
    result_directory: Path | None = None
    input_data: PipelineInput | None = None


class PipelineBackend(Protocol):
    """Backend contract implemented by V1 adapters and future V2 execution."""

    def execute(self, request: PipelineRequest) -> PipelineResult:
        ...


class Pipeline:
    """Validate orchestration-level invariants around an injected backend."""

    def __init__(self, backend: PipelineBackend) -> None:
        self._backend = backend

    def run(self, request: PipelineRequest) -> PipelineResult:
        result = self._backend.execute(request)
        if result.schema_version != RESULT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported result schema {result.schema_version}; "
                f"expected {RESULT_SCHEMA_VERSION}"
            )
        if result.dataset_id != request.config.dataset_id:
            raise ValueError("backend returned a result for a different dataset")
        if result.track is not request.config.track:
            raise ValueError("backend returned a result for a different track")
        return result
