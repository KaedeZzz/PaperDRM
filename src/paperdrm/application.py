"""Application service that sequences native execution and atomic storage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from paperdrm.config import PipelineConfig
from paperdrm.detection import NativeDetectorBackend
from paperdrm.io import FilesystemInputProvider, PreparedInput
from paperdrm.models import PipelineResult
from paperdrm.persistence import RunStore
from paperdrm.pipeline import Pipeline, PipelineRequest


class InputProvider(Protocol):
    def prepare(self, config: PipelineConfig) -> PreparedInput:
        ...


class ArtifactBuilder(Protocol):
    def build(
        self,
        result: PipelineResult,
        prepared: PreparedInput,
        directory: Path,
    ) -> Mapping[str, str | Path]:
        ...


@dataclass(frozen=True)
class ApplicationRun:
    result: PipelineResult
    run_directory: Path


class ApplicationRunner:
    """Own effect sequencing while core detection remains file-free."""

    def __init__(
        self,
        pipeline: Pipeline,
        store: RunStore,
        input_provider: InputProvider,
        *,
        artifact_builder: ArtifactBuilder | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._store = store
        self._input_provider = input_provider
        self._artifact_builder = artifact_builder

    @classmethod
    def native(
        cls,
        runs_root: str | Path,
        *,
        artifact_builder: ArtifactBuilder | None = None,
    ) -> "ApplicationRunner":
        return cls(
            Pipeline(NativeDetectorBackend()),
            RunStore(runs_root),
            FilesystemInputProvider(),
            artifact_builder=artifact_builder,
        )

    def run(self, config: PipelineConfig, *, run_id: str) -> ApplicationRun:
        self._store.ensure_available(config.dataset_id, run_id)
        prepared = self._input_provider.prepare(config)
        if prepared.config.dataset_id != config.dataset_id:
            raise ValueError("input provider changed dataset_id")
        if prepared.config.track is not config.track:
            raise ValueError("input provider changed detector track")
        result = self._pipeline.run(
            PipelineRequest(
                config=prepared.config,
                input_data=prepared.input_data,
            )
        )
        result = replace(
            result,
            provenance={
                **result.provenance,
                "application_runner": "paperdrm.application.ApplicationRunner",
                "requested_fov_width_cm": config.fov_width_cm,
                "effective_fov_width_cm": prepared.config.fov_width_cm,
                "input_files": [str(path) for path in prepared.input_paths],
            },
        )

        with TemporaryDirectory(prefix="paperdrm-artifacts-") as temporary:
            workspace = Path(temporary).resolve()
            artifacts: Mapping[str, str | Path] = {}
            if self._artifact_builder is not None:
                artifacts = self._artifact_builder.build(
                    result,
                    prepared,
                    workspace,
                )
                for source in artifacts.values():
                    try:
                        Path(source).resolve(strict=True).relative_to(workspace)
                    except (FileNotFoundError, ValueError) as exc:
                        raise ValueError(
                            "artifact builder sources must be files inside its workspace"
                        ) from exc
            run_directory = self._store.save(
                result,
                prepared.config,
                run_id=run_id,
                inputs=prepared.input_paths,
                artifacts=artifacts,
            )
        return ApplicationRun(result=result, run_directory=run_directory)
