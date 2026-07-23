"""Publish complete V2 runs without exposing partial output directories."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import uuid4

from paperdrm.config import PipelineConfig
from paperdrm.models import ArtifactManifestEntry, PipelineResult, RunManifest


_ARTIFACT_GROUPS = frozenset({"diagnostics", "overlays", "reports"})
_IDENTIFIER_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)
_IDENTIFIER_START_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)


def _validate_identifier(value: str, *, label: str) -> str:
    if not value or len(value) > 128:
        raise ValueError(f"{label} must contain between 1 and 128 characters")
    if value in {".", ".."} or value[0] not in _IDENTIFIER_START_CHARACTERS:
        raise ValueError(f"{label} must start with an alphanumeric character")
    if any(character not in _IDENTIFIER_CHARACTERS for character in value):
        raise ValueError(
            f"{label} may contain only letters, numbers, dots, underscores, and hyphens"
        )
    return value


def _artifact_destination(value: str) -> PurePosixPath:
    if "\\" in value:
        raise ValueError("artifact destinations must use forward slashes")
    destination = PurePosixPath(value)
    if (
        destination.is_absolute()
        or len(destination.parts) < 2
        or destination.parts[0] not in _ARTIFACT_GROUPS
        or any(part in {"", ".", ".."} for part in destination.parts)
    ):
        groups = ", ".join(sorted(_ARTIFACT_GROUPS))
        raise ValueError(
            "artifact destination must be a safe relative path below one of "
            f"{groups}"
        )
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, payload: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync; some filesystems do not support it."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


class RunStore:
    """Persist immutable runs under ``<root>/<dataset>/<run-id>/``.

    All files are prepared in a private sibling directory. The completed
    directory is published with one same-filesystem rename, so readers see
    either no run or the complete run. A per-run lock prevents cooperating
    writers from replacing an existing run during the final rename.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def save(
        self,
        result: PipelineResult,
        config: PipelineConfig,
        *,
        run_id: str,
        inputs: Sequence[str | Path] = (),
        artifacts: Mapping[str, str | Path] | None = None,
    ) -> Path:
        """Write and atomically publish one run, returning its directory."""

        dataset_id = _validate_identifier(result.dataset_id, label="dataset_id")
        run_id = _validate_identifier(run_id, label="run_id")
        if config.dataset_id != result.dataset_id:
            raise ValueError("result and config dataset_id values differ")
        if config.track is not result.track:
            raise ValueError("result and config detector tracks differ")

        artifact_sources: list[tuple[PurePosixPath, Path]] = []
        for raw_destination, raw_source in sorted((artifacts or {}).items()):
            destination = _artifact_destination(raw_destination)
            source = Path(raw_source)
            if not source.is_file():
                raise FileNotFoundError(f"artifact source is not a file: {source}")
            artifact_sources.append((destination, source))

        # Validate both payloads before creating any persistence directories.
        result_payload = result.to_json() + "\n"
        config_payload = config.to_dict()
        json.dumps(config_payload, allow_nan=False)

        dataset_directory = self.root / dataset_id
        final_directory = dataset_directory / run_id
        lock_path = dataset_directory / f".{run_id}.lock"
        temporary_directory = dataset_directory / f".{run_id}.tmp-{uuid4().hex}"
        dataset_directory.mkdir(parents=True, exist_ok=True)
        if dataset_directory.is_symlink():
            raise ValueError(
                f"dataset directory must not be a symbolic link: {dataset_directory}"
            )

        lock_descriptor: int | None = None
        lock_acquired = False
        try:
            try:
                lock_descriptor = os.open(
                    lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
                lock_acquired = True
            except FileExistsError as exc:
                raise FileExistsError(
                    f"run is already being written: {dataset_id}/{run_id}"
                ) from exc

            if final_directory.exists():
                raise FileExistsError(
                    f"run already exists and will not be overwritten: "
                    f"{dataset_id}/{run_id}"
                )

            temporary_directory.mkdir()
            artifacts_directory = temporary_directory / "artifacts"
            for group in sorted(_ARTIFACT_GROUPS):
                (artifacts_directory / group).mkdir(parents=True)

            _write_text(temporary_directory / "result.json", result_payload)

            artifact_entries: list[ArtifactManifestEntry] = []
            for destination, source in artifact_sources:
                target = artifacts_directory.joinpath(*destination.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise FileExistsError(
                        f"artifact destinations collide on this filesystem: {destination}"
                    )
                shutil.copy2(source, target)
                _fsync_file(target)
                artifact_entries.append(
                    ArtifactManifestEntry(
                        path=f"artifacts/{destination.as_posix()}",
                        size_bytes=target.stat().st_size,
                        sha256=_sha256(target),
                    )
                )

            manifest = RunManifest(
                run_id=run_id,
                dataset_id=dataset_id,
                track=result.track,
                config=config_payload,
                inputs=tuple(str(path) for path in inputs),
                created_at_utc=datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
                policy_version=(
                    result.confidence.policy_version
                    if result.confidence is not None
                    else None
                ),
                artifacts=tuple(artifact_entries),
            )
            manifest_payload = json.dumps(
                manifest.to_dict(),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            _write_text(
                temporary_directory / "manifest.json",
                manifest_payload + "\n",
            )

            _fsync_directory(artifacts_directory)
            _fsync_directory(temporary_directory)
            temporary_directory.rename(final_directory)
            _fsync_directory(dataset_directory)
            return final_directory
        finally:
            if lock_descriptor is not None:
                os.close(lock_descriptor)
            if temporary_directory.exists():
                shutil.rmtree(temporary_directory)
            if lock_acquired and lock_path.exists():
                lock_path.unlink()
