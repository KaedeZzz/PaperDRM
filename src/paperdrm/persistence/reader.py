"""Read and verify one immutable V2 run directory."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from paperdrm.models import MANIFEST_SCHEMA_VERSION, RESULT_SCHEMA_VERSION


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON value is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key is not allowed: {key}")
        value[key] = item
    return value


def _load_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular JSON file: {path}")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite,
        object_pairs_hook=_unique_object,
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _contained_file(run_directory: Path, raw_path: object) -> tuple[str, Path]:
    if not isinstance(raw_path, str) or "\\" in raw_path:
        raise ValueError("manifest file paths must be forward-slash strings")
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"unsafe manifest file path: {raw_path!r}")

    candidate = run_directory.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"manifest file is missing: {candidate}") from exc
    if not resolved.is_relative_to(run_directory.resolve()):
        raise ValueError(f"manifest file escapes the run directory: {raw_path}")
    if candidate.is_symlink() or not resolved.is_file():
        raise ValueError(f"manifest path is not a regular file: {raw_path}")
    return relative.as_posix(), resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class StoredRun:
    directory: Path
    manifest: dict[str, Any]
    result: dict[str, Any]
    artifacts: dict[str, Path]


def read_verified_artifact(stored: StoredRun, relative_path: str) -> bytes:
    """Read one artifact and verify the bytes against its manifest entry."""

    try:
        path = stored.artifacts[relative_path]
    except KeyError as exc:
        raise KeyError(f"artifact is not declared by the run: {relative_path}") from exc
    metadata = next(
        entry
        for entry in stored.manifest["artifacts"]
        if entry["path"] == relative_path
    )
    payload = path.read_bytes()
    if len(payload) != metadata["size_bytes"]:
        raise ValueError(f"artifact size mismatch while reading: {relative_path}")
    if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
        raise ValueError(f"artifact checksum mismatch while reading: {relative_path}")
    return payload


def load_run(
    run_directory: str | Path,
    *,
    verify_artifacts: bool = True,
) -> StoredRun:
    """Load a V2 run and reject schema, identity, path or integrity drift."""

    directory = Path(run_directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"run directory must be a real directory: {directory}")

    manifest = _load_object(directory / "manifest.json")
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema version")
    if manifest.get("run_id") != directory.name:
        raise ValueError("manifest run_id does not match its directory")
    if manifest.get("dataset_id") != directory.parent.name:
        raise ValueError("manifest dataset_id does not match its directory")
    if not isinstance(manifest.get("config"), dict):
        raise ValueError("manifest config must be an object")

    result_name, result_path = _contained_file(
        directory,
        manifest.get("result_file"),
    )
    if result_name != "result.json":
        raise ValueError("manifest result_file must be result.json")
    result = _load_object(result_path)
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported result schema version")
    if manifest.get("result_schema_version") != result.get("schema_version"):
        raise ValueError("manifest and result schema versions differ")
    for field in ("dataset_id", "track"):
        if manifest.get(field) != result.get(field):
            raise ValueError(f"manifest and result {field} values differ")

    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ValueError("manifest artifacts must be a list")

    artifacts: dict[str, Path] = {}
    for entry in raw_artifacts:
        if not isinstance(entry, dict):
            raise ValueError("manifest artifact entries must be objects")
        relative, artifact_path = _contained_file(directory, entry.get("path"))
        if not relative.startswith("artifacts/"):
            raise ValueError(f"artifact is outside the artifacts directory: {relative}")
        if relative in artifacts:
            raise ValueError(f"duplicate artifact path in manifest: {relative}")

        expected_size = entry.get("size_bytes")
        expected_digest = entry.get("sha256")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise ValueError(f"invalid artifact size metadata: {relative}")
        if (
            not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(character not in "0123456789abcdef" for character in expected_digest)
        ):
            raise ValueError(f"invalid artifact digest metadata: {relative}")
        if verify_artifacts:
            if artifact_path.stat().st_size != expected_size:
                raise ValueError(f"artifact size mismatch: {relative}")
            if _sha256(artifact_path) != expected_digest:
                raise ValueError(f"artifact checksum mismatch: {relative}")
        artifacts[relative] = artifact_path

    return StoredRun(
        directory=directory.resolve(),
        manifest=manifest,
        result=result,
        artifacts=artifacts,
    )
