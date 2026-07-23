"""Stable cache fingerprints for expensive image-derived artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


CACHE_SCHEMA_VERSION = 1


def build_cache_fingerprint(
    files: Iterable[str | Path],
    parameters: Mapping[str, Any],
) -> str:
    """
    Hash the selected input-file identities and all preprocessing parameters.

    File contents are not hashed because DRP datasets can contain hundreds of
    large images. Resolved path, size and nanosecond mtime provide a fast,
    deterministic invalidation key for normal research workflows.
    """
    file_records = []
    for value in files:
        path = Path(value).resolve()
        stat = path.stat()
        file_records.append({
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })

    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "files": file_records,
        "parameters": _jsonable(parameters),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
