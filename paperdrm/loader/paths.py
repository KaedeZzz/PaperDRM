from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataPaths:
    """
    Convenience wrapper for repository data locations.
    Creates the expected subdirectories if they are missing.
    """

    root: Path
    raw: Path
    processed: Path
    cache: Path

    @classmethod
    def from_root(cls, root: Path | str) -> "DataPaths":
        base = Path(root)
        if not base.is_absolute():
            base = Path.cwd() / base

        raw = base / "raw"
        processed = base / "processed"
        cache = base / "cache"
        for path in (raw, processed, cache):
            path.mkdir(parents=True, exist_ok=True)

        return cls(root=base, raw=raw, processed=processed, cache=cache)
