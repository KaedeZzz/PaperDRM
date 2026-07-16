"""Safe, track-aware archiving for pipeline outputs."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Protocol


class _PackWithSerial(Protocol):
    data_serial: object


MANAGED_RESULT_FILES = frozenset({
    "evaluation_report.json",
    "interval_distribution.json",
    "fit_quality.json",
    "wire_width_stats.json",
    "wire_width_segments.png",
    "split_half_stability.json",
    "split_half_stability.png",
    "self_contrast.json",
    "self_contrast.png",
    "self_contrast.simple.json",
    "self_contrast.simple.png",
    "laid_lines_overlay.png",
    "laid_lines_overlay_bands.png",
    "laid_lines_overlay_legacy.png",
    "report_en.html",
    "report_zh.html",
})

SINGLE_IMAGE_ARTIFACTS = (
    "interval_distribution.json",
    "fit_quality.json",
    "wire_width_stats.json",
    "wire_width_segments.png",
    "self_contrast.json",
    "self_contrast.png",
    "laid_lines_overlay.png",
    "laid_lines_overlay_bands.png",
)

MULTI_PHI_ARTIFACTS = SINGLE_IMAGE_ARTIFACTS + (
    "split_half_stability.json",
    "split_half_stability.png",
)

LEGACY_ARTIFACTS = (
    "evaluation_report.json",
    "interval_distribution.json",
    "fit_quality.json",
    "laid_lines_overlay_legacy.png",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def archive_results(
    pack: _PackWithSerial | None,
    config_path: str | Path,
    *,
    artifacts: Iterable[str | Path],
    serial: str | None = None,
    root: Path | None = None,
    generate_reports: bool = True,
) -> Path:
    """
    Archive only explicitly declared outputs from the current pipeline run.

    Old pipeline-managed files are removed first so results from another track
    cannot contaminate this dataset. User-managed files such as manual_gt.json
    and bbox overlays are preserved.
    """
    if serial is None:
        value = pack.data_serial if pack is not None else None
        serial = str(value) if value is not None else "unknown"

    root = repository_root() if root is None else Path(root)
    archive_dir = root / "results" / serial
    archive_dir.mkdir(parents=True, exist_ok=True)

    sources = []
    for artifact in artifacts:
        src = Path(artifact)
        if not src.is_absolute():
            src = root / src
        if not src.is_file():
            raise FileNotFoundError(f"Expected pipeline artifact was not created: {src}")
        sources.append(src)

    removed = []
    for name in sorted(MANAGED_RESULT_FILES):
        stale = archive_dir / name
        if stale.is_file():
            stale.unlink()
            removed.append(name)

    copied = []
    for src in sources:
        shutil.copy2(src, archive_dir / src.name)
        copied.append(src.name)

    cfg_src = Path(config_path)
    if cfg_src.exists():
        cfg_dst = archive_dir / cfg_src.name
        if cfg_src.resolve() != cfg_dst.resolve():
            shutil.copy2(cfg_src, cfg_dst)
            copied.append(cfg_src.name)

    cleaned = []
    for name in sorted(MANAGED_RESULT_FILES):
        staged = root / name
        if staged.is_file():
            staged.unlink()
            cleaned.append(name)

    print(
        f"[Archive] removed {len(removed)} stale managed files; "
        f"copied {len(copied)} current files -> results/{serial}/; "
        f"cleaned {len(cleaned)} root staging files"
        f"  ({', '.join(copied)})"
    )

    if generate_reports:
        try:
            report_script = root / "scripts" / "generate_report.py"
            subprocess.run(
                [
                    sys.executable,
                    str(report_script),
                    "--serial",
                    serial,
                    "--results-dir",
                    str(root / "results"),
                ],
                check=True,
            )
        except Exception as exc:
            print(f"[Archive] Report generation failed: {exc}")

    return archive_dir
