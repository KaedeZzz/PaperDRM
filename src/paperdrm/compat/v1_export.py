"""Create a disposable V1-compatible view from a canonical V2 run."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from paperdrm.persistence import load_run


def _present(**values: Any) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_v1_documents(
    result: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Map the compact V2 schema to report-readable V1 JSON documents."""

    config = config or {}
    measurement = result.get("measurement") or {}
    diagnostics = result.get("diagnostics") or {}
    evaluation = result.get("evaluation") or {}
    interval = evaluation.get("interval") or {}
    fit = evaluation.get("fit") or {}
    contrast = evaluation.get("contrast") or {}
    split_half = evaluation.get("split_half")
    wire_width = result.get("wire_width")
    grid = result.get("grid") or {}

    period_px = measurement.get("period_px")
    cm_per_px = measurement.get("cm_per_px")
    interval_cm = measurement.get("interval_cm")
    lines_per_cm = measurement.get("lines_per_cm")
    local_gap_px = interval.get("median_gap_px")
    gap_iqr_px = interval.get("gap_iqr_px")

    physical: dict[str, Any] | None = None
    if cm_per_px is not None:
        local_interval_cm = (
            local_gap_px * cm_per_px if local_gap_px is not None else None
        )
        gap_iqr_cm = (
            [gap_iqr_px[0] * cm_per_px, gap_iqr_px[1] * cm_per_px]
            if gap_iqr_px is not None
            else None
        )
        physical = _present(
            cm_per_px=cm_per_px,
            spectral_interval_cm=interval_cm,
            spectral_lines_per_cm=lines_per_cm,
            median_interval_cm=local_interval_cm,
            lines_per_cm_median=(
                1.0 / local_interval_cm
                if local_interval_cm is not None and local_interval_cm > 0
                else None
            ),
            gap_iqr_cm=gap_iqr_cm,
            gap_median_relative_error_vs_spectral=interval.get(
                "gap_median_relative_error_vs_spectral"
            ),
        )

    documents: dict[str, dict[str, Any]] = {
        "interval_distribution.json": _present(
            n_peaks=interval.get("n_peaks", 0),
            n_gaps=interval.get("n_gaps", 0),
            period_px_used=period_px,
            px=_present(median=local_gap_px, iqr=gap_iqr_px),
            physical=physical,
        ),
        "fit_quality.json": _present(
            period_px_used=period_px,
            r2_fundamental_only=fit.get("r2_fundamental"),
            r2_with_harmonics=fit.get("r2_with_harmonics"),
            r2_gaussian_comb=fit.get("r2_gaussian_comb"),
            frequency_concentration=fit.get("frequency_concentration"),
            best_period_by_r2=fit.get("best_period_by_r2_px"),
            best_r2=fit.get("best_r2"),
            agrees_with_dominant=fit.get("agrees_with_dominant"),
            period_at_search_boundary=diagnostics.get(
                "period_at_search_boundary", False
            ),
            period_boundary_side=diagnostics.get("period_boundary_side"),
            period_warning=diagnostics.get("period_warning"),
        ),
        "self_contrast.json": _present(
            n_lines=contrast.get("n_lines", 0),
            contrast_rel=contrast.get("contrast_relative"),
            contrast_z=contrast.get("contrast_z"),
            warning=contrast.get("warning"),
            wire_is_darker=config.get("wire_is_darker"),
            period_px_used=period_px,
            line_dir_deg=grid.get("line_direction_deg"),
        ),
    }

    if split_half is not None:
        documents["split_half_stability.json"] = _present(
            n_images=split_half.get("n_images"),
            n_splits=split_half.get("n_splits"),
            period_diff_std=split_half.get("period_difference_std_px"),
            agree_rate_within_1px=split_half.get("agree_rate_within_1px"),
            agree_rate_within_0p5px=split_half.get(
                "agree_rate_within_half_px"
            ),
        )

    if wire_width is not None:
        median_fwhm_mm = wire_width.get("median_fwhm_mm")
        wire_document = _present(
            period_px=period_px,
            line_dir_deg=grid.get("line_direction_deg"),
            n_segments=wire_width.get("segment_count", 0),
            aggregate={
                "fwhm_px": _present(
                    n_valid=wire_width.get("segment_valid_count", 0),
                    median=wire_width.get("segment_median_fwhm_px"),
                )
            },
            physical=(
                {"fwhm_mm": {"median": median_fwhm_mm}}
                if median_fwhm_mm is not None
                else None
            ),
            experimental=wire_width.get("experimental", True),
        )
        wire_document["global"] = _present(
            fwhm_px=wire_width.get("fwhm_px"),
            model_ok=wire_width.get("model_ok"),
            warning=wire_width.get("warning"),
        )
        documents["wire_width_stats.json"] = wire_document

    return documents


class V1RunExporter:
    """Export a verified run into a new, disposable flat V1 directory."""

    def export(
        self,
        run_directory: str | Path,
        destination: str | Path,
    ) -> Path:
        stored = load_run(run_directory)
        target = Path(destination)
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"V1 export destination already exists: {target}"
            )

        documents = build_v1_documents(
            stored.result,
            config=stored.manifest.get("config") or {},
        )
        artifact_metadata = {
            entry["path"]: entry for entry in stored.manifest["artifacts"]
        }
        temporary = target.parent / f".{target.name}.tmp-{uuid4().hex}"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            temporary.mkdir()
            for name, document in sorted(documents.items()):
                payload = json.dumps(
                    document,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                (temporary / name).write_text(payload + "\n", encoding="utf-8")

            copied_names: set[str] = set()
            for relative, source in sorted(stored.artifacts.items()):
                name = Path(relative).name
                if name in documents:
                    continue
                if name in copied_names or (temporary / name).exists():
                    raise ValueError(
                        f"V1 flat export has an artifact name collision: {name}"
                    )
                copied = temporary / name
                shutil.copy2(source, copied)
                metadata = artifact_metadata[relative]
                if (
                    copied.stat().st_size != metadata["size_bytes"]
                    or _sha256(copied) != metadata["sha256"]
                ):
                    raise ValueError(
                        f"artifact changed while creating the V1 export: {relative}"
                    )
                copied_names.add(name)

            if target.exists() or target.is_symlink():
                raise FileExistsError(
                    f"V1 export destination already exists: {target}"
                )
            temporary.rename(target)
            return target
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
