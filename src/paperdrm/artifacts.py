"""Concrete visual artifacts for a completed native V2 result."""

from __future__ import annotations

from pathlib import Path

import cv2

from paperdrm.io import PreparedInput
from paperdrm.models import PipelineResult
from paperdrm.reporting import render_bilingual_reports, report_values_from_v2
from paperdrm.stage3_detect.simple_detector import overlay_grid, overlay_grid_bands


def _write_image(path: Path, image) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write artifact image: {path}")


class StandardArtifactBuilder:
    """Build standard overlays and canonical bilingual V2 reports."""

    def build(
        self,
        result: PipelineResult,
        prepared: PreparedInput,
        directory: Path,
    ) -> dict[str, Path]:
        if result.grid is None:
            raise ValueError("standard overlays require a grid estimate")
        representative = int(result.provenance.get("representative_index", 0))
        if representative < 0 or representative >= len(prepared.display_images):
            raise ValueError("representative image index is outside prepared display images")
        image = prepared.display_images[representative]
        grid_path = directory / "laid_lines_overlay.png"
        grid = overlay_grid(
            image,
            result.grid.positions_px,
            line_dir_deg=result.grid.line_direction_deg,
            color=(0, 0, 255),
            thickness=1,
            alpha=0.55,
        )
        _write_image(grid_path, grid)
        artifacts = {"overlays/laid_lines_overlay.png": grid_path}
        report_overlay = grid_path

        wire_width = result.wire_width
        if wire_width is not None:
            fwhm = (
                wire_width.segment_median_fwhm_px
                if wire_width.segment_median_fwhm_px is not None
                else wire_width.fwhm_px
            )
            if fwhm is not None:
                bands_path = directory / "laid_lines_overlay_bands.png"
                bands = overlay_grid_bands(
                    image,
                    result.grid.positions_px,
                    fwhm,
                    line_dir_deg=result.grid.line_direction_deg,
                    color=(0, 0, 255),
                    alpha=0.4,
                )
                _write_image(bands_path, bands)
                artifacts["overlays/laid_lines_overlay_bands.png"] = bands_path
                report_overlay = bands_path

        values = report_values_from_v2(
            result.to_dict(),
            serial=result.dataset_id,
            technical_location=(
                "manifest.json and result.json in this immutable run"
            ),
        )
        english, chinese = render_bilingual_reports(
            values,
            report_overlay.read_bytes(),
        )
        english_path = directory / "report_en.html"
        chinese_path = directory / "report_zh.html"
        english_path.write_text(english, encoding="utf-8")
        chinese_path.write_text(chinese, encoding="utf-8")
        artifacts["reports/report_en.html"] = english_path
        artifacts["reports/report_zh.html"] = chinese_path
        return artifacts
