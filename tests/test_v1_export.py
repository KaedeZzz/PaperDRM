import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paperdrm.compat import V1RunExporter
from paperdrm.config import PipelineConfig
from paperdrm.models import (
    ConfidenceAssessment,
    ConfidenceLevel,
    ConfidenceReason,
    ContrastEvaluation,
    DetectionDiagnostics,
    DetectorTrack,
    EvaluationSummary,
    FitEvaluation,
    GridEstimate,
    IntervalEvaluation,
    PipelineResult,
    ResultDisposition,
    SpacingMeasurement,
    SplitHalfEvaluation,
    WireWidthEstimate,
)
from paperdrm.persistence import RunStore, load_run
from scripts.generate_report import interpret, load_results


class V1RunExporterTests(unittest.TestCase):
    def _config(self):
        return PipelineConfig(
            dataset_id="folio",
            track=DetectorTrack.MULTI_PHI,
            fov_width_cm=10.0,
            wire_is_darker=True,
        )

    def _result(self):
        return PipelineResult(
            dataset_id="folio",
            track=DetectorTrack.MULTI_PHI,
            measurement=SpacingMeasurement.from_period(20.0, cm_per_px=0.005),
            diagnostics=DetectionDiagnostics(
                self_contrast_z=3.5,
                split_half_period_diff_std_px=0.25,
                split_half_agree_rate_within_1px=0.95,
            ),
            grid=GridEstimate(
                line_direction_deg=90.0,
                phase_rad=0.5,
                positions_px=(10, 30, 50),
            ),
            wire_width=WireWidthEstimate(
                fwhm_px=4.0,
                model_ok=True,
                segment_median_fwhm_px=4.2,
                segment_valid_count=15,
                segment_count=16,
                median_fwhm_mm=0.21,
            ),
            evaluation=EvaluationSummary(
                interval=IntervalEvaluation(
                    n_peaks=11,
                    n_gaps=10,
                    median_gap_px=20.5,
                    gap_iqr_px=(19.0, 22.0),
                    gap_median_relative_error_vs_spectral=0.025,
                ),
                fit=FitEvaluation(
                    r2_fundamental=0.3,
                    r2_with_harmonics=0.5,
                    r2_gaussian_comb=0.2,
                    frequency_concentration=0.7,
                    best_period_by_r2_px=20.0,
                    best_r2=0.5,
                    agrees_with_dominant=True,
                ),
                contrast=ContrastEvaluation(
                    n_lines=11,
                    contrast_relative=0.12,
                    contrast_z=3.5,
                ),
                split_half=SplitHalfEvaluation(
                    n_images=8,
                    n_splits=20,
                    period_difference_std_px=0.25,
                    agree_rate_within_1px=0.95,
                    agree_rate_within_half_px=0.8,
                ),
            ),
            confidence=ConfidenceAssessment(
                disposition=ResultDisposition.ACCEPTED,
                level=ConfidenceLevel.HIGH,
                primary_reason=ConfidenceReason.STRONG_SELF_CONTRAST,
            ),
        )

    def _stored_run(self, root: Path, *, artifacts=None) -> Path:
        return RunStore(root / "runs").save(
            self._result(),
            self._config(),
            run_id="run-001",
            inputs=("folio-a.tif", "folio-b.tif"),
            artifacts=artifacts,
        )

    def test_exports_report_readable_v1_documents_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            overlay = root / "overlay.png"
            overlay.write_bytes(b"overlay")
            run = self._stored_run(
                root,
                artifacts={"overlays/laid_lines_overlay_bands.png": overlay},
            )

            destination = V1RunExporter().export(run, root / "legacy-view")

            self.assertTrue((destination / "interval_distribution.json").is_file())
            self.assertTrue((destination / "fit_quality.json").is_file())
            self.assertTrue((destination / "self_contrast.json").is_file())
            self.assertTrue((destination / "split_half_stability.json").is_file())
            self.assertTrue((destination / "wire_width_stats.json").is_file())
            self.assertEqual(
                (destination / "laid_lines_overlay_bands.png").read_bytes(),
                b"overlay",
            )

            interval = json.loads(
                (destination / "interval_distribution.json").read_text()
            )
            self.assertEqual(interval["period_px_used"], 20.0)
            self.assertEqual(interval["physical"]["spectral_lines_per_cm"], 10.0)
            self.assertEqual(interval["physical"]["gap_iqr_cm"], [0.095, 0.11])

            values = interpret(load_results(destination), "folio")
            self.assertEqual(values["detect_confidence_en"], "High")
            self.assertAlmostEqual(values["period_mm"], 1.0)
            self.assertAlmostEqual(values["fwhm_mm_median"], 0.21)

    def test_reader_rejects_artifact_checksum_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "overlay.png"
            source.write_bytes(b"original")
            run = self._stored_run(
                root,
                artifacts={"overlays/overlay.png": source},
            )
            (run / "artifacts/overlays/overlay.png").write_bytes(b"modified")

            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_run(run)
            self.assertFalse((root / "legacy-view").exists())

    def test_reader_rejects_manifest_result_identity_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._stored_run(root)
            result_path = run / "result.json"
            result = json.loads(result_path.read_text())
            result["dataset_id"] = "different"
            result_path.write_text(json.dumps(result), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "dataset_id values differ"):
                load_run(run)

    def test_reader_rejects_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._stored_run(root)
            result_path = run / "result.json"
            payload = result_path.read_text().replace(
                '"period_px": 20.0',
                '"period_px": NaN',
            )
            result_path.write_text(payload, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "non-finite JSON"):
                load_run(run)

    def test_reader_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._stored_run(root)
            manifest_path = run / "manifest.json"
            payload = manifest_path.read_text().replace(
                '"run_id": "run-001"',
                '"run_id": "run-001", "run_id": "run-001"',
            )
            manifest_path.write_text(payload, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
                load_run(run)

    def test_export_never_overwrites_existing_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._stored_run(root)
            destination = root / "legacy-view"
            destination.mkdir()
            sentinel = destination / "manual.txt"
            sentinel.write_text("preserve me", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                V1RunExporter().export(run, destination)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve me")

    def test_flat_artifact_name_collision_fails_without_partial_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")
            run = self._stored_run(
                root,
                artifacts={
                    "diagnostics/shared.txt": first,
                    "reports/shared.txt": second,
                },
            )
            destination = root / "legacy-view"

            with self.assertRaisesRegex(ValueError, "name collision"):
                V1RunExporter().export(run, destination)

            self.assertFalse(destination.exists())
            self.assertFalse(
                any(path.name.startswith(".legacy-view") for path in root.iterdir())
            )

    def test_artifact_change_during_copy_fails_without_partial_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "overlay.png"
            source.write_bytes(b"original")
            run = self._stored_run(
                root,
                artifacts={"overlays/overlay.png": source},
            )
            destination = root / "legacy-view"

            def copy_modified(_source, target):
                Path(target).write_bytes(b"modified")

            with patch(
                "paperdrm.compat.v1_export.shutil.copy2",
                side_effect=copy_modified,
            ):
                with self.assertRaisesRegex(ValueError, "changed while creating"):
                    V1RunExporter().export(run, destination)

            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
