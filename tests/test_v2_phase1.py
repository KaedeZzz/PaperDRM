import json
import tempfile
import unittest
from pathlib import Path

from paperdrm.cli import build_parser
from paperdrm.compat.v1 import (
    V1ResultDirectoryBackend,
    load_v1_config,
    result_from_directory,
)
from paperdrm.config import PipelineConfig
from paperdrm.models import (
    DetectionDiagnostics,
    DetectorTrack,
    PipelineResult,
    RunManifest,
    SpacingMeasurement,
)
from paperdrm.pipeline import Pipeline, PipelineRequest


ROOT = Path(__file__).resolve().parents[1]


class Phase1ConfigAndCliTests(unittest.TestCase):
    def test_parser_keeps_v1_defaults_and_choices(self):
        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.config, "exp_param.yaml")
        self.assertIsNone(args.image)
        self.assertEqual(args.track, "multi_phi")

        selected = parser.parse_args(["--track", "simple", "--config", "x.yaml"])
        self.assertEqual(selected.track, "simple")
        self.assertEqual(selected.config, "x.yaml")

    def test_image_path_in_yaml_overrides_requested_drp_track(self):
        config = load_v1_config(
            ROOT / "configs/Kk1-5_f5v.yaml",
            requested_track=DetectorTrack.LEGACY,
        )
        self.assertEqual(config.track, DetectorTrack.SINGLE_IMAGE)
        self.assertIsNotNone(config.image_path)
        self.assertEqual(config.dataset_id, "Kk1-5_f5v")
        self.assertEqual(config.input_mode.value, "single_image")

    def test_cli_image_override_uses_filename_when_serial_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "empty.yaml"
            yaml_path.write_text("{}\n", encoding="utf-8")
            config = load_v1_config(yaml_path, image_override="folio.png")

        self.assertEqual(config.track, DetectorTrack.SINGLE_IMAGE)
        self.assertEqual(config.dataset_id, "folio")
        self.assertEqual(config.image_path, Path("folio.png"))


class Phase1ResultModelTests(unittest.TestCase):
    def test_spectral_measurement_derives_physical_units(self):
        measurement = SpacingMeasurement.from_period(25.0, cm_per_px=0.004)
        self.assertAlmostEqual(measurement.interval_cm, 0.1)
        self.assertAlmostEqual(measurement.lines_per_cm, 10.0)
        self.assertEqual(measurement.source.value, "global_spectral_period")

    def test_v1_results_aggregate_into_versioned_v2_result(self):
        config = load_v1_config(ROOT / "configs/Kk1-5_f5v.yaml")
        result = Pipeline(V1ResultDirectoryBackend()).run(
            PipelineRequest(
                config=config,
                result_directory=ROOT / "results/Kk1-5_f5v",
            )
        )

        self.assertEqual(result.schema_version, 2)
        self.assertEqual(result.dataset_id, "Kk1-5_f5v")
        self.assertEqual(result.track, DetectorTrack.SINGLE_IMAGE)
        self.assertAlmostEqual(result.measurement.lines_per_cm, 9.137045846, places=6)
        self.assertFalse(result.diagnostics.polarity_contradiction)
        self.assertIn("manual_gt.json", result.artifacts)

        payload = result.to_dict()
        self.assertEqual(payload["measurement"]["source"], "global_spectral_period")
        json.dumps(payload)

    def test_negative_v1_self_contrast_remains_a_contradiction(self):
        config = load_v1_config(ROOT / "configs/Ff2-6_f140r.yaml")
        result = result_from_directory(ROOT / "results/Ff2-6_f140r", config)
        self.assertLess(result.diagnostics.self_contrast_z, 0.0)
        self.assertTrue(result.diagnostics.polarity_contradiction)

    def test_pipeline_rejects_backend_identity_mismatch(self):
        config = PipelineConfig(dataset_id="expected", track=DetectorTrack.MULTI_PHI)

        class WrongDatasetBackend:
            def execute(self, request):
                return PipelineResult(
                    dataset_id="wrong",
                    track=request.config.track,
                    measurement=SpacingMeasurement(period_px=20.0),
                )

        with self.assertRaisesRegex(ValueError, "different dataset"):
            Pipeline(WrongDatasetBackend()).run(
                PipelineRequest(config=config, result_directory=Path("unused"))
            )

    def test_manifest_is_json_compatible_and_versioned(self):
        manifest = RunManifest(
            run_id="run-001",
            dataset_id="folio",
            track=DetectorTrack.SIMPLE,
            config={"period_range_cm": [0.08, 0.2]},
            inputs=("image-a.jpg",),
        )
        payload = manifest.to_dict()
        self.assertEqual(payload["manifest_schema_version"], 1)
        self.assertEqual(payload["track"], "simple")
        self.assertEqual(payload["inputs"], ["image-a.jpg"])
        json.dumps(payload)

    def test_boundary_side_requires_boundary_hit(self):
        with self.assertRaisesRegex(ValueError, "requires a boundary hit"):
            DetectionDiagnostics(period_boundary_side="upper")


if __name__ == "__main__":
    unittest.main()
