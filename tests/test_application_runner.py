import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from paperdrm.application import ApplicationRunner
from paperdrm.config import PipelineConfig
from paperdrm.io import FilesystemInputProvider, PreparedInput
from paperdrm.models import DetectorTrack, PipelineResult, SpacingMeasurement
from paperdrm.persistence import RunStore, load_run
from paperdrm.pipeline import Pipeline, SingleImageInput
from paperdrm.v2_cli import build_parser as build_v2_parser
from paperdrm.v2_cli import main as v2_main


def _laid_lines(shape=(128, 128), period=16.0):
    height, width = shape
    x = np.arange(width, dtype=np.float64)
    distance = (x + period / 2.0) % period - period / 2.0
    profile = 1.0 - np.exp(-0.5 * (distance / 1.8) ** 2)
    return (np.repeat(profile[None, :], height, axis=0) * 255).astype(np.uint8)


class FilesystemInputProviderTests(unittest.TestCase):
    def test_single_image_crop_and_square_adjust_effective_fov(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "single.png"
            image = np.arange(200, dtype=np.uint8).reshape(10, 20)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            config = PipelineConfig(
                dataset_id="single",
                track=DetectorTrack.SINGLE_IMAGE,
                image_path=image_path,
                subtract_background=False,
                crop_roi=(2, 1, 10, 8),
                square_crop=True,
                fov_width_cm=20.0,
            )

            prepared = FilesystemInputProvider().prepare(config)

            self.assertIsInstance(prepared.input_data, SingleImageInput)
            self.assertEqual(prepared.input_data.image.shape, (8, 8))
            self.assertEqual(prepared.display_images[0].shape, (8, 8))
            self.assertAlmostEqual(prepared.config.fov_width_cm, 8.0)
            self.assertEqual(prepared.input_paths, (image_path.resolve(),))

    def test_crop_outside_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "single.png"
            self.assertTrue(cv2.imwrite(str(image_path), np.zeros((10, 10), np.uint8)))
            config = PipelineConfig(
                dataset_id="single",
                track=DetectorTrack.SINGLE_IMAGE,
                image_path=image_path,
                subtract_background=False,
                crop_roi=(8, 8, 4, 4),
            )

            with self.assertRaisesRegex(ValueError, "outside image bounds"):
                FilesystemInputProvider().prepare(config)

    def test_multi_phi_loads_only_highest_theta_in_numeric_phi_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            values = {
                "0_10.png": 10,
                "0_20.png": 20,
                "90_10.png": 110,
                "90_20.png": 120,
            }
            for name, value in values.items():
                self.assertTrue(
                    cv2.imwrite(
                        str(raw / name),
                        np.full((12, 16), value, dtype=np.uint8),
                    )
                )
            config = PipelineConfig(
                dataset_id="drp",
                track=DetectorTrack.MULTI_PHI,
                data_root=root,
                image_format="png",
                subtract_background=False,
            )

            prepared = FilesystemInputProvider().prepare(config)

            self.assertEqual(prepared.input_data.phi_deg, (0.0, 90.0))
            self.assertEqual(
                tuple(int(image[0, 0]) for image in prepared.input_data.images),
                (20, 120),
            )
            self.assertEqual(
                tuple(path.name for path in prepared.input_paths),
                ("0_20.png", "90_20.png"),
            )

    def test_simple_drp_route_reads_only_one_grazing_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            for phi in (0, 90):
                for theta in (10, 20):
                    self.assertTrue(
                        cv2.imwrite(
                            str(raw / f"{phi}_{theta}.png"),
                            np.full((8, 8), phi + theta, dtype=np.uint8),
                        )
                    )
            config = PipelineConfig(
                dataset_id="simple",
                track=DetectorTrack.SIMPLE,
                data_root=root,
                image_format="png",
                subtract_background=False,
            )

            prepared = FilesystemInputProvider().prepare(config)

            self.assertIsInstance(prepared.input_data, SingleImageInput)
            self.assertEqual(prepared.input_paths[0].name, "0_20.png")
            self.assertEqual(len(prepared.input_paths), 1)


class ApplicationRunnerTests(unittest.TestCase):
    @staticmethod
    def _config():
        return PipelineConfig(
            dataset_id="runner",
            track=DetectorTrack.SINGLE_IMAGE,
            image_path=Path("provided-in-memory.png"),
            subtract_background=False,
        )

    @staticmethod
    def _prepared(config):
        image = np.full((32, 32), 128, dtype=np.uint8)
        return PreparedInput(
            config=config,
            input_data=SingleImageInput(image),
            input_paths=(Path("input-a.png"),),
            display_images=(image,),
        )

    def test_sequences_pipeline_artifacts_and_atomic_store(self):
        class Backend:
            def execute(self, request):
                return PipelineResult(
                    dataset_id=request.config.dataset_id,
                    track=request.config.track,
                    measurement=SpacingMeasurement(period_px=16.0),
                )

        class Provider:
            def prepare(inner_self, config):
                return self._prepared(config)

        class Builder:
            def build(self, result, prepared, directory):
                report = directory / "report.html"
                report.write_text("<html>report</html>", encoding="utf-8")
                return {"reports/report.html": report}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = ApplicationRunner(
                Pipeline(Backend()),
                RunStore(root / "runs"),
                Provider(),
                artifact_builder=Builder(),
            )

            application_run = runner.run(self._config(), run_id="run-001")
            stored = load_run(application_run.run_directory)

            self.assertEqual(stored.result["measurement"]["period_px"], 16.0)
            self.assertEqual(
                stored.result["provenance"]["application_runner"],
                "paperdrm.application.ApplicationRunner",
            )
            self.assertEqual(stored.manifest["inputs"], ["input-a.png"])
            self.assertEqual(
                (application_run.run_directory / "artifacts/reports/report.html").read_text(),
                "<html>report</html>",
            )
            json.dumps(stored.result, allow_nan=False)

    def test_provider_cannot_change_run_identity(self):
        class Provider:
            def prepare(inner_self, config):
                return self._prepared(replace(config, dataset_id="changed"))

        class Backend:
            def execute(self, request):
                raise AssertionError("pipeline must not run")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = ApplicationRunner(
                Pipeline(Backend()),
                RunStore(root / "runs"),
                Provider(),
            )
            with self.assertRaisesRegex(ValueError, "changed dataset_id"):
                runner.run(self._config(), run_id="run-001")
            self.assertFalse((root / "runs").exists())

    def test_invalid_run_id_fails_before_input_or_pipeline_work(self):
        class Provider:
            def prepare(self, config):
                raise AssertionError("input provider must not run")

        class Backend:
            def execute(self, request):
                raise AssertionError("pipeline must not run")

        with tempfile.TemporaryDirectory() as temporary:
            runner = ApplicationRunner(
                Pipeline(Backend()),
                RunStore(Path(temporary) / "runs"),
                Provider(),
            )
            with self.assertRaisesRegex(ValueError, "run_id"):
                runner.run(self._config(), run_id="../unsafe")

    def test_artifact_builder_cannot_escape_workspace(self):
        class Backend:
            def execute(self, request):
                return PipelineResult(
                    dataset_id=request.config.dataset_id,
                    track=request.config.track,
                    measurement=SpacingMeasurement(period_px=16.0),
                )

        class Provider:
            def prepare(inner_self, config):
                return self._prepared(config)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside.html"
            outside.write_text("outside", encoding="utf-8")

            class Builder:
                def build(self, result, prepared, directory):
                    return {"reports/report.html": outside}

            runner = ApplicationRunner(
                Pipeline(Backend()),
                RunStore(root / "runs"),
                Provider(),
                artifact_builder=Builder(),
            )
            with self.assertRaisesRegex(ValueError, "inside its workspace"):
                runner.run(self._config(), run_id="run-001")
            self.assertFalse((root / "runs").exists())

    def test_native_runner_executes_single_image_from_disk(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "laid-lines.png"
            self.assertTrue(cv2.imwrite(str(image_path), _laid_lines()))
            config = PipelineConfig(
                dataset_id="native",
                track=DetectorTrack.SINGLE_IMAGE,
                image_path=image_path,
                subtract_background=False,
                fov_width_cm=1.28,
                period_range_cm=(0.12, 0.24),
                line_direction_deg=90.0,
            )

            application_run = ApplicationRunner.native(root / "runs").run(
                config,
                run_id="run-001",
            )

            stored = load_run(application_run.run_directory)
            self.assertAlmostEqual(
                stored.result["measurement"]["period_px"],
                16.0,
            )
            self.assertEqual(stored.result["schema_version"], 2)

    def test_v2_cli_publishes_native_run_with_standard_overlays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "laid-lines.png"
            config_path = root / "config.yaml"
            runs_root = root / "runs"
            self.assertTrue(cv2.imwrite(str(image_path), _laid_lines()))
            config_path.write_text(
                "\n".join(
                    (
                        "data_serial: cli-synthetic",
                        "subtract_background: false",
                        "fov_width_cm: 1.28",
                        "period_range_cm: [0.12, 0.24]",
                        "line_dir_deg: 90.0",
                        "wire_is_darker: true",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()) as output:
                exit_code = v2_main(
                    [
                        "--config",
                        str(config_path),
                        "--image",
                        str(image_path),
                        "--run-id",
                        "run-001",
                        "--runs-root",
                        str(runs_root),
                    ]
                )

            run = runs_root / "cli-synthetic" / "run-001"
            stored = load_run(run)
            self.assertEqual(exit_code, 0)
            self.assertIn("Run published:", output.getvalue())
            self.assertEqual(stored.result["track"], "single_image")
            self.assertTrue(
                (run / "artifacts/overlays/laid_lines_overlay.png").is_file()
            )
            self.assertTrue(
                (run / "artifacts/overlays/laid_lines_overlay_bands.png").is_file()
            )
            english = run / "artifacts/reports/report_en.html"
            chinese = run / "artifacts/reports/report_zh.html"
            self.assertTrue(english.is_file())
            self.assertTrue(chinese.is_file())
            self.assertIn(
                f"Policy {stored.result['confidence']['policy_version']}",
                english.read_text(encoding="utf-8"),
            )
            self.assertEqual(
                set(stored.artifacts),
                {
                    "artifacts/overlays/laid_lines_overlay.png",
                    "artifacts/overlays/laid_lines_overlay_bands.png",
                    "artifacts/reports/report_en.html",
                    "artifacts/reports/report_zh.html",
                },
            )

    def test_report_failure_does_not_publish_partial_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "laid-lines.png"
            config_path = root / "config.yaml"
            runs_root = root / "runs"
            self.assertTrue(cv2.imwrite(str(image_path), _laid_lines()))
            config_path.write_text(
                "\n".join(
                    (
                        "data_serial: report-failure",
                        "subtract_background: false",
                        "fov_width_cm: 1.28",
                        "period_range_cm: [0.12, 0.24]",
                        "line_dir_deg: 90.0",
                        "wire_is_darker: true",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with patch(
                "paperdrm.artifacts.render_bilingual_reports",
                side_effect=RuntimeError("simulated report failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated report failure"):
                    v2_main(
                        [
                            "--config",
                            str(config_path),
                            "--image",
                            str(image_path),
                            "--run-id",
                            "run-001",
                            "--runs-root",
                            str(runs_root),
                        ]
                    )

            self.assertFalse(
                (runs_root / "report-failure" / "run-001").exists()
            )

    def test_v2_cli_exposes_only_native_drp_tracks(self):
        parser = build_v2_parser()
        args = parser.parse_args(["--run-id", "run-001"])
        self.assertEqual(args.track, "multi_phi")
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    ["--run-id", "run-001", "--track", "legacy"]
                )


if __name__ == "__main__":
    unittest.main()
