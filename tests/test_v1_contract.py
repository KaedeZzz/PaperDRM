import json
import subprocess
import sys
import unittest
from dataclasses import fields
from pathlib import Path

from paperdrm.result_archive import (
    LEGACY_ARTIFACTS,
    MULTI_PHI_ARTIFACTS,
    SINGLE_IMAGE_ARTIFACTS,
)
from paperdrm.stage0_loader.settings import Settings


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "contracts/v1/contract.json").read_text())
BENCHMARK = json.loads((ROOT / "benchmarks/v1-manual-gt.json").read_text())


def _get_path(value, dotted_path):
    for part in dotted_path.split("."):
        value = value[part]
    return value


class V1ContractTests(unittest.TestCase):
    def test_cli_help_preserves_arguments_choices_and_defaults(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / CONTRACT["cli"]["entry_point"]), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        help_text = completed.stdout

        self.assertIn("--config CONFIG", help_text)
        self.assertIn("--image IMAGE", help_text)
        self.assertIn("--track {multi_phi,simple,legacy}", help_text)
        self.assertIn("default: exp_param.yaml", help_text)
        self.assertIn("default: multi_phi", help_text)

    def test_settings_accept_at_least_the_frozen_v1_fields(self):
        actual = {field.name for field in fields(Settings)}
        required = set(CONTRACT["configuration"]["accepted_settings_fields"])
        self.assertTrue(required <= actual, required - actual)

    def test_artifact_sets_match_the_archiver(self):
        artifacts = CONTRACT["artifacts"]
        self.assertEqual(artifacts["single_image"], list(SINGLE_IMAGE_ARTIFACTS))
        self.assertEqual(artifacts["simple"], list(SINGLE_IMAGE_ARTIFACTS))
        self.assertEqual(artifacts["multi_phi"], list(MULTI_PHI_ARTIFACTS))
        self.assertEqual(artifacts["legacy"], list(LEGACY_ARTIFACTS))

    def test_nine_folio_results_satisfy_minimum_json_contract(self):
        required_paths = CONTRACT["result_json_required_paths"]
        for item in BENCHMARK["datasets"]:
            result_dir = ROOT / "results" / item["serial"]
            for filename, paths in required_paths.items():
                with self.subTest(serial=item["serial"], filename=filename):
                    value = json.loads((result_dir / filename).read_text())
                    for path in paths:
                        _get_path(value, path)

    def test_benchmark_matches_preserved_manual_gt_and_result_files(self):
        threshold = BENCHMARK["acceptance_threshold_abs_error_pct"]
        status_counts = {"within_threshold": 0, "known_failure": 0}

        for expected in BENCHMARK["datasets"]:
            result_dir = ROOT / "results" / expected["serial"]
            manual = json.loads((result_dir / "manual_gt.json").read_text())
            interval = json.loads(
                (result_dir / "interval_distribution.json").read_text()
            )
            physical = interval["physical"]
            measured = 1.0 / (
                interval["period_px_used"] * physical["cm_per_px"]
            )
            error_pct = (
                (measured - manual["lpc_mean"]) / manual["lpc_mean"] * 100.0
            )

            with self.subTest(serial=expected["serial"]):
                self.assertAlmostEqual(
                    manual["lpc_mean"], expected["manual_lines_per_cm"], places=6
                )
                self.assertAlmostEqual(
                    measured, expected["pipeline_lines_per_cm"], places=6
                )
                self.assertAlmostEqual(
                    error_pct, expected["relative_error_pct"], places=5
                )
                if expected["status"] == "within_threshold":
                    self.assertLessEqual(abs(error_pct), threshold)
                else:
                    self.assertGreater(abs(error_pct), threshold)
                status_counts[expected["status"]] += 1

        self.assertEqual(
            status_counts["within_threshold"],
            BENCHMARK["summary"]["within_threshold"],
        )
        self.assertEqual(
            status_counts["known_failure"], BENCHMARK["summary"]["known_failures"]
        )


if __name__ == "__main__":
    unittest.main()
