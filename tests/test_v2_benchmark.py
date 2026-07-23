import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from paperdrm.benchmark import evaluate_v2_runs, load_benchmark
from paperdrm.config import PipelineConfig
from paperdrm.models import (
    ConfidenceAssessment,
    ConfidenceLevel,
    ConfidenceReason,
    DetectorTrack,
    PipelineResult,
    ResultDisposition,
    SpacingMeasurement,
)
from paperdrm.persistence import RunStore
from scripts.benchmark_v2 import main as benchmark_main


ROOT = Path(__file__).resolve().parents[1]


class V2BenchmarkTests(unittest.TestCase):
    def test_frozen_nine_folio_benchmark_is_supported(self):
        benchmark = load_benchmark(ROOT / "benchmarks" / "v1-manual-gt.json")

        self.assertEqual(len(benchmark["datasets"]), 9)
        self.assertEqual(
            sum(
                item["baseline_status"] == "known_failure"
                for item in benchmark["datasets"]
            ),
            1,
        )

    def _write_benchmark(self, root: Path) -> Path:
        path = root / "benchmark.json"
        path.write_text(
            json.dumps(
                {
                    "benchmark_version": 1,
                    "baseline_ref": "baseline-v1",
                    "baseline_commit": "abc1234",
                    "acceptance_threshold_abs_error_pct": 10.0,
                    "datasets": [
                        {
                            "serial": "known-good",
                            "manual_lines_per_cm": 10.0,
                            "pipeline_lines_per_cm": 9.5,
                            "status": "within_threshold",
                        },
                        {
                            "serial": "known-failure",
                            "manual_lines_per_cm": 10.0,
                            "pipeline_lines_per_cm": 5.0,
                            "status": "known_failure",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    def _save_run(
        self,
        runs: Path,
        serial: str,
        lines_per_cm: float,
        disposition: ResultDisposition,
    ) -> None:
        if disposition is ResultDisposition.ACCEPTED:
            level = ConfidenceLevel.HIGH
            reason = ConfidenceReason.STRONG_SELF_CONTRAST
        else:
            level = ConfidenceLevel.LOW
            reason = ConfidenceReason.WEAK_SELF_CONTRAST
        result = PipelineResult(
            dataset_id=serial,
            track=DetectorTrack.SIMPLE,
            measurement=SpacingMeasurement.from_period(
                10.0,
                cm_per_px=1.0 / (10.0 * lines_per_cm),
            ),
            confidence=ConfidenceAssessment(
                disposition=disposition,
                level=level,
                primary_reason=reason,
                policy_version="v-test",
            ),
        )
        RunStore(runs).save(
            result,
            PipelineConfig(dataset_id=serial, track=DetectorTrack.SIMPLE),
            run_id="benchmark-001",
        )

    def test_known_failure_must_improve_or_be_flagged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = self._write_benchmark(root)
            runs = root / "runs"
            self._save_run(runs, "known-good", 10.0, ResultDisposition.ACCEPTED)
            self._save_run(
                runs,
                "known-failure",
                5.0,
                ResultDisposition.REVIEW_REQUIRED,
            )

            report = evaluate_v2_runs(
                benchmark,
                runs,
                run_id="benchmark-001",
            )

            self.assertTrue(report["summary"]["gate_pass"])
            self.assertEqual(report["summary"]["within_threshold"], 1)
            self.assertEqual(report["summary"]["known_failure_flagged"], 1)
            self.assertEqual(
                report["datasets"][1]["outcome"],
                "known_failure_flagged",
            )

    def test_accepted_known_failure_is_a_gate_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = self._write_benchmark(root)
            runs = root / "runs"
            self._save_run(runs, "known-good", 10.0, ResultDisposition.ACCEPTED)
            self._save_run(
                runs,
                "known-failure",
                5.0,
                ResultDisposition.ACCEPTED,
            )

            report = evaluate_v2_runs(
                benchmark,
                runs,
                run_id="benchmark-001",
            )

            self.assertFalse(report["summary"]["gate_pass"])
            self.assertEqual(report["summary"]["unsafe_known_failures"], 1)
            self.assertEqual(
                report["summary"]["gate_failures"],
                ["known-failure"],
            )

    def test_missing_run_fails_instead_of_being_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = self._write_benchmark(root)
            self._save_run(
                root / "runs",
                "known-good",
                10.0,
                ResultDisposition.ACCEPTED,
            )

            with self.assertRaises(FileNotFoundError):
                evaluate_v2_runs(
                    benchmark,
                    root / "runs",
                    run_id="benchmark-001",
                )

    def test_integrity_invalid_run_fails_instead_of_being_scored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = self._write_benchmark(root)
            runs = root / "runs"
            self._save_run(runs, "known-good", 10.0, ResultDisposition.ACCEPTED)
            self._save_run(
                runs,
                "known-failure",
                5.0,
                ResultDisposition.REVIEW_REQUIRED,
            )
            result_path = runs / "known-good" / "benchmark-001" / "result.json"
            result_path.write_text("{}\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                evaluate_v2_runs(
                    benchmark,
                    runs,
                    run_id="benchmark-001",
                )

    def test_cli_writes_new_report_and_never_overwrites_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = self._write_benchmark(root)
            runs = root / "runs"
            output = root / "reports" / "benchmark.json"
            self._save_run(runs, "known-good", 10.0, ResultDisposition.ACCEPTED)
            self._save_run(
                runs,
                "known-failure",
                5.0,
                ResultDisposition.REVIEW_REQUIRED,
            )
            args = [
                "--benchmark",
                str(benchmark),
                "--runs-root",
                str(runs),
                "--run-id",
                "benchmark-001",
                "--output",
                str(output),
            ]

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(benchmark_main(args), 0)
            original = output.read_bytes()
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(benchmark_main(args), 2)
            self.assertEqual(output.read_bytes(), original)

    def test_benchmark_rejects_duplicate_or_unsafe_dataset_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = self._write_benchmark(root)
            value = json.loads(benchmark.read_text(encoding="utf-8"))
            value["datasets"][1]["serial"] = "../known-good"
            benchmark.write_text(json.dumps(value), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "safe identifier"):
                load_benchmark(benchmark)


if __name__ == "__main__":
    unittest.main()
