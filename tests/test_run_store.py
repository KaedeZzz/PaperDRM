import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


class RunStoreTests(unittest.TestCase):
    def _config(self, dataset_id="folio"):
        return PipelineConfig(dataset_id=dataset_id, track=DetectorTrack.SIMPLE)

    def _result(self, dataset_id="folio", *, provenance=None):
        return PipelineResult(
            dataset_id=dataset_id,
            track=DetectorTrack.SIMPLE,
            measurement=SpacingMeasurement(period_px=20.0),
            confidence=ConfidenceAssessment(
                disposition=ResultDisposition.ACCEPTED,
                level=ConfidenceLevel.HIGH,
                primary_reason=ConfidenceReason.STRONG_SELF_CONTRAST,
                policy_version="v1",
            ),
            provenance=provenance or {},
        )

    def test_publishes_expected_layout_manifest_and_artifact_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "overlay.png"
            source.write_bytes(b"example-overlay")

            run = RunStore(root / "runs").save(
                self._result(),
                self._config(),
                run_id="run-001",
                inputs=("input-a.tif",),
                artifacts={"overlays/grid.png": source},
            )

            self.assertEqual(run, root / "runs" / "folio" / "run-001")
            self.assertTrue((run / "result.json").is_file())
            self.assertTrue((run / "manifest.json").is_file())
            for group in ("diagnostics", "overlays", "reports"):
                self.assertTrue((run / "artifacts" / group).is_dir())
            self.assertEqual(
                (run / "artifacts/overlays/grid.png").read_bytes(),
                b"example-overlay",
            )

            result = json.loads((run / "result.json").read_text(encoding="utf-8"))
            manifest = json.loads(
                (run / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["schema_version"], 2)
            self.assertEqual(manifest["manifest_schema_version"], 1)
            self.assertEqual(manifest["result_schema_version"], 2)
            self.assertEqual(manifest["policy_version"], "v1")
            self.assertEqual(manifest["inputs"], ["input-a.tif"])
            self.assertEqual(
                manifest["artifacts"],
                [
                    {
                        "path": "artifacts/overlays/grid.png",
                        "sha256": hashlib.sha256(b"example-overlay").hexdigest(),
                        "size_bytes": len(b"example-overlay"),
                    }
                ],
            )
            self.assertFalse(
                any(
                    path.name.startswith(".run-001")
                    for path in run.parent.iterdir()
                )
            )

    def test_existing_run_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = RunStore(Path(temporary) / "runs")
            first = store.save(self._result(), self._config(), run_id="same-run")
            original = (first / "result.json").read_bytes()

            with self.assertRaisesRegex(FileExistsError, "will not be overwritten"):
                store.save(self._result(), self._config(), run_id="same-run")

            self.assertEqual((first / "result.json").read_bytes(), original)

    def test_rejects_unsafe_dataset_run_and_artifact_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = RunStore(Path(temporary) / "runs")
            with self.assertRaisesRegex(ValueError, "dataset_id"):
                store.save(
                    self._result("../folio"),
                    self._config("../folio"),
                    run_id="run-001",
                )
            with self.assertRaisesRegex(ValueError, "run_id"):
                store.save(self._result(), self._config(), run_id="../run")

            source = Path(temporary) / "artifact.txt"
            source.write_text("artifact", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact destination"):
                store.save(
                    self._result(),
                    self._config(),
                    run_id="run-001",
                    artifacts={"overlays/../escape.txt": source},
                )

            self.assertFalse((Path(temporary) / "runs").exists())

    def test_strict_json_failure_creates_no_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            with self.assertRaises(ValueError):
                RunStore(root).save(
                    self._result(provenance={"invalid": float("nan")}),
                    self._config(),
                    run_id="run-001",
                )
            self.assertFalse(root.exists())

    def test_copy_failure_removes_private_directory_and_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "artifact.txt"
            source.write_text("artifact", encoding="utf-8")
            runs = root / "runs"

            with patch(
                "paperdrm.persistence.store.shutil.copy2",
                side_effect=OSError("simulated copy failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated copy failure"):
                    RunStore(runs).save(
                        self._result(),
                        self._config(),
                        run_id="run-001",
                        artifacts={"reports/report.html": source},
                    )

            dataset_directory = runs / "folio"
            self.assertTrue(dataset_directory.is_dir())
            self.assertEqual(list(dataset_directory.iterdir()), [])

    def test_result_and_config_identity_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = RunStore(Path(temporary) / "runs")
            with self.assertRaisesRegex(ValueError, "dataset_id values differ"):
                store.save(self._result("one"), self._config("two"), run_id="run")

    def test_competing_writer_lock_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset_directory = Path(temporary) / "runs" / "folio"
            dataset_directory.mkdir(parents=True)
            lock = dataset_directory / ".run-001.lock"
            lock.write_text("owned by another writer\n", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "already being written"):
                RunStore(Path(temporary) / "runs").save(
                    self._result(),
                    self._config(),
                    run_id="run-001",
                )

            self.assertEqual(
                lock.read_text(encoding="utf-8"),
                "owned by another writer\n",
            )

    def test_rejects_dataset_directory_symbolic_link(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside"
            outside.mkdir()
            runs = root / "runs"
            runs.mkdir()
            (runs / "folio").symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "symbolic link"):
                RunStore(runs).save(
                    self._result(),
                    self._config(),
                    run_id="run-001",
                )

            self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
