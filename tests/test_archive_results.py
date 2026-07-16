import tempfile
import unittest
from pathlib import Path

from result_archive import archive_results


class ArchiveResultsTests(unittest.TestCase):
    def test_archives_only_current_run_and_removes_stale_managed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "results" / "dataset-b"
            result_dir.mkdir(parents=True)

            # Existing dataset state: one stale pipeline result plus user data.
            (result_dir / "split_half_stability.json").write_text('{"dataset": "a"}')
            (result_dir / "manual_gt.json").write_text('{"keep": true}')

            # Repo-root state: current outputs plus an unrelated stale output.
            (root / "interval_distribution.json").write_text('{"dataset": "b"}')
            (root / "laid_lines_overlay.png").write_bytes(b"current")
            (root / "self_contrast.json").write_text('{"dataset": "a"}')
            config = root / "dataset-b.yaml"
            config.write_text("data_serial: dataset-b\n")

            archive_results(
                None,
                str(config),
                serial="dataset-b",
                artifacts=("interval_distribution.json", "laid_lines_overlay.png"),
                root=root,
                generate_reports=False,
            )

            self.assertEqual(
                (result_dir / "interval_distribution.json").read_text(),
                '{"dataset": "b"}',
            )
            self.assertEqual(
                (result_dir / "laid_lines_overlay.png").read_bytes(),
                b"current",
            )
            self.assertFalse((result_dir / "split_half_stability.json").exists())
            self.assertFalse((result_dir / "self_contrast.json").exists())
            self.assertTrue((result_dir / "manual_gt.json").exists())
            self.assertTrue((result_dir / "dataset-b.yaml").exists())

    def test_missing_declared_artifact_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "dataset.yaml"
            config.write_text("data_serial: dataset\n")

            with self.assertRaisesRegex(FileNotFoundError, "was not created"):
                archive_results(
                    None,
                    str(config),
                    serial="dataset",
                    artifacts=("missing.json",),
                    root=root,
                    generate_reports=False,
                )


if __name__ == "__main__":
    unittest.main()
