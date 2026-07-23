import json
import unittest
from pathlib import Path

import numpy as np

from paperdrm.config import PipelineConfig
from paperdrm.detection import NativeDetectorBackend
from paperdrm.evaluation import EvaluationOptions, EvaluationService
from paperdrm.models import DetectorTrack
from paperdrm.models import ConfidenceLevel, ResultDisposition
from paperdrm.pipeline import (
    MultiPhiInput,
    Pipeline,
    PipelineRequest,
    SingleImageInput,
)


def _synthetic_laid_lines(
    *,
    period_px: float = 16.0,
    sigma_px: float = 1.8,
    shape: tuple[int, int] = (128, 128),
    noise_std: float = 0.025,
    seed: int = 0,
) -> np.ndarray:
    """Deterministic vertical dark-line comb used as an integration fixture."""

    height, width = shape
    x = np.arange(width, dtype=np.float64)
    distance = (x + period_px / 2.0) % period_px - period_px / 2.0
    profile = 1.0 - np.exp(-0.5 * (distance / sigma_px) ** 2)
    image = np.repeat(profile[None, :], height, axis=0)
    noise = np.random.default_rng(seed).normal(0.0, noise_std, image.shape)
    return (np.clip(image + noise, 0.0, 1.0) * 255.0).astype(np.uint8)


class NativeDetectorBackendTests(unittest.TestCase):
    @staticmethod
    def _backend() -> NativeDetectorBackend:
        return NativeDetectorBackend(
            EvaluationService(
                EvaluationOptions(wire_segments=4, split_half_splits=20)
            )
        )

    def test_single_image_runs_end_to_end_through_v2_boundary(self):
        image = _synthetic_laid_lines(seed=1)
        config = PipelineConfig(
            dataset_id="synthetic-single",
            track=DetectorTrack.SINGLE_IMAGE,
            image_path=Path("synthetic.pgm"),
            fov_width_cm=1.28,
            period_range_cm=(0.12, 0.24),
            line_direction_deg=90.0,
            wire_is_darker=True,
        )
        result = Pipeline(self._backend()).run(
            PipelineRequest(config=config, input_data=SingleImageInput(image))
        )

        self.assertAlmostEqual(result.measurement.period_px, 16.0, places=6)
        self.assertAlmostEqual(result.measurement.lines_per_cm, 6.25, places=6)
        self.assertFalse(result.diagnostics.period_at_search_boundary)
        self.assertIsNotNone(result.grid)
        self.assertGreater(len(result.grid.positions_px), 5)
        self.assertTrue(result.wire_width.experimental)
        self.assertEqual(result.wire_width.segment_valid_count, 4)
        self.assertIsNotNone(result.evaluation)
        self.assertEqual(result.evaluation.interval.median_gap_px, 16.0)
        self.assertTrue(result.evaluation.fit.agrees_with_dominant)
        self.assertGreater(result.evaluation.contrast.contrast_z, 0.0)
        self.assertFalse(result.diagnostics.polarity_contradiction)
        self.assertEqual(result.confidence.disposition, ResultDisposition.ACCEPTED)
        self.assertEqual(result.confidence.level, ConfidenceLevel.HIGH)
        self.assertEqual(result.to_dict()["confidence"]["level"], "high")
        self.assertEqual(result.provenance["period_range_px"], [12.0, 24.0])
        self.assertEqual(result.artifacts, {})
        json.dumps(result.to_dict(), allow_nan=False)
        result.to_json()

    def test_multi_phi_runs_end_to_end_through_v2_boundary(self):
        images = tuple(_synthetic_laid_lines(seed=seed) for seed in range(4))
        config = PipelineConfig(
            dataset_id="synthetic-multi",
            track=DetectorTrack.MULTI_PHI,
            fov_width_cm=1.28,
            period_range_cm=(0.12, 0.24),
            line_direction_deg=90.0,
            wire_is_darker=True,
        )
        result = Pipeline(self._backend()).run(
            PipelineRequest(config=config, input_data=MultiPhiInput(images))
        )

        self.assertAlmostEqual(result.measurement.period_px, 16.0, places=6)
        self.assertAlmostEqual(result.measurement.lines_per_cm, 6.25, places=6)
        self.assertEqual(result.diagnostics.n_images, 4)
        self.assertGreaterEqual(result.diagnostics.phase_resultant_length, 0.0)
        self.assertLessEqual(result.diagnostics.phase_resultant_length, 1.0)
        self.assertFalse(result.diagnostics.period_at_search_boundary)
        self.assertIsNotNone(result.evaluation.split_half)
        self.assertEqual(result.evaluation.split_half.n_splits, 20)
        self.assertEqual(result.evaluation.split_half.period_difference_std_px, 0.0)
        self.assertEqual(result.evaluation.split_half.agree_rate_within_1px, 1.0)
        self.assertEqual(result.confidence.disposition, ResultDisposition.ACCEPTED)

    def test_backend_rejects_input_type_mismatch(self):
        config = PipelineConfig(
            dataset_id="mismatch",
            track=DetectorTrack.MULTI_PHI,
        )
        with self.assertRaisesRegex(ValueError, "requires MultiPhiInput"):
            Pipeline(self._backend()).run(
                PipelineRequest(
                    config=config,
                    input_data=SingleImageInput(_synthetic_laid_lines()),
                )
            )

    def test_legacy_stays_behind_v1_compatibility_path(self):
        config = PipelineConfig(dataset_id="legacy", track=DetectorTrack.LEGACY)
        with self.assertRaisesRegex(NotImplementedError, "V1 compatibility"):
            Pipeline(self._backend()).run(PipelineRequest(config=config))


if __name__ == "__main__":
    unittest.main()
