import unittest

import numpy as np

from paperdrm.stage5_evaluation.interval_distribution import (
    gap_distribution_from_signal,
)


class IntervalDistributionTests(unittest.TestCase):
    def test_physical_summary_keeps_spectral_period_separate_from_local_gaps(self):
        x = np.arange(200, dtype=np.float32)
        signal = np.cos(2.0 * np.pi * x / 10.0)

        result = gap_distribution_from_signal(
            signal,
            10.0,
            fov_width_cm=20.0,
            image_width_px=200,
        )

        physical = result["physical"]
        self.assertAlmostEqual(physical["spectral_interval_cm"], 1.0)
        self.assertAlmostEqual(physical["spectral_lines_per_cm"], 1.0)
        self.assertIn("median_interval_cm", physical)
        self.assertIn("gap_median_relative_error_vs_spectral", physical)


if __name__ == "__main__":
    unittest.main()
