import unittest

from paperdrm.detection_diagnostics import period_boundary_diagnostic


class DetectionDiagnosticsTests(unittest.TestCase):
    def test_upper_period_boundary_is_flagged(self):
        result = period_boundary_diagnostic(
            peak_index=0,
            n_bins=100,
            period_range_px=(8.0, 50.2),
            detected_period_px=49.95,
        )

        self.assertTrue(result["period_at_search_boundary"])
        self.assertEqual(result["period_boundary_side"], "upper")
        self.assertIn("upper search boundary", result["period_warning"])

    def test_internal_peak_is_not_flagged(self):
        result = period_boundary_diagnostic(
            peak_index=10,
            n_bins=100,
            period_range_px=(8.0, 65.0),
            detected_period_px=55.35,
        )

        self.assertFalse(result["period_at_search_boundary"])
        self.assertIsNone(result["period_boundary_side"])
        self.assertIsNone(result["period_warning"])
