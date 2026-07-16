import unittest

from scripts.generate_report import build_html_en, interpret


def _report_data(z: float, *, boundary: bool = False) -> dict:
    return {
        "interval": {
            "physical": {
                "lines_per_cm_mean": 10.0,
                "lines_per_cm_median": 10.0,
                "mean_interval_cm": 0.1,
                "median_interval_cm": 0.1,
                "std_interval_cm": 0.01,
                "cm_per_px": 0.01,
                "spectral_interval_cm": 0.08,
                "spectral_lines_per_cm": 12.5,
                "gap_iqr_cm": [0.09, 0.11],
                "gap_median_relative_error_vs_spectral": 0.25,
            },
            "period_px_used": 8.0,
            "px": {"iqr": [9.0, 11.0]},
            "n_peaks": 11,
            "n_gaps": 10,
        },
        "wire_width": {
            "physical": {"fwhm_mm": {"median": 0.2, "ci_t": [0.1, 0.3]}},
            "n_segments": 4,
            "aggregate": {"fwhm_px": {"n_valid": 4}},
        },
        "split_half": {
            "n_images": 8,
            "n_splits": 20,
            "period_diff_std": 0.0,
            "agree_rate_within_1px": 1.0,
            "agree_rate_within_0p5px": 1.0,
        },
        "self_contrast": {
            "contrast_z": z,
            "contrast_rel": -0.12 if z < 0 else 0.12,
            "n_lines": 10,
        },
        "fit_quality": {
            "r2_with_harmonics": 0.4,
            "frequency_concentration": 0.5,
            "period_at_search_boundary": boundary,
            "period_warning": (
                "Detected period is pinned to the upper search boundary."
                if boundary else None
            ),
        },
    }


class GenerateReportTests(unittest.TestCase):
    def test_positive_z_can_be_high_confidence(self):
        values = interpret(_report_data(3.5), "positive")
        self.assertEqual(values["detect_confidence_en"], "High")
        self.assertEqual(values["period_mm"], 0.8)
        self.assertEqual(values["lines_per_cm"], 12.5)

    def test_negative_z_is_not_converted_to_positive_confidence(self):
        values = interpret(_report_data(-3.5), "negative")
        self.assertEqual(values["detect_confidence_en"], "Contradictory polarity")

        html = build_html_en(values, None)
        self.assertIn("z = -3.50", html)
        self.assertIn("opposite intensity polarity", html)
        self.assertNotIn("z = 3.50", html)

    def test_boundary_hit_overrides_positive_z_confidence(self):
        values = interpret(_report_data(5.0, boundary=True), "boundary")
        self.assertEqual(values["detect_confidence_en"], "Search boundary hit")

        html = build_html_en(values, None)
        self.assertIn("Invalid period search range", html)
        self.assertIn("not validated", html)


if __name__ == "__main__":
    unittest.main()
