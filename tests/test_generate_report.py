import tempfile
import unittest
from pathlib import Path

from paperdrm.config import PipelineConfig
from paperdrm.models import (
    ConfidenceAssessment,
    ConfidenceLevel,
    ConfidenceReason,
    DetectionDiagnostics,
    DetectorTrack,
    PipelineResult,
    ResultDisposition,
    SpacingMeasurement,
)
from paperdrm.persistence import RunStore
from paperdrm.reporting import render_bilingual_reports, report_values_from_v2
from scripts.generate_report import (
    build_html_en,
    generate_reports_from_run,
    interpret,
)


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

    def test_v2_report_uses_stored_policy_instead_of_reclassifying_z(self):
        result = PipelineResult(
            dataset_id="folio",
            track=DetectorTrack.SIMPLE,
            measurement=SpacingMeasurement.from_period(20.0, cm_per_px=0.005),
            diagnostics=DetectionDiagnostics(self_contrast_z=10.0),
            confidence=ConfidenceAssessment(
                disposition=ResultDisposition.REVIEW_REQUIRED,
                level=ConfidenceLevel.LOW,
                primary_reason=ConfidenceReason.WEAK_SELF_CONTRAST,
                warnings=(ConfidenceReason.FIT_PERIOD_DISAGREEMENT,),
                policy_version="v-test",
            ),
        )

        values = report_values_from_v2(result.to_dict())

        self.assertEqual(values["z"], 10.0)
        self.assertEqual(values["detect_confidence_en"], "Low")
        self.assertEqual(values["confidence_policy_version"], "v-test")
        english, chinese = render_bilingual_reports(values)
        self.assertIn(
            "Policy v-test · review_required · weak_self_contrast",
            english,
        )
        self.assertIn("warnings: fit_period_disagreement", english)
        self.assertIn(
            "策略 v-test · review_required · weak_self_contrast",
            chinese,
        )
        self.assertNotIn(">High<", english)

    def test_v2_renderer_escapes_text_and_embeds_overlay(self):
        values = report_values_from_v2({}, serial="<script>alert(1)</script>")
        values["confidence_warnings"] = ["<unsafe>"]
        values["technical_location"] = 'run/<bad>&"'

        english, chinese = render_bilingual_reports(values, b"png-bytes")

        for html in (english, chinese):
            self.assertNotIn("<script>alert(1)</script>", html)
            self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
            self.assertIn("&lt;unsafe&gt;", html)
            self.assertIn("data:image/png;base64,cG5nLWJ5dGVz", html)

    def test_v2_report_is_written_outside_and_does_not_mutate_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = PipelineConfig(
                dataset_id="folio",
                track=DetectorTrack.SIMPLE,
            )
            result = PipelineResult(
                dataset_id="folio",
                track=DetectorTrack.SIMPLE,
                measurement=SpacingMeasurement.from_period(
                    20.0,
                    cm_per_px=0.005,
                ),
                diagnostics=DetectionDiagnostics(self_contrast_z=3.5),
                confidence=ConfidenceAssessment(
                    disposition=ResultDisposition.ACCEPTED,
                    level=ConfidenceLevel.HIGH,
                    primary_reason=ConfidenceReason.STRONG_SELF_CONTRAST,
                    policy_version="v-test",
                ),
            )
            run = RunStore(root / "runs").save(
                result,
                config,
                run_id="run-001",
            )
            before = {
                path.relative_to(run): path.read_bytes()
                for path in run.rglob("*")
                if path.is_file()
            }

            english, chinese, values = generate_reports_from_run(
                run,
                root / "reports" / "folio-run-001",
            )

            self.assertTrue(english.is_file())
            self.assertTrue(chinese.is_file())
            self.assertIn("Policy v-test", english.read_text(encoding="utf-8"))
            self.assertEqual(values["detect_confidence_en"], "High")
            after = {
                path.relative_to(run): path.read_bytes()
                for path in run.rglob("*")
                if path.is_file()
            }
            self.assertEqual(after, before)

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                generate_reports_from_run(
                    run,
                    root / "reports" / "folio-run-001",
                )


if __name__ == "__main__":
    unittest.main()
