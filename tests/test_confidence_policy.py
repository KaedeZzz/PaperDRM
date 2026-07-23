import unittest

from paperdrm.confidence import ConfidencePolicyV1
from paperdrm.models import (
    ConfidenceAssessment,
    ConfidenceLevel,
    ConfidenceReason,
    ContrastEvaluation,
    DetectionDiagnostics,
    EvaluationSummary,
    FitEvaluation,
    IntervalEvaluation,
    ResultDisposition,
    SplitHalfEvaluation,
)


def _evaluation(
    *,
    fit_agrees: bool = True,
    gap_error: float = 0.0,
    split_std: float | None = 0.0,
) -> EvaluationSummary:
    return EvaluationSummary(
        interval=IntervalEvaluation(
            n_peaks=10,
            n_gaps=9,
            median_gap_px=20.0,
            gap_iqr_px=(19.0, 21.0),
            gap_median_relative_error_vs_spectral=gap_error,
        ),
        fit=FitEvaluation(
            r2_fundamental=0.3,
            r2_with_harmonics=0.5,
            r2_gaussian_comb=0.4,
            frequency_concentration=0.6,
            best_period_by_r2_px=20.0,
            best_r2=0.5,
            agrees_with_dominant=fit_agrees,
        ),
        contrast=ContrastEvaluation(
            n_lines=10,
            contrast_relative=0.1,
            contrast_z=3.0,
        ),
        split_half=(
            SplitHalfEvaluation(
                n_images=8,
                n_splits=20,
                period_difference_std_px=split_std,
                agree_rate_within_1px=1.0,
                agree_rate_within_half_px=1.0,
            )
            if split_std is not None else None
        ),
    )


class ConfidencePolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = ConfidencePolicyV1()

    def test_boundary_hit_has_highest_precedence(self):
        result = self.policy.assess(
            DetectionDiagnostics(
                period_at_search_boundary=True,
                period_boundary_side="upper",
                self_contrast_z=-5.0,
            ),
            _evaluation(),
        )
        self.assertEqual(result.disposition, ResultDisposition.REJECTED)
        self.assertEqual(
            result.primary_reason, ConfidenceReason.PERIOD_SEARCH_BOUNDARY
        )

    def test_strong_negative_contrast_is_rejected_as_polarity_contradiction(self):
        result = self.policy.assess(
            DetectionDiagnostics(self_contrast_z=-2.0), _evaluation()
        )
        self.assertEqual(result.disposition, ResultDisposition.REJECTED)
        self.assertEqual(
            result.primary_reason, ConfidenceReason.POLARITY_CONTRADICTION
        )

    def test_missing_contrast_is_insufficient_evidence(self):
        result = self.policy.assess(DetectionDiagnostics(), _evaluation())
        self.assertEqual(
            result.disposition, ResultDisposition.INSUFFICIENT_EVIDENCE
        )
        self.assertEqual(result.level, ConfidenceLevel.UNKNOWN)

    def test_positive_thresholds_preserve_v1_report_semantics(self):
        high = self.policy.assess(
            DetectionDiagnostics(self_contrast_z=3.0), _evaluation()
        )
        moderate = self.policy.assess(
            DetectionDiagnostics(self_contrast_z=2.0), _evaluation()
        )
        low = self.policy.assess(
            DetectionDiagnostics(self_contrast_z=1.99), _evaluation()
        )

        self.assertEqual(
            (high.disposition, high.level),
            (ResultDisposition.ACCEPTED, ConfidenceLevel.HIGH),
        )
        self.assertEqual(
            (moderate.disposition, moderate.level),
            (ResultDisposition.ACCEPTED, ConfidenceLevel.MODERATE),
        )
        self.assertEqual(
            (low.disposition, low.level),
            (ResultDisposition.REVIEW_REQUIRED, ConfidenceLevel.LOW),
        )

    def test_uncalibrated_diagnostics_are_warnings_not_hard_rejections(self):
        result = self.policy.assess(
            DetectionDiagnostics(self_contrast_z=4.0),
            _evaluation(fit_agrees=False, gap_error=0.2, split_std=1.5),
        )
        self.assertEqual(result.disposition, ResultDisposition.ACCEPTED)
        self.assertEqual(result.level, ConfidenceLevel.HIGH)
        self.assertEqual(
            result.warnings,
            (
                ConfidenceReason.FIT_PERIOD_DISAGREEMENT,
                ConfidenceReason.LOCAL_GAP_DISAGREEMENT,
                ConfidenceReason.SPLIT_HALF_INSTABILITY,
            ),
        )

    def test_warning_thresholds_are_strict_and_versioned(self):
        result = self.policy.assess(
            DetectionDiagnostics(self_contrast_z=3.0),
            _evaluation(gap_error=0.15, split_std=1.49),
        )
        self.assertEqual(result.warnings, ())
        self.assertEqual(result.policy_version, "v1")

    def test_model_rejects_inconsistent_disposition_and_level(self):
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            ConfidenceAssessment(
                disposition=ResultDisposition.ACCEPTED,
                level=ConfidenceLevel.LOW,
                primary_reason=ConfidenceReason.WEAK_SELF_CONTRAST,
            )


if __name__ == "__main__":
    unittest.main()
