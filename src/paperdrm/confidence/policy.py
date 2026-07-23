"""Conservative, explainable confidence policy for V2 results."""

from __future__ import annotations

from paperdrm.models import (
    ConfidenceAssessment,
    ConfidenceLevel,
    ConfidenceReason,
    DetectionDiagnostics,
    EvaluationSummary,
    ResultDisposition,
)


class ConfidencePolicyV1:
    """Preserve V1 thresholds while making precedence explicit."""

    version = "v1"
    contradictory_z_max = -2.0
    moderate_z_min = 2.0
    high_z_min = 3.0
    local_gap_warning_relative_error = 0.15
    split_half_warning_std_px = 1.5

    def assess(
        self,
        diagnostics: DetectionDiagnostics,
        evaluation: EvaluationSummary | None,
    ) -> ConfidenceAssessment:
        warnings = self._warnings(evaluation)
        z = diagnostics.self_contrast_z

        if diagnostics.period_at_search_boundary:
            return ConfidenceAssessment(
                disposition=ResultDisposition.REJECTED,
                level=ConfidenceLevel.UNKNOWN,
                primary_reason=ConfidenceReason.PERIOD_SEARCH_BOUNDARY,
                warnings=warnings,
                policy_version=self.version,
            )
        if z is not None and z <= self.contradictory_z_max:
            return ConfidenceAssessment(
                disposition=ResultDisposition.REJECTED,
                level=ConfidenceLevel.UNKNOWN,
                primary_reason=ConfidenceReason.POLARITY_CONTRADICTION,
                warnings=warnings,
                policy_version=self.version,
            )
        if z is None:
            return ConfidenceAssessment(
                disposition=ResultDisposition.INSUFFICIENT_EVIDENCE,
                level=ConfidenceLevel.UNKNOWN,
                primary_reason=ConfidenceReason.MISSING_SELF_CONTRAST,
                warnings=warnings,
                policy_version=self.version,
            )
        if z >= self.high_z_min:
            return ConfidenceAssessment(
                disposition=ResultDisposition.ACCEPTED,
                level=ConfidenceLevel.HIGH,
                primary_reason=ConfidenceReason.STRONG_SELF_CONTRAST,
                warnings=warnings,
                policy_version=self.version,
            )
        if z >= self.moderate_z_min:
            return ConfidenceAssessment(
                disposition=ResultDisposition.ACCEPTED,
                level=ConfidenceLevel.MODERATE,
                primary_reason=ConfidenceReason.MODERATE_SELF_CONTRAST,
                warnings=warnings,
                policy_version=self.version,
            )
        return ConfidenceAssessment(
            disposition=ResultDisposition.REVIEW_REQUIRED,
            level=ConfidenceLevel.LOW,
            primary_reason=ConfidenceReason.WEAK_SELF_CONTRAST,
            warnings=warnings,
            policy_version=self.version,
        )

    def _warnings(
        self, evaluation: EvaluationSummary | None
    ) -> tuple[ConfidenceReason, ...]:
        if evaluation is None:
            return ()

        warnings: list[ConfidenceReason] = []
        if not evaluation.fit.agrees_with_dominant:
            warnings.append(ConfidenceReason.FIT_PERIOD_DISAGREEMENT)

        gap_error = evaluation.interval.gap_median_relative_error_vs_spectral
        if (
            gap_error is not None
            and gap_error > self.local_gap_warning_relative_error
        ):
            warnings.append(ConfidenceReason.LOCAL_GAP_DISAGREEMENT)

        split_half = evaluation.split_half
        if (
            split_half is not None
            and split_half.period_difference_std_px
            >= self.split_half_warning_std_px
        ):
            warnings.append(ConfidenceReason.SPLIT_HALF_INSTABILITY)
        return tuple(warnings)
