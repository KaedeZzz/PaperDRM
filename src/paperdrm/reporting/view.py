"""Build the presentation values used by the bilingual HTML report."""

from __future__ import annotations

from datetime import date
from math import isfinite
from typing import Any


def _number(value: object, default: float = float("nan")) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if isfinite(number) else default


def _count(value: object) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _confidence_labels(confidence: dict[str, Any]) -> tuple[str, str]:
    disposition = confidence.get("disposition")
    level = confidence.get("level")
    reason = confidence.get("primary_reason")
    if reason == "period_search_boundary":
        return "Search boundary hit", "命中搜索边界"
    if reason == "polarity_contradiction":
        return "Contradictory polarity", "极性矛盾"
    if disposition == "accepted" and level == "high":
        return "High", "高"
    if disposition == "accepted" and level == "moderate":
        return "Moderate", "中"
    if disposition == "review_required":
        return "Low", "低"
    if disposition == "insufficient_evidence":
        return "Insufficient evidence", "证据不足"
    if disposition == "rejected":
        return "Rejected", "已拒绝"
    return "Unknown", "未知"


def _stability_labels(split_half: dict[str, Any] | None) -> tuple[str, str]:
    if split_half is None:
        return "Unavailable", "不可用"
    difference = _number(split_half.get("period_difference_std_px"))
    if not isfinite(difference):
        return "Unavailable", "不可用"
    if difference == 0.0:
        return (
            "Perfect (no variation across all splits)",
            "完全一致（所有分组结果相同）",
        )
    if difference < 0.5:
        return f"Excellent (±{difference:.3f} px)", f"优秀（±{difference:.3f} 像素）"
    if difference < 1.5:
        return f"Good (±{difference:.3f} px)", f"良好（±{difference:.3f} 像素）"
    return f"Fair (±{difference:.3f} px)", f"一般（±{difference:.3f} 像素）"


def _historical_context(lines_per_cm: float) -> tuple[str, str]:
    if not isfinite(lines_per_cm):
        return "not available", "暂无数据"
    if lines_per_cm < 8:
        return (
            "below the typical historical range (8–14 lines/cm)",
            "低于历史常见范围（8–14 条/cm）",
        )
    if lines_per_cm <= 14:
        return (
            "within the typical historical range (8–14 lines/cm)",
            "处于历史常见范围（8–14 条/cm）之内",
        )
    return (
        "above the typical historical range (8–14 lines/cm)",
        "高于历史常见范围（8–14 条/cm）",
    )


def report_values_from_v2(
    result: dict[str, Any],
    *,
    serial: str | None = None,
    technical_location: str | None = None,
) -> dict[str, Any]:
    """Map canonical V2 evidence and policy into stable report values."""

    measurement = result.get("measurement") or {}
    diagnostics = result.get("diagnostics") or {}
    evaluation = result.get("evaluation") or {}
    interval = evaluation.get("interval") or {}
    fit = evaluation.get("fit") or {}
    contrast = evaluation.get("contrast") or {}
    split_half = evaluation.get("split_half")
    wire_width = result.get("wire_width") or {}
    confidence = result.get("confidence") or {}

    period_px = _number(measurement.get("period_px"))
    cm_per_px = _number(measurement.get("cm_per_px"))
    interval_cm = _number(measurement.get("interval_cm"))
    if not isfinite(interval_cm) and isfinite(period_px) and isfinite(cm_per_px):
        interval_cm = period_px * cm_per_px
    period_mm = interval_cm * 10.0
    lines_per_cm = _number(measurement.get("lines_per_cm"))
    if not isfinite(lines_per_cm) and interval_cm > 0:
        lines_per_cm = 1.0 / interval_cm

    local_gap_px = _number(interval.get("median_gap_px"))
    local_median_cm = local_gap_px * cm_per_px
    local_median_mm = local_median_cm * 10.0
    lines_per_cm_med = (
        1.0 / local_median_cm if local_median_cm > 0 else float("nan")
    )
    raw_iqr = interval.get("gap_iqr_px")
    if isinstance(raw_iqr, (list, tuple)) and len(raw_iqr) == 2:
        local_iqr_lo_mm = _number(raw_iqr[0]) * cm_per_px * 10.0
        local_iqr_hi_mm = _number(raw_iqr[1]) * cm_per_px * 10.0
    else:
        local_iqr_lo_mm = local_iqr_hi_mm = float("nan")
    local_error = _number(interval.get("gap_median_relative_error_vs_spectral"))

    difference = (
        _number(split_half.get("period_difference_std_px"))
        if isinstance(split_half, dict)
        else float("nan")
    )
    stability_en, stability_zh = _stability_labels(
        split_half if isinstance(split_half, dict) else None
    )
    context_en, context_zh = _historical_context(lines_per_cm)
    confidence_en, confidence_zh = _confidence_labels(confidence)

    disposition = str(confidence.get("disposition") or "unknown")
    reason = str(confidence.get("primary_reason") or "unknown")
    policy_version = str(confidence.get("policy_version") or "unknown")
    warnings = confidence.get("warnings")
    warning_codes = (
        [str(value) for value in warnings]
        if isinstance(warnings, (list, tuple))
        else []
    )

    return {
        "serial": serial or str(result.get("dataset_id") or "unknown"),
        "today": date.today().strftime("%Y-%m-%d"),
        "lines_per_cm": lines_per_cm,
        "lines_per_cm_med": lines_per_cm_med,
        "period_mm": period_mm,
        "local_median_mm": local_median_mm,
        "local_iqr_lo_mm": local_iqr_lo_mm,
        "local_iqr_hi_mm": local_iqr_hi_mm,
        "local_median_error_pct": local_error * 100.0,
        "n_gaps": _count(interval.get("n_gaps")),
        "n_peaks": _count(interval.get("n_peaks")),
        "fwhm_mm_median": _number(wire_width.get("median_fwhm_mm")),
        "fwhm_mm_ci_lo": float("nan"),
        "fwhm_mm_ci_hi": float("nan"),
        "n_segments": _count(wire_width.get("segment_count")),
        "seg_valid": _count(wire_width.get("segment_valid_count")),
        "n_phi": (
            _count(split_half.get("n_images"))
            if isinstance(split_half, dict)
            else _count(diagnostics.get("n_images"))
        ),
        "n_splits": (
            _count(split_half.get("n_splits"))
            if isinstance(split_half, dict)
            else 0
        ),
        "diff_std": difference,
        "agree_1px": (
            _number(split_half.get("agree_rate_within_1px")) * 100.0
            if isinstance(split_half, dict)
            else float("nan")
        ),
        "agree_05px": (
            _number(split_half.get("agree_rate_within_half_px")) * 100.0
            if isinstance(split_half, dict)
            else float("nan")
        ),
        "z": _number(
            contrast.get("contrast_z"),
            _number(diagnostics.get("self_contrast_z")),
        ),
        "contrast_rel": _number(contrast.get("contrast_relative")),
        "n_lines": _count(contrast.get("n_lines")),
        "r2_k4": _number(fit.get("r2_with_harmonics")),
        "fc": _number(fit.get("frequency_concentration")),
        "period_at_boundary": bool(
            diagnostics.get("period_at_search_boundary", False)
        ),
        "period_warning": diagnostics.get("period_warning"),
        "detect_confidence_en": confidence_en,
        "detect_confidence_zh": confidence_zh,
        "confidence_disposition": disposition,
        "confidence_reason": reason,
        "confidence_policy_version": policy_version,
        "confidence_warnings": warning_codes,
        "stability_en": stability_en,
        "stability_zh": stability_zh,
        "context_en": context_en,
        "context_zh": context_zh,
        "technical_location": technical_location or "stored V2 run",
    }
