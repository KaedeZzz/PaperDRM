"""Evaluate verified immutable V2 runs against the frozen manual-GT benchmark."""

from __future__ import annotations

import json
import re
from math import isfinite
from pathlib import Path
from typing import Any

from paperdrm.persistence import load_run


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_BASELINE_STATUSES = frozenset({"within_threshold", "known_failure"})
_DISPOSITIONS = frozenset(
    {"accepted", "review_required", "rejected", "insufficient_evidence"}
)


def _identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field} is not a safe identifier: {value!r}")
    return value


def _positive_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive finite number")
    number = float(value)
    if not isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be a positive finite number")
    return number


def load_benchmark(path: str | Path) -> dict[str, Any]:
    """Load and validate the compact manual-GT benchmark definition."""

    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("benchmark root must be an object")
    if value.get("benchmark_version") != 1:
        raise ValueError("unsupported benchmark_version")
    threshold = _positive_number(
        value.get("acceptance_threshold_abs_error_pct"),
        field="acceptance_threshold_abs_error_pct",
    )
    datasets = value.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("benchmark datasets must be a non-empty list")

    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, item in enumerate(datasets):
        if not isinstance(item, dict):
            raise ValueError(f"datasets[{index}] must be an object")
        serial = _identifier(item.get("serial"), field=f"datasets[{index}].serial")
        if serial in seen:
            raise ValueError(f"duplicate benchmark dataset: {serial}")
        seen.add(serial)
        status = item.get("status")
        if status not in _BASELINE_STATUSES:
            raise ValueError(f"invalid baseline status for {serial}: {status!r}")
        validated.append(
            {
                "serial": serial,
                "manual_lines_per_cm": _positive_number(
                    item.get("manual_lines_per_cm"),
                    field=f"{serial}.manual_lines_per_cm",
                ),
                "baseline_lines_per_cm": _positive_number(
                    item.get("pipeline_lines_per_cm"),
                    field=f"{serial}.pipeline_lines_per_cm",
                ),
                "baseline_status": status,
            }
        )

    return {
        "benchmark_version": 1,
        "baseline_ref": str(value.get("baseline_ref") or "unknown"),
        "baseline_commit": str(value.get("baseline_commit") or "unknown"),
        "acceptance_threshold_abs_error_pct": threshold,
        "datasets": validated,
    }


def evaluate_v2_runs(
    benchmark_path: str | Path,
    runs_root: str | Path,
    *,
    run_id: str,
) -> dict[str, Any]:
    """Compare one verified V2 run per benchmark dataset with manual GT.

    A previously accurate dataset must remain within the declared error
    threshold. The frozen known failure may either improve or remain outside
    the threshold only when the stored policy does not mark it ``accepted``.
    """

    run_id = _identifier(run_id, field="run_id")
    benchmark = load_benchmark(benchmark_path)
    threshold = benchmark["acceptance_threshold_abs_error_pct"]
    root = Path(runs_root)
    rows: list[dict[str, Any]] = []

    for case in benchmark["datasets"]:
        serial = case["serial"]
        run_directory = root / serial / run_id
        if not run_directory.exists():
            raise FileNotFoundError(
                f"missing V2 benchmark run: {serial}/{run_id}"
            )
        stored = load_run(run_directory)
        measurement = stored.result.get("measurement")
        if not isinstance(measurement, dict):
            raise ValueError(f"V2 result has no measurement object: {serial}")
        measured = _positive_number(
            measurement.get("lines_per_cm"),
            field=f"{serial}.measurement.lines_per_cm",
        )
        confidence = stored.result.get("confidence")
        if not isinstance(confidence, dict):
            raise ValueError(f"V2 result has no stored confidence policy: {serial}")
        disposition = confidence.get("disposition")
        if disposition not in _DISPOSITIONS:
            raise ValueError(f"invalid confidence disposition for {serial}")
        policy_version = confidence.get("policy_version")
        reason = confidence.get("primary_reason")
        if not isinstance(policy_version, str) or not policy_version:
            raise ValueError(f"invalid confidence policy version for {serial}")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"invalid confidence reason for {serial}")

        manual = case["manual_lines_per_cm"]
        error_pct = (measured - manual) / manual * 100.0
        within_threshold = abs(error_pct) <= threshold
        if within_threshold:
            outcome = "within_threshold"
            gate_pass = True
        elif case["baseline_status"] == "known_failure":
            if disposition == "accepted":
                outcome = "known_failure_accepted"
                gate_pass = False
            else:
                outcome = "known_failure_flagged"
                gate_pass = True
        else:
            outcome = "accuracy_regression"
            gate_pass = False

        rows.append(
            {
                **case,
                "v2_lines_per_cm": measured,
                "relative_error_pct": error_pct,
                "within_threshold": within_threshold,
                "policy_version": policy_version,
                "disposition": disposition,
                "primary_reason": reason,
                "outcome": outcome,
                "gate_pass": gate_pass,
            }
        )

    failures = [row["serial"] for row in rows if not row["gate_pass"]]
    return {
        "report_schema_version": 1,
        "benchmark_version": benchmark["benchmark_version"],
        "baseline_ref": benchmark["baseline_ref"],
        "baseline_commit": benchmark["baseline_commit"],
        "run_id": run_id,
        "acceptance_threshold_abs_error_pct": threshold,
        "datasets": rows,
        "summary": {
            "dataset_count": len(rows),
            "within_threshold": sum(row["within_threshold"] for row in rows),
            "known_failure_flagged": sum(
                row["outcome"] == "known_failure_flagged" for row in rows
            ),
            "accuracy_regressions": sum(
                row["outcome"] == "accuracy_regression" for row in rows
            ),
            "unsafe_known_failures": sum(
                row["outcome"] == "known_failure_accepted" for row in rows
            ),
            "gate_failures": failures,
            "gate_pass": not failures,
        },
    }
