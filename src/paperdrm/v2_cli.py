"""Dedicated command-line entry point for native immutable V2 runs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from paperdrm.models import DetectorTrack


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the native PaperDRM V2 pipeline into an immutable run."
    )
    parser.add_argument("--config", default="exp_param.yaml")
    parser.add_argument(
        "--image",
        help="Single image override; selects the single_image route.",
    )
    parser.add_argument(
        "--track",
        choices=(DetectorTrack.MULTI_PHI.value, DetectorTrack.SIMPLE.value),
        default=DetectorTrack.MULTI_PHI.value,
        help="Native DRP detector route when --image/image_path is absent.",
    )
    parser.add_argument("--run-id", required=True, help="Immutable run identifier.")
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="V2 runs root directory (default: runs).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from paperdrm.application import ApplicationRunner
    from paperdrm.artifacts import StandardArtifactBuilder
    from paperdrm.compat.v1 import load_v1_config

    config = load_v1_config(
        args.config,
        requested_track=args.track,
        image_override=args.image,
    )
    runner = ApplicationRunner.native(
        Path(args.runs_root),
        artifact_builder=StandardArtifactBuilder(),
    )
    completed = runner.run(config, run_id=args.run_id)
    confidence = completed.result.confidence
    print(f"Run published: {completed.run_directory}")
    print(
        f"Measurement: {completed.result.measurement.period_px:.3f} px"
        + (
            f" | {completed.result.measurement.lines_per_cm:.3f} lines/cm"
            if completed.result.measurement.lines_per_cm is not None
            else ""
        )
    )
    if confidence is not None:
        print(
            f"Policy {confidence.policy_version}: {confidence.disposition.value} "
            f"({confidence.primary_reason.value})"
        )
    return 0
