"""Command-line contract for PaperDRM.

Phase 1 centralises argument parsing here while ``main.py`` remains the V1
pipeline compatibility entry point. Moving execution behind this boundary is a
later migration step; keeping parsing separate already prevents CLI drift.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from paperdrm.models import DetectorTrack


DRP_TRACKS = (
    DetectorTrack.MULTI_PHI.value,
    DetectorTrack.SIMPLE.value,
    DetectorTrack.LEGACY.value,
)


def build_parser(
    *, default_track: str | DetectorTrack = DetectorTrack.MULTI_PHI
) -> argparse.ArgumentParser:
    """Build the frozen V1-compatible argument parser."""

    default = DetectorTrack(default_track).value
    if default not in DRP_TRACKS:
        raise ValueError(f"default_track must be one of {DRP_TRACKS}, got {default!r}")

    parser = argparse.ArgumentParser(description="PaperDRM pipeline entry point.")
    parser.add_argument(
        "--config",
        default="exp_param.yaml",
        help="Path to the settings yaml (default: exp_param.yaml).",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to a single image for the single_image track. "
        "Overrides image_path in the yaml.",
    )
    parser.add_argument(
        "--track",
        choices=DRP_TRACKS,
        default=default,
        help=f"DRP detector route (default: {default}).",
    )
    return parser


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    default_track: str | DetectorTrack = DetectorTrack.MULTI_PHI,
) -> argparse.Namespace:
    """Parse CLI arguments without starting a pipeline run."""

    return build_parser(default_track=default_track).parse_args(argv)
