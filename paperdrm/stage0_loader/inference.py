"""
Infer DRP acquisition parameters from a folder of images.

Filenames are expected to follow ``<phi>_<theta>.<ext>`` with integer
angles in degrees (e.g. ``0_15.jpg``, ``108_60.jpg``). Non-matching
files in the folder are skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from paperdrm.stage0_loader.settings import DRPConfig


# Captures "<phi>_<theta>.<ext>" with integer angles. Anything else is skipped.
_FNAME_RE = re.compile(r"^(?P<phi>\d+)_(?P<theta>\d+)\.[^.]+$")


@dataclass
class InferenceReport:
    folder: Path
    img_format: str
    matched: int
    skipped: list[str]
    phi_values: list[int]
    theta_values: list[int]
    ph_step: float
    th_step: float
    missing_pairs: list[tuple[int, int]] = field(default_factory=list)
    duplicate_pairs: list[tuple[int, int]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"folder:        {self.folder}",
            f"format:        *.{self.img_format}",
            f"matched files: {self.matched}",
            f"skipped files: {len(self.skipped)}"
            + (f"  e.g. {self.skipped[:3]}" if self.skipped else ""),
            f"phi:   {len(self.phi_values)} values, "
            f"[{self.phi_values[0]}, {self.phi_values[-1]}], step {self.ph_step:g}",
            f"theta: {len(self.theta_values)} values, "
            f"[{self.theta_values[0]}, {self.theta_values[-1]}], step {self.th_step:g}",
            f"missing (phi, theta) pairs:   {len(self.missing_pairs)}",
            f"duplicate (phi, theta) pairs: {len(self.duplicate_pairs)}",
        ]
        return "\n".join(lines)


def infer_drp_config_from_folder(
    folder: str | Path,
    img_format: str = "jpg",
    strict: bool = True,
) -> tuple[DRPConfig, InferenceReport]:
    """
    Infer a DRPConfig (th_min/max/num, ph_min/max/num) from filenames.

    :param folder:     directory containing ``<phi>_<theta>.<ext>`` files.
    :param img_format: file extension to glob (without the dot).
    :param strict:     when True, raise on missing or duplicate (phi, theta)
                       pairs and on non-uniform phi/theta spacing.
    :return: (cfg, report) — cfg is a validated DRPConfig with the six
             acquisition fields filled; report carries the parsed grid
             and any diagnostics.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    paths = sorted(folder.glob(f"*.{img_format}"))
    if not paths:
        raise ValueError(f"No *.{img_format} files in {folder}")

    pairs: list[tuple[int, int]] = []
    skipped: list[str] = []
    for p in paths:
        m = _FNAME_RE.match(p.name)
        if m is None:
            skipped.append(p.name)
            continue
        pairs.append((int(m.group("phi")), int(m.group("theta"))))

    if not pairs:
        raise ValueError(
            f"No filenames in {folder} matched <phi>_<theta>.{img_format}"
        )

    phi_values = sorted({phi for phi, _ in pairs})
    theta_values = sorted({theta for _, theta in pairs})

    seen: set[tuple[int, int]] = set()
    duplicates: list[tuple[int, int]] = []
    for pair in pairs:
        if pair in seen:
            duplicates.append(pair)
        else:
            seen.add(pair)

    expected = {(phi, th) for phi in phi_values for th in theta_values}
    missing = sorted(expected - seen)

    ph_step = _uniform_step(phi_values, "phi", strict=strict)
    th_step = _uniform_step(theta_values, "theta", strict=strict)

    if strict:
        if duplicates:
            raise ValueError(f"Duplicate (phi, theta) pairs: {duplicates}")
        if missing:
            raise ValueError(
                f"Missing (phi, theta) pairs ({len(missing)} of "
                f"{len(expected)}): first few = {missing[:5]}"
            )

    cfg = DRPConfig(
        th_min=theta_values[0],
        th_max=theta_values[-1],
        th_num=len(theta_values),
        ph_min=phi_values[0],
        ph_max=phi_values[-1],
        ph_num=len(phi_values),
    )
    cfg.validate()

    report = InferenceReport(
        folder=folder,
        img_format=img_format,
        matched=len(pairs),
        skipped=skipped,
        phi_values=phi_values,
        theta_values=theta_values,
        ph_step=ph_step,
        th_step=th_step,
        missing_pairs=missing,
        duplicate_pairs=duplicates,
    )
    return cfg, report


def verify_drp_match(expected: DRPConfig, inferred: DRPConfig, source: str | Path) -> None:
    """
    Raise ValueError if ``expected`` and ``inferred`` disagree on any of the
    six acquisition fields. ``source`` is the folder whose files were inferred
    from — used only for the error message.
    """
    fields = ("th_min", "th_max", "th_num", "ph_min", "ph_max", "ph_num")
    mismatches = [
        f"{f}: yaml={getattr(expected, f)} inferred={getattr(inferred, f)}"
        for f in fields
        if getattr(expected, f) != getattr(inferred, f)
    ]
    if mismatches:
        lines = "\n  ".join(mismatches)
        raise ValueError(
            f"DRP yaml fields disagree with files in {source}:\n  {lines}\n"
            "Fix the yaml, or omit the six acquisition fields to enable inference."
        )


def _uniform_step(values: list[int], label: str, strict: bool) -> float:
    if len(values) < 2:
        raise ValueError(f"Need at least 2 distinct {label} values, got {len(values)}.")
    diffs = np.diff(np.asarray(values, dtype=float))
    step = float(diffs[0])
    if not np.allclose(diffs, step):
        if strict:
            raise ValueError(
                f"{label} values are not uniformly spaced: diffs={diffs.tolist()}"
            )
        return float(np.median(diffs))
    return step
