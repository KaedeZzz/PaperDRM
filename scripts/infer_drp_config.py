"""
Infer a DRPConfig from a data folder by parsing filenames, and compare
the result against an existing exp_param.yaml.

Usage:
    python scripts/infer_drp_config.py
    python scripts/infer_drp_config.py --folder data/raw --yaml exp_param.yaml --ext jpg
    python scripts/infer_drp_config.py --no-strict   # tolerate gaps in the grid

Exit code 1 if any acquisition field disagrees with the yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paperdrm.stage0_loader.inference import infer_drp_config_from_folder
from paperdrm.stage0_loader.settings import load_drp_config
ACQ_FIELDS = ("th_min", "th_max", "th_num", "ph_min", "ph_max", "ph_num")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--yaml", type=Path, default=REPO_ROOT / "exp_param.yaml")
    parser.add_argument("--ext", default="jpg")
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="tolerate missing/duplicate (phi, theta) pairs and non-uniform spacing",
    )
    args = parser.parse_args()

    cfg, report = infer_drp_config_from_folder(
        args.folder, img_format=args.ext, strict=not args.no_strict
    )

    print("== Inference report ==")
    print(report.summary())

    print("\n== Inferred DRPConfig ==")
    for key in ACQ_FIELDS:
        print(f"  {key} = {getattr(cfg, key)}")

    if not args.yaml.exists():
        print(f"\nNo yaml at {args.yaml}; nothing to compare.")
        return 0

    yaml_cfg = load_drp_config(args.yaml)
    print(f"\n== Comparison vs {args.yaml} ==")
    diffs: list[str] = []
    for key in ACQ_FIELDS:
        inferred = getattr(cfg, key)
        from_yaml = getattr(yaml_cfg, key)
        marker = " OK " if inferred == from_yaml else "DIFF"
        print(f"  [{marker}] {key:<7s}  inferred={inferred:<6}  yaml={from_yaml}")
        if inferred != from_yaml:
            diffs.append(key)

    if diffs:
        print(f"\nMismatched fields: {diffs}")
        return 1
    print("\nAll inferred acquisition fields match yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
