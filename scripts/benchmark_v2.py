"""Evaluate immutable V2 runs against the frozen nine-folio manual GT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paperdrm.benchmark import evaluate_v2_runs


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--benchmark",
        default=str(ROOT / "benchmarks" / "v1-manual-gt.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional new JSON output path; existing files are never overwritten.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output is not None and (
        args.output.exists() or args.output.is_symlink()
    ):
        print(f"ERROR: output already exists: {args.output}", file=sys.stderr)
        return 2
    try:
        report = evaluate_v2_runs(
            args.benchmark,
            args.runs_root,
            run_id=args.run_id,
        )
        payload = json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
        if args.output is None:
            print(payload, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
            print(f"Benchmark report: {args.output}")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    summary = report["summary"]
    print(
        f"Gate: {'PASS' if summary['gate_pass'] else 'FAIL'} "
        f"({summary['within_threshold']}/{summary['dataset_count']} within threshold)",
        file=sys.stderr,
    )
    return 0 if summary["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
