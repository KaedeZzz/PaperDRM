"""
Fetch a DRP dataset from Google Drive (or copy from a local folder) into
``data/drp/<serial>/raw/`` and write a per-dataset sample.yaml ready for the
pipeline.

Usage:
    # From Google Drive (requires `pip install gdown`):
    python scripts/fetch_dataset.py <gdrive_folder_url> --serial 12 --fov 8.65
    python scripts/fetch_dataset.py <url> --serial 12 --fov 8.65 \\
        --theta-min 30 --bg-url <gdrive_url_for_backgrounds>

    # From a local source (e.g. an already-downloaded folder, useful for
    # testing or when Google Drive is not reachable):
    python scripts/fetch_dataset.py --from-local /path/to/folder --serial 12 --fov 8.65

After running, point the pipeline at the new bundle:
    python main.py --config data/drp/<serial>/sample.yaml
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from paperdrm.stage0_loader.inference import infer_drp_config_from_folder


ACQ_FIELDS = ("th_min", "th_max", "th_num", "ph_min", "ph_max", "ph_num")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="Google Drive folder URL. Omit when using --from-local.",
    )
    parser.add_argument(
        "--from-local",
        dest="from_local",
        type=Path,
        help="Skip the Google Drive download; copy *.<ext> files from this "
             "local folder instead. Useful for testing.",
    )
    parser.add_argument(
        "--serial",
        help="Dataset identifier (becomes data/drp/<serial>/). Auto-derived "
             "from the URL tail or local folder name if absent.",
    )
    parser.add_argument("--fov", type=float, required=True,
                        help="fov_width_cm in cm (required).")
    parser.add_argument("--theta-min", type=float, dest="theta_min", default=None,
                        help="Optional theta_min_deg processing threshold.")
    parser.add_argument("--bg-url", dest="bg_url", default=None,
                        help="Optional Google Drive folder URL for background images. "
                             "Downloads into <dataset>/background/.")
    parser.add_argument("--ext", default="jpg", help="Image extension (default: jpg).")
    parser.add_argument("--data-root", dest="data_root", type=Path,
                        default=REPO_ROOT / "data",
                        help="Override data root (default: <repo>/data).")
    parser.add_argument("--notes", default=None,
                        help="Free-form note saved into sample.yaml under notes.note.")
    parser.add_argument("--copy-mode", choices=("copy", "symlink"), default="copy",
                        help="When --from-local, copy files or symlink them (default: copy).")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip the post-download inference check.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing into an existing non-empty dataset folder.")
    args = parser.parse_args()

    if (args.url is None) == (args.from_local is None):
        parser.error("Provide exactly one of <url> or --from-local.")

    serial = args.serial or _auto_serial(args.url, args.from_local)
    dataset_dir = args.data_root / "drp" / str(serial)
    raw_dir = dataset_dir / "raw"
    if raw_dir.exists() and any(raw_dir.iterdir()) and not args.overwrite:
        print(
            f"ERROR: {dataset_dir} already exists and is non-empty. "
            "Pick a different --serial or pass --overwrite.",
            file=sys.stderr,
        )
        return 2
    raw_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("processed", "cache", "background"):
        (dataset_dir / sub).mkdir(parents=True, exist_ok=True)

    if args.from_local is not None:
        _ingest_local(args.from_local, raw_dir, args.ext, args.copy_mode)
        source_descriptor = f"local:{args.from_local.resolve()}"
    else:
        _gdown_folder(args.url, raw_dir)
        source_descriptor = args.url

    if args.bg_url:
        bg_dir = dataset_dir / "background"
        bg_dir.mkdir(exist_ok=True)
        _gdown_folder(args.bg_url, bg_dir)

    if not args.no_validate:
        cfg, report = infer_drp_config_from_folder(
            dataset_dir, img_format=args.ext, strict=True
        )
        print("\n== Inference report ==")
        print(report.summary())
        print("\n== Inferred DRPConfig ==")
        for key in ACQ_FIELDS:
            print(f"  {key} = {getattr(cfg, key)}")

    sample_path = dataset_dir / "sample.yaml"
    sample = _build_sample_yaml(
        serial=serial,
        fov=args.fov,
        theta_min=args.theta_min,
        source=source_descriptor,
        note=args.notes,
    )
    sample_path.write_text(yaml.safe_dump(sample, sort_keys=False))
    print(f"\nWrote {sample_path}")

    print("\nNext step:")
    rel = sample_path.relative_to(REPO_ROOT) if sample_path.is_relative_to(REPO_ROOT) else sample_path
    print(f"  python main.py --config {rel}")
    return 0


def _auto_serial(url: str | None, local: Path | None) -> str:
    if local is not None:
        return local.resolve().name
    if url is not None:
        # Strip query string, take the last non-empty path component, cap length.
        last = url.rstrip("/").split("?")[0].split("/")[-1]
        if last:
            return last[:16]
    return time.strftime("%Y%m%d_%H%M%S")


def _ingest_local(src: Path, dst: Path, ext: str, mode: str) -> None:
    if not src.is_dir():
        raise SystemExit(f"Local source must be a directory: {src}")
    files = sorted(src.glob(f"*.{ext}"))
    if not files:
        raise SystemExit(f"No *.{ext} files in {src}")
    action = "Symlinking" if mode == "symlink" else "Copying"
    print(f"{action} {len(files)} *.{ext} files from {src} -> {dst}")
    for f in files:
        target = dst / f.name
        if target.exists() or target.is_symlink():
            target.unlink()
        if mode == "symlink":
            target.symlink_to(f.resolve())
        else:
            shutil.copy2(f, target)


def _gdown_folder(url: str, output: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit(
            "gdown is not installed. Run `pip install gdown` "
            "(also tracked in requirements.txt) and retry."
        ) from exc
    print(f"Downloading from {url} -> {output}")
    gdown.download_folder(url, output=str(output), quiet=False, use_cookies=False)


def _build_sample_yaml(
    *, serial: str, fov: float, theta_min: float | None,
    source: str, note: str | None,
) -> dict:
    sample: dict = {
        "data_serial": serial,
        "folder": f"data/drp/{serial}/raw",
        "fov_width_cm": fov,
    }
    if theta_min is not None:
        sample["theta_min_deg"] = theta_min
    notes: dict = {"source": source, "fetched": date.today().isoformat()}
    if note:
        notes["note"] = note
    sample["notes"] = notes
    return sample


if __name__ == "__main__":
    raise SystemExit(main())
