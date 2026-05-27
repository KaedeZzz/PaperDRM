"""
Downscale the large results/ PNG files into report-friendly versions.

The full-resolution overlays in results/<serial>/laid_lines_overlay.png
are 4K-ish images of order 20-45 MB. Including them directly in the
report blows the LaTeX PDF up to 100+ MB, which is too large to email
to a supervisor. This script writes downscaled copies into
report/figures/ that the LaTeX \\includegraphics calls reference by
short name.

Photo-style overlays are saved as JPEG with quality 85, which gives an
order-of-magnitude file-size reduction without visible loss at the
target print width. Matplotlib-output metric plots are saved as PNG
because line art compresses badly under JPEG.

Run from project root: .venv/Scripts/python scripts/downscale_figures_for_report.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MAX_WIDTH = 1500       # px; matches roughly 6 in at 250 dpi printed
JPEG_QUALITY = 85       # visually lossless on photo content at this scale


# (source path relative to results/, output basename, output format)
FIGURES: list[tuple[str, str, str]] = [
    # Reference good case: Kk.1.5 f5v
    ("Kk1-5_f5v/laid_lines_overlay.png",      "f5v_overlay",      "jpeg"),
    ("Kk1-5_f5v/wire_width_segments.png",     "f5v_wire",         "png"),
    ("Kk1-5_f5v/split_half_stability.png",    "f5v_split",        "png"),
    # Octave-alias failure case
    ("Ff4-15_f24r/laid_lines_overlay.png",    "ff4_15_overlay",   "jpeg"),
    # Da Rold controlled reproduction
    ("10/laid_lines_overlay.png",             "darold_overlay",   "jpeg"),
    ("10/wire_width_segments.png",            "darold_wire",      "png"),
    ("10/self_contrast.png",                  "darold_selfc",     "png"),
]


def downscale_one(src_rel: str, out_basename: str, fmt: str) -> None:
    src = RESULTS / src_rel
    if not src.exists():
        print(f"[skip] {src_rel} -- source missing")
        return
    img = Image.open(src)
    orig_size = src.stat().st_size
    if img.width > MAX_WIDTH:
        ratio = MAX_WIDTH / img.width
        new_size = (MAX_WIDTH, int(round(img.height * ratio)))
        img = img.resize(new_size, Image.LANCZOS)
    out = OUT / f"{out_basename}.{fmt}"
    if fmt == "jpeg":
        img.convert("RGB").save(out, "JPEG", quality=JPEG_QUALITY, optimize=True)
    else:
        img.save(out, "PNG", optimize=True)
    new_size = out.stat().st_size
    ratio_pct = 100.0 * new_size / orig_size
    print(f"[ok] {src_rel:48s} -> figures/{out.name:24s} "
          f"({orig_size/1e6:6.1f} MB -> {new_size/1e6:5.2f} MB, {ratio_pct:5.1f}%)")


def main() -> None:
    for src, name, fmt in FIGURES:
        downscale_one(src, name, fmt)


if __name__ == "__main__":
    main()
