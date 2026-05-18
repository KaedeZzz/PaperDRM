"""
Detect the paper (manuscript) region in a transmitted-light DRM image.

In this setup the image has three brightness zones:
  - Black  : camera / equipment masking the edges
  - White  : light panel visible around the paper
  - Gray   : the actual paper (the region we want)

The script finds the gray region, computes its axis-aligned bounding box,
and prints a crop_roi line ready to paste into exp_param.yaml.

Usage:
    python scripts/detect_paper_roi.py <image_path>
    python scripts/detect_paper_roi.py <image_path> --preview
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def detect_paper_roi(
    image: np.ndarray,
    *,
    low: int = 50,
    high: int = 210,
    morph_kernel: int = 40,
) -> tuple[int, int, int, int]:
    """
    Return (x, y, w, h) bounding box of the paper (gray) region.

    Parameters
    ----------
    image : grayscale uint8 array
    low   : pixels below this are treated as black equipment
    high  : pixels above this are treated as white light panel
    morph_kernel : size of morphological close/open to remove small gaps
    """
    # Isolate the "paper" brightness band
    mask = ((image > low) & (image < high)).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    # Largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels < 2:
        raise RuntimeError("Could not find a paper region — try adjusting --low / --high.")

    # Skip label 0 (background); pick the largest component
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1

    x = int(stats[best, cv2.CC_STAT_LEFT])
    y = int(stats[best, cv2.CC_STAT_TOP])
    w = int(stats[best, cv2.CC_STAT_WIDTH])
    h = int(stats[best, cv2.CC_STAT_HEIGHT])
    return x, y, w, h


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect paper ROI in a transmitted-light DRM image.")
    parser.add_argument("image", help="Path to a representative image from the dataset.")
    parser.add_argument("--low",  type=int, default=50,  help="Lower brightness threshold (default 50).")
    parser.add_argument("--high", type=int, default=210, help="Upper brightness threshold (default 210).")
    parser.add_argument("--morph", type=int, default=40, help="Morphological kernel size in px (default 40).")
    parser.add_argument("--preview", action="store_true", help="Write a preview image next to the input.")
    args = parser.parse_args()

    path = Path(args.image)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"ERROR: could not open {path}")

    x, y, w, h = detect_paper_roi(img, low=args.low, high=args.high, morph_kernel=args.morph)
    H, W = img.shape

    print(f"\nImage size : {W} x {H} px")
    print(f"Paper ROI  : x={x}  y={y}  w={w}  h={h}")
    print(f"\nPaste into exp_param.yaml:")
    print(f"  crop_roi: [{x}, {y}, {w}, {h}]")

    if args.preview:
        preview = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), max(2, W // 300))
        out = path.parent / (path.stem + "_roi_preview.jpg")
        cv2.imwrite(str(out), preview)
        print(f"\nPreview saved to: {out}")


if __name__ == "__main__":
    main()
