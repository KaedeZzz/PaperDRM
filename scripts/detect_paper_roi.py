"""
Detect the paper (manuscript) region in a transmitted-light MSI/DRM image.

Two methods are available:

  texture (default)
    Segments paper from background using local intensity variance.
    Paper has texture (laid lines, writing); the light panel and equipment
    frame are uniform, so their variance is near zero regardless of brightness.
    Works at 1/8 downscale for speed, then maps back to full resolution.

  legacy
    Brightness-band thresholding (low < pixel < high) + largest connected
    component.  Simple but fails when the light-panel brightness overlaps
    the paper brightness (common in TX-NIR transmitted images).

Usage:
    python scripts/detect_paper_roi.py <image_path> [--preview]
    python scripts/detect_paper_roi.py <image_path> --method legacy --preview
    python scripts/detect_paper_roi.py <image_path> --aspect 1.36 --preview
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Texture-variance method
# ---------------------------------------------------------------------------

def _two_level_otsu(img_u8: np.ndarray) -> tuple[int, int]:
    """Return (T_lo, T_hi) that maximise between-class variance for 3 classes."""
    hist = np.bincount(img_u8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    cdf_n  = np.cumsum(hist) / total          # cumulative weight
    cdf_mu = np.cumsum(hist * np.arange(256)) / total  # cumulative mean * total
    mu_all = cdf_mu[-1]

    best_var, best = -1.0, (0, 128)
    for t1 in range(1, 254):
        w0 = cdf_n[t1]
        if w0 < 1e-6:
            continue
        for t2 in range(t1 + 1, 255):
            w1 = cdf_n[t2] - cdf_n[t1]
            w2 = 1.0 - cdf_n[t2]
            if w1 < 1e-6 or w2 < 1e-6:
                continue
            mu0 = cdf_mu[t1] / w0
            mu1 = (cdf_mu[t2] - cdf_mu[t1]) / w1
            mu2 = (mu_all - cdf_mu[t2]) / w2
            var = w0*(mu0-mu_all)**2 + w1*(mu1-mu_all)**2 + w2*(mu2-mu_all)**2
            if var > best_var:
                best_var, best = var, (t1, t2)
    return best


def detect_paper_roi_texture(
    image: np.ndarray,
    *,
    downsample: int = 8,
    var_ksize: int = 7,
    grow_px: int = 80,
    morph_open_px: int = 10,
    expected_aspect: float | None = None,
    aspect_tol: float = 0.15,
) -> tuple[int, int, int, int]:
    """
    Return (x, y, w, h) bounding box of the paper region.

    Strategy
    --------
    1. Downsample (default 8×) for speed.
    2. Two-level Otsu on brightness → paper brightness range [T_lo, T_hi].
       Separates equipment (dark) / paper (mid) / light-panel (bright).
    3. Local variance → Otsu mask → high-texture seeds (definitely paper).
    4. Dilate seeds by grow_px, then AND with brightness mask.
       Region grows from texture seeds but stops at bright/dark boundaries,
       so blank paper margins are absorbed while the light panel is excluded.
    5. Morphological open (removes stray noise).
    6. Largest connected component → bounding box.
    7. Optional aspect-ratio sanity check; raises RuntimeError on failure.

    Parameters
    ----------
    image          : grayscale uint8 ndarray
    downsample     : integer scale factor
    var_ksize      : variance window size in downsampled pixels
    grow_px        : dilation radius in downsampled pixels (bridges blank margins)
    morph_open_px  : opening kernel in downsampled pixels (removes stray noise)
    expected_aspect: expected h/w ratio (optional sanity check)
    aspect_tol     : allowed relative deviation from expected_aspect
    """
    H, W = image.shape
    s = downsample

    # --- 1. Downsample ---
    small = cv2.resize(image, (W // s, H // s), interpolation=cv2.INTER_AREA)
    Hs, Ws = small.shape

    # --- 2. Two-level Otsu → paper brightness band ---
    T_lo, T_hi = _two_level_otsu(small)
    brightness_mask = ((small > T_lo) & (small < T_hi)).astype(np.uint8) * 255

    # --- 3. Local variance → high-texture seeds ---
    img_f = small.astype(np.float32)
    kw    = (var_ksize, var_ksize)
    mu    = cv2.boxFilter(img_f,         -1, kw, normalize=True)
    mu2   = cv2.boxFilter(img_f * img_f, -1, kw, normalize=True)
    var   = np.clip(mu2 - mu * mu, 0.0, None)
    var_u8 = cv2.normalize(var, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    _, tex_seeds = cv2.threshold(var_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tex_seeds = cv2.bitwise_and(tex_seeds, brightness_mask)  # seeds must be in brightness range

    # --- 4. Grow seeds within brightness band ---
    # Dilation expands seeds; AND with brightness_mask stops growth at light panel / equipment.
    grow_k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow_px * 2 + 1, grow_px * 2 + 1))
    grown     = cv2.dilate(tex_seeds, grow_k)
    paper_mask = cv2.bitwise_and(grown, brightness_mask)

    # --- 5. Morphological open (remove noise) ---
    def _rect(px: int) -> np.ndarray:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (px, px))

    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN,  _rect(morph_open_px))

    # --- 6. Largest connected component ---
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(paper_mask, connectivity=8)
    if n_labels < 2:
        raise RuntimeError(
            "Texture-based detection found no paper region. "
            "Check that the image contains a manuscript with visible texture."
        )

    areas = stats[1:, cv2.CC_STAT_AREA]
    best  = int(np.argmax(areas)) + 1
    xs = int(stats[best, cv2.CC_STAT_LEFT])
    ys = int(stats[best, cv2.CC_STAT_TOP])
    ws = int(stats[best, cv2.CC_STAT_WIDTH])
    hs = int(stats[best, cv2.CC_STAT_HEIGHT])

    # Map back to full resolution
    x, y, w, h = xs * s, ys * s, ws * s, hs * s
    # Clamp to image bounds
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = min(w, W - x)
    h = min(h, H - y)

    # --- 7. Aspect-ratio sanity check ---
    if expected_aspect is not None:
        detected = h / w
        dev = abs(detected - expected_aspect) / expected_aspect
        if dev > aspect_tol:
            raise RuntimeError(
                f"Aspect ratio check failed: detected {detected:.3f}, "
                f"expected {expected_aspect:.3f} "
                f"(deviation {dev*100:.1f}% > tolerance {aspect_tol*100:.0f}%).\n"
                f"  crop_roi would be [{x}, {y}, {w}, {h}] — inspect with --preview."
            )

    return x, y, w, h


# ---------------------------------------------------------------------------
# Legacy brightness-band method (kept for comparison)
# ---------------------------------------------------------------------------

def detect_paper_roi(
    image: np.ndarray,
    *,
    low: int = 50,
    high: int = 210,
    morph_kernel: int = 40,
) -> tuple[int, int, int, int]:
    """
    Return (x, y, w, h) via brightness-band thresholding.
    Fails when light-panel brightness overlaps paper brightness.
    """
    mask = ((image > low) & (image < high)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels < 2:
        raise RuntimeError("Could not find a paper region — try adjusting --low / --high.")

    areas = stats[1:, cv2.CC_STAT_AREA]
    best  = int(np.argmax(areas)) + 1
    return (
        int(stats[best, cv2.CC_STAT_LEFT]),
        int(stats[best, cv2.CC_STAT_TOP]),
        int(stats[best, cv2.CC_STAT_WIDTH]),
        int(stats[best, cv2.CC_STAT_HEIGHT]),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect paper ROI in a transmitted-light MSI/DRM image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", help="Path to the image file.")
    parser.add_argument(
        "--method", choices=["texture", "legacy"], default="texture",
        help="Detection method (default: texture).",
    )
    parser.add_argument(
        "--aspect", type=float, default=None, metavar="H_W",
        help="Expected height/width ratio from physical dimensions (e.g. 1.36). "
             "If given, triggers a sanity check and raises on large deviation.",
    )
    parser.add_argument(
        "--downsample", type=int, default=8,
        help="Downscale factor for texture method (default 8).",
    )
    parser.add_argument("--preview", action="store_true",
                        help="Save a preview image with the detected bbox drawn.")
    # Legacy-only options
    parser.add_argument("--low",   type=int, default=50)
    parser.add_argument("--high",  type=int, default=210)
    parser.add_argument("--morph", type=int, default=40)
    args = parser.parse_args()

    path = Path(args.image)
    img  = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"ERROR: could not open {path}")
    H, W = img.shape

    if args.method == "texture":
        x, y, w, h = detect_paper_roi_texture(
            img,
            downsample=args.downsample,
            expected_aspect=args.aspect,
        )
    else:
        x, y, w, h = detect_paper_roi(
            img, low=args.low, high=args.high, morph_kernel=args.morph,
        )

    print(f"\nImage size : {W} x {H} px")
    print(f"Method     : {args.method}")
    print(f"Paper ROI  : x={x}  y={y}  w={w}  h={h}  (aspect h/w={h/w:.3f})")
    print(f"\nPaste into exp_param.yaml:")
    print(f"  crop_roi: [{x}, {y}, {w}, {h}]")

    if args.preview:
        preview = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), max(2, W // 300))
        out = path.parent / (path.stem + "_roi_preview.jpg")
        cv2.imwrite(str(out), preview)
        print(f"\nPreview -> {out}")


if __name__ == "__main__":
    main()
