"""
Test: extract 1 cm × 1 cm centre patch from each folio and run detection.
Compare against full-image result and spreadsheet GT.
"""
import sys, json
sys.path.insert(0, ".")
import cv2, numpy as np, yaml
from pathlib import Path
from paperdrm.stage3_detect.simple_detector import (
    detect_laid_lines_simple, auto_detect_line_dir
)

DATASETS = [
    # (serial,         gt_lpc,  note)
    ("Kk1-5_f5v",    9.0,   "manual GT"),
    ("Kk1-5_f9v",    9.0,   "manual GT"),
    ("Hh2-12_f190",  10.0,  ""),
    ("Ee5-22_f328r", 10.0,  ""),
    ("Ff2-6_f140r",  11.0,  ""),
    ("Ff4-9_f42r",    6.0,  ""),
    ("Ff4-15_f24r",  13.5,  "GT uncertain 13-14"),
    ("Hh2-10_f24r",  13.5,  "GT uncertain 13-14"),
    ("Ii3-8_f135v",   9.0,  ""),
]

SIGMA_BG = 100   # same as main.py background subtraction

def load_and_crop(cfg: dict) -> tuple:
    """Load image, optionally subtract background, apply crop_roi.
    Returns (image_processed, effective_fov_cm, cm_per_px)."""
    path = Path(cfg["image_path"])
    raw  = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise IOError(f"Cannot read {path}")

    img = raw.astype(np.float32)
    if cfg.get("subtract_background", False):
        bg  = cv2.GaussianBlur(raw, (0, 0), sigmaX=SIGMA_BG,
                               borderType=cv2.BORDER_REFLECT_101).astype(np.float32)
        img = np.clip(img - bg, 0, None)
        ref = float(np.percentile(img, 99.5))
        img = np.clip(img * (255.0 / max(ref, 1.0)), 0, 255)
    img = img.astype(np.uint8)

    crop = cfg.get("crop_roi")
    orig_w = raw.shape[1]
    fov = cfg["fov_width_cm"]
    if crop:
        x, y, w, h = crop
        img  = img[y:y+h, x:x+w]
        fov  = fov * w / orig_w    # effective fov after crop
    cm_per_px = fov / img.shape[1]
    return img, fov, cm_per_px


def centre_patch(img: np.ndarray, side_cm: float, cm_per_px: float) -> np.ndarray:
    """Extract a square patch of `side_cm` × `side_cm` from the image centre."""
    h, w = img.shape
    px   = int(round(side_cm / cm_per_px))
    px   = min(px, w, h)          # clamp to image size
    cy, cx = h // 2, w // 2
    y0, x0 = cy - px // 2, cx - px // 2
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(h, y0 + px), min(w, x0 + px)
    return img[y0:y1, x0:x1]


print(f"{'Serial':<22} {'GT':>5}  {'Full/cm':>8}  {'Patch/cm':>9}  {'Patch err':>10}  patch_px")
print("-"*80)

for serial, gt, note in DATASETS:
    cfg_path = Path("configs") / f"{serial}.yaml"
    res_path = Path("results") / serial / "interval_distribution.json"
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        full_iv = json.loads(res_path.read_text())
        full_lpc = full_iv["physical"]["lines_per_cm_mean"]
    except Exception as e:
        print(f"{serial:<22}  -- {e}"); continue

    try:
        img, fov, cm_per_px = load_and_crop(cfg)
    except Exception as e:
        print(f"{serial:<22}  load error: {e}"); continue

    patch = centre_patch(img, 1.0, cm_per_px)
    ph, pw = patch.shape
    patch_fov = pw * cm_per_px   # ≈ 1.0 cm

    period_range_cm = cfg.get("period_range_cm", [0.05, 0.20])
    period_range_px = (
        period_range_cm[0] / cm_per_px,
        period_range_cm[1] / cm_per_px,
    )
    wire_is_darker = cfg.get("wire_is_darker", False)

    try:
        # Direction: constrained search centred on 0° (portrait folios)
        line_dir = auto_detect_line_dir(patch, period_range_px=period_range_px,
                                        center_deg=0.0)
        result = detect_laid_lines_simple(
            patch,
            line_dir_deg=line_dir,
            period_range_px=period_range_px,
            wire_is_darker=wire_is_darker,
        )
        period_px  = result["dominant_period_px"]
        patch_lpc  = 1.0 / (period_px * cm_per_px)   # lines per cm from FFT period
        patch_err  = (patch_lpc - gt) / gt * 100
        full_err   = (full_lpc  - gt) / gt * 100

        # also get gap-based estimate
        n_peaks = len(result["grid_positions_x"])
        gap_lpc = None
        if n_peaks > 2:
            gaps_px = np.diff(np.sort(result["grid_positions_x"]))
            if len(gaps_px) > 0:
                gap_lpc = 1.0 / (float(np.median(gaps_px)) * cm_per_px)

        gap_str = f"(gap:{gap_lpc:.2f})" if gap_lpc else ""
        print(f"{serial:<22} {gt:>5.1f}  {full_lpc:>8.2f}  {patch_lpc:>9.2f} {patch_err:>+10.1f}%  "
              f"{pw}x{ph}px dir={line_dir:+.1f}° n={n_peaks} {gap_str}")
    except Exception as e:
        print(f"{serial:<22}  detect error: {e}")
