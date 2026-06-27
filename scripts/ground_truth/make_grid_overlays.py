"""
Generate 1 cm × 1 cm grid overlays for manual laid-line counting.

For each dataset:
  - Load raw image with crop_roi applied (no bg subtraction, so lines are visible)
  - Draw a 1 cm grid (cyan semi-transparent lines + labels)
  - Save to results/<serial>/grid_1cm_overlay.jpg  (50 % scale to keep file small)
"""
import sys, yaml
sys.path.insert(0, ".")
import cv2, numpy as np
from pathlib import Path

DATASETS = [
    "Kk1-5_f5v",
    "Kk1-5_f9v",
    "Hh2-12_f190",
    "Ee5-22_f328r",
    "Ff2-6_f140r",
    "Ff4-9_f42r",
    "Ff4-15_f24r",
    "Hh2-10_f24r",
    "Ii3-8_f135v",
]

SCALE = 0.40          # output image scale (40 % of cropped size)
GRID_COLOR   = (0, 220, 220)   # BGR cyan
LABEL_COLOR  = (0, 200, 0)     # BGR green
ALPHA        = 0.45            # grid line opacity


def draw_grid(bgr: np.ndarray, cm_per_px: float, scale: float) -> np.ndarray:
    """Draw 1 cm grid on a BGR image (already at `scale`)."""
    out   = bgr.copy()
    h, w  = out.shape[:2]
    step  = cm_per_px * scale          # 1 cm in scaled pixels
    thick = max(1, round(step * 0.015))

    overlay = out.copy()
    # vertical lines (x = k * step)
    x = step
    col_idx = 1
    while x < w:
        xi = int(round(x))
        cv2.line(overlay, (xi, 0), (xi, h - 1), GRID_COLOR, thick)
        # label at top
        cv2.putText(overlay, f"{col_idx}cm", (xi + 3, int(step * 0.25)),
                    cv2.FONT_HERSHEY_SIMPLEX, step * 0.035, LABEL_COLOR, max(1, thick), cv2.LINE_AA)
        x += step
        col_idx += 1

    # horizontal lines (y = k * step)
    y = step
    row_idx = 1
    while y < h:
        yi = int(round(y))
        cv2.line(overlay, (0, yi), (w - 1, yi), GRID_COLOR, thick)
        cv2.putText(overlay, f"{row_idx}cm", (int(step * 0.03), yi - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, step * 0.035, LABEL_COLOR, max(1, thick), cv2.LINE_AA)
        y += step
        row_idx += 1

    return cv2.addWeighted(out, 1 - ALPHA, overlay, ALPHA, 0)


for serial in DATASETS:
    cfg_path = Path("configs") / f"{serial}.yaml"
    out_dir  = Path("results") / serial
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grid_1cm_overlay.jpg"

    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[{serial}] config read error: {e}")
        continue

    img_path = Path(cfg["image_path"])
    raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        print(f"[{serial}] LOAD FAILED: {img_path}")
        continue

    orig_w = raw.shape[1]
    fov    = cfg["fov_width_cm"]
    crop   = cfg.get("crop_roi")
    if crop:
        x, y, w, h = crop
        raw  = raw[y:y+h, x:x+w]
        fov  = fov * w / orig_w

    ch, cw = raw.shape
    cm_per_px = fov / cw    # cm per full-res pixel

    # Scale down
    sw = int(round(cw * SCALE))
    sh = int(round(ch * SCALE))
    small = cv2.resize(raw, (sw, sh), interpolation=cv2.INTER_AREA)
    bgr   = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

    result = draw_grid(bgr, 1.0 / cm_per_px, SCALE)

    # Add title bar
    bar_h = max(30, int(sh * 0.025))
    bar   = np.zeros((bar_h, sw, 3), dtype=np.uint8)
    title = f"{serial}   fov={fov:.2f}cm  scale={cm_per_px*10:.2f}px/mm  1cm grid"
    cv2.putText(bar, title, (8, bar_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, bar_h * 0.028, (200, 200, 200), 1, cv2.LINE_AA)
    result = np.vstack([bar, result])

    cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[{serial}]  {cw}x{ch} px  1cm={1/cm_per_px:.0f}px  "
          f"grid={int(sw//(1/cm_per_px*SCALE))+1}x{int(sh//(1/cm_per_px*SCALE))+1} cells"
          f"  -> {out_path}")
