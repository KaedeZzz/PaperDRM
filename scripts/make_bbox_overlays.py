"""
Draw the crop_roi bounding box (red) on the full original image
and save to results/<serial>/bbox_overlay.jpg.
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

SCALE = 0.10   # 10 % of original — full image is 6132×8176, so output ~613×818 px

for serial in DATASETS:
    cfg_path = Path("configs") / f"{serial}.yaml"
    out_dir  = Path("results") / serial
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bbox_overlay.jpg"

    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[{serial}] config error: {e}"); continue

    raw = cv2.imread(cfg["image_path"], cv2.IMREAD_GRAYSCALE)
    if raw is None:
        print(f"[{serial}] LOAD FAILED: {cfg['image_path']}"); continue

    orig_h, orig_w = raw.shape
    sw = int(round(orig_w * SCALE))
    sh = int(round(orig_h * SCALE))
    small = cv2.resize(raw, (sw, sh), interpolation=cv2.INTER_AREA)
    bgr   = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

    crop = cfg.get("crop_roi")
    if crop:
        x, y, w, h = crop
        x2, y2 = x + w, y + h
        # scale to thumbnail coords
        sx  = int(round(x  * SCALE))
        sy  = int(round(y  * SCALE))
        sx2 = int(round(x2 * SCALE))
        sy2 = int(round(y2 * SCALE))
        thick = max(2, int(sw * 0.005))
        cv2.rectangle(bgr, (sx, sy), (sx2, sy2), (0, 0, 255), thick)
        label = f"crop [{x},{y},{w},{h}]"
        cv2.putText(bgr, label, (sx + 4, sy + thick + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, sw * 0.0012, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(bgr, "no crop_roi", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # title bar
    bar_h = max(22, int(sh * 0.03))
    bar   = np.zeros((bar_h, sw, 3), dtype=np.uint8)
    cv2.putText(bar, f"{serial}  {orig_w}x{orig_h}px",
                (6, bar_h - 5), cv2.FONT_HERSHEY_SIMPLEX,
                bar_h * 0.030, (180, 180, 180), 1, cv2.LINE_AA)
    out_img = np.vstack([bar, bgr])

    cv2.imwrite(str(out_path), out_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    crop_str = f"[{x},{y},{w},{h}]" if crop else "none"
    print(f"[{serial}]  bbox={crop_str}  -> {out_path}")
