"""Draw a 1cm x 1cm grid over the full cropped paper image."""
import cv2
import numpy as np
from paperdrm.stage0_loader.settings import Settings

cfg = Settings.from_yaml("results/Kk1-5_f9v/exp_param.yaml")

img = cv2.imread(str(cfg.image_path), cv2.IMREAD_GRAYSCALE)
x, y, w, h = cfg.crop_roi
crop = img[y:y + h, x:x + w]

phys_w_mm, phys_h_mm = 197.0, 273.0
crop_h, crop_w = crop.shape
px_per_mm_x = crop_w / phys_w_mm
px_per_mm_y = crop_h / phys_h_mm

bw = int(round(10 * px_per_mm_x))   # 1 cm in X pixels
bh = int(round(10 * px_per_mm_y))   # 1 cm in Y pixels

base = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

for col in range(0, crop_w, bw):
    cv2.line(base, (col, 0), (col, crop_h - 1), (0, 0, 255), 2)
for row in range(0, crop_h, bh):
    cv2.line(base, (0, row), (crop_w - 1, row), (0, 0, 255), 2)

# Scale bar label in top-left corner
cv2.putText(base, "Grid: 1cm x 1cm", (12, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
cv2.putText(base, f"({bw} x {bh} px | {phys_w_mm:.0f} x {phys_h_mm:.0f} mm paper)",
            (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

out = "results/Kk1-5_f9v/scale_grid_1cm.png"
cv2.imwrite(out, base)
print(f"Saved {out}  ({crop_w}x{crop_h} px, grid {bw}x{bh} px/cell)")
