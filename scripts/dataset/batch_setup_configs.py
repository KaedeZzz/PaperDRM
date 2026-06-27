"""
Detect paper crop for each new MSI image, compute correct fov_width_cm,
generate config yaml files, and print a summary table.
"""
import sys, os, cv2
import pathlib as _pl
sys.path.insert(0, ".")
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from detect_paper_roi import detect_paper_roi_texture
import yaml, pathlib

BASE = pathlib.Path("E:/Downloads/MSI photography")
CFG_DIR = pathlib.Path("configs")

# (serial, image_filename, paper_width_mm, period_range_cm)
DATASETS = [
    ("Ee5-22_f328r",  "MS-EE-00005-00022-000-00328-R+TX940IR_027.jpg",  210, [0.05, 0.20]),
    ("Ff2-6_f140r",   "MS-FF-00002-00006-000-00140-R+TX940IR_018.jpg",  210, [0.05, 0.20]),
    ("Ff4-9_f42r",    "MS-FF-00004-00009-000-00042-R+TX940IR_018.jpg",  210, [0.05, 0.20]),
    ("Ff4-15_f24r",   "MS-FF-00004-00015-000-00024-R+TX940IR_018.jpg",  190, [0.05, 0.20]),
    ("Hh2-10_f24r",   "MS-HH-00002-00010-000-00024-R+TX940IR_018.jpg",  205, [0.05, 0.20]),
    ("Ii3-8_f135v",   "MS-II-00003-00008-000-00135-V+TX940IR_018.jpg",  212, [0.05, 0.20]),
]

print(f"{'Serial':<22} {'Img W':>6} {'Crop x,y,w,h':<26} {'fov_width_cm':>13}")
print("-"*75)

for serial, fname, paper_mm, period_range in DATASETS:
    img_path = BASE / fname
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"{serial:<22}  LOAD FAILED: {img_path}")
        continue

    orig_h, orig_w = img.shape
    crop = detect_paper_roi_texture(img)
    x, y, w, h = crop
    fov = orig_w * paper_mm / (w * 10.0)

    print(f"{serial:<22} {orig_w:>6}  [{x},{y},{w},{h}]  {' ':>14}  fov={fov:.4f} cm")

    cfg = {
        "data_serial": serial,
        "image_path":  str(img_path).replace("\\", "/"),
        "fov_width_cm": round(fov, 4),
        "crop_roi": [x, y, w, h],
        "auto_line_dir": True,
        "period_range_cm": period_range,
        "wire_is_darker": False,
        "subtract_background": False,
    }

    out_path = CFG_DIR / f"{serial}.yaml"
    with open(out_path, "w") as f:
        f.write(f"data_serial: {serial}\n")
        f.write(f"image_path: \"{cfg['image_path']}\"\n\n")
        f.write(f"# Paper width from spreadsheet: {paper_mm} mm\n")
        f.write(f"# crop detected automatically; fov = full_img_w * paper_mm / (crop_w * 10)\n")
        f.write(f"fov_width_cm: {cfg['fov_width_cm']}\n")
        f.write(f"crop_roi: {list(crop)}\n\n")
        f.write(f"auto_line_dir: true\n")
        f.write(f"period_range_cm: {period_range}\n")
        f.write(f"wire_is_darker: false   # TX940IR transmitted light\n")
        f.write(f"subtract_background: false\n")

    print(f"  -> wrote {out_path}")
