# PaperDRM

PaperDRM is a Python toolkit for directional reflectance profile (DRP) image stacks, with a workflow focused on laid-line analysis.

It includes:
- image stack loading and optional background subtraction,
- DRP stack caching to disk,
- per-pixel DRP direction estimation,
- patchwise trigonometric orientation masks,
- patchwise Gabor frequency estimation,
- orientation and score visualizations.

## Current Pipeline
`main.py` runs this sequence:
1. Load settings from `exp_param.yaml`.
2. Build/load an `ImagePack` (and cached DRP memmap).
3. Compute DRP direction map (`deg_map`).
4. Build:
- baseline trig mask (`azimuth_to_laidline_gray`),
- patchwise trig mask (`patchwise_trigonometric_mask`).
5. Visualize:
- raw azimuth vs patch dominant orientation,
- trig baseline vs patchwise trig outputs,
- patchwise Gabor confidence map.
6. Run patchwise Gabor using the **patchwise trig enhanced grayscale image** as input.
7. Overlay inferred laid-line grid and save `laid_lines_overlay_grid.png`.

## Repository Layout
- `main.py`: end-to-end runnable pipeline.
- `paperdrm/`: core library modules.
- `paperdrm/imagepack.py`: image loading, background subtraction, angle slicing, DRP cache management.
- `paperdrm/drp_compute.py`, `paperdrm/drp_direction.py`, `paperdrm/drp_plot.py`: DRP compute and direction/plot utilities.
- `paperdrm/trig_mask.py`: trig-mask builders and orientation comparison map helpers.
- `paperdrm/gabor_laidlines.py`: Gabor-based laid-line period/orientation estimation and overlay utilities.
- `paperdrm/visualization.py`: plotting helpers used by the pipeline.
- `scripts/rgb2grey.py`: convert `data/raw` images to grayscale in `data/processed`.
- `scripts/bg_blur.py`: generate blurred backgrounds in `data/background`.
- `exp_param.yaml`: DRP and runtime settings.

## Installation
Python 3.10+ is required.

PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data Preparation
Expected folders under `data/`:
- `data/raw`
- `data/processed`
- `data/background`
- `data/cache` (auto-created)

Typical prep:
```powershell
python scripts/rgb2grey.py
python scripts/bg_blur.py
```

## Configuration
Edit `exp_param.yaml`.

Important fields:
- `th_min`, `th_max`, `th_num`: theta sampling.
- `ph_min`, `ph_max`, `ph_num`: phi sampling.
- `data_serial`: bump to force DRP cache rebuild when dataset changes.
- `subtract_background`: enable/disable subtraction.
- `square_crop`: optional center crop to square.
- `theta_min_deg`: optional lower bound; keeps only theta samples greater than or equal to this value.
- `fov_width_cm`: horizontal field-of-view width in centimeters (optional, used to report laid-line interval in cm).

`main.py` currently overrides settings with:
- `angle_slice=(2, 2)`
- `verbose=True`

## Run
```powershell
python main.py
```

Outputs:
- orientation comparison plot,
- trig mask comparison plot,
- patchwise Gabor score heatmap,
- console estimates of laid-line interval (`cm`) and density (`lines/cm`) when `fov_width_cm` is set,
- saved overlay image: `laid_lines_overlay_grid.png`.

## Library Usage Example
```python
from paperdrm import ImagePack, Settings
from paperdrm.drp_direction import drp_direction_map
from paperdrm.trig_mask import patchwise_trigonometric_mask
from paperdrm.gabor_laidlines import estimate_laidline_frequency_gabor_patches

settings = Settings.from_yaml("exp_param.yaml").with_overrides(angle_slice=(2, 2))
imp = ImagePack(settings=settings)

_, deg_map = drp_direction_map(imp, verbose=False)
_, patch_img, _, _ = patchwise_trigonometric_mask(deg_map, patch_size=(512, 512), stride=(256, 256))

out = estimate_laidline_frequency_gabor_patches(
    patch_img,
    line_dir_deg=90.0,
    patch_size=(512, 512),
    stride=(256, 256),
    periods_px=list(range(6, 41, 2)),
)
print(out["dominant_period_px"], out["dominant_freq_cpp"])
```

## Caching Notes
- DRP stack is stored at `data/cache/drp.dat` (NumPy memmap).
- Cache metadata is stored in `data/cache/data_config.yaml`.
- Cache rebuild occurs when serial/slice/shape expectations do not match.

## Troubleshooting
- No images loaded: verify `data/processed` contains files matching configured format.
- Background mismatch errors: regenerate `data/background` from the same processed set.
- Unexpected stale results: bump `data_serial` or clear `data/cache`.
- `cv2.imshow` window issues (headless environments): comment GUI display lines in `main.py` and rely on saved outputs.
