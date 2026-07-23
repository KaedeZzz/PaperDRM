# PaperDRM

PaperDRM is a research codebase for measuring laid-line spacing in historical
paper from directional reflectance photography (DRP) stacks or individual
manuscript images.

The current recommended method aggregates the one-dimensional spectral power
from multiple grazing-light azimuths, estimates the dominant laid-line period,
aligns per-image phase and polarity, and reconstructs a line grid. A
single-image spectral detector is also available for transmitted-light MSI
images. The older trig-mask and patchwise-Gabor route remains only as a legacy
ablation because it has a known half-period bias.

This repository is a research prototype. Period and line-density estimates are
the most mature outputs. Wire-width estimates remain experimental and should
not be treated as validated physical measurements without further calibration.

The current technical audit, manual-GT benchmark interpretation, literature
review and proposed research roadmap are documented in
[`docs/repo_audit_and_research_roadmap_zh.md`](docs/repo_audit_and_research_roadmap_zh.md).
The incremental V2 rewrite is specified in [`docs/v2/`](docs/v2/); its Phase 0
contract freezes V1 behaviour before any package or algorithm migration.

## Quick start

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python main.py --config exp_param.yaml
```

For an MSI/single-image configuration:

```bash
python main.py --config configs/Kk1-5_f5v.yaml
```

If the YAML contains `image_path`, PaperDRM automatically selects the
single-image route. Otherwise it loads a DRP stack.

Run the lightweight regression tests with:

```bash
python -m unittest discover -s tests -v
```

## Detection routes

### Multi-phi DRP — recommended for DRP acquisitions

1. Load and angularly filter the DRP image stack.
2. Select the steepest grazing-light image at each azimuth.
3. Compute and normalize the laid-line-normal spectrum for every image.
4. Aggregate spectral power and estimate the dominant period.
5. Fit per-image phase, correct opposite polarity by a half-period shift, and
   combine phases with a circular mean.
6. Reconstruct the laid-line grid and evaluate interval distribution,
   split-half stability, self-contrast, spectral fit and wire-width diagnostics.

### Single image — recommended for transmitted-light MSI

1. Load the image, optionally detect/crop the paper ROI and subtract background.
2. Estimate or use the configured laid-line direction.
3. Estimate period spectrally and apply fixed-period Gabor refinement.
4. Reconstruct the grid and run the available single-image diagnostics.

### Legacy route

The DRP direction-map, trigonometric-mask and patchwise-Gabor pipeline is kept
for historical comparison only. It is slow and can detect half the true period.

Select a DRP route with `--track`:

```bash
python main.py --config exp_param.yaml --track multi_phi
python main.py --config exp_param.yaml --track simple
python main.py --config exp_param.yaml --track legacy
```

## Configuration

Important YAML fields include:

- `data_serial`: dataset identifier and result-directory name.
- `folder`: DRP image directory. Filenames should encode phi and theta.
- `image_path`: single-image input; its presence selects the single-image route.
- `img_format`: image extension, default `jpg`.
- `angle_slice`: `[phi_stride, theta_stride]`.
- `theta_min_deg`: discard lower-elevation DRP samples.
- `subtract_background`: enable background subtraction.
- `subtraction_scale_percentile`: scaling percentile after subtraction.
- `crop_roi`: `[x, y, width, height]`.
- `square_crop`: optionally centre-crop DRP images to a square.
- `fov_width_cm`: horizontal field of view used for physical-unit conversion.
- `period_range_cm`: expected laid-line interval range in centimetres.
- `line_dir_deg`: configured laid-line direction.
- `auto_line_dir`: estimate direction near the expected portrait/landscape axis.
- `wire_is_darker`: expected line polarity. Use `false` when wire marks appear
  brighter, as in many transmitted-light images.
- `use_cached_stack`: set to `false` to force a DRP cache rebuild.

If the six DRP acquisition fields (`th_min/max/num`, `ph_min/max/num`) are
omitted, they are inferred from the image filenames. Partial acquisition
definitions are rejected.

## Outputs and result integrity

Pipeline stages use temporary files in the repository root, archive them under
`results/<data_serial>/`, and then remove the root staging files.

The archive step uses an explicit artifact list for each detector route. Before
copying, it removes old pipeline-managed outputs from that dataset directory so
that a stale split-half result or image from another run cannot enter a new
report. User-managed files such as `manual_gt.json`, bounding boxes and manual
overlays are preserved.

Typical outputs:

- `interval_distribution.json`
- `fit_quality.json`
- `self_contrast.json`
- `split_half_stability.json` for multi-phi runs
- `wire_width_stats.json`
- `laid_lines_overlay.png`
- `laid_lines_overlay_bands.png`
- `report_en.html` and `report_zh.html`

A negative self-contrast z-score is reported as a polarity contradiction, not
converted to a positive confidence score.

The fit-quality output also records whether the dominant spectral peak is
pinned to either edge of the configured period range. A boundary hit is treated
as an invalid search range and overrides otherwise favorable stability or
contrast scores in the generated reports.

The primary spacing/density measurement is the global spectral period. Local
peak-to-peak gaps remain available as a descriptive distribution (median and
IQR); their arithmetic mean is not used as the headline measurement because
missed weak peaks create long-gap outliers.

## Cache behaviour

The DRP memmap is stored under `<data_root>/cache/drp.dat`, with metadata in
`data_config.yaml`.

Cache reuse requires a fingerprint match covering:

- selected image paths, sizes and modification times;
- any selected background files;
- angular slicing and theta filtering;
- background-subtraction mode and scaling;
- ROI and square cropping;
- filtered acquisition geometry and final stack shape.

Changing any of these inputs invalidates the cache. Older cache metadata has no
fingerprint and is rebuilt automatically on first use.

## Repository layout

- `main.py` — pipeline entry point and detector-route orchestration.
- `src/paperdrm/stage0_loader/` — settings, image loading and cache management.
- `src/paperdrm/stage0_drp/` — DRP slicing and stack operations.
- `src/paperdrm/stage1_features/` — DRP direction estimation used by the legacy route.
- `src/paperdrm/stage2_enhance/` — legacy trigonometric enhancement.
- `src/paperdrm/stage3_detect/` — multi-phi, single-image and legacy detectors.
- `src/paperdrm/stage4_viz/` — visualization helpers.
- `src/paperdrm/stage5_evaluation/` — interval, stability, contrast and fit metrics.
- `configs/` — dataset-specific configurations.
- `scripts/` — data preparation, benchmarking and report utilities.
- `tests/` — lightweight regression tests.
- `results/` — archived experimental outputs.
- `report/` and `logbook/` — report source and project logbook.

## Data preparation

To create a local DRP dataset bundle from an existing folder:

```bash
python scripts/fetch_dataset.py \
  --from-local /path/to/images \
  --serial example \
  --fov 8.65
```

Google Drive ingestion is also supported:

```bash
python scripts/fetch_dataset.py \
  "https://drive.google.com/drive/folders/..." \
  --serial example \
  --fov 8.65
```

The script creates `data/drp/<serial>/` with raw, processed, background and
cache directories plus a configuration file.

## Known limitations

- There is not yet a full end-to-end test using a versioned real-image fixture.
- The current benchmark set is small and contains manually configured physical
  scales and ROIs.
- Period aliases remain possible outside the validated frequency range.
- Split-half agreement measures repeatability, not correctness; stable aliases
  are possible.
- Self-contrast depends on correct phase and polarity configuration.
- Wire-width estimation is sensitive to line profile, period and image SNR.
- Report values must be regenerated after algorithm or configuration changes;
  committed historical results are not automatically updated.

See `report/README.md` for building the written project report.
