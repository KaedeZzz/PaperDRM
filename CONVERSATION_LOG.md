# Conversation Log

This file records summaries of all working conversations with Claude on PaperDRM.
Updated before each git commit so the history of decisions and intent stays in-repo.

Format: newest entry on top. Each entry includes date, what was discussed/changed,
and any open follow-ups.

---

## 2026-05-24（续）— 手动 GT 标注工具 + 全数据集对比

### 背景

上一 session 分析了系统性误差，用户提出用手动标注作为 ground truth 代替 spreadsheet。本 session 实现并运行了完整标注流程。

### 新建工具

**`scripts/gt_builder.py`** — 两阶段交互式 GT 构建 app

- **Phase 1**：bbox 选择（复用 `select_bbox.py` 的 `BBoxSelector`，通过 `importlib` 动态加载）
- **Phase 2**：laid-line 标注
  - 图像以自然方向显示（不旋转），因为 laid line 在 portrait 图像中是**水平**的
  - 左键点击放置**横向**标记线（Y 坐标），D 删除最近标记，滚轮缩放，中键/右键拖动平移
  - 实时显示 N、mean_gap_px、lpc 估计值
  - 保存到 `results/<serial>/manual_gt.json`（含 `y_positions_px`、`lpc_mean`、`lpc_median`、`cm_per_px`）

**Bug 修复（当场发现）**：初版错误地将标记线做成竖线（X 坐标），因为混淆了"旋转使帘线变竖后标 X"的逻辑——实际图像未旋转时帘线是横的，应该标 Y 坐标放横线。修复后改为直接用自然方向图像 + 横线标记。

**`scripts/compare_vs_manual_gt.py`** — 对比表：手动 GT vs pipeline vs spreadsheet

### 手动标注结果（全 9 个 folio）

每个 folio 标记 10–16 根帘线，以 mean Y 间距计算 lpc：

| 数据集 | N标记 | 手动 lpc | Pipeline lpc | 误差(vs手动) | Spreadsheet |
|---|---|---|---|---|---|
| Kk1-5_f5v | 14 | **9.35** | 8.97 | −4.1% | 9.0 |
| Kk1-5_f9v | 14 | **9.35** | 9.04 | −3.3% | 9.0 |
| Hh2-12_f190 | 14 | **8.91** | 8.91 | **0.0%** | 10.0 |
| Ee5-22_f328r | 14 | **7.57** | 7.19 | −5.1% | 10.0 |
| Ff2-6_f140r | 11 | **10.16** | 9.43 | −7.2% | 11.0 |
| Ff4-9_f42r | 16 | **5.90** | 5.27 | −10.7% | 6.0 |
| Ff4-15_f24r | 10 | **12.08** | 3.92 | −67.6% ❌ | 13.5 |
| Hh2-10_f24r | 10 | **11.52** | 11.20 | −2.8% | 13.5 |
| Ii3-8_f135v | 12 | **7.40** | 6.75 | −8.8% | 9.0 |

### 关键结论

1. **Spreadsheet GT 不可靠**：Hh2-12（10 vs 8.91）、Ee5-22（10 vs 7.57）、Hh2-10（13.5 vs 11.52）等偏差极大，手动量才是真值。

2. **Pipeline 系统性低估 3–11%**（Ff4-15 除外）。方向一致，说明 FFT 检测的 `dominant_period_px` 系统性偏大（找到的频率偏低）。

3. **Ff4-15_f24r 完全失败（−68%）**：手动 GT 12.08/cm 接近 spreadsheet 13.5，pipeline 只报 3.92——检测到的是 chain line（间距宽），不是 laid line。需要单独处理。

### 涉及文件

- `scripts/gt_builder.py`（新建）
- `scripts/compare_vs_manual_gt.py`（新建）
- `results/*/manual_gt.json`（全 9 个，新建）

### 下一步

- 分析 pipeline 系统性低估 5–10% 的根因（FFT period 偏大）
- 修复 Ff4-15_f24r 检测到 chain line 而非 laid line 的问题

---

## 2026-05-24 — 系统性误差分析 + Spreadsheet GT 可靠性审查

### 背景

上一 session 结束时手动框选了所有 9 个数据集的 bbox 并重跑了 pipeline。本 session 追查结果变差的原因，并深入分析系统性偏低的来源。

### fov_width_cm 换算验证

`select_bbox.py` 的反推公式代数上等价于直接用已知 paper_mm，对所有数据集数值验证吻合（误差 < 0.01%）。

结果变差的真实原因（以 Hh2-12_f190 为例）：改前 fov=22.0 无 crop_roi → pipeline 把全图 6132px 当 22cm（错误比例尺）；改后正确。之前的"好结果"是比例尺错误带来的虚假精度。

### pipeline 估计 lpc 的完整方法

**Step 1** — 2D FFT → `dominant_period_px`（在 `period_range_px` 内找峰值频率）  
**Step 2** — Gabor 在检测到的 period 处滤波 → 干净的 `dominant_signal_1d`  
**Step 3** — `peaks_from_signal`：平滑后找局部极大值，`min_dist = 0.7 × period_px`  
**Step 4** — `gaps_px = np.diff(peaks_x)`，`lpc_mean = 1 / (mean(gaps_px) × cm_per_px)`

**根源**：FFT 检测到的 `dominant_period_px` 决定整条链路。

### 系统性偏低分析

当前结果（手动 bbox 后重跑）：

| 数据集 | GT | lpc_mean | error |
|---|---|---|---|
| Kk1-5_f5v/f9v | 9.0 | 8.97/9.04 | **< 0.5%** ✓ |
| Hh2-12_f190 | 10.0 | 8.91 | −10.9% |
| Ee5-22_f328r | 10.0 | 7.19 | −28.1% |
| Ff2-6_f140r | 11.0 | 9.43 | −14.3% |
| Ff4-9_f42r | 6.0 | 5.27 | −12.2% |
| Ff4-15_f24r | 13.5 | 3.92 | −71% **FAILED** |
| Hh2-10_f24r | 13.5 | 11.20 | −17.1% |
| Ii3-8_f135v | 9.0 | 6.75 | −25.0% |

**Wire/period 比值**：所有数据集均在 0.35–0.41 之间高度一致，说明 pipeline 在每张图上检测的是同一类物理结构（帘线），内部自洽。

**FFT period vs GT 期望对比**：Kk1-5 完全吻合（ratio=0.986），其他数据集 FFT 找到的 period 比 GT 期望值大 10–27%，说明不是 gap-mean 膨胀问题，而是 FFT 本身找到了错误（或不同的）频率。

### Spreadsheet GT 可靠性审查

查询 `Detailed Paper Info` sheet，所有 folio 均在正确 locus 区间内：

| 数据集 | 匹配 locus | Spreadsheet lpc |
|---|---|---|
| Hh2-12_f190 | 176r–197v | 10 |
| Ee5-22_f328r | 324,325–328,329 | 10 |
| Ff2-6_f140r | 138r–141v | **11 或 10（两行冲突）** |
| Ff4-9_f42r | 1r–83v | 6 |
| Ff4-15_f24r | 23r–40v | 13 (14? hard to tell) |
| Hh2-10_f24r | 1r–36v | **13(14) 或 12(14)（两行冲突）** |
| Ii3-8_f135v | 133r–144v | 9 |
| Kk1-5 | 不在 spreadsheet | 手动数 |

**Spreadsheet 可靠性问题**：多处有冲突行、明确标注不确定性（"hard to tell"）。

**关键验证**：用户肉眼看 Ee5-22 约 8/cm，与 pipeline 的 ~7.9 lpc 一致，而非 spreadsheet 的 10。结论：spreadsheet 本身计数可能有误，pipeline 对 Kk1-5（直接手动验证）完全准确，对其他数据集的检测结果很可能也是正确的。

### 待实施

用户提出了新的解决方案（待下一 session 说明）。

### 涉及文件

- `data/manuscript_db/Master Paper Spreadsheet.xlsx`（GT 来源）
- `results/pipeline_vs_spreadsheet.json`（最新对比）
- `scripts/save_comparison.py`（GT 硬编码需更新）

---

## 2026-05-23 — feat(detect): fix auto_detect_line_dir + wire_is_darker + auto crop_roi integration

### 背景

本次 session 延续上一次对 `auto_detect_line_dir` 的修复工作（上一 session 实现了 projection autocorrelation，但未经测试）。测试发现三个数据集全部检测错误，经过诊断和重写，最终完全修好。

### 问题诊断

测试初始结果：

| 数据集 | 错误结果 | 期望 |
|---|---|---|
| Kk1-5_f5v | −62° | ~0° |
| Kk1-5_f9v | 77° | ~2° |
| Hh2-12_f190 | −78° | ~0° |

加入谱功率诊断后发现：

- **根本原因 1 — 归一化自相关被低频趋势欺骗。** 归一化 AC = AC[lag]/AC[0]，若投影信号存在宽缓的低频趋势（背景渐变等），AC[0]（总功率）被拉大，所有短 lag 处的 AC 接近 1.0，算法无法区分真正的帘线周期性。
- **根本原因 2 — 二次谐波竞争。** 帘线周期 T 的半周期 T/2（二次谐波）在某些角度下 AC 值高于基频，导致误检。
- **对照：谱功率（绝对 |FFT|²）在正确角度处比任何错误角度高 100–1000 倍**，对低频趋势免疫。

### 修复内容

**`paperdrm/stage3_detect/simple_detector.py` — `auto_detect_line_dir`**

将 scoring 从归一化自相关改为目标频带内的峰值谱功率：

```python
# 旧（autocorrelation）
n2 = 2 * len(proj)
F  = np.fft.rfft(proj, n=n2)
ac = np.fft.irfft(F * F.conj())[:len(proj)]
if ac[0] > 0: ac /= ac[0]
scores[i] = float(ac[lag_min: lag_max + 1].max())

# 新（spectral power）
F     = np.fft.rfft(proj)
power = np.abs(F) ** 2
freqs = np.fft.rfftfreq(len(proj))
band  = (freqs >= freq_lo) & (freqs <= freq_hi)
scores[i] = float(power[band].max())
```

**`configs/Kk1-5_f5v.yaml`**

将手动指定的 `line_dir_deg: 0.0` 改为 `auto_line_dir: true`。

### 修复后测试结果

| 数据集 | 检测角度 | 期望 | 线密度 | SNR（vs 次优角度） |
|---|---|---|---|---|
| Kk1-5_f5v | **1°** ✅ | ~0° | 9.2 线/cm | 3.0x (4.7 dB) |
| Kk1-5_f9v | **2°** ✅ | ~2° | 9.1 线/cm | 10.5x (10.2 dB) |
| Hh2-12_f190 | **0°** ✅ | ~0° | 10.2 线/cm | 1.6x (2.2 dB) |

f190 的 SNR 最低（1.6x），次优方向均为 90°（竖向），可能来自 JPEG 压缩伪影或图像纹理。目前三个数据集均正确通过，但若遇到帘线对比度更低的图像需留意。

三个 config 现在全部使用 `auto_line_dir: true`，可完全取代手动指定方向。

### 涉及文件

- `paperdrm/stage3_detect/simple_detector.py`（`auto_detect_line_dir` 重写）
- `configs/Kk1-5_f5v.yaml`（改用 `auto_line_dir: true`）
- `results/Kk1-5_f5v/`（用自动检测结果刷新）

### Commit

`caecc39` feat(detect): replace projection autocorrelation with spectral power in auto_detect_line_dir

---

## 2026-05-22 — feat(validation): synthetic phantom accuracy characterisation

Added `scripts/phantom_synthetic.py`: generates images with known period / direction / wire-width (Gaussian comb, dark dips on bright background, matching real grazing/MSI data), runs the full detector pipeline, and measures detection error across four parameter sweeps. Results saved to `results/phantom/`.

**Key findings:**

| Metric | Result |
|---|---|
| Period accuracy (T=10–50 px) | <2% error |
| Period SNR robustness | error unchanged SNR 2→50 (quantisation-limited, not noise) |
| Direction accuracy (60°–90°) | ≤1° error; +3° boundary effect at exactly 90° |
| Wire-width accuracy (σ=2–5 px, T~24 px) | −10% to +22% systematic bias |

**Limitations documented:**
- T≥60 px with thin wires (σ/T≤0.03): FFT picks second harmonic → −50% period error
- Wire-width model overestimates +46–75% at intermediate periods (T=18–28 px, σ=2 px); this range covers typical manuscript data → wire-width numbers should be treated with caution

**Note on dataset 10:** already identified as a physical phantom with known wire spacing; pipeline results against that dataset provide complementary real-hardware validation.

---

## 2026-05-22 — fix(detector): overlay rotation direction reversed (auto_detect_line_dir)

**Bug.** Overlay red lines leaned in the opposite direction from the actual laid lines in the base image (base: higher on right; overlay: higher on left — mirror image).

**Root cause.** In `auto_detect_line_dir` (`simple_detector.py`), the 2D FFT spectral angle was computed as `arctan2(FY, FX)` where `FY` is the row-frequency axis — image coordinates with y pointing down. But `line_dir_deg` is defined in display coordinates (y pointing up). The sign mismatch caused the detected angle to be the negation of the true tilt (e.g. −2° instead of +2°).

For nearly-vertical lines (small tilt), `rot_angle = 90 − line_dir_deg` becomes ≈178° instead of ≈2°. The period detection still worked (a nearly-vertical line rotated by ≈180° stays nearly vertical), but `overlay_grid` applied the inverse rotation in the wrong direction, flipping the overlay tilt.

**Fix.** Changed `arctan2(FY, FX)` → `arctan2(-FY, FX)` in `auto_detect_line_dir`. This converts the FFT row-frequency from image-y to display-y convention before computing the spectral angle. After the fix `auto_detect_line_dir` returns +2.0° for the f9v dataset (was −2.0°); overlay now matches the base image.

**File changed.** `paperdrm/stage3_detect/simple_detector.py` line 303.

---

## 2026-05-21 — chore: scale grid + HTML report exclusion for f9v

- `.gitignore`: suppress `results/**/report_*.html` (50–60 MB each, over GitHub's 50 MB soft limit). Untracked already-committed HTML files from Kk1-5 runs.
- `results/Kk1-5_f9v/scale_grid_1cm.png`: 1 cm × 1 cm grid overlay for physical scale verification (197 × 273 mm paper, 223 × 227 px/cell).

---

## 2026-05-21 — chore(results): archive Kk1-5_f5v and Kk1-5_f9v MSI runs

First real manuscript runs using the single-image track on MSI transmitted-light (TX940IR) images of two folios of MS Kk.01.05 pts 5–6. Line direction set to `line_dir_deg=0` (horizontal laid lines visible in transmitted light). ROI crop derived from `detect_paper_roi.py`.

---

## 2026-05-19 — feat(pipeline): MSI single-image support + detect_paper_roi utility

**MSI single-image support (`52f85cc`).**
- `Settings`: added `line_dir_deg` field (default 90.0) so per-config laid-line orientation is yaml-driven; single-image track reads it instead of hardcoding 90°.
- `Settings`: auto-derive `data_root` from `data_serial` (`data/drp/<serial>`) when not explicitly set.
- `main.py`: single-image track now auto-activates when `image_path` is set in yaml, removing dependency on `DETECTOR_TRACK` constant.
- `image_io`: default image folder changed from `paths.processed` to `paths.raw`.
- `fetch_dataset`: restructured output to `data/drp/<serial>/raw/`; create `processed/`, `cache/`, `background/` subdirs; added `data_serial` to `sample.yaml`.
- `infer_drp_config`: made `--folder` required.

**detect_paper_roi utility (`42d77f5`).**
- New `scripts/detect_paper_roi.py`: scans horizontal and vertical cross-sections to locate the gray paper region between black background and bright lightbox frame, then prints a ready-to-paste `crop_roi` line for `exp_param.yaml`.

---

## 2026-05-19 — feat(pipeline): single_image track (`112ade0`)

Lightweight track that bypasses `ImagePack` and the DRP stack entirely, accepting a single image via `--image <path>` or `image_path:` in yaml.

Pipeline: Gaussian-blur background subtraction (optional, σ=100) → ROI crop with `fov_width_cm` auto-scaling → `period_range_cm`→px conversion → `detect_laid_lines_simple` (FFT + Gabor) → full Stage-5 evaluation (gap distribution, fit quality, wire width, self-contrast; split-half skipped) → overlay + archive + HTML report.

---

## 2026-05-18 — feat + fix: period_range_px wiring, phase auto-correction, report cards (`b6d727a`, `06c145e`, `7b3bddd`)

**period_range_px wiring (`b6d727a`).** Passed `settings.period_range_px` into `stage_detect_multi_phi`, `stage_split_half`, and `stage_detect_simple` so the search window is fully yaml-driven. `exp_param.yaml` gained `period_range_px=[8,50]` to avoid a spurious 2nd-harmonic peak at ~73 px and lock onto the true ~41 px period for dataset 2b (~11.6 lines/cm).

**Half-period phase auto-correction (`06c145e`).** After computing the weighted-mean phase across phi images, sample the representative image at the detected grid vs. the half-period-shifted positions. If the grid lands on the brighter side, shift `phi_mean` by π and recompute `grid_x`. Makes absolute grid position independent of which phi happens to be the anchor. `phase_auto_corrected` key added to result dict.

**Report measurement cards (`7b3bddd`).** HTML reports now open with two prominent blue-bordered metric cards: Laid Line Spacing (mean in mm, 95% t-CI from gap distribution) and Wire Shadow Width / FWHM (segment-median in mm, 95% t-CI from segments).

---

## 2026-05-17 — run(pipeline): first real manuscript dataset (data_serial=10)

**Dataset.** New manuscript data loaded into `data/processed/` + `data/background/`.
Grid: 72 phi × 12 theta = 864 images, each 2160 × 4096 px.
Config: `theta_min_deg=30`, `fov_width_cm=8.65`, `angle_slice=(2,2)` → 36φ × 4θ = 144 images
actually loaded. `data_serial=10`.

**Memory fix.** Loading all 864 images (≈14.5 GB) caused OOM during background subtraction.
Fixed by pre-selecting image paths with `apply_angle_slice` + `apply_theta_min_filter` *before*
`load_images`, reducing peak load to 144 images (≈1.2 GB). Background images now loaded one
at a time (streaming), never held in memory simultaneously.
Changed files: `stage0_loader/image_io.py` (add `load_images_from_paths`),
`stage0_loader/imagepack.py` (pre-filter paths, streaming bg subtraction).

**Results (multi_phi track, 36 phi images).**

| metric | value | notes |
|---|---|---|
| period | 56.11 px → **8.44 lines/cm** | matches data_serial=9 (8.55) within 1.3% |
| R_raw → R_aligned | 0.369 → **0.935** | polarity correction effective |
| polarity-flipped phi | 18/36 (50%) | same rate as before; multi_phi handles automatically |
| split-half std | **0.000 px** (200 splits, half=18) | perfect stability (36φ > 20φ previously) |
| wire FWHM (segment median) | 17.91 px = **0.378 mm** | |
| self-contrast z | **−3.53**, contrast = −11.55% | laid lines appear as *bright* lines in this data |

**Note on polarity.** Self-contrast z is negative (contrast_rel = −11.55%), meaning grid
columns are *brighter* than background — opposite of data_serial=9 (z=+2.27, dark lines).
The multi_phi polarity correction still works (R_aligned=0.935) because it aligns phases
internally; `wire_is_darker=True` is the initial assumption but the anchor-based flip
corrects the consensus. Physical interpretation: this manuscript's laid-line wires reflect
more light in the grazing geometry used.

**Result files saved (root).**
`interval_distribution.json`, `fit_quality.json`, `wire_width_stats.json`,
`split_half_stability.json`, `self_contrast.json`, `laid_lines_overlay.png`,
`laid_lines_overlay_bands.png`, `split_half_stability.png`, `self_contrast.png`,
`wire_width_segments.png`.

**Follow-up.** User switching to another dataset next. Per-dataset result archiving
(e.g. `results/<serial>/`) not yet implemented — current JSONs will be overwritten.

---

## 2026-05-17 — feat(stage0_loader): infer DRP config from filenames + Google Drive fetch

**Discussed.** Planning the next IIB direction: validate pipeline
robustness on *other* datasets (multi-phi numbers so far rest entirely on
data_serial=9 internal consistency, no external reference). Phantom data
was discussed as the cheapest path to ground-truth wire-width σ but
deferred; this entry covers the prerequisite — making it cheap to point
the pipeline at a new dataset without hand-writing 6 acquisition fields
into yaml. Pain point: filenames already encode (phi, theta), but
`exp_param.yaml` still required `th_min/max/num` and `ph_min/max/num`
plus `data_serial` per sample, of which only `fov_width_cm` and
`theta_min_deg` are genuinely per-sample knobs.

**Why.** Three frictions blocked "try another dataset":
  1. Six acquisition fields per yaml are redundant with the data on disk.
  2. yaml expected one global dataset; multi-dataset workflow had no
     story for `data_serial`, background folders, or sample-level config.
  3. Downloading from Google Drive + writing the yaml by hand for every
     new sample was the slowest part of trying new data.

**How.** Three layered steps, each runnable independently:

  1. **DRP config inference from filenames** — `<phi>_<theta>.<ext>`
     regex with leading-zero tolerance, set-based dedup,
     uniform-step + complete-grid validation in strict mode. Returns a
     validated `DRPConfig` and an `InferenceReport` for diagnostics.

  2. **Settings/ImagePack integration** — `Settings.drp` may now be
     None; `Settings.from_yaml` enforces all-or-nothing on the six acq
     keys (partial set raises with the missing field list). When yaml
     omits them, `ImagePack` populates `DRPConfig` from inference;
     otherwise it cross-checks yaml vs inferred and errors on any field
     disagreement (`verify_drp_match`, exact yaml=X inferred=Y). The
     `data_serial_hint` field carries the yaml's `data_serial` through
     until inference attaches it.

  3. **Google Drive fetch + dataset bundle** — `scripts/fetch_dataset.py`
     downloads (gdown) or copies (`--from-local`) into
     `data/raw/<serial>/`, optionally pulls a background subfolder, runs
     the same inference for download validation, then writes a
     per-dataset `sample.yaml` (folder, fov_width_cm, optional
     theta_min_deg, notes.{source,fetched,note}). `ImagePack` falls back
     to `<image_folder>/background/` before the legacy global
     `data/background/`, and `data_serial` is taken from folder basename
     when yaml/hint provide none (numeric → int, otherwise string,
     generic names like `raw`/`processed` ignored). `main.py` gained a
     `--config` arg so per-sample yamls can be loaded directly.

**Changes.**
- `paperdrm/stage0_loader/inference.py` (new):
  `infer_drp_config_from_folder`, `InferenceReport`, `verify_drp_match`.
- `paperdrm/stage0_loader/settings.py`: `Settings.drp` optional;
  `data_serial_hint` field; `resolve_drp_from_yaml` helper used by both
  `from_yaml` and `ImagePack`; `validate` skips drp-dependent checks
  when drp is None.
- `paperdrm/stage0_loader/imagepack.py`: post-folder-resolve, run
  inference for populate-or-cross-check; `_data_serial_from_folder`
  fallback; background folder lookup prefers sibling then global.
- `main.py`: argparse entry-point with `--config` (default
  `exp_param.yaml`).
- `requirements.txt`: add `gdown`; rewrite from UTF-16 LE back to UTF-8
  (regression from a previous cleanup).
- `scripts/infer_drp_config.py` (new): CLI that runs inference on a
  folder and diffs the result against an existing yaml; exit 1 on any
  field mismatch.
- `scripts/fetch_dataset.py` (new): Google Drive / local ingest with
  inference-based validation and sample.yaml generation.

**Verification.**
- Current `data/raw/` (480 jpgs): inference produces
  `th_min=10 th_max=65 th_num=12 ph_min=0 ph_max=351 ph_num=40`,
  identical to existing yaml.
- Padded filenames (`009_010.jpg`): parsed identically (`\d+` +
  `int()`).
- Partial yaml (only `th_min`, `th_max`): `Settings.from_yaml` raises
  with the four missing field names.
- Mismatched yaml (`ph_num: 39` vs 40 files in folder): `ImagePack`
  raises from `verify_drp_match` with `ph_num: yaml=39 inferred=40`.
- Minimal yaml (no acq fields, just `data_serial`, `theta_min_deg`,
  `fov_width_cm`): `ImagePack` infers cleanly, identical
  `base_config` and `num_images_after_filter=320` to the full-yaml run.
- `fetch_dataset.py --from-local data/raw --serial test_local
  --copy-mode symlink --fov 8.65 --theta-min 30`: 480 symlinks +
  sample.yaml; subsequent `ImagePack` run logs
  `data_serial inferred from folder name: 'test_local'` and reproduces
  the same pipeline state.
- gdown-missing path: clear `pip install gdown` message, exit 1.
- Overwrite guard: existing non-empty target → exit 2 unless
  `--overwrite`.

**Follow-up.**
1. Real Google Drive end-to-end test still pending — gdown is in
   requirements but the user's venv was relocated from
   `/Users/kaedez/Documents/DRP-Processing/` (broken shebangs); after
   `pip install gdown` succeeded, no actual Drive URL has been run
   through the script yet.
2. Single-zip mode (`--zip-url`) was discussed as a more reliable
   alternative to gdown folder downloads but not implemented.
3. Cache (`data/cache/data.dat` + `data_config.yaml`) is still global;
   switching datasets triggers a rebuild. Per-dataset cache would
   avoid the ~100s rebuild cost when alternating between samples but
   isn't needed for single-active-dataset workflows.
4. Phantom-data validation for wire-width σ is still the unfinished
   prerequisite for publishing the multi_phi results — not in this
   commit's scope but next on the IIB roadmap.

---

## 2026-05-16 — feat(pipeline): multi-phi spectral aggregation + split-half / self-contrast evaluators

**Discussed.** Reintroduced DRP information into the predictor without
falling back to the bias-prone LEGACY track. The SIMPLE detector uses
exactly one grazing image and throws away the other ~19 per-phi
observations; the LEGACY detector uses all of them but via a trig-mask +
patchwise Gabor path that detects half the true period. New track sits
between: keeps the simple-detector machinery, but feeds it averaged
information from N phi images.

**Why.** Single-image weaknesses: (1) ink/text occludes whichever
section of laid lines that frame happens to capture; (2) SNR at the
period peak is just whatever that one phi gives. The laid-line peak is
phi-stationary (wires sit at the same physical pixels in every image),
while ink reflectance and texture noise vary across phi. Power-spectrum
averaging therefore boosts the signal coherently and washes out the
contamination.

**How.**
- For each phi (steepest theta), compute the radial power spectrum
  P_phi(f) by integrating the 2D FFT magnitude along the line direction.
- Normalise each P_phi (default: divide by total in-band power so a
  bright phi doesn't dominate) and sum across phi → P_agg(f). Pick the
  peak on P_agg, not on any single image.
- Period locked from P_agg, per-image phase at that period is amplitude-
  weighted into a circular mean. Wire-width / broadband signal use the
  highest-weight phi as representative.
- **Polarity correction at the phase-aggregation step**: per-image phases
  are first computed with the `wire_is_darker` convention, then aligned
  to the highest-weight phi (anchor). Any per-image phase whose
  anchor-relative offset exceeds π/2 is flipped by π before the weighted
  circular mean is taken. This handles the (real, observed) phenomenon
  where some phi values show wires as *brighter* than substrate, giving
  a π-shifted phase that would otherwise destroy the circular-mean
  consensus.

Two evaluators added since there is no ground truth:
- **Split-half stability**: random A/B partitions of the 20 phi images,
  aggregate each half, compare period estimates. Quantifies "is the
  aggregated detector measuring something stable?"
- **Self-consistency contrast**: in the spatial domain, compare mean
  intensity at predicted grid columns vs. half-period-shifted columns.
  Independent of FFT — corroborates that the grid landed on real wires.

**Changes.**
- `paperdrm/stage3_detect/multi_phi_detector.py` (new): `collect_grazing_per_phi`,
  `aggregate_radial_power`, `detect_laid_lines_multi_phi`. Returns the
  same keys as `detect_laid_lines_simple` plus per-image phases/weights,
  per-image polarity-flip mask, anchor index, raw vs. aligned phase
  resultant length, and per-image spectra.
- `paperdrm/stage5_evaluation/split_half.py` (new): `split_half_period_stability`
  + print/save/plot. Reuses pre-computed per-image normalised spectra so
  200 splits is cheap.
- `paperdrm/stage5_evaluation/self_contrast.py` (new): `self_consistency_contrast`
  + print/save/plot. Works on any track's output (multi-phi or simple).
- `main.py`: replaced binary `USE_SIMPLE_DETECTOR` with string
  `DETECTOR_TRACK = "multi_phi" | "simple" | "legacy"`. Added
  `stage_detect_multi_phi`, `stage_split_half`, `stage_self_contrast`.
  SIMPLE track now also runs self-contrast (writing to
  `self_contrast.simple.{json,png}`) for apples-to-apples comparison.
- Stage 3 / Stage 5 `__init__.py`: export new symbols.

**Numbers (data_serial=9, 20 phi × 4 theta after filter).**
- Period 55.35 px → 0.1169 cm → **8.55 lines/cm** (consistent with the
  paleographic 8–14 lines/cm range).
- Split-half (200 splits, half=10): diff std = **0.684 px** (CV 1.2%);
  100% of splits agree within ±1 px, 19% within ±0.5 px.
- Self-contrast: z = **+2.27**, contrast_rel = +5.55% — grid columns
  are systematically darker than half-period-shifted columns.
- Phase coherence: **R_raw = 0.280 → R_aligned = 0.981** after anchor-
  based polarity correction (8/20 phi were polarity-flipped). The first
  run without correction had R = 0.28 (circ_var 0.72), confirming the
  flip hypothesis empirically.

**Head-to-head on all 5 detectors (data_serial=9, ref = phi index 0).**
Reproducible via `python scripts/compare_detectors.py`.

| detector       | period_px | lines/cm | R²(k=4) | med\|z\| | med z | frac+ | sc_ref | time_s |
|----------------|----------:|---------:|--------:|--------:|------:|------:|-------:|-------:|
| radial_fft     |    56.110 |    8.439 |  0.0203 |    1.55 | −0.56 |   50% |  +1.29 |   0.25 |
| gabor_full     |    30.000 |   15.784 |  0.0003 |    0.19 | −0.06 |   40% |  −0.49 |  66.19 |
| gabor_patches  |    26.000 |   18.213 |  0.0002 |    0.34 | +0.17 |   65% |  +0.40 | 252.64 |
| simple         |    56.110 |    8.439 |  0.0203 |    1.52 | −0.54 |   50% |  +1.27 |   1.17 |
| **multi_phi**  |    55.351 |    8.555 |  0.0097 |    1.13 | +0.19 |   55% |  −1.22 |   5.23 |

Reading:
- `gabor_full` / `gabor_patches`: still period/2-biased (30 / 26 vs. the
  correct ≈55), `med|z|` essentially noise-level — confirms the legacy
  track is unrecoverable on this data.
- `radial_fft` ≈ `simple` ≈ `multi_phi` on lines/cm (8.44 vs 8.44 vs
  8.55, agreement within 1.4%). The radial-FFT family is the only
  family producing a usable answer.
- `R²` and `sc_ref` are computed on phi=0's broadband signal, so methods
  whose period/phase were derived from phi=0 score higher by
  construction. `med|z|` (median |z| across all 20 phi) is the
  polarity-robust apples-to-apples score; `multi_phi` is slightly lower
  because its grid is a cross-phi consensus rather than phi=0-optimal.
- `frac+` ≈ 50% for the single-image methods independently corroborates
  that ~half the phi observations have polarity opposite the assumed
  `wire_is_darker`. This is the same phenomenon the multi-phi anchor
  alignment fixes internally.

**Polarity-flip diagnostic figure.**
- `scripts/plot_polarity_flip.py`: picks the anchor (max-weight phi) and
  the highest-weight polarity-flipped phi, plots their broadband
  column-mean signals over 4 periods with the global grid overlaid.
  In data_serial=9: anchor = phi 72°, flipped exemplar = phi 0°,
  Δ_phase = −162.9° (≈ −π). Output: `polarity_flip.png`.

**Verdict on multi_phi reliability vs. SIMPLE.**

| dimension                 | SIMPLE                  | MULTI_PHI                    |
|---------------------------|-------------------------|------------------------------|
| period estimate           | 56.11 px (one image)    | 55.35 px (20 images)         |
| period confidence         | none                    | ±0.68 px (split-half CV ≈ 1.2%) |
| robustness to ink         | one image fails silently | 1/20 contribution            |
| polarity handling         | implicit single-image    | explicit anchor + π-flip     |
| wall time                 | 1.2 s                   | 5.2 s                        |
| beats legacy gabor?       | yes                     | yes                          |

The headline numerical change is small (≈1.4% on lines/cm) but multi_phi
is the first track with an *internal* reliability measure. SIMPLE could
silently return the wrong answer if its single grazing image happens to
be occluded; multi_phi cannot fail the same way.

**Follow-up.**
1. The 0.76 px gap between `simple` (56.11) and `multi_phi` (55.35) —
   is multi_phi's cross-phi consensus *more* accurate, or is phi=0
   alone? Need an independent measure (e.g. manual count over a region,
   or comparison against another phi). Split-half says multi_phi's 55.35
   is reproducible to ±0.68 px so it's not noise.
2. Wire-width still uses the representative phi only. Could average
   harmonic amplitudes `|c_n|` across phi for a multi-phi σ estimate.
3. Add peak SNR (P_agg peak / median in-band power) as a third
   reliability metric, and compare against worst-case single-phi P_phi.

---

## 2026-05-15 — feat(pipeline): wire-shadow width via Gaussian-comb harmonic fit

**Discussed.** First wire-shadow *width* estimator end-to-end. Until now the
pipeline only produced period + grid positions; this adds the third
parameter (σ / FWHM of each wire's shadow) and a parametric consistency
check against the broadband signal.

**Why.** Period + phase alone leaves the wire shape unspecified. For
paleographic comparison we need the *thickness* of the laid-line shadow
(reflects the original wire's diameter and how deep it pressed into the
mould). Also: the existing fit-quality R² was measured against the
Gabor-cleaned narrow-band signal, which by construction has no harmonics,
so high R² was uninformative.

**How.** A periodic comb of Gaussians of width σ at spacing T has Fourier
coefficients `|c_n| ~ exp(-2π²σ²n²/T²)`. So `ln|c_n|` is linear in `n²`
with slope `-2π²σ²/T²` → σ from least-squares. To avoid spectral leakage
from FFT bins at non-integer 1/T, the DTFT is sampled directly at the
exact harmonic frequencies `n/T`.

**Changes.**
- `paperdrm/stage3_detect/wire_width.py` (new): `estimate_wire_width`
  returns σ, FWHM, harmonic amplitudes, regression diagnostics, model_ok
  flag. Requires broadband signal (Gabor-cleaned input is rejected with
  a warning because harmonics are suppressed).
- `paperdrm/stage3_detect/simple_detector.py`: factored out
  `_broadband_signal_1d` (always computed); `detect_laid_lines_simple`
  now also returns `broadband_signal_1d` and the wire-width fields. New
  `overlay_grid_bands` draws filled FWHM bands instead of 1-px lines.
- `paperdrm/stage5_evaluation/wire_width_stats.py` (new, 326 lines):
  per-segment wire-width with two CIs (t-based mean CI + percentile
  spread). Print/save/plot helpers.
- `paperdrm/stage5_evaluation/fit_quality.py`: switched to read
  `broadband_signal_1d`; added `gaussian_comb_r2` (template with T,
  phase, σ all fixed — checks (period, phase, width) consistency
  jointly); default `n_harmonics` 2 → 4; report now carries three R²
  values with a "Gaussian-fit gap" diagnostic.
- `main.py`: loads pre-bg-subtraction raw image for overlays
  (`pick_grazing_image_raw`); `stage_evaluate` returns ww stats and
  takes `image=`; new `stage_overlay_simple_bands` uses
  segment-median FWHM.
- `.gitignore`: ignore `/wire_width_stats*.json` (matches existing
  Stage 5 report ignores).

**Follow-up.** A/B JSON artifacts (`.before`, `.prev`) are kept locally
but ignored. Real-data σ values from segment statistics still need
validation against the phantom test images planned for Stage 5.

---

## 2026-05-15 — refactor(scripts): simplify bg_blur + add freq-response comparison

**Discussed.** Replaced the multi-step background-blur in `scripts/bg_blur.py`
with a single direct Gaussian. Added a diagnostic script to verify the
substitution is frequency-equivalent.

**Why.** The old pipeline did
`GaussianBlur(σ=5) → resize 0.2× → GaussianBlur(σ=20) → resize back`,
which mixed interpolation artifacts with the lowpass. A single
`GaussianBlur(σ=100, BORDER_REFLECT_101)` is cleaner and (per the comparison
script) has near-identical frequency response on real data.

**Changes.**
- `scripts/bg_blur.py`: replaced the 4-step lowpass with one GaussianBlur
  call.
- `scripts/bg_blur_freq_compare.py` (new, 184 lines): impulse-response and
  real-image residual-spectrum comparison of OLD vs NEW; writes PNGs to
  `scripts/_bg_blur_freq_out/`.
- `.gitignore`: added `scripts/_bg_blur_freq_out/` (diagnostic outputs).

**Follow-up.** None — bg subtraction in the main pipeline now uses the
simpler call. If downstream changes regress, the comparison script lets us
diff old vs new spectra directly.

---

## 2026-05-15 — Set up conversation logging

**Discussed.** User asked to start summarizing all conversations and keep the
record in git, updated alongside each commit.

**Outcome.**
- Created this `CONVERSATION_LOG.md` at repo root.
- Auto-hook configuration via `settings.json` (Stop / PreToolUse) was declined
  by the permission classifier; instead Claude will manually update this file
  before each commit. A feedback memory was saved so this persists across
  conversations.

**Follow-up.** If user wants true automation (independent of Claude
remembering), they can authorize a `.claude/settings.json` hook change.

**Repo state at time of writing** (uncommitted, from recent work on radial-FFT
detector / fit-quality evaluator branch):
- modified: `main.py`, `paperdrm/stage3_detect/__init__.py`,
  `paperdrm/stage3_detect/simple_detector.py`,
  `paperdrm/stage5_evaluation/__init__.py`,
  `paperdrm/stage5_evaluation/fit_quality.py`, `scripts/bg_blur.py`
- new: `paperdrm/stage3_detect/wire_width.py`,
  `paperdrm/stage5_evaluation/wire_width_stats.py`,
  `scripts/bg_blur_freq_compare.py`, `wire_width_stats*.json`
- last commit: `e6c2657` feat(pipeline): add radial-FFT simple detector and
  interval/fit-quality evaluators

---

## Backfill: May 2026 commits

Historical entries reconstructed from `git log` after the logging convention
was introduced on 2026-05-15. Commit summaries below; full bodies in
`git show <hash>`.

### 2026-05-13 — `e6c2657` feat(pipeline): radial-FFT simple detector + fit-quality evaluators

Introduced a dual-track `main.py`: legacy DRP→trig→patchwise-Gabor path kept
for ablation, plus a new single-image radial-FFT path that produces an
unbiased period estimate (~56 px vs legacy ~27 px on test data — matches the
documented period/2 abs-response bias). Added Stage 5 interval-distribution
and fit-quality evaluators that work for both tracks.
Files: `main.py`, `paperdrm/stage3_detect/{__init__,simple_detector}.py`,
`paperdrm/stage5_evaluation/{__init__,fit_quality,interval_distribution}.py`.

### 2026-05-12 — `7afdc1c` feat(stage3): score-weighted consensus replaces winner-takes-all

Stage 3 previously picked the dominant period from the single highest-scoring
patch, making the global estimate fragile to outlier patches; `weight_scale`
was unused. Now accumulates `best_score ** weight_scale` across the candidate-
period grid; ties broken by single-patch best score. `weight_scale=0` →
majority vote, `1` → linear, `3` → current default, `→∞` → old behaviour.
Also fixed a consistency bug where returned `theta` came from the best patch
while `signal_1d` came from a full-image rescan, so they could disagree.
On current data the dominant 26 px is unchanged but the algorithm is now
robust to outlier single-patch scores. File: `paperdrm/stage3_detect/gabor.py`.

### 2026-05-12 — `eb7376f` feat(stage5): patch consistency diagnostics

First concrete evaluation tool (Tier 0 item 3 of the eval framework):
quantifies how much per-patch Gabor estimates agree on period and orientation,
exposing whether the dominant output is a true consensus or an outlier-driven
pick. Adds `patch_period_stats`, `patch_orientation_stats`,
`patch_consistency_report`, `print_consistency_report`,
`save_consistency_report`, `plot_patch_consistency` (3-panel figure). Wired
into `main.py` as `stage_evaluate`; JSON lands at `./evaluation_report.json`.
On current data: 105/105 valid patches, median 26 mode 26 MAD 2, but a long
tail at 28–32 px → score-weighted mean 27.6 (bimodal, not tight consensus).
Orientation well-consistent (R=0.98, circ-mean 89.9°).
File: `paperdrm/stage5_evaluation/consistency.py`.

### 2026-05-12 — `15826df` fix(scripts): update import to new stage0_loader path

`from paperdrm.paths import DataPaths` was broken by the stage-based refactor.
`DataPaths` now lives under `paperdrm.stage0_loader` (re-exported from package
`__init__`). Files: `scripts/bg_blur.py`, `scripts/rgb2grey.py`.

### 2026-05-12 — `f7d4d6b` chore: clean repo root and tighten .gitignore

Untracked two large pipeline-output PNGs (~35 MB combined) and four
exploratory notebooks (`.ipynb` now treated as personal scratch, ignored).
Rewrote `requirements.txt` as UTF-8 with actual deps
(numpy/scipy/matplotlib/opencv/tqdm/pillow/PyYAML), replacing a UTF-16 file
that had unrelated Django entries. Also ignored `*.egg-info`, `build/`,
`dist/`, `*.bak`.

### 2026-05-12 — `bcec65f` refactor: prefix pipeline subpackages with stage numbers

Renamed subpackages so pipeline order is visible in directory listing:
`loader → stage0_loader`, `drp → stage0_drp`, `features → stage1_features`,
`enhance → stage2_enhance`, `detect → stage3_detect`, `viz → stage4_viz`,
`evaluation → stage5_evaluation`. `legacy/` keeps no number. Pipeline output
unchanged (period=26 px, 18.2 lines/cm), verified end-to-end.

### 2026-05-12 — `3b157fb` refactor: reorganize paperdrm into stage-based subpackages

Reshaped package layout to mirror pipeline stages and clarify where new code
goes: `loader/` (Stage 0 data loading + DRP cache), `drp/` (DRP stack ops),
`features/` (per-pixel: direction + spherical, Stage 1), `enhance/`
(orientation→laid-line grayscale via trig masks, Stage 2), `detect/` (Gabor,
Stage 3), `viz/` (plotting split per stage, Stage 4), `evaluation/`
(placeholder for consistency/phantom/GT, Stage 5), `legacy/` (Hough, spectral
TV, ROI, ImageParam, drp_analysis). Deleted empty `src/` shim and its
`pyproject.toml` declaration. Split `drp_compute.py` →
`drp/{slicing,stack,mask}.py`; split `drp_direction.py` →
`features/{direction,spherical}.py`; pulled inline matplotlib out into
`viz/direction.py`. Rewrote `main.py` as a sequence of `stage_*` functions.
Replaced `from .X import *` wildcards with explicit `__all__`. Pipeline
output unchanged (26 px / 18.2 lines/cm), verified.

### 2026-05-12 — `6a60e48` Reroll to old data

Reverted `exp_param.yaml` and the overlay PNG to the previous data set.
