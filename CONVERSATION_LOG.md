# Conversation Log

This file records summaries of all working conversations with Claude on PaperDRM.
Updated before each git commit so the history of decisions and intent stays in-repo.

Format: newest entry on top. Each entry includes date, what was discussed/changed,
and any open follow-ups.

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
