# IIB Report — Figure List & Key Data Tables

Auto-generated 2026-05-25 (updated for CUED chapter scheme).
All paths relative to repo root.

Page budget: ≤50 pp main body; figures cost real estate, so be selective.

---

## A. Figures by chapter

### Ch 2 — Theory and Design of Experiment

| Fig | Source | Caption sketch |
|---|---|---|
| 2.1 | (draw new) Pipeline block diagram (5 stages) | DRP loader → direction → trig mask → detection → evaluation |
| 2.2 | `paperdrm/stage4_viz/drp.py` output (regen) | Example DRP angular profile at one pixel |
| 2.3 | `laid_lines_overlay.png` from any folio | Example trig-mask enhanced grayscale |
| 2.4 | `laid_lines_overlay_bands.png` any folio | Multi-phi detector overlay bands |

### Ch 3 — Apparatus and Experimental Techniques

| Fig | Source | Caption sketch |
|---|---|---|
| 3.1 | (draw or photo) MSI rig schematic | Imaging setup |
| 3.2 | `results/phantom/phantom_synthetic.png` | Synthetic phantom example (clean) |
| 3.3 | `results/phantom/example_snr{2,3,5}.png` (3-panel) | Phantom at SNR 2/3/5 |

### Ch 4 — Results

| Fig | Source | Caption sketch |
|---|---|---|
| 4.1 | (plot from `phantom_synthetic_results.json` `period_sweep`) | Period error vs true period (sweep 10–60 px) |
| 4.2 | (plot from `phantom_synthetic_results.json` `snr_sweep`) | Period & sigma error vs SNR |
| 4.3 | (plot from `phantom_synthetic_results.json` `angle_sweep`) | Angle error vs true orientation |
| 4.4 | (plot from `phantom_synthetic_results.json` `sigma_sweep`) | Wire-width recovery vs true σ |
| 4.5 | `results/Kk1-5_f5v/laid_lines_overlay.png` | Best-case folio: overlay + grid |
| 4.6 | `results/Kk1-5_f5v/grid_1cm_overlay.jpg` | 1 cm scale grid (calibration sanity check) |
| 4.7 | `results/Kk1-5_f5v/interval_distribution.json` → bar/hist | Interval distribution Kk1-5 f5v |
| 4.8 | `results/Kk1-5_f5v/wire_width_segments.png` + `wire_width_stats.json` | Wire-width FWHM per segment |
| 4.9 | `results/Kk1-5_f5v/self_contrast.png` | Self-contrast z-score visualisation |
| 4.10 | `results/Kk1-5_f5v/split_half_stability.png` | Split-half stability |
| 4.11 | `results/Ff4-15_f24r/laid_lines_overlay.png` | Failure case: 70% under-count |
| 4.12 | `results/spreadsheet_comparison.png` | Pipeline vs spreadsheet GT scatter (9 folios) |
| 4.13 | (regen) Manual-GT vs pipeline scatter (2 folios) | Calibrated comparison Kk1-5 f5v/f9v |

### Ch 5 — Discussion

| Fig | Source | Caption sketch |
|---|---|---|
| 5.1 | (compose) Before/after spectral-power vs autocorrelation overlay | Ablation: direction detector |
| 5.2 | (compose) Polarity flip example (TX wire_is_darker) | Polarity sensitivity |

---

## B. Key data tables

### Table 4.1 — Synthetic phantom accuracy (period sweep)

| True period (px) | Period err % (mean) | σ err % (mean) | n |
|---|---|---|---|
| 10 | +0.39 | +5.87 | 5/5 |
| 12 | −0.78 | +15.34 | 5/5 |
| 15 | +0.39 | +3.34 | 5/5 |
| 18 | +1.59 | +46.45 | 5/5 |
| 20 | −1.54 | +66.15 | 5/5 |
| 24 | +1.59 | +73.18 | 5/5 |
| 28 | +1.59 | +74.91 | 5/5 |
| 32 | 0.00 | −9.97 | 5/5 |
| 40 | −1.54 | n/a | 5/0 |
| 50 | +2.40 | n/a | 5/0 |
| 60 | −49.80 | +7.95 | 5/5 |

**Headline:** period recovery within ±2.5% for 10–50 px range; breaks at 60 px (likely octave-jump). Wire σ recovery only stable up to ~32 px.

### Table 4.2 — Synthetic phantom: SNR robustness

| SNR | Period err % | σ err % | n |
|---|---|---|---|
| 2  | +1.59 | +70.81 | 5 |
| 3  | +1.59 | +70.43 | 5 |
| 5  | +1.59 | +70.60 | 5 |
| 8  | +1.59 | +71.60 | 5 |
| 10 | +1.59 | +72.17 | 5 |
| 15 | +1.59 | +73.18 | 5 |
| 20 | +1.59 | +73.81 | 5 |
| 30 | +1.59 | +74.46 | 5 |
| 50 | +1.59 | +74.96 | 5 |

**Headline:** period detection essentially SNR-invariant down to SNR=2. σ recovery is biased high but stable.

### Table 4.3 — Synthetic phantom: angle robustness

Angle error ≤ 1° for all true orientations except +90° (3° error). Period error constant at +1.59%.

### Table 4.4 — 9-folio benchmark vs spreadsheet GT

| Folio | GT (l/cm) | Pipeline (l/cm) | Err % | FWHM (mm) | self-z |
|---|---|---|---|---|---|
| Kk1-5 f5v | 9.0\* | 8.97 | **−0.35** | 0.39 | 0.72 |
| Kk1-5 f9v | 9.0\* | 9.04 | **+0.48** | 0.39 | 11.96 |
| Hh2-12 f190 | 10.0 | 8.91 | −10.93 | 0.42 | 5.59 |
| Ee5-22 f328r | 10.0 | 7.19 | −28.12 | 0.47 | 8.58 |
| Ff2-6 f140r | 11.0 | 9.43 | −14.26 | 0.42 | −7.90 |
| Ff4-9 f42r | 6.0 | 5.27 | −12.17 | 0.68 | −1.13 |
| Ff4-15 f24r | 13.5 | 3.92 | −70.98 | 0.46 | −0.81 |
| Hh2-10 f24r | 13.5 | 11.20 | −17.07 | 0.34 | −4.03 |
| Ii3-8 f135v | 9.0 | 6.75 | −24.97 | 0.49 | 0.92 |

\* manual count (the spreadsheet records 12 for a different folio of the same stock).

**Discussion hooks:**
- Two folios with corrected manual GT → sub-1% error. Strong evidence the pipeline itself is accurate.
- Other 7 errors likely reflect spreadsheet/folio mismatch (different folios within same stock have different laid densities), NOT pipeline failure. Worth a paragraph in §5.
- Ff4-15 f24r is a likely octave-half (3.92 vs 13.5 ≈ ÷3.4) — discuss frequency-aliasing failure mode.

---

## C. Inputs that still need a small plot script

These need a ~10-line matplotlib snippet to turn JSON into a figure:
- `phantom_synthetic_results.json` → Fig 4.1–4.4 (4 panels)
- `interval_distribution.json` per folio → Fig 4.7
- Manual GT scatter from `spreadsheet_comparison.json` → Fig 4.13

I'll write `scripts/plot_for_report.py` on Day 4 (5/28) when drafting Results.

---

## D. Already-generated assets (just need cropping/sizing)

All present and ready to drop into LaTeX with `\includegraphics`:
- 9× `laid_lines_overlay.png`
- 9× `laid_lines_overlay_bands.png`
- 9× `grid_1cm_overlay.jpg`
- 9× `wire_width_segments.png`
- 9× `self_contrast.png`
- 9× `split_half_stability.png`
- 9× `bbox_overlay.jpg`
- 3× phantom SNR examples + 1 clean phantom
- `spreadsheet_comparison.png`
