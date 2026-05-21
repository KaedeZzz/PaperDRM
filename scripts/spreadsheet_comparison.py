"""
Compare pipeline measurements against spreadsheet ground truth
for Kk.01.05 pts. 5-6 (f5v and f9v MSI datasets).

Outputs:
  results/spreadsheet_comparison.json
  results/spreadsheet_comparison.png
"""

from __future__ import annotations
import json
from pathlib import Path

import openpyxl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# 1. Read spreadsheet GT for Kk.01.05 pts 5-6
# ---------------------------------------------------------------------------

def load_spreadsheet_gt() -> dict:
    wb = openpyxl.load_workbook(
        r"data\manuscript_db\Master Paper Spreadsheet.xlsx",
        read_only=True, data_only=True,
    )
    ws = wb["Detailed Paper Info"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    col = {h: i for i, h in enumerate(header) if h}

    laid_col   = col["laidline every 1:CM"]
    chain_col  = col["distance between chainlines (gutter to edge/ top to bottom)"]
    locus_col  = col["locus from (conjugate)"]

    records = []
    for row in rows[1:]:
        shelf = str(row[4]) if row[4] else ""
        if "Kk.01.05" in shelf and ("pts. 5" in shelf or "pts.5" in shelf):
            records.append({
                "id":           row[0],
                "shelfmark":    shelf,
                "locus":        row[locus_col],
                "laid_per_cm":  row[laid_col],
                "chainline_mm": row[chain_col],
            })

    # Aggregate: mean of the two stock measurements
    lpc_vals = [r["laid_per_cm"] for r in records if r["laid_per_cm"] is not None]
    return {
        "records":         records,
        "laid_per_cm_mean": float(np.mean(lpc_vals)) if lpc_vals else None,
        "laid_per_cm_vals": lpc_vals,
        "interval_mm_mean": float(10.0 / np.mean(lpc_vals)) if lpc_vals else None,
    }


# ---------------------------------------------------------------------------
# 2. Read pipeline results
# ---------------------------------------------------------------------------

def load_pipeline(serial: str) -> dict:
    base = Path("results") / serial
    iv  = json.loads((base / "interval_distribution.json").read_text())
    ww  = json.loads((base / "wire_width_stats.json").read_text())
    sc  = json.loads((base / "self_contrast.json").read_text())
    fq  = json.loads((base / "fit_quality.json").read_text())

    phys    = iv.get("physical", {})
    ww_phys = ww.get("physical", {})
    fwhm_mm = ww_phys.get("fwhm_mm", {})
    ci      = fwhm_mm.get("ci_t", [None, None])

    # self-contrast key varies by schema version
    z   = sc.get("z_score", sc.get("contrast_z"))
    rel = sc.get("contrast_rel_pct", sc.get("contrast_rel", 0) * 100)
    r2  = fq.get("fourier_r2_k4",   fq.get("r2_with_harmonics"))

    return {
        "serial":          serial,
        "lines_per_cm":    phys.get("lines_per_cm_mean"),
        "lines_per_cm_med":phys.get("lines_per_cm_median"),
        "interval_mm":     phys.get("mean_interval_cm", 0) * 10,
        "n_peaks":         iv.get("n_peaks"),
        "fwhm_mm_median":  fwhm_mm.get("median"),
        "fwhm_mm_ci_lo":   ci[0],
        "fwhm_mm_ci_hi":   ci[1],
        "self_z":          z,
        "self_rel_pct":    rel,
        "r2_k4":           r2,
    }


# ---------------------------------------------------------------------------
# 3. Build comparison table + figure
# ---------------------------------------------------------------------------

def make_comparison(gt: dict, pipelines: list[dict]) -> dict:
    rows = []
    for p in pipelines:
        gt_lpc = gt["laid_per_cm_mean"]
        err    = ((p["lines_per_cm"] - gt_lpc) / gt_lpc * 100) if gt_lpc else None
        rows.append({
            **p,
            "gt_lines_per_cm": gt_lpc,
            "gt_interval_mm":  gt["interval_mm_mean"],
            "error_pct":       err,
        })
    return {"ground_truth": gt, "pipeline_results": rows}


def plot_comparison(comp: dict, save_path: str = "results/spreadsheet_comparison.png"):
    rows  = comp["pipeline_results"]
    gt    = comp["ground_truth"]
    gt_lpc = gt["laid_per_cm_mean"]

    serials = [r["serial"].replace("Kk1-5_", "") for r in rows]
    lpc_pip = [r["lines_per_cm"]     for r in rows]
    lpc_med = [r["lines_per_cm_med"] for r in rows]
    fwhm    = [r["fwhm_mm_median"]   for r in rows]
    fwhm_lo = [r["fwhm_mm_ci_lo"]    for r in rows]
    fwhm_hi = [r["fwhm_mm_ci_hi"]    for r in rows]
    errs    = [r["error_pct"]        for r in rows]

    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    fig.suptitle("Pipeline vs. Spreadsheet Ground Truth — Kk.01.05 pts. 5–6", fontsize=13)

    x = np.arange(len(serials))
    w = 0.35

    # Panel 1: lines/cm comparison
    ax1.bar(x - w/2, lpc_pip, w, label="Pipeline (mean)", color="steelblue")
    ax1.bar(x + w/2, lpc_med, w, label="Pipeline (median)", color="cornflowerblue")
    ax1.axhline(gt_lpc, color="crimson", lw=2, ls="--", label=f"Spreadsheet GT ({gt_lpc:.0f} lines/cm)")
    ax1.set_xticks(x); ax1.set_xticklabels(serials)
    ax1.set_ylabel("Laid lines / cm")
    ax1.set_title("Line density")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: error %
    colors = ["tomato" if e and e < 0 else "steelblue" for e in errs]
    ax2.bar(x, errs, color=colors)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(serials)
    ax2.set_ylabel("Error vs. GT (%)")
    ax2.set_title("Relative error")
    ax2.grid(axis="y", alpha=0.3)
    for i, e in enumerate(errs):
        if e is not None:
            ax2.text(i, e + (1 if e >= 0 else -2), f"{e:+.1f}%", ha="center", fontsize=9)

    # Panel 3: FWHM with 95% CI
    ax3.bar(x, fwhm, color="darkorange", label="FWHM (segment median)")
    for i in range(len(rows)):
        if fwhm_lo[i] and fwhm_hi[i]:
            ax3.errorbar(i, fwhm[i],
                         yerr=[[fwhm[i]-fwhm_lo[i]], [fwhm_hi[i]-fwhm[i]]],
                         fmt="none", color="black", capsize=5)
    ax3.set_xticks(x); ax3.set_xticklabels(serials)
    ax3.set_ylabel("Wire shadow FWHM (mm)")
    ax3.set_title("Wire width (no GT)")
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Comparison] Figure -> {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gt = load_spreadsheet_gt()
    print(f"[GT] Spreadsheet (different folios 235a/235b): {gt['laid_per_cm_vals']} lines/cm  "
          f"(mean={gt['laid_per_cm_mean']:.1f})")

    # Spreadsheet rows 235a/235b record 12 lines/cm but refer to different folios
    # within Kk.01.05 pts.5-6. Manual count on the imaged folios (f5v, f9v) = 9/cm.
    MANUAL_GT_LPC = 9.0
    gt["laid_per_cm_mean"] = MANUAL_GT_LPC
    gt["laid_per_cm_vals"] = [MANUAL_GT_LPC]
    gt["interval_mm_mean"] = 10.0 / MANUAL_GT_LPC
    gt["note"] = "GT overridden to manual count (9/cm); spreadsheet value 12 is from different folios"
    print(f"[GT] Manual count (imaged folios): {MANUAL_GT_LPC} lines/cm  "
          f"interval={gt['interval_mm_mean']:.3f}mm")

    pipes = [load_pipeline("Kk1-5_f5v"), load_pipeline("Kk1-5_f9v")]
    for p in pipes:
        print(f"[Pipeline] {p['serial']}: {p['lines_per_cm']:.3f} lines/cm  "
              f"interval={p['interval_mm']:.3f}mm  FWHM={p['fwhm_mm_median']:.3f}mm")

    comp = make_comparison(gt, pipes)

    print("\n--- Comparison ---")
    for r in comp["pipeline_results"]:
        print(f"  {r['serial']}: pipeline={r['lines_per_cm']:.3f}  "
              f"GT={r['gt_lines_per_cm']:.1f}  error={r['error_pct']:+.1f}%")

    out = Path("results/spreadsheet_comparison.json")
    out.write_text(json.dumps(comp, indent=2, default=str))
    print(f"[Comparison] JSON -> {out}")

    plot_comparison(comp)
