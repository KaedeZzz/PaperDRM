"""
Generate figures for the IIB report from the existing results/ JSONs.

Outputs PDF files into report/figures/ so they can be included with
\\includegraphics from the LaTeX source. PDFs are vector and scale
cleanly when LaTeX sets the figure width.

Each function below is self-contained: it reads one or two JSON files
and writes one PDF. Calling main() runs them all.

Style choices:
- Serif font matching Computer Modern (LaTeX body font)
- No top/right axis spines
- A4-friendly figure widths around 5-6 inches (fits inside the 25 mm
  margins of the report layout when scaled to \\linewidth)
- Black + a small red accent for highlighted points
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Repository root, relative to this script.
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_DIR = ROOT / "report" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
    "figure.dpi": 100,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

BLACK = "#000000"
ACCENT = "#cc0000"  # muted red for highlights
GREY = "#888888"


def _save(fig, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] wrote {path.relative_to(ROOT)}")


# ===========================================================================
# Phantom characterisation figures (Section 4.1)
# ===========================================================================
def phantom_period_sweep() -> None:
    """Figure 4.1: recovered period error vs true period."""
    with open(RESULTS / "phantom" / "phantom_synthetic_results.json") as f:
        data = json.load(f)
    sweep = data["period_sweep"]
    periods = np.array([r["period_true"] for r in sweep], dtype=float)
    errors = np.array([r["period_error_pct_mean"] for r in sweep], dtype=float)

    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.axhline(0, color=GREY, linewidth=0.6)
    ax.plot(periods, errors, "o-", color=BLACK, linewidth=1.0, markersize=4)
    # Highlight the 60 px alias failure
    fail_mask = periods == 60
    if fail_mask.any():
        ax.plot(periods[fail_mask], errors[fail_mask], "o",
                color=ACCENT, markersize=7, label="alias failure")
        ax.legend(loc="lower left")
    ax.set_xlabel("True period (pixels)")
    ax.set_ylabel("Recovered period error (%)")
    ax.set_title("Period sweep on synthetic phantom")
    _save(fig, "phantom_period_sweep.pdf")


def phantom_snr_sweep() -> None:
    """Figure 4.2: period and sigma error vs SNR."""
    with open(RESULTS / "phantom" / "phantom_synthetic_results.json") as f:
        data = json.load(f)
    sweep = data["snr_sweep"]
    snrs = np.array([r["snr"] for r in sweep], dtype=float)
    period_err = np.array([r["period_error_pct_mean"] for r in sweep], dtype=float)
    sigma_err = np.array([r["sigma_error_pct_mean"] for r in sweep], dtype=float)

    fig, ax_left = plt.subplots(figsize=(5.5, 3.3))
    ax_right = ax_left.twinx()
    # Right axis is twin -- restore its spine because rcParams hid it
    ax_right.spines["right"].set_visible(True)
    ax_right.spines["top"].set_visible(False)
    ax_right.grid(False)

    l1, = ax_left.plot(snrs, period_err, "o-", color=BLACK,
                       linewidth=1.0, markersize=4, label="period error (left)")
    ax_left.axhline(0, color=GREY, linewidth=0.4)
    ax_left.set_xlabel("Signal-to-noise ratio")
    ax_left.set_ylabel("Recovered period error (%)")
    ax_left.set_xscale("log")

    l2, = ax_right.plot(snrs, sigma_err, "s--", color=ACCENT,
                        linewidth=1.0, markersize=4, label="wire-width error (right)")
    ax_right.set_ylabel("Recovered wire width error (%)", color=ACCENT)
    ax_right.tick_params(axis="y", labelcolor=ACCENT)

    ax_left.legend(handles=[l1, l2], loc="center right")
    ax_left.set_title("SNR sweep on synthetic phantom")
    _save(fig, "phantom_snr_sweep.pdf")


def phantom_angle_sweep() -> None:
    """Figure 4.3: angular error vs true line angle."""
    with open(RESULTS / "phantom" / "phantom_synthetic_results.json") as f:
        data = json.load(f)
    sweep = data["angle_sweep"]
    angles = np.array([r["angle_true"] for r in sweep], dtype=float)
    errors = np.array([r["angle_error_deg_mean"] for r in sweep], dtype=float)
    # Sort by true angle for a clean plot
    order = np.argsort(angles)
    angles = angles[order]
    errors = errors[order]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.axhline(0, color=GREY, linewidth=0.6)
    ax.axhline(1, color=GREY, linewidth=0.4, linestyle=":")
    ax.axhline(-1, color=GREY, linewidth=0.4, linestyle=":")
    ax.plot(angles, errors, "o-", color=BLACK, linewidth=1.0, markersize=4)
    ax.set_xlabel("True orientation (degrees)")
    ax.set_ylabel("Recovered orientation error (degrees)")
    ax.set_title("Orientation sweep on synthetic phantom")
    ax.set_ylim(-4, 4)
    _save(fig, "phantom_angle_sweep.pdf")


def phantom_sigma_sweep() -> None:
    """Figure 4.4: recovered FWHM error vs true sigma."""
    with open(RESULTS / "phantom" / "phantom_synthetic_results.json") as f:
        data = json.load(f)
    sweep = data["sigma_sweep"]
    sigmas = np.array([r["sigma_true"] for r in sweep], dtype=float)
    errors = np.array([r["sigma_error_pct_mean"] for r in sweep], dtype=float)

    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.axhline(0, color=GREY, linewidth=0.6)
    ax.axhspan(-10, 10, color="black", alpha=0.05, label="10% band")
    ax.plot(sigmas, errors, "o-", color=BLACK, linewidth=1.0, markersize=4)
    ax.set_xlabel("True wire sigma (pixels)")
    ax.set_ylabel("Recovered wire-width error (%)")
    ax.set_title("Wire-width sweep on synthetic phantom")
    ax.legend(loc="lower left")
    _save(fig, "phantom_sigma_sweep.pdf")


# ===========================================================================
# Per-folio figures (Section 4.2)
# ===========================================================================
def interval_histogram(serial: str, name: str | None = None) -> None:
    """Bar histogram of inter-peak distances for one folio."""
    src = RESULTS / serial / "interval_distribution.json"
    with open(src) as f:
        data = json.load(f)
    gap_hist = data["gap_histogram"]
    period = data["period_px_used"]
    bins = sorted(int(k) for k in gap_hist.keys())
    counts = [gap_hist[str(b)] for b in bins]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.bar(bins, counts, width=0.8, color=BLACK, alpha=0.75)
    ax.axvline(period, color=ACCENT, linestyle="--",
               label=f"locked period = {period:.1f} px")
    ax.set_xlabel("Inter-peak distance (pixels)")
    ax.set_ylabel("Count")
    title_name = name or serial
    ax.set_title(f"Laid-line interval distribution: {title_name}")
    ax.legend(loc="upper right")
    out_name = f"interval_{serial.replace('-', '_')}.pdf"
    _save(fig, out_name)


# ===========================================================================
# Spreadsheet vs manual GT (Section 4.3 and 4.4)
# ===========================================================================
def spreadsheet_scatter() -> None:
    """Figure 4.7: pipeline vs spreadsheet density scatter (9 folios)."""
    with open(RESULTS / "pipeline_vs_spreadsheet.json") as f:
        data = json.load(f)
    manual_serials = {"Kk1-5_f5v", "Kk1-5_f9v"}

    spread_x, spread_y, spread_s = [], [], []
    manual_x, manual_y, manual_s = [], [], []
    for r in data:
        x = r["gt_lines_per_cm"]
        y = r["lines_per_cm_mean"]
        s = r["serial"].replace("_", "\n", 1)  # break long folio names
        if r["serial"] in manual_serials:
            manual_x.append(x); manual_y.append(y); manual_s.append(s)
        else:
            spread_x.append(x); spread_y.append(y); spread_s.append(s)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    lo = min(min(spread_x + manual_x), min(spread_y + manual_y)) - 1
    hi = max(max(spread_x + manual_x), max(spread_y + manual_y)) + 1
    ax.plot([lo, hi], [lo, hi], color=GREY, linewidth=0.6, linestyle="--",
            label="y = x (perfect agreement)")
    ax.scatter(spread_x, spread_y, marker="o", color=BLACK, s=40,
               label="spreadsheet GT")
    ax.scatter(manual_x, manual_y, marker="^", color=ACCENT, s=80,
               label="manual GT", zorder=5)
    for x, y, s in zip(spread_x + manual_x, spread_y + manual_y, spread_s + manual_s):
        ax.annotate(s, (x, y), fontsize=6.5,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Ground-truth density (lines/cm)")
    ax.set_ylabel("Pipeline density (lines/cm)")
    ax.set_title("Pipeline vs ground-truth density on the nine-folio benchmark")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    _save(fig, "spreadsheet_scatter.pdf")


def three_way_comparison() -> None:
    """Pipeline vs spreadsheet vs per-folio manual GT on the 9-folio MSI benchmark."""
    import glob
    # Load per-folio manual GTs
    manuals = {}
    for p in sorted(glob.glob(str(RESULTS / "*" / "manual_gt.json"))):
        d = json.load(open(p))
        manuals[d["serial"]] = d["lpc_median"]
    # Load pipeline + spreadsheet
    with open(RESULTS / "pipeline_vs_spreadsheet.json") as f:
        data = json.load(f)

    serials, ss, pipe, man = [], [], [], []
    for r in data:
        s = r["serial"]
        if s not in manuals:
            continue
        serials.append(s)
        ss.append(r["gt_lines_per_cm"])
        pipe.append(r["lines_per_cm_mean"])
        man.append(manuals[s])

    n = len(serials)
    x = np.arange(n)
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - w, man, w, color=BLACK, label="manual count")
    ax.bar(x,     pipe, w, color=ACCENT, label="pipeline")
    ax.bar(x + w, ss,   w, color=GREY, label="spreadsheet")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in serials],
                       rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Density (lines/cm)")
    ax.set_title("Three-way comparison on the nine-folio MSI benchmark")
    ax.legend(loc="upper right", ncol=3)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
    _save(fig, "three_way_comparison.pdf")


def manual_gt_bars() -> None:
    """Figure 4.8 panel A: bar chart pipeline vs manual GT for two calibrated folios."""
    with open(RESULTS / "pipeline_vs_spreadsheet.json") as f:
        data = json.load(f)
    rows = [r for r in data if r["serial"] in {"Kk1-5_f5v", "Kk1-5_f9v"}]
    labels = [r["serial"].replace("_", " ") for r in rows]
    pipeline = [r["lines_per_cm_mean"] for r in rows]
    gt = [r["gt_lines_per_cm"] for r in rows]
    err = [r["error_pct"] for r in rows]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.bar(x - w/2, gt, width=w, color=GREY, label="manual GT")
    ax.bar(x + w/2, pipeline, width=w, color=BLACK, label="pipeline")
    for i, (p, e) in enumerate(zip(pipeline, err)):
        ax.text(i + w/2, p + 0.1, f"{e:+.2f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Density (lines/cm)")
    ax.set_title("Pipeline vs manual count on the two calibrated folios")
    ax.set_ylim(0, max(max(pipeline), max(gt)) * 1.18)
    ax.legend(loc="upper right")
    _save(fig, "manual_gt_bars.pdf")


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    phantom_period_sweep()
    phantom_snr_sweep()
    phantom_angle_sweep()
    phantom_sigma_sweep()
    interval_histogram("Kk1-5_f5v", name="Kk.1.5 f5v")
    interval_histogram("10", name="Dataset 10 (Da Rold reproduction)")
    spreadsheet_scatter()
    manual_gt_bars()
    three_way_comparison()


if __name__ == "__main__":
    main()
