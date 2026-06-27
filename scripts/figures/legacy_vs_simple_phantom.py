"""
Quantitative demonstration of the legacy patchwise-Gabor detector's
period/2 bias on a synthetic phantom, alongside the final
single-image FFT detector for reference.

Generates Gaussian-comb phantoms at a sweep of true periods, runs
both detectors, and saves a 2-panel figure to
report/figures/legacy_vs_simple_phantom.pdf.

Run: .venv/bin/python scripts/legacy_vs_simple_phantom.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make repo root importable BEFORE the phantom module pulls in paperdrm.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phantom_synthetic import make_synthetic_image
from paperdrm.stage3_detect.simple_detector import detect_laid_lines_simple
from paperdrm.stage3_detect.gabor import estimate_laidline_frequency_gabor_patches

REPO = Path(__file__).resolve().parent.parent
FIG_OUT = REPO / "report" / "figures" / "legacy_vs_simple_phantom.pdf"
JSON_OUT = REPO / "results" / "phantom" / "legacy_vs_simple.json"


def run_one(period_px: float, *, snr: float = 15.0, sigma_px: float = 2.0,
            seed: int = 0) -> dict:
    img = make_synthetic_image(
        period_px=period_px,
        line_dir_deg=90.0,    # vertical lines
        sigma_px=sigma_px,
        snr=snr,
        shape=(512, 512),
        seed=seed,
    )

    # Final single-image FFT + Gabor detector
    simple = detect_laid_lines_simple(
        img,
        line_dir_deg=90.0,
        period_range_px=(8.0, 80.0),
        wire_is_darker=True,
    )
    simple_period = float(simple["dominant_period_px"])

    # Legacy patchwise-Gabor detector with main.py's original
    # candidate-period grid (range 4..80 px). The proportional
    # bandwidth bias of the Gabor bank scores higher-frequency
    # responses more strongly, and the abs-response harmonic doubling
    # introduces extra alias peaks; the patch-level "winners" are
    # concentrated at the boundary of the candidate grid rather than
    # at the true period.
    legacy = estimate_laidline_frequency_gabor_patches(
        img,
        line_dir_deg=90.0,
        patch_size=(256, 256),
        stride=(128, 128),
        periods_px=list(range(4, 81)),
        min_score=0.02,
        weight_scale=3.0,
        show_progress=False,
    )
    legacy_period = float(legacy["dominant_period_px"])

    return {
        "period_true": period_px,
        "period_simple": simple_period,
        "period_legacy": legacy_period,
        "simple_err_pct": (simple_period - period_px) / period_px * 100.0,
        "legacy_err_pct": (legacy_period - period_px) / period_px * 100.0,
        "legacy_ratio": legacy_period / period_px,
    }


def main():
    periods = [12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 40.0]
    results = []
    for p in periods:
        print(f"running period={p:5.1f} px...")
        r = run_one(p)
        print(f"  simple={r['period_simple']:5.2f}  legacy={r['period_legacy']:5.2f}  "
              f"legacy/true={r['legacy_ratio']:.3f}")
        results.append(r)

    # Save numbers
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {JSON_OUT}")

    # Plot
    true_p = np.array([r["period_true"] for r in results])
    simple_p = np.array([r["period_simple"] for r in results])
    legacy_p = np.array([r["period_legacy"] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

    # Left: recovered vs true period
    ax = axes[0]
    diag = np.linspace(0, max(true_p) * 1.1, 50)
    ax.plot(diag, diag, color="black", lw=0.8, ls="--", label="$y = x$ (correct)")
    ax.plot(true_p, simple_p, "o-", color="#1f77b4", lw=1.5, ms=7,
            label="final FFT detector")
    ax.plot(true_p, legacy_p, "s-", color="#d62728", lw=1.5, ms=7,
            label="legacy patchwise Gabor")
    ax.set_xlabel("true phantom period (px)")
    ax.set_ylabel("recovered period (px)")
    ax.set_xlim(0, max(true_p) * 1.1)
    ax.set_ylim(0, max(true_p) * 1.1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # Right: absolute error vs true period (log-y, signed)
    ax = axes[1]
    err_simple = simple_p - true_p
    err_legacy = legacy_p - true_p
    ax.axhline(0.0, color="black", lw=0.8, ls="--")
    ax.plot(true_p, err_simple, "o-", color="#1f77b4", lw=1.5, ms=7,
            label="final FFT detector")
    ax.plot(true_p, err_legacy, "s-", color="#d62728", lw=1.5, ms=7,
            label="legacy patchwise Gabor")
    ax.set_xlabel("true phantom period (px)")
    ax.set_ylabel("recovered $-$ true period (px)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_OUT, bbox_inches="tight", pad_inches=0.05, dpi=300)
    print(f"saved {FIG_OUT}")


if __name__ == "__main__":
    main()
