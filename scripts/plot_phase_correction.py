"""
Standalone script: re-run the multi-phi detector on dataset 10 and
plot the per-image phase polar diagram before and after the
polarity correction described in Section 3.3.3 of the report.

Output: report/figures/phase_correction.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paperdrm import ImagePack, Settings  # noqa: E402
from paperdrm.stage3_detect.multi_phi_detector import (  # noqa: E402
    collect_grazing_per_phi,
    detect_laid_lines_multi_phi,
)

OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CONFIG = ROOT / "results" / "10" / "exp_param.yaml"

# Matplotlib style matched to plot_for_report.py
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 100,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

BLACK = "#000000"
ACCENT = "#cc0000"


def run() -> None:
    print(f"[phase] loading {CONFIG}")
    settings = Settings.from_yaml(CONFIG).with_overrides(
        angle_slice=(2, 2), verbose=False
    )
    pack = ImagePack(settings=settings)
    images, phi_deg = collect_grazing_per_phi(pack)
    print(f"[phase] {len(images)} grazing images, running multi-phi detector")
    out = detect_laid_lines_multi_phi(
        images,
        line_dir_deg=90.0,
        period_range_px=(8.0, 80.0),
        wire_is_darker=True,
    )

    phases_raw = np.asarray(out["per_image_phase_rad"])
    phases_aligned = np.asarray(out["per_image_phase_aligned_rad"])
    weights = np.asarray(out["per_image_weight"])
    flipped = np.asarray(out["per_image_polarity_flipped"]).astype(bool)
    anchor = int(out["anchor_index"])
    R_raw = float(out["phase_resultant_length_raw"])
    R = float(out["phase_resultant_length"])

    # Normalise marker sizes by weight
    if weights.max() > 0:
        w_norm = (weights / weights.max())
    else:
        w_norm = np.ones_like(weights)
    sizes = 25.0 + 200.0 * w_norm

    fig = plt.figure(figsize=(7.2, 3.6))

    # Left: before correction
    ax_l = fig.add_subplot(1, 2, 1, projection="polar")
    ax_l.set_title(f"before correction\nresultant length $R = {R_raw:.2f}$",
                    fontsize=9)
    ax_l.scatter(phases_raw, np.ones_like(phases_raw),
                 s=sizes, c=BLACK, alpha=0.6, edgecolors="none")
    ax_l.scatter(phases_raw[anchor], 1.0, s=sizes[anchor] + 60,
                 facecolor="none", edgecolor=ACCENT, linewidth=1.4,
                 label="anchor", zorder=5)
    ax_l.set_rticks([])
    ax_l.set_ylim(0, 1.25)
    ax_l.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), frameon=False)

    # Right: after correction
    ax_r = fig.add_subplot(1, 2, 2, projection="polar")
    ax_r.set_title(f"after $\\pi$-flip correction\nresultant length $R = {R:.2f}$",
                    fontsize=9)
    ax_r.scatter(phases_aligned[~flipped], np.ones(np.sum(~flipped)),
                 s=sizes[~flipped], c=BLACK, alpha=0.6, edgecolors="none",
                 label="kept")
    ax_r.scatter(phases_aligned[flipped], np.ones(np.sum(flipped)),
                 s=sizes[flipped], c=ACCENT, alpha=0.6, edgecolors="none",
                 label="flipped by $\\pi$")
    ax_r.set_rticks([])
    ax_r.set_ylim(0, 1.25)
    ax_r.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18),
                 frameon=False, ncol=2)

    path = OUT / "phase_correction.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"[phase] wrote {path.relative_to(ROOT)}")
    print(f"[phase] {int(flipped.sum())}/{len(flipped)} images flipped; "
          f"R_raw={R_raw:.3f} -> R={R:.3f}")


if __name__ == "__main__":
    run()
