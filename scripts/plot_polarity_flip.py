"""
Diagnostic: visualise polarity inversion between two phi observations.

Picks the multi-phi anchor (max-weight phi, polarity = +) and the phi
that is the most flipped (largest anchor-relative phase offset within
the flipped set, polarity = -), and overlays their broadband column-mean
signals over a few periods. Same physical wire positions; opposite
brightness sign on the cosine model.

Output: polarity_flip.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from paperdrm import ImagePack, Settings
from paperdrm.stage3_detect.multi_phi_detector import (
    collect_grazing_per_phi,
    detect_laid_lines_multi_phi,
)
from paperdrm.stage3_detect.simple_detector import _broadband_signal_1d


YAML = "exp_param.yaml"
LINE_DIR_DEG = 90.0
N_PERIODS = 4         # how many periods to display
PAD_PX = 30           # left margin in image so we start clear of borders
OUT_PATH = "polarity_flip.png"


def _smooth(s: np.ndarray, win: int) -> np.ndarray:
    if win < 3 or win >= s.size:
        return s
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=np.float64) / win
    return np.convolve(s, k, mode="same")


def main() -> None:
    settings = Settings.from_yaml(YAML).with_overrides(angle_slice=(2, 2), verbose=False)
    pack = ImagePack(settings=settings)
    images, phi_deg = collect_grazing_per_phi(pack)

    res = detect_laid_lines_multi_phi(
        images, line_dir_deg=LINE_DIR_DEG, wire_is_darker=True,
    )
    period = float(res["dominant_period_px"])
    phases = np.asarray(res["per_image_phase_rad"])
    flipped_mask = np.asarray(res["per_image_polarity_flipped"], dtype=bool)
    weights = np.asarray(res["per_image_weight"])
    anchor_idx = int(res["anchor_index"])
    anchor_phi = float(phases[anchor_idx])

    # Anchor relative offset, wrapped to [-pi, pi]
    delta = ((phases - anchor_phi) + np.pi) % (2 * np.pi) - np.pi

    # Pick the most strongly-flipped phi (largest |delta| among flipped ones)
    flipped_indices = np.where(flipped_mask)[0]
    if flipped_indices.size == 0:
        raise SystemExit("No polarity-flipped phi found in this dataset.")
    # Among flipped, prefer the one with highest weight (cleanest counter-example).
    flipped_weights = weights[flipped_indices]
    flip_idx = int(flipped_indices[int(np.argmax(flipped_weights))])

    anchor_img = images[anchor_idx]
    flip_img = images[flip_idx]
    s_anchor = _broadband_signal_1d(anchor_img, LINE_DIR_DEG)
    s_flip = _broadband_signal_1d(flip_img, LINE_DIR_DEG)
    smooth_win = max(3, int(period / 12) | 1)
    s_anchor_sm = _smooth(s_anchor, smooth_win)
    s_flip_sm = _smooth(s_flip, smooth_win)

    # Display window: a few periods, centred where the signal is healthy.
    x_start = PAD_PX
    x_end = x_start + int(N_PERIODS * period)
    x_end = min(x_end, s_anchor.size - PAD_PX)
    xs = np.arange(x_start, x_end, dtype=np.float64)

    # Wire positions in the rotated frame, from the detected phase
    omega = 2.0 * np.pi / period
    phi_final = float(res["phase"])
    x0 = -phi_final / omega
    k_start = int(np.floor((x_start - x0) / period))
    k_end = int(np.ceil((x_end - x0) / period))
    wire_x = x0 + period * np.arange(k_start, k_end + 1)
    wire_x = wire_x[(wire_x >= x_start) & (wire_x < x_end)]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    axes[0].plot(xs, s_anchor[x_start:x_end], color="lightcoral", alpha=0.4,
                 linewidth=0.8, label="raw")
    axes[0].plot(xs, s_anchor_sm[x_start:x_end], color="darkred", linewidth=1.6,
                 label=f"smoothed (win={smooth_win})")
    axes[0].axhline(0.0, color="gray", linewidth=0.6, alpha=0.5)
    for wx in wire_x:
        axes[0].axvline(wx, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    axes[0].set_ylabel("brightness − local mean")
    axes[0].set_title(
        f"phi index {anchor_idx} ({phi_deg[anchor_idx]:.0f}°) — anchor (polarity +).  "
        f"phase = {phases[anchor_idx]:+.2f} rad,  weight = {weights[anchor_idx]:.4f}"
    )
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(xs, s_flip[x_start:x_end], color="lightsteelblue", alpha=0.4,
                 linewidth=0.8, label="raw")
    axes[1].plot(xs, s_flip_sm[x_start:x_end], color="navy", linewidth=1.6,
                 label=f"smoothed (win={smooth_win})")
    axes[1].axhline(0.0, color="gray", linewidth=0.6, alpha=0.5)
    for wx in wire_x:
        axes[1].axvline(wx, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    axes[1].set_xlabel("x (pixels in rotated frame)")
    axes[1].set_ylabel("brightness − local mean")
    axes[1].set_title(
        f"phi index {flip_idx} ({phi_deg[flip_idx]:.0f}°) — flipped (polarity −).  "
        f"phase = {phases[flip_idx]:+.2f} rad,  Δ vs anchor = {delta[flip_idx]:+.2f} rad "
        f"({np.degrees(delta[flip_idx]):+.0f}°),  weight = {weights[flip_idx]:.4f}"
    )
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(
        f"Polarity flip diagnostic — period={period:.2f} px, "
        f"dotted lines = global grid (multi-phi consensus, {len(wire_x)} wires shown)",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(OUT_PATH, dpi=130)
    print(f"wrote {OUT_PATH}")
    print(f"  anchor: phi index {anchor_idx}, phi = {phi_deg[anchor_idx]:.1f}°")
    print(f"  flipped: phi index {flip_idx}, phi = {phi_deg[flip_idx]:.1f}°, "
          f"Δ = {np.degrees(delta[flip_idx]):+.1f}°")
    print(f"  total polarity-flipped: {int(flipped_mask.sum())}/{flipped_mask.size}")


if __name__ == "__main__":
    main()
