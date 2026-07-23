"""
Split-half period stability for the multi-phi detector.

Randomly partitions the N phi images into halves A and B, re-runs spectral
aggregation on each half, and compares the two dominant-period estimates.
Repeats over many random splits to produce a reliability distribution.

This is a ground-truth-free reliability indicator: low spread of (A-B) is
evidence that the multi-phi aggregation is locked onto a phi-stationary
feature (the laid lines), not phi-random structure that would shuffle
between halves.

Public API:
    split_half_period_stability -- compute statistics over N random splits
    print_split_half            -- console summary
    save_split_half             -- JSON dump (trimmed)
    plot_split_half             -- A vs B scatter + difference histogram
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from paperdrm.stage3_detect.multi_phi_detector import aggregate_radial_power


def split_half_period_stability(
    images: list[np.ndarray],
    *,
    line_dir_deg: float = 90.0,
    period_range_px: tuple[float, float] = (8.0, 80.0),
    normalize: str = "sum",
    n_splits: int = 100,
    seed: int = 0,
    fov_width_cm: float | None = None,
) -> dict:
    """
    Per-split: choose `half = N // 2` images at random into A, the next
    `half` into B, aggregate per-image normalised spectra in each half,
    pick peak freq -> period. Repeat `n_splits` times.
    """
    n = len(images)
    if n < 4:
        raise ValueError(f"need >= 4 images for split-half; got {n}.")

    # Precompute per-image normalised spectra once -> per-split aggregation
    # is just a sum of rows, no FFTs in the loop.
    agg_all = aggregate_radial_power(
        images,
        line_dir_deg=line_dir_deg,
        period_range_px=period_range_px,
        normalize=normalize,
    )
    freqs = agg_all["freqs"]
    P_per = agg_all["power_per_image"]  # (N, K)

    rng = np.random.default_rng(int(seed))
    periods_a = np.zeros(int(n_splits), dtype=np.float64)
    periods_b = np.zeros(int(n_splits), dtype=np.float64)
    half = n // 2

    for s in range(int(n_splits)):
        order = rng.permutation(n)
        a = order[:half]
        b = order[half:half * 2]
        Pa = P_per[a].sum(axis=0)
        Pb = P_per[b].sum(axis=0)
        periods_a[s] = 1.0 / float(freqs[int(np.argmax(Pa))])
        periods_b[s] = 1.0 / float(freqs[int(np.argmax(Pb))])

    diff = periods_a - periods_b
    pooled = np.concatenate([periods_a, periods_b])

    out: dict = {
        "n_images": int(n),
        "half_size": int(half),
        "n_splits": int(n_splits),
        "seed": int(seed),
        "normalize": str(normalize),
        "period_a_mean": float(np.mean(periods_a)),
        "period_b_mean": float(np.mean(periods_b)),
        "period_a_std": float(np.std(periods_a, ddof=1)) if n_splits > 1 else 0.0,
        "period_b_std": float(np.std(periods_b, ddof=1)) if n_splits > 1 else 0.0,
        "period_pooled_mean": float(np.mean(pooled)),
        "period_pooled_std": float(np.std(pooled, ddof=1)) if n_splits > 1 else 0.0,
        "period_diff_mean": float(np.mean(diff)),
        "period_diff_std": float(np.std(diff, ddof=1)) if n_splits > 1 else 0.0,
        "period_diff_mad": float(np.median(np.abs(diff - np.median(diff)))),
        "agree_rate_within_1px": float(np.mean(np.abs(diff) <= 1.0)),
        "agree_rate_within_0p5px": float(np.mean(np.abs(diff) <= 0.5)),
        "period_a": periods_a.tolist(),
        "period_b": periods_b.tolist(),
    }
    if fov_width_cm is not None and n > 0:
        img_w = int(images[0].shape[1])
        cm_per_px = float(fov_width_cm) / float(img_w)
        out["cm_per_px"] = cm_per_px
        out["period_pooled_mean_cm"] = out["period_pooled_mean"] * cm_per_px
        out["period_pooled_std_cm"] = out["period_pooled_std"] * cm_per_px
        out["period_diff_std_cm"] = out["period_diff_std"] * cm_per_px
    return out


def print_split_half(stats: dict) -> None:
    print("[Eval] Split-half period stability")
    print(f"  n_images={stats['n_images']} half={stats['half_size']}"
          f" splits={stats['n_splits']} normalize={stats['normalize']}")
    print(f"  period A: mean={stats['period_a_mean']:.3f}"
          f"  std={stats['period_a_std']:.3f} px")
    print(f"  period B: mean={stats['period_b_mean']:.3f}"
          f"  std={stats['period_b_std']:.3f} px")
    print(f"  diff (A-B): mean={stats['period_diff_mean']:+.3f}"
          f"  std={stats['period_diff_std']:.3f}"
          f"  mad={stats['period_diff_mad']:.3f} px")
    print(f"  agree |diff| <= 1.0 px: {stats['agree_rate_within_1px']*100:.1f}%"
          f"  | <= 0.5 px: {stats['agree_rate_within_0p5px']*100:.1f}%")
    if "period_pooled_mean_cm" in stats:
        print(f"  cm    : pooled mean={stats['period_pooled_mean_cm']:.5f} cm"
              f"  std={stats['period_pooled_std_cm']:.5f} cm"
              f"  diff_std={stats['period_diff_std_cm']:.5f} cm")


def save_split_half(stats: dict, path: str | Path) -> None:
    keep = {k: v for k, v in stats.items() if k not in {"period_a", "period_b"}}
    keep["period_a_sample"] = stats["period_a"][:20]
    keep["period_b_sample"] = stats["period_b"][:20]
    Path(path).write_text(json.dumps(keep, indent=2))


def plot_split_half(stats: dict, save_path: str | Path | None = None) -> None:
    a = np.asarray(stats["period_a"], dtype=np.float64)
    b = np.asarray(stats["period_b"], dtype=np.float64)
    if a.size == 0:
        print("[Eval] plot_split_half: nothing to plot")
        return
    diff = a - b

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(a, b, s=14, alpha=0.6, color="steelblue")
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    pad = max(0.5, 0.05 * (hi - lo))
    axes[0].plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", alpha=0.4, linewidth=1)
    axes[0].set_xlabel("Period from split A (px)")
    axes[0].set_ylabel("Period from split B (px)")
    axes[0].set_title("Split-half period agreement")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(diff, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    axes[1].axvline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("Period A − B (px)")
    axes[1].set_ylabel("Splits")
    axes[1].set_title(f"Difference  (std={stats['period_diff_std']:.3f} px,"
                      f" mad={stats['period_diff_mad']:.3f})")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
