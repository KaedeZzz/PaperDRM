"""
Generate Figs 1.1, 1.3, 2.1, 2.2, 2.3 for the IIB report (ch1 + ch2).
Fig 1.2 is a TikZ schematic written inline in chapters/01_background.tex.

Outputs go to report/figures/.
Run from repo root: .venv/bin/python scripts/make_report_figures.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# DRP cache shape and grid (see data/cache/data_config.yaml and data/raw/config.txt).
H, W = 2160, 4096
PH_NUM, TH_NUM = 40, 12
PH_GRID = np.arange(PH_NUM) * 9.0          # 0..351 step 9 (deg)
TH_GRID = np.arange(TH_NUM) * 5.0 + 10.0   # 10..65 step 5 (deg)

DRP_CACHE = REPO / "data" / "cache" / "drp.dat"
RAW_DIR = REPO / "data" / "raw"

# The raw frames carry a horizontal duplication artefact at offset ~1364 px in
# the 4096-wide image (verified by cache-mean autocorrelation: peak xcorr 0.64
# at shift (0, +/-1364), absent in any single frame). We crop all figures that
# rely on the multi-angle stack to ROI_X_MAX so only one copy of the folio is
# shown. 2000 px sits safely below the 1364 px offset; gives a single folio
# with the watermark on the right of the crop.
ROI_X_MAX = 2000        # in raw 4096-wide coordinates
ROI_X_MAX_DS8 = ROI_X_MAX // 8  # downsampled-by-8 coordinate (= 250)


def open_drp_memmap() -> np.memmap:
    return np.memmap(DRP_CACHE, dtype=np.uint8, mode="r",
                     shape=(H, W, PH_NUM, TH_NUM))


# ---------------------------------------------------------------------------
# Fig 1.1 -- labelled folio close-up showing laid (and chain) lines
# ---------------------------------------------------------------------------
def fig_folio_anatomy():
    """Side-by-side: a grazing image plus an enhanced inset with arrows."""
    raw = cv2.imread(str(RAW_DIR / "0_15.jpg"), cv2.IMREAD_GRAYSCALE)

    # Crop inside the single-folio ROI (cols 0..ROI_X_MAX) so the horizontal
    # duplication artefact at offset ~1364 px isn't visible.
    cy = raw.shape[0] // 2
    cx = ROI_X_MAX // 2
    hy, hx = 600, 800
    crop = raw[cy - hy:cy + hy, cx - hx:cx + hx]
    h2, w2 = crop.shape

    # CLAHE boost so the periodic banding (laid wires) is unambiguous in print.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    crop_eq = clahe.apply(crop)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.imshow(crop_eq, cmap="gray", origin="upper", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])

    # Bracket-style annotation across two adjacent laid lines. In this folio
    # the laid lines run vertically, so spacing is measured horizontally.
    y_bar = int(h2 * 0.20)
    x0 = int(w2 * 0.62)
    x1 = int(w2 * 0.78)
    ax.annotate("", xy=(x1, y_bar), xytext=(x0, y_bar),
                arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.8))
    ax.text((x0 + x1) / 2, y_bar - 18,
            "laid-line spacing",
            color="#d62728", fontsize=12,
            va="bottom", ha="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#d62728", lw=0.8))

    # Direction label
    ax.annotate("laid wires",
                xy=(w2 * 0.22, h2 * 0.55),
                xytext=(w2 * 0.03, h2 * 0.12),
                color="#1f77b4", fontsize=12,
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.8),
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#1f77b4", lw=0.8))

    # Scale bar: 1 cm. Full raw image is 4096 px wide at fov_width_cm=8.65,
    # so 1 cm = 473 px and the same scale applies to this crop.
    px_per_cm = 4096.0 / 8.65
    bar_px = px_per_cm
    bar_y = h2 - 55
    bar_x0 = 60
    # White rectangle (filled) so the bar reads on dark backgrounds.
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((bar_x0, bar_y - 4), bar_px, 8,
                           facecolor="white", edgecolor="black", linewidth=0.6))
    ax.text(bar_x0 + bar_px / 2, bar_y - 14,
            r"$1\,\mathrm{cm}$",
            color="white", ha="center", va="bottom", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.15", fc="black",
                      ec="none", alpha=0.55))

    out = FIG_DIR / "folio_anatomy.pdf"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close()
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Fig 1.3 -- DRP polar plot at a single (well-textured) pixel
# ---------------------------------------------------------------------------
def fig_drp_polar():
    mm = open_drp_memmap()
    # Sample a small region inside the single-folio ROI, average to denoise.
    # x is at quarter-width so we are well inside cols [0, ROI_X_MAX] and away
    # from the duplication-zone gutter.
    cy, cx = H // 2, ROI_X_MAX // 2
    half = 25
    block = mm[cy - half:cy + half, cx - half:cx + half, :, :]
    drp = block.astype(np.float32).mean(axis=(0, 1))   # (PH_NUM, TH_NUM)

    # Polar heatmap: phi around the circle, theta as radius.
    fig = plt.figure(figsize=(5.2, 5.2))
    ax = fig.add_subplot(111, projection="polar")

    phi_rad = np.deg2rad(PH_GRID)
    # Close the loop in phi for clean rendering.
    phi_edges = np.deg2rad(np.r_[PH_GRID - 4.5, PH_GRID[-1] + 4.5])
    th_edges = np.r_[TH_GRID - 2.5, TH_GRID[-1] + 2.5]

    pcm = ax.pcolormesh(phi_edges, th_edges, drp.T, cmap="viridis", shading="auto")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rticks([15, 30, 45, 60])
    ax.set_rlabel_position(135)
    ax.tick_params(labelsize=8)
    cb = plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.12, shrink=0.7)
    cb.set_label("background-subtracted intensity (a.u.)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    out = FIG_DIR / "drp_polar.pdf"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close()
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Direction-map helpers (replicate paperdrm.stage1_features.direction logic
# but on a downsampled stack so we can run in seconds).
# ---------------------------------------------------------------------------
def downsampled_drp_stack(stride: int = 8, crop_x: bool = True):
    """Downsampled and ROI-cropped DRP stack.

    Two crops are applied:
    - Vertical: the cached DRP has the bottom ~30 % of rows zeroed out by
      the background-subtraction step; we drop dead rows.
    - Horizontal: the raw frames carry a horizontal duplication artefact at
      offset ~1364 px (raw), so we keep only cols 0..ROI_X_MAX to show
      one copy of the folio. Set crop_x=False to opt out.
    """
    mm = open_drp_memmap()
    stack = np.array(mm[::stride, ::stride, :, :], dtype=np.float32)
    # Live-paper rows: keep rows whose mean (across phi, theta, x) > threshold.
    row_act = stack.mean(axis=(1, 2, 3))
    live = np.where(row_act > 5.0)[0]
    if len(live):
        stack = stack[: live.max() + 1]
    if crop_x:
        max_x_ds = ROI_X_MAX // stride
        stack = stack[:, :max_x_ds]
    return stack


def drp_direction_map_local(stack: np.ndarray):
    """Replicates paperdrm.stage1_features.direction.drp_direction_map."""
    h, w, ph_num, _ = stack.shape
    phi_vec = stack.mean(axis=3)               # (h, w, ph_num), average over theta
    mean_mat = phi_vec.mean(axis=2, keepdims=True)
    phi_angles = np.linspace(0, 2 * np.pi, ph_num, endpoint=False)[:, None]
    phi_cos = np.cos(phi_angles)
    phi_sin = np.sin(phi_angles)

    X = (phi_vec - mean_mat).reshape(h * w, ph_num) @ phi_cos
    Y = (phi_vec - mean_mat).reshape(h * w, ph_num) @ phi_sin
    X = X.reshape(h, w)
    Y = Y.reshape(h, w)
    mag_map = np.sqrt(X ** 2 + Y ** 2)
    deg_map = np.degrees(np.arctan2(Y, X))

    mag_mean = mag_map.mean()
    mag_map = np.clip(mag_map, None, 2 * mag_mean)
    norm_mag = (mag_map - mag_map.min()) / (mag_map.max() - mag_map.min() + 1e-9)
    return norm_mag, deg_map


# ---------------------------------------------------------------------------
# Fig 2.1 -- anisotropy + azimuth map pair
# ---------------------------------------------------------------------------
def fig_anisotropy_azimuth(stack: np.ndarray):
    mag, deg = drp_direction_map_local(stack)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    im0 = axes[0].imshow(mag, cmap="magma", aspect="equal")
    axes[0].set_title("anisotropy magnitude $\\|M\\|$", fontsize=11)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.ax.tick_params(labelsize=8)

    # Use a circular colormap (twilight) for azimuth angles, range [-180, 180].
    im1 = axes[1].imshow(deg, cmap="twilight", vmin=-180, vmax=180, aspect="equal")
    axes[1].set_title("azimuth $\\arg(M)$ (deg)", fontsize=11)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, ticks=[-180, -90, 0, 90, 180])
    cb1.ax.tick_params(labelsize=8)

    out = FIG_DIR / "anisotropy_azimuth.pdf"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close()
    print(f"  saved {out}")
    return mag, deg


# ---------------------------------------------------------------------------
# Fig 2.2 -- patchwise trigonometric mask (greyscale enhancement)
# ---------------------------------------------------------------------------
def fig_trigmask(deg_map: np.ndarray):
    # Run the project's own patchwise routine; sensible patch_size for the
    # downsampled map (~ 270 x 512). Use 96 px patch, 48 px stride.
    sys.path.insert(0, str(REPO))
    from paperdrm.stage2_enhance.trig_mask import patchwise_trigonometric_mask
    raw, _enhanced, _, _ = patchwise_trigonometric_mask(
        deg_map, patch_size=(96, 96), stride=(48, 48), sigma_deg=12.0, enhance=False
    )

    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    ax.imshow(raw, cmap="gray", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])

    out = FIG_DIR / "trigmask.pdf"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close()
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Fig 2.3 -- two grazing images at different azimuth (same elevation)
# ---------------------------------------------------------------------------
def fig_grazing_pair():
    # Same theta (15 deg, grazing), two perpendicular azimuths.
    im_a = cv2.imread(str(RAW_DIR / "0_15.jpg"), cv2.IMREAD_GRAYSCALE)
    im_b = cv2.imread(str(RAW_DIR / "90_15.jpg"), cv2.IMREAD_GRAYSCALE)

    # Crop inside the single-folio ROI (cols 0..ROI_X_MAX) so the horizontal
    # duplication artefact (offset ~1364 px) isn't visible.
    def centre_crop(im, hy=600, hx=800):
        cy = im.shape[0] // 2
        cx = ROI_X_MAX // 2
        return im[cy - hy:cy + hy, cx - hx:cx + hx]

    a = centre_crop(im_a)
    b = centre_crop(im_b)
    # Match display range so brightness comparison is honest.
    vmin = min(int(np.percentile(a, 2)), int(np.percentile(b, 2)))
    vmax = max(int(np.percentile(a, 98)), int(np.percentile(b, 98)))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].imshow(a, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    axes[0].set_title(r"$\phi = 0^\circ$, $\theta = 15^\circ$", fontsize=11)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].imshow(b, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
    axes[1].set_title(r"$\phi = 90^\circ$, $\theta = 15^\circ$", fontsize=11)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    out = FIG_DIR / "grazing_pair.pdf"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close()
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Fig 3.1 -- four grazing images at different phi (same theta)
# ---------------------------------------------------------------------------
def fig_grazing_quad():
    phis = [0, 45, 90, 135]
    theta = 15
    imgs = [cv2.imread(str(RAW_DIR / f"{p}_{theta}.jpg"), cv2.IMREAD_GRAYSCALE)
            for p in phis]

    # Crop inside the single-folio ROI to avoid the horizontal duplication.
    def centre_crop(im, hy=500, hx=700):
        cy = im.shape[0] // 2
        cx = ROI_X_MAX // 2
        return im[cy - hy:cy + hy, cx - hx:cx + hx]

    crops = [centre_crop(im) for im in imgs]
    # Common display range across all four for honest brightness comparison.
    vmin = int(min(np.percentile(c, 2) for c in crops))
    vmax = int(max(np.percentile(c, 98) for c in crops))

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4))
    for ax, p, c in zip(axes, phis, crops):
        ax.imshow(c, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(rf"$\phi = {p}^\circ$", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    out = FIG_DIR / "grazing_quad.pdf"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close()
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Fig 3.4 -- zoomed overlay of the final grid on Kk1-5 f5v
# ---------------------------------------------------------------------------
def fig_overlay_zoom():
    overlay_path = REPO / "results" / "Kk1-5_f5v" / "grid_1cm_overlay.jpg"
    im = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    # Pick a high-contrast region: ~1/3 down, mid-x, ~700x900 crop so the
    # individual grid lines are legible at print size.
    cy = int(h * 0.40)
    cx = int(w * 0.50)
    hy, hx = 450, 600
    crop = im[cy - hy:cy + hy, cx - hx:cx + hx]

    fig, ax = plt.subplots(figsize=(6.5, 4.9))
    ax.imshow(crop, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])

    out = FIG_DIR / "f5v_overlay_zoom.pdf"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close()
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
def main():
    print("[Fig 1.1] folio anatomy")
    fig_folio_anatomy()

    print("[Fig 1.3] DRP polar plot")
    fig_drp_polar()

    print("[downsampled DRP stack for Fig 2.1, 2.2]")
    stack = downsampled_drp_stack(stride=8)
    print(f"  shape={stack.shape} ({stack.nbytes / 1e6:.1f} MB float32)")

    print("[Fig 2.1] anisotropy + azimuth map")
    _, deg = fig_anisotropy_azimuth(stack)

    print("[Fig 2.2] patchwise trig-mask")
    fig_trigmask(deg)

    print("[Fig 2.3] grazing pair at two phi")
    fig_grazing_pair()

    print("[Fig 3.1] grazing quad at four phi")
    fig_grazing_quad()

    print("[Fig 3.4] overlay zoom on Kk1-5 f5v")
    fig_overlay_zoom()
    # Fig 3.5 reuses existing report/figures/interval_Kk1_5_f5v.pdf.


if __name__ == "__main__":
    main()
