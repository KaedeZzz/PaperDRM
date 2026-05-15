"""
Compare frequency response of two background-blur implementations:
  OLD: GaussianBlur(sigma=5) -> resize 0.2x -> GaussianBlur(sigma=20) -> resize back
  NEW: GaussianBlur(sigma=100)

Outputs:
  - 1D radial frequency response on a delta impulse (analytical-ish)
  - Radial power spectrum of (img - lowpass) on a real image
  - Per-pixel difference statistics between OLD and NEW lowpass on a real image
"""
from pathlib import Path
import sys

import numpy as np
import cv2
from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def blur_old(img: np.ndarray) -> np.ndarray:
    x = cv2.GaussianBlur(img, (0, 0), 5)
    small = cv2.resize(x, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 20)
    return cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)


def blur_new(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigmaX=100, borderType=cv2.BORDER_REFLECT_101)


def radial_profile(power2d: np.ndarray) -> np.ndarray:
    h, w = power2d.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices(power2d.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    tbin = np.bincount(r.ravel(), power2d.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def impulse_response_freq(size: int = 1024) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (freq_cycles_per_pixel, |H_old|, |H_new|) via radial avg of FFT(impulse)."""
    delta = np.zeros((size, size), dtype=np.float32)
    delta[size // 2, size // 2] = 1.0
    h_old = blur_old(delta)
    h_new = blur_new(delta)
    F_old = np.abs(np.fft.fftshift(np.fft.fft2(h_old)))
    F_new = np.abs(np.fft.fftshift(np.fft.fft2(h_new)))
    prof_old = radial_profile(F_old)
    prof_new = radial_profile(F_new)
    # freq in cycles/pixel: bin index k -> k / size
    k = np.arange(prof_old.size)
    freq = k / size
    return freq, prof_old, prof_new


def real_image_compare(img_path: Path) -> dict:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot read {img_path}")
    img_f = img.astype(np.float32)
    lp_old = blur_old(img).astype(np.float32)
    lp_new = blur_new(img).astype(np.float32)

    hp_old = img_f - lp_old
    hp_new = img_f - lp_new

    # Use a central square crop to avoid boundary effects
    h, w = img_f.shape
    s = min(h, w)
    s = s - (s % 2)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    crop_old = hp_old[y0:y0 + s, x0:x0 + s]
    crop_new = hp_new[y0:y0 + s, x0:x0 + s]

    win = np.outer(np.hanning(s), np.hanning(s)).astype(np.float32)
    P_old = np.abs(np.fft.fftshift(np.fft.fft2(crop_old * win))) ** 2
    P_new = np.abs(np.fft.fftshift(np.fft.fft2(crop_new * win))) ** 2
    prof_old = radial_profile(P_old)
    prof_new = radial_profile(P_new)
    freq = np.arange(prof_old.size) / s

    diff = lp_old - lp_new
    stats = {
        "img_shape": img.shape,
        "lp_diff_mean": float(diff.mean()),
        "lp_diff_std": float(diff.std()),
        "lp_diff_absmax": float(np.abs(diff).max()),
        "lp_old_mean": float(lp_old.mean()),
        "lp_new_mean": float(lp_new.mean()),
        "hp_old_std": float(hp_old.std()),
        "hp_new_std": float(hp_new.std()),
        "hp_corr": float(np.corrcoef(hp_old.ravel(), hp_new.ravel())[0, 1]),
        "freq": freq,
        "prof_old": prof_old,
        "prof_new": prof_new,
    }
    return stats


def main() -> None:
    out_dir = REPO_ROOT / "scripts" / "_bg_blur_freq_out"
    out_dir.mkdir(exist_ok=True)

    # 1) impulse-response frequency comparison
    freq, H_old, H_new = impulse_response_freq(size=1024)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    nyq_keep = freq < 0.05  # only show low end — both are heavy lowpass
    ax[0].plot(freq[nyq_keep], H_old[nyq_keep], label="OLD (5 → 0.2x → 20)", lw=1.5)
    ax[0].plot(freq[nyq_keep], H_new[nyq_keep], label="NEW (sigma=100)", lw=1.5)
    ax[0].set_xlabel("frequency [cycles / pixel]")
    ax[0].set_ylabel("|H(f)|  (radial avg)")
    ax[0].set_title("Impulse-response magnitude (linear)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].semilogy(freq[nyq_keep], np.maximum(H_old[nyq_keep], 1e-12), label="OLD")
    ax[1].semilogy(freq[nyq_keep], np.maximum(H_new[nyq_keep], 1e-12), label="NEW")
    ax[1].set_xlabel("frequency [cycles / pixel]")
    ax[1].set_ylabel("|H(f)|  (log)")
    ax[1].set_title("Impulse-response magnitude (log)")
    ax[1].legend()
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "impulse_response.png", dpi=140)
    plt.close(fig)

    # Equivalent sigma estimate from H_new vs H_old: where |H| drops to ~0.5
    def cutoff(freq: np.ndarray, H: np.ndarray, level: float = 0.5) -> float:
        Hn = H / H.max()
        idx = np.argmax(Hn < level)
        return float(freq[idx]) if idx > 0 else float("nan")

    fc_old = cutoff(freq, H_old, 0.5)
    fc_new = cutoff(freq, H_new, 0.5)

    # For a Gaussian: |H(f)| = exp(-2 pi^2 sigma^2 f^2)
    # |H(f)| = 0.5  =>  f = sqrt(ln 2) / (pi * sqrt(2) * sigma) = sqrt(ln 2 / 2) / (pi sigma)
    def sigma_from_cutoff(fc: float, level: float = 0.5) -> float:
        if not np.isfinite(fc) or fc <= 0:
            return float("nan")
        return float(np.sqrt(-np.log(level) / 2) / (np.pi * fc))

    sigma_eff_old = sigma_from_cutoff(fc_old)
    sigma_eff_new = sigma_from_cutoff(fc_new)

    # 2) real-image comparison
    img_path = REPO_ROOT / "data" / "processed" / "0_10.jpg"
    stats = real_image_compare(img_path)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    keep = stats["freq"] < 0.05
    ax.loglog(stats["freq"][keep][1:], stats["prof_old"][keep][1:], label="img - OLD lowpass")
    ax.loglog(stats["freq"][keep][1:], stats["prof_new"][keep][1:], label="img - NEW lowpass")
    ax.set_xlabel("frequency [cycles / pixel]")
    ax.set_ylabel("radial power spectrum  (log-log)")
    ax.set_title(f"High-pass residual spectrum  ({img_path.name})")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "real_image_residual_spectrum.png", dpi=140)
    plt.close(fig)

    # Print summary
    print("=== impulse response ===")
    print(f"OLD: half-magnitude cutoff ≈ {fc_old:.5f} cyc/px  -> effective sigma ≈ {sigma_eff_old:.2f} px")
    print(f"NEW: half-magnitude cutoff ≈ {fc_new:.5f} cyc/px  -> effective sigma ≈ {sigma_eff_new:.2f} px")
    print()
    print(f"=== real image: {img_path.name} shape={stats['img_shape']} ===")
    print(f"lowpass diff (OLD - NEW): mean={stats['lp_diff_mean']:.4f}  std={stats['lp_diff_std']:.4f}  |max|={stats['lp_diff_absmax']:.4f}")
    print(f"lowpass means         : OLD={stats['lp_old_mean']:.3f}  NEW={stats['lp_new_mean']:.3f}")
    print(f"high-pass std         : OLD={stats['hp_old_std']:.4f}  NEW={stats['hp_new_std']:.4f}")
    print(f"high-pass corr(OLD,NEW)= {stats['hp_corr']:.6f}")
    print()
    print(f"plots saved under: {out_dir}")


if __name__ == "__main__":
    main()
