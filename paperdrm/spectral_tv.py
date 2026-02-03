"""Spectral TV decomposition utilities for separating text and paper features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpectralTVResult:
    times: np.ndarray
    u_history: list[np.ndarray]


def _tv_divergence(u: np.ndarray, eps: float) -> np.ndarray:
    """Compute divergence of normalized gradient with Neumann boundaries."""
    grad_x = np.zeros_like(u)
    grad_y = np.zeros_like(u)
    grad_x[:, :-1] = u[:, 1:] - u[:, :-1]
    grad_y[:-1, :] = u[1:, :] - u[:-1, :]

    norm = np.sqrt(grad_x**2 + grad_y**2 + eps**2)
    px = grad_x / norm
    py = grad_y / norm

    div = np.zeros_like(u)
    div[:, :-1] += px[:, :-1]
    div[:, 1:] -= px[:, :-1]
    div[:-1, :] += py[:-1, :]
    div[1:, :] -= py[:-1, :]
    return div


def spectral_tv_decomposition(
    image: np.ndarray,
    *,
    num_steps: int = 160,
    dt: float = 0.2,
    eps: float = 1e-6,
    verbose: bool = False,
    progress_every: int | None = None,
) -> SpectralTVResult:
    """Run spectral TV flow and return the full flow history.

    Args:
        image: 2D grayscale image (float32/float64 recommended).
        num_steps: Number of TV flow steps.
        dt: Time step for TV flow (smaller is more stable).
        eps: Small stabilizer for gradient norm.
        verbose: Print progress information during the flow.
        progress_every: Optional manual stride for step logging when verbose.

    Returns:
        SpectralTVResult with time samples and the full TV-flow history.
    """
    u = image.astype(np.float32, copy=True)
    u_history = [u]
    times = np.arange(0, num_steps + 1, dtype=np.float32) * dt
    log_every = progress_every or max(1, num_steps // 10)
    if verbose:
        print(f"[spectral_tv] start flow: num_steps={num_steps}, dt={dt}, eps={eps}")

    for step in range(num_steps):
        div = _tv_divergence(u, eps)
        u = u + dt * div
        u_history.append(u)
        if verbose and ((step + 1) % log_every == 0 or step == num_steps - 1):
            print(f"[spectral_tv] step {step + 1}/{num_steps} complete")

    if verbose:
        print("[spectral_tv] decomposition complete")
    return SpectralTVResult(times=times, u_history=u_history)


def split_text_background(
    image: np.ndarray,
    *,
    text_max_time: float = 4.0,
    num_steps: int = 160,
    dt: float = 0.1,
    eps: float = 1e-7,
    verbose: bool = False,
    progress_every: int | None = None,
    text_is_coarse: bool = False,
) -> tuple[np.ndarray, np.ndarray, SpectralTVResult]:
    """Split an image into text (small scales) and background (large scales)."""
    result = spectral_tv_decomposition(
        image,
        num_steps=num_steps,
        dt=dt,
        eps=eps,
        verbose=verbose,
        progress_every=progress_every,
    )
    if verbose:
        print(f"[spectral_tv] integrating text/background split at t<={text_max_time}")
    k = int(round(text_max_time / dt))
    background = result.u_history[k]
    text_layer = image - background
    if text_is_coarse:
        # Swap interpretation: text is the coarse component, paper texture is fine.
        text_layer, background = background, text_layer
    return text_layer, background, result
