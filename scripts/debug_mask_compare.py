"""Minimal reproduction of mask brightness differences without extra processing.

This script generates white-noise input, a circular mask, and a simple vector
field, then runs the bryLIC convolution both with and without mask awareness.
The raw float outputs are saved (no normalization) alongside a quick matplotlib
comparison plot that shares a global color scale.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import brylic
import rlic as vanilla_rlic

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RESOLUTION = (512, 512)  # (height, width)
STREAM_LENGTH = 40  # kernel half-width in pixels
EDGE_GAIN_STRENGTH = 3.0
EDGE_GAIN_POWER = 1.0
ITERATIONS = 2
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def make_noise(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


def make_mask(shape: tuple[int, int], radius_fraction: float = 0.25) -> np.ndarray:
    h, w = shape
    cy = (h - 1) * 0.5
    cx = (w - 1) * 0.5
    radius = radius_fraction * min(h, w)
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2


def make_vector_field(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float32),
        np.linspace(-1.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    radius = np.hypot(xx, yy).astype(np.float32)
    radius[radius == 0] = 1.0  # avoid division by zero at the center
    u = (xx / radius).astype(np.float32)
    v = (yy / radius).astype(np.float32)
    u[h // 2, w // 2] = 0.0
    v[h // 2, w // 2] = 0.0
    return u, v


def raised_cosine_kernel(half_width: int) -> np.ndarray:
    hw = max(int(half_width), 1)
    positions = np.arange(-hw, hw + 1, dtype=np.float32)
    return 0.5 * (1.0 + np.cos(np.pi * positions / hw))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    texture = make_noise(RESOLUTION, seed=1)
    mask = make_mask(RESOLUTION)
    u, v = make_vector_field(RESOLUTION)
    kernel = raised_cosine_kernel(STREAM_LENGTH)

    print("Running vanilla rlic baseline...")
    result_vanilla = vanilla_rlic.convolve(
        texture,
        u,
        v,
        kernel=kernel,
        uv_mode="velocity",
        boundaries="closed",
        iterations=ITERATIONS,
    )

    u_masked = u.copy()
    v_masked = v.copy()
    u_masked[mask] = 0.0
    v_masked[mask] = 0.0

    print("Running vanilla rlic with masked flow...")
    result_vanilla_masked = vanilla_rlic.convolve(
        texture,
        u_masked,
        v_masked,
        kernel=kernel,
        uv_mode="velocity",
        boundaries="closed",
        iterations=ITERATIONS,
    )

    print("Running convolution without mask (edge_gain=0)...")
    result_nomask_raw = brylic.convolve(
        texture,
        u,
        v,
        kernel=kernel,
        iterations=ITERATIONS,
        edge_gain_strength=0.0,
        edge_gain_power=EDGE_GAIN_POWER,
        mask=None,
    )

    print("Running convolution without mask...")
    result_mask_nogain = brylic.convolve( #stupid name, with maksk
        texture,
        u,
        v,
        kernel=kernel,
        iterations=ITERATIONS,
        edge_gain_strength=0.0,
        edge_gain_power=EDGE_GAIN_POWER,
        mask=mask,
    )

    print("Running convolution with mask awareness...")
    result_masked = brylic.convolve(
        texture,
        u,
        v,
        kernel=kernel,
        iterations=ITERATIONS,
        edge_gain_strength=EDGE_GAIN_STRENGTH,
        edge_gain_power=EDGE_GAIN_POWER,
        mask=mask,
    )

    bundle_path = OUTPUT_DIR / "debug_mask_compare_raw_outputs.npz"
    np.savez(
        bundle_path,
        texture=texture,
        mask=mask,
        vanilla=result_vanilla,
        vanilla_masked=result_vanilla_masked,
        nomask_raw=result_nomask_raw,
        nomask_gain=result_mask_nogain,
        masked=result_masked,
    )
    print(f"Saved raw arrays to {bundle_path}")

    maybe_plot(
        texture,
        mask,
        result_vanilla,
        result_vanilla_masked,
        result_nomask_raw,
        result_mask_nogain,
        result_masked,
    )


def maybe_plot(
    texture: np.ndarray,
    mask: np.ndarray,
    result_vanilla: np.ndarray,
    result_vanilla_masked: np.ndarray,
    result_nomask_raw: np.ndarray,
    result_mask_nogain: np.ndarray,
    result_masked: np.ndarray,
) -> None:
    results = [
        texture,
        result_nomask_raw,
        result_mask_nogain,
        result_masked,
        result_vanilla,
        result_vanilla_masked,
    ]
    global_min = min(float(arr.min()) for arr in results)
    global_max = max(float(arr.max()) for arr in results)

    output_dir = OUTPUT_DIR  # local alias
    mpl_dir = output_dir / "mpl_config"
    fontcache_dir = output_dir / "fontcache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    fontcache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(fontcache_dir))
    try:
        import matplotlib.pyplot as plt  # local import to honor env vars
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"Skipping matplotlib figure ({exc})")
        return

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    axes = np.atleast_2d(axes)

    axes[0, 0].imshow(texture, cmap="gray", vmin=global_min, vmax=global_max)
    axes[0, 0].set_title("Input noise")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(result_nomask_raw, cmap="gray", vmin=global_min, vmax=global_max)
    axes[0, 1].set_title("Raw")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(
        result_mask_nogain, cmap="gray", vmin=global_min, vmax=global_max
    )
    axes[0, 2].set_title("Mask, no edge gain gain")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(result_masked, cmap="gray", vmin=global_min, vmax=global_max)
    axes[0, 3].set_title("Mask, edge gain")
    axes[0, 3].axis("off")

    axes[1, 0].imshow(result_vanilla, cmap="gray", vmin=global_min, vmax=global_max)
    axes[1, 0].set_title("Vanilla rlic")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(
        result_vanilla_masked, cmap="gray", vmin=global_min, vmax=global_max
    )
    axes[1, 1].set_title("Vanilla rlic (flow masked)")
    axes[1, 1].axis("off")

    # Hide remaining panels
    for col in range(2, axes.shape[1]):
        axes[1, col].axis("off")

    fig_path = output_dir / "debug_mask_compare.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Wrote comparison figure to {fig_path}")


if __name__ == "__main__":
    main()
