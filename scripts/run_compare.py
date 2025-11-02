"""Compare baseline rlic with brylic implementation.

Usage
-----
    python scripts/run_compare.py

Dependencies
------------
    pip install -r scripts/requirements.txt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

# ============================================================================
# CONFIGURATION
# ============================================================================
RESOLUTION = (1024, 1024)  # (height, width)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# LIC parameters
STREAMLENGTH_FACTOR = 60.0 / 1024.0
ITERATIONS = 2
NOISE_SEED = 0
NOISE_OVERSAMPLE = 1.5
NOISE_SIGMA = 0.0

# Mask and edge gain
MASK_RADIUS_FRACTION = 0.14
EDGE_GAIN_STRENGTH = 0.12
EDGE_GAIN_POWER = 1.0

# Tiling
TILE_SHAPE = (512, 512)
TILE_OVERLAP = None  # Auto
NUM_THREADS = None  # Auto

# Display
CLIP_PERCENT = 1.5
CONTRAST = 1.5
EDGE_DISPLAY_BOOST = 1.0

# ============================================================================
# SETUP
# ============================================================================
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rlic as vanilla_rlic
import brylic


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_noise(shape: tuple[int, int]) -> np.ndarray:
    """Generate band-limited noise texture."""
    rng = np.random.default_rng(NOISE_SEED)
    height, width = shape

    # Oversample
    hi_h = max(1, int(round(height * NOISE_OVERSAMPLE)))
    hi_w = max(1, int(round(width * NOISE_OVERSAMPLE)))
    base = rng.random((hi_h, hi_w), dtype=np.float32)

    # Low-pass filter
    if NOISE_SIGMA > 0.0:
        base = gaussian_filter(base, sigma=NOISE_SIGMA)

    # Downsample
    scale_y = height / base.shape[0]
    scale_x = width / base.shape[1]  # type: ignore[misc]
    base = zoom(base, (scale_y, scale_x), order=1)

    # Normalize to [0, 1]
    base_min = float(base.min())  # type: ignore[attr-defined]
    base_max = float(base.max())  # type: ignore[attr-defined]
    if base_max > base_min:
        base = (base - base_min) / (base_max - base_min)
    else:
        base = np.zeros_like(base)

    return base.astype(np.float32)  # type: ignore[attr-defined]


def circular_mask(shape: tuple[int, int]) -> np.ndarray:
    """Create circular mask at center."""
    height, width = shape
    cy = (height - 1) * 0.5
    cx = (width - 1) * 0.5
    radius = MASK_RADIUS_FRACTION * min(height, width)
    yy, xx = np.ogrid[:height, :width]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def radial_field(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Generate radial vector field."""
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[1], dtype=np.float32),
        indexing="ij",
    )
    radius = np.hypot(xx, yy).astype(np.float32) + 1e-6
    u = (xx / radius).astype(np.float32)
    v = (yy / radius).astype(np.float32)
    u[shape[0] // 2, shape[1] // 2] = 0.0
    v[shape[0] // 2, shape[1] // 2] = 0.0
    return u, v


def cosine_kernel(streamlength: int) -> np.ndarray:
    """Raised-cosine kernel."""
    streamlength = max(int(streamlength), 1)
    positions = np.arange(1 - streamlength, streamlength, dtype=np.float32)
    return 0.5 * (1.0 + np.cos(np.pi * positions / streamlength))


def lic_to_unit(arr: np.ndarray) -> np.ndarray:
    """Convert LIC output to [0, 1]."""
    arr = arr.astype(np.float32, copy=False)
    max_abs = float(np.max(np.abs(arr)))
    if max_abs > 1e-12:
        arr = arr / max_abs
    return 0.5 * (arr + 1.0)


def normalize_panels(baseline: np.ndarray, brylic_gain: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize both panels using baseline percentiles."""
    base_unit = lic_to_unit(baseline)
    vmin = float(np.percentile(base_unit, CLIP_PERCENT))
    vmax = float(np.percentile(base_unit, 100.0 - CLIP_PERCENT))

    if vmax <= vmin:
        vmin, vmax = float(np.min(base_unit)), float(np.max(base_unit))
        if vmax <= vmin:
            vmin, vmax = 0.0, 1.0

    def normalize(arr: np.ndarray) -> np.ndarray:
        unit = lic_to_unit(arr)
        if vmax > vmin:
            norm = np.clip((unit - vmin) / (vmax - vmin), 0.0, 1.0)
        else:
            norm = np.zeros_like(unit, dtype=np.float32)
        if not np.isclose(CONTRAST, 1.0):
            norm = np.clip((norm - 0.5) * CONTRAST + 0.5, 0.0, 1.0)
        return norm

    baseline_norm = normalize(baseline)
    brylic_norm = normalize(brylic_gain)

    # Match brylic gain panel brightness to baseline, then apply optional boost
    sel = ~mask
    ref_high = float(np.percentile(baseline_norm[sel], 99.0))
    gain_high = float(np.percentile(brylic_norm[sel], 99.0))
    if gain_high > 1e-6 and ref_high > 1e-6:
        scale = (ref_high / gain_high) * EDGE_DISPLAY_BOOST
        brylic_norm = np.clip(brylic_norm * scale, 0.0, 1.0)

    # Fill mask with black
    baseline_norm[mask] = 0.0
    brylic_norm[mask] = 0.0

    return baseline_norm, brylic_norm


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def run_comparison() -> None:
    """Run the comparison between rlic and brylic."""
    print(f"Resolution: {RESOLUTION[1]}×{RESOLUTION[0]}")
    print(f"Iterations: {ITERATIONS}")
    print(f"Tile shape: {TILE_SHAPE}")
    print()

    # Generate test case
    texture = generate_noise(RESOLUTION)
    u, v = radial_field(RESOLUTION)
    mask = circular_mask(RESOLUTION)

    # Zero out vectors in masked region
    u_masked = u.copy()
    v_masked = v.copy()
    u_masked[mask] = 0.0
    v_masked[mask] = 0.0

    # Kernel
    streamlength_pixels = max(1, int(round(STREAMLENGTH_FACTOR * min(RESOLUTION))))
    kernel = cosine_kernel(streamlength_pixels)

    # Run baseline
    start = time.perf_counter()
    baseline = vanilla_rlic.convolve(
        texture,
        u_masked,
        v_masked,
        kernel=kernel,
        uv_mode="velocity",
        boundaries="closed",
        iterations=ITERATIONS,
    )
    baseline_time = time.perf_counter() - start

    # Run brylic with gain
    start = time.perf_counter()
    brylic_gain = brylic.tiled_convolve(
        texture,
        u_masked,
        v_masked,
        kernel=kernel,
        uv_mode="velocity",
        boundaries="closed",
        iterations=ITERATIONS,
        mask=mask,
        edge_gain_strength=EDGE_GAIN_STRENGTH,
        edge_gain_power=EDGE_GAIN_POWER,
        tile_shape=TILE_SHAPE,
        overlap=TILE_OVERLAP,
        num_threads=NUM_THREADS,
    )
    brylic_time = time.perf_counter() - start

    # Print results
    print(f"Baseline rlic runtime:      {baseline_time:.3f} s")
    print(f"brylic runtime:             {brylic_time:.3f} s")
    print(f"Speedup (baseline/brylic):  {baseline_time / brylic_time:.2f}×")
    print()

    # Save comparison image
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_norm, brylic_norm = normalize_panels(baseline, brylic_gain, mask)
    stacked = np.concatenate([baseline_norm, brylic_norm], axis=1)
    png = (stacked * 255.0).clip(0, 255).astype(np.uint8)
    output_path = OUTPUT_DIR / "comparison.png"
    Image.fromarray(png, mode="L").save(output_path)
    print(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    run_comparison()
