"""Scan different tile sizes to find optimal performance.

Usage
-----
    python scripts/tile_scan.py

Dependencies
------------
    pip install -r scripts/requirements.txt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

# ============================================================================
# CONFIGURATION
# ============================================================================
RESOLUTION = (2048, 2048)  # (height, width)
TILE_SIZES = [256, 512, 1024]  # Square tile sizes to test

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
TILE_OVERLAP = None  # Auto
NUM_THREADS = None  # Auto

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


# ============================================================================
# MAIN SCAN
# ============================================================================

def run_tile_scan() -> None:
    """Scan different tile sizes to find optimal performance."""
    print(f"Resolution: {RESOLUTION[1]}×{RESOLUTION[0]}")
    print(f"Iterations: {ITERATIONS}")
    print(f"Tile sizes to test: {TILE_SIZES}")
    print()

    # Generate test case (same for all runs)
    print("Generating test case...")
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

    # Run baseline once
    print("Running baseline rlic...")
    start = time.perf_counter()
    vanilla_rlic.convolve(
        texture,
        u_masked,
        v_masked,
        kernel=kernel,
        uv_mode="velocity",
        boundaries="closed",
        iterations=ITERATIONS,
    )
    baseline_time = time.perf_counter() - start
    print(f"  Baseline runtime: {baseline_time:.3f} s")
    print()

    # Test each tile size
    results = []
    for tile_size in TILE_SIZES:
        tile_shape = (tile_size, tile_size)
        print(f"Testing tile size {tile_size}×{tile_size}...")

        start = time.perf_counter()
        brylic.tiled_convolve(
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
            tile_shape=tile_shape,
            overlap=TILE_OVERLAP,
            num_threads=NUM_THREADS,
        )
        brylic_time = time.perf_counter() - start
        speedup = baseline_time / brylic_time

        results.append((tile_size, brylic_time, speedup))
        print(f"  Runtime: {brylic_time:.3f} s")
        print(f"  Speedup: {speedup:.2f}×")
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Tile Size':<15} {'Runtime (s)':<15} {'Speedup':<15}")
    print("-" * 60)
    for tile_size, brylic_time, speedup in results:
        print(f"{tile_size}×{tile_size:<10} {brylic_time:<15.3f} {speedup:<15.2f}×")

    # Find best
    best_tile_size, best_time, best_speedup = min(results, key=lambda x: x[1])
    print("-" * 60)
    print(f"Best: {best_tile_size}×{best_tile_size} ({best_time:.3f} s, {best_speedup:.2f}× speedup)")


if __name__ == "__main__":
    run_tile_scan()
