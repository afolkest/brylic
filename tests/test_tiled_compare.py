"""Validate that tiled_convolve matches convolve for representative inputs."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from brylic import convolve, tiled_convolve

DEFAULT_STREAMLENGTH_FACTOR = 60.0 / 1024.0


def _circular_mask(shape: tuple[int, int], radius_fraction: float = 0.18) -> np.ndarray:
    height, width = shape
    cy = (height - 1) * 0.5
    cx = (width - 1) * 0.5
    radius = radius_fraction * min(height, width)
    yy, xx = np.ogrid[:height, :width]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def _generate_noise(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


def _make_field(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    texture = _generate_noise(shape)
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[1], dtype=np.float32),
        indexing="ij",
    )
    radius = np.hypot(xx, yy).astype(np.float32) + 1e-6
    u = (-yy / radius).astype(np.float32)
    v = (xx / radius).astype(np.float32)
    streamlength = max(1, int(round(DEFAULT_STREAMLENGTH_FACTOR * min(shape))))
    positions = np.arange(1 - streamlength, streamlength, dtype=np.float32)
    kernel = 0.5 * (1.0 + np.cos(np.pi * positions / streamlength))
    return texture, u, v, kernel


def test_tiled_matches_full_convolution() -> None:
    shape = (256, 256)
    texture, u, v, kernel = _make_field(shape)
    mask = _circular_mask(shape)

    baseline = convolve(
        texture,
        u,
        v,
        kernel=kernel,
        iterations=2,
        boundaries="closed",
        mask=mask,
        edge_gain_strength=0.08,
        edge_gain_power=2.0,
    )
    tiled = tiled_convolve(
        texture,
        u,
        v,
        kernel=kernel,
        iterations=2,
        boundaries="closed",
        mask=mask,
        edge_gain_strength=0.08,
        edge_gain_power=2.0,
        tile_shape=(128, 128),
    )

    assert_allclose(tiled, baseline, rtol=1e-6, atol=5e-6)
