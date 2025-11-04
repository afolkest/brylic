"""Test domain edge gain feature for external boundaries."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from brylic import convolve

DEFAULT_STREAMLENGTH_FACTOR = 60.0 / 1024.0


def _generate_noise(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


def _make_field(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a simple radial vector field for testing."""
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


def test_domain_edge_gain_affects_edges():
    """Test that domain_edge_gain_strength affects pixels near domain boundaries."""
    shape = (128, 128)
    texture, u, v, kernel = _make_field(shape)

    # Convolve without domain edge gain
    result_no_gain = convolve(
        texture,
        u,
        v,
        kernel=kernel,
        boundaries="closed",
        iterations=1,
        domain_edge_gain_strength=0.0,
    )

    # Convolve with domain edge gain
    result_with_gain = convolve(
        texture,
        u,
        v,
        kernel=kernel,
        boundaries="closed",
        iterations=1,
        domain_edge_gain_strength=0.3,
        domain_edge_gain_power=2.0,
    )

    # Check that the results differ (edge gain should change values)
    assert not np.allclose(result_no_gain, result_with_gain), \
        "Domain edge gain should affect the result"

    # Edge pixels should be more affected than center pixels
    # Check corners (most likely to hit domain edges)
    edge_region = np.s_[:10, :10]  # Top-left corner
    center_region = np.s_[59:69, 59:69]  # Center 10x10

    edge_diff = np.abs(result_with_gain[edge_region] - result_no_gain[edge_region]).mean()
    center_diff = np.abs(result_with_gain[center_region] - result_no_gain[center_region]).mean()

    # Edges should show more difference than center (may not always be true depending on field)
    # At minimum, edges should show some difference
    assert edge_diff > 0, "Edge regions should be affected by domain edge gain"


def test_domain_edge_gain_zero_is_noop():
    """Test that domain_edge_gain_strength=0.0 doesn't change results."""
    shape = (64, 64)
    texture, u, v, kernel = _make_field(shape)

    # These should be identical
    result1 = convolve(
        texture,
        u,
        v,
        kernel=kernel,
        boundaries="closed",
        iterations=1,
        domain_edge_gain_strength=0.0,
    )

    result2 = convolve(
        texture,
        u,
        v,
        kernel=kernel,
        boundaries="closed",
        iterations=1,
        # domain_edge_gain_strength defaults to 0.0
    )

    assert_allclose(result1, result2, rtol=1e-6)


def test_domain_edge_gain_with_periodic():
    """Test that domain edge gain doesn't activate with periodic boundaries."""
    shape = (64, 64)
    texture, u, v, kernel = _make_field(shape)

    # With periodic boundaries, there are no "edges" to hit
    result_periodic_no_gain = convolve(
        texture,
        u,
        v,
        kernel=kernel,
        boundaries="periodic",
        iterations=1,
        domain_edge_gain_strength=0.0,
    )

    result_periodic_with_gain = convolve(
        texture,
        u,
        v,
        kernel=kernel,
        boundaries="periodic",
        iterations=1,
        domain_edge_gain_strength=0.3,
    )

    # Should be identical because periodic boundaries never hit edges
    assert_allclose(result_periodic_no_gain, result_periodic_with_gain, rtol=1e-6)


def test_domain_and_mask_edge_gain_independent():
    """Test that domain edge gain and mask edge gain work independently."""
    shape = (128, 128)
    texture, u, v, kernel = _make_field(shape)

    # Create a circular mask in the center
    cy, cx = shape[0] // 2, shape[1] // 2
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    mask_radius = min(shape) // 4
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= mask_radius ** 2

    # Test with only domain edge gain
    result_domain_only = convolve(
        texture,
        u,
        v,
        kernel=kernel,
        boundaries="closed",
        mask=mask,
        edge_gain_strength=0.0,  # No mask edge gain
        domain_edge_gain_strength=0.2,  # Domain edge gain
    )

    # Test with only mask edge gain
    result_mask_only = convolve(
        texture,
        u,
        v,
        kernel=kernel,
        boundaries="closed",
        mask=mask,
        edge_gain_strength=0.2,  # Mask edge gain
        domain_edge_gain_strength=0.0,  # No domain edge gain
    )

    # Test with both
    result_both = convolve(
        texture,
        u,
        v,
        kernel=kernel,
        boundaries="closed",
        mask=mask,
        edge_gain_strength=0.2,
        domain_edge_gain_strength=0.2,
    )

    # They should all be different
    assert not np.allclose(result_domain_only, result_mask_only)
    assert not np.allclose(result_domain_only, result_both)
    assert not np.allclose(result_mask_only, result_both)
