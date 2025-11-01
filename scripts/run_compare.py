"""Compare baseline bryLIC runs against mask-aware and tiled variants.

Usage
-----
    python scripts/run_compare.py --resolutions 512x512 1024x1024

Dependencies
------------
Install minimal extras with::

    pip install -r scripts/requirements.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Literal

import numpy as np
from PIL import Image

try:  # Optional – improves noise filtering and resampling.
    from scipy.ndimage import gaussian_filter, zoom
except Exception:  # pragma: no cover - SciPy not installed
    gaussian_filter = None
    zoom = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Import vanilla rlic as baseline (from system/flowcol installation)
import rlic as vanilla_rlic

# Import local brylic implementation (from src/)
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import rlic as brylic  # Local version is also called 'rlic' but we alias it as brylic

# Roughly mirrors FlowCol defaults for comparable aesthetics.
DEFAULT_STREAMLENGTH_FACTOR = 60.0 / 1024.0
DEFAULT_RENDER_PASSES = 2
DEFAULT_NOISE_SEED = 0
DEFAULT_NOISE_SIGMA = 0.0

UVMode = Literal["velocity", "polarization"]


def _circular_mask(shape: tuple[int, int], radius_fraction: float = 0.14) -> np.ndarray:
    """Binary mask marking a filled circle at the array centre."""
    height, width = shape
    cy = (height - 1) * 0.5
    cx = (width - 1) * 0.5
    radius = radius_fraction * min(height, width)
    yy, xx = np.ogrid[:height, :width]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def _generate_noise(
    shape: tuple[int, int],
    *,
    seed: int = DEFAULT_NOISE_SEED,
    oversample: float = 1.5,
    lowpass_sigma: float = DEFAULT_NOISE_SIGMA,
) -> np.ndarray:
    """Band-limited noise similar to FlowCol's render pipeline."""
    rng = np.random.default_rng(seed)
    height, width = shape
    if oversample > 1.0:
        hi_h = max(1, int(round(height * oversample)))
        hi_w = max(1, int(round(width * oversample)))
        base = rng.random((hi_h, hi_w), dtype=np.float32)
    else:
        base = rng.random((height, width), dtype=np.float32)

    if lowpass_sigma > 0.0 and gaussian_filter is not None:
        base = gaussian_filter(base, sigma=lowpass_sigma)

    if oversample > 1.0:
        if zoom is not None:
            scale_y = height / base.shape[0]
            scale_x = width / base.shape[1]
            base = zoom(base, (scale_y, scale_x), order=1)
        else:  # fallback: simple cropping
            base = base[:height, :width]

    base_min = float(base.min())
    base_max = float(base.max())
    if base_max > base_min:
        base = (base - base_min) / (base_max - base_min)
    else:
        base = np.zeros_like(base)
    return base.astype(np.float32)


def _cosine_kernel(streamlength: int) -> np.ndarray:
    """Classic raised-cosine kernel sized by streamlength."""
    streamlength = max(int(streamlength), 1)
    positions = np.arange(1 - streamlength, streamlength, dtype=np.float32)
    return 0.5 * (1.0 + np.cos(np.pi * positions / streamlength))


def _lic_to_unit(arr: np.ndarray) -> np.ndarray:
    """Convert raw LIC array to [0,1] by max-abs scaling and remap."""
    arr = arr.astype(np.float32, copy=False)
    max_abs = float(np.max(np.abs(arr)))
    if max_abs > 1e-12:
        arr = arr / max_abs
    return 0.5 * (arr + 1.0)  # [-1,1] -> [0,1]


def _make_test_case(
    shape: tuple[int, int],
    *,
    uv_mode: UVMode,
    field: Literal["swirl", "radial"] = "swirl",
    streamlength_factor: float = DEFAULT_STREAMLENGTH_FACTOR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a reproducible LIC test case."""
    texture = _generate_noise(shape)
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[1], dtype=np.float32),
        indexing="ij",
    )
    radius = np.hypot(xx, yy).astype(np.float32) + 1e-6

    if field == "radial":
        u = (xx / radius).astype(np.float32)
        v = (yy / radius).astype(np.float32)
        u[shape[0] // 2, shape[1] // 2] = 0.0
        v[shape[0] // 2, shape[1] // 2] = 0.0
    elif field == "swirl":
        u = (-yy / radius).astype(np.float32)
        v = (xx / radius).astype(np.float32)
        u += 0.05 * xx
        v += 0.05 * yy
    else:  # pragma: no cover
        raise ValueError(f"Unknown field type {field!r}")

    streamlength_pixels = max(1, int(round(streamlength_factor * min(shape))))
    kernel = _cosine_kernel(streamlength_pixels)
    return texture, u, v, kernel


def measure_once(
    shape: tuple[int, int],
    *,
    uv_mode: UVMode = "velocity",
    iterations: int = DEFAULT_RENDER_PASSES,
    boundaries: object = "closed",
    edge_gain_strength: float = 0.08,
    edge_gain_power: float = 2.0,
    tile_shape: tuple[int, int] | None = (512, 512),
    overlap: int | None = None,
    num_threads: int | None = None,
) -> dict[str, float | bool]:
    texture, u, v, kernel = _make_test_case(shape, uv_mode=uv_mode, field="radial")
    mask = _circular_mask(shape)

    u_masked = u.copy()
    v_masked = v.copy()
    u_masked[mask] = 0.0
    v_masked[mask] = 0.0

    start = time.perf_counter()
    baseline = vanilla_rlic.convolve(
        texture,
        u_masked,
        v_masked,
        kernel=kernel,
        uv_mode=uv_mode,
        boundaries=boundaries,
        iterations=iterations,
    )
    baseline_time = time.perf_counter() - start

    start = time.perf_counter()
    masked = brylic.convolve(
        texture,
        u_masked,
        v_masked,
        kernel=kernel,
        uv_mode=uv_mode,
        boundaries=boundaries,
        iterations=iterations,
        mask=mask,
        edge_gain_strength=0.0,
        edge_gain_power=edge_gain_power,
    )
    masked_time = time.perf_counter() - start

    masked_gain = brylic.convolve(
        texture,
        u_masked,
        v_masked,
        kernel=kernel,
        uv_mode=uv_mode,
        boundaries=boundaries,
        iterations=iterations,
        mask=mask,
        edge_gain_strength=edge_gain_strength,
        edge_gain_power=edge_gain_power,
    )

    start = time.perf_counter()
    tiled = brylic.tiled_convolve(
        texture,
        u_masked,
        v_masked,
        kernel=kernel,
        uv_mode=uv_mode,
        boundaries=boundaries,
        iterations=iterations,
        mask=mask,
        edge_gain_strength=0.0,
        edge_gain_power=edge_gain_power,
        tile_shape=tile_shape,
        overlap=overlap,
        num_threads=num_threads,
    )
    tiled_time = time.perf_counter() - start

    diff_masked = masked - baseline
    diff_tiled = tiled - masked

    return {
        "baseline_time": baseline_time,
        "masked_time": masked_time,
        "tiled_time": tiled_time,
        "speedup_masked": baseline_time / masked_time if masked_time else float("inf"),
        "speedup_tiled": masked_time / tiled_time if tiled_time else float("inf"),
        "max_abs_masked": float(np.max(np.abs(diff_masked))),
        "rms_masked": float(np.sqrt(np.mean(diff_masked**2))),
        "max_abs_tiled": float(np.max(np.abs(diff_tiled))),
        "rms_tiled": float(np.sqrt(np.mean(diff_tiled**2))),
        "baseline": baseline,
        "masked": masked,
        "tiled": tiled,
        "masked_gain": masked_gain,
        "mask": mask,
    }


def print_metrics(shape: tuple[int, int], metrics: dict[str, float | bool]) -> None:
    print(f"Resolution: {shape[1]}×{shape[0]}")
    print(f"Baseline rlic runtime:      {metrics['baseline_time']:.3f} s")
    print(f"Experimental runtime:       {metrics['masked_time']:.3f} s")
    print(f"Speedup (baseline/exp):     {metrics['speedup_masked']:.2f}×")
    print(f"Max |Δ|:                    {metrics['max_abs_masked']:.3e}")
    print(f"RMS Δ:                      {metrics['rms_masked']:.3e}")
    print(f"Tiled runtime:              {metrics['tiled_time']:.3f} s")
    print(f"Speedup (exp/tiled):        {metrics['speedup_tiled']:.2f}×")
    print(f"Tiled Max |Δ|:              {metrics['max_abs_tiled']:.3e}")
    print(f"Tiled RMS Δ:                {metrics['rms_tiled']:.3e}")


def _locked_display_panels(
    panels: dict[str, np.ndarray],
    order: list[str],
    *,
    clip_percent: float = 1.5,
    contrast: float = 1.5,
    mask: np.ndarray | None = None,
    edge_display_boost: float = 1.0,
) -> list[np.ndarray]:
    """Normalize multiple panels using baseline percentiles."""
    if "baseline" not in panels:
        raise ValueError("panels dictionary must include a 'baseline' entry.")

    base_unit = _lic_to_unit(panels["baseline"])
    vmin = float(np.percentile(base_unit, clip_percent))
    vmax = float(np.percentile(base_unit, 100.0 - clip_percent))
    if vmax <= vmin:
        vmin, vmax = float(np.min(base_unit)), float(np.max(base_unit))
        if vmax <= vmin:
            vmin, vmax = 0.0, 1.0

    def normalize_with_ref(unit_arr: np.ndarray) -> np.ndarray:
        if vmax > vmin:
            norm = np.clip((unit_arr - vmin) / (vmax - vmin), 0.0, 1.0)
        else:
            norm = np.zeros_like(unit_arr, dtype=np.float32)
        if not np.isclose(contrast, 1.0):
            norm = np.clip((norm - 0.5) * contrast + 0.5, 0.0, 1.0)
        return norm

    norms: dict[str, np.ndarray] = {}
    for name, arr in panels.items():
        unit = _lic_to_unit(arr)
        norms[name] = normalize_with_ref(unit)

    if "experimental_gain" in norms:
        sel = ~mask if mask is not None else slice(None)
        base_norm = norms["baseline"]
        gain_norm = norms["experimental_gain"]
        ref_high = float(np.percentile(base_norm[sel], 99.0))
        gain_high = float(np.percentile(gain_norm[sel], 99.0))
        eps = 1e-6
        if gain_high > eps and ref_high > eps:
            scale = (ref_high / gain_high) * float(edge_display_boost)
            norms["experimental_gain"] = np.clip(gain_norm * scale, 0.0, 1.0)

    return [norms[name] for name in order if name in norms]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline rlic with experimental implementation.")
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Field width for single-run mode",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Field height for single-run mode",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_RENDER_PASSES,
        help=f"Number of LIC iterations (default: {DEFAULT_RENDER_PASSES})",
    )
    parser.add_argument(
        "--uv-mode",
        choices=["velocity", "polarization"],
        default="velocity",
        help="Vector field interpretation (default: velocity)",
    )
    parser.add_argument(
        "--resolutions",
        nargs="*",
        help="WIDTHxHEIGHT (ignored if multiple are given; last one is used).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of repetitions per resolution to average timings.",
    )
    parser.add_argument(
        "--edge-gain-strength",
        type=float,
        default=0.12,
        help="Edge gain strength for the 3rd panel (default: 0.06).",
    )
    parser.add_argument(
        "--edge-gain-power",
        type=float,
        default=1.0,
        help="Edge gain power for the 3rd panel (default: 2.0).",
    )
    parser.add_argument(
        "--edge-display-boost",
        type=float,
        default=1.0,
        help="Display-only boost applied to the right panel after normalization (default: 1.0).",
    )
    parser.add_argument(
        "--tile-size",
        type=str,
        default="512x512",
        help="Tile core size HEIGHTxWIDTH for tiled LIC (use 'full' to disable tiling).",
    )
    parser.add_argument(
        "--tiled-overlap",
        type=int,
        default=None,
        help="Override tile overlap in pixels (default: auto).",
    )
    parser.add_argument(
        "--tiled-processes",
        type=int,
        default=None,
        help="Number of worker processes for tiled LIC (default: CPU count).",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=HERE / "output" / "comparison.png",
        help="Path to save a baseline|experimental PNG comparison (default: output/comparison.png).",
    )
    args = parser.parse_args()

    # Decide a single target resolution (final only).
    if args.resolutions:
        # Use only the last provided resolution token.
        item = args.resolutions[-1]
        try:
            width_str, height_str = item.lower().split("x")
            width = int(width_str)
            height = int(height_str)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"Invalid resolution '{item}'. Expected format WIDTHxHEIGHT.") from exc
        resolution = (height, width)
    else:
        if args.width is None or args.height is None:
            resolution = (1024, 1024)
        else:
            resolution = (args.height, args.width)

    repeat = max(1, args.repeat)

    last_baseline_arr: np.ndarray | None = None
    last_experimental_arr: np.ndarray | None = None
    last_tiled_arr: np.ndarray | None = None
    last_shape: tuple[int, int] | None = None

    tile_shape: tuple[int, int] | None
    tile_arg = args.tile_size.strip().lower() if isinstance(args.tile_size, str) else ""
    if tile_arg in ("full", "none", "all"):
        tile_shape = None
    else:
        try:
            h_str, w_str = tile_arg.split("x")
            tile_shape = (max(1, int(h_str)), max(1, int(w_str)))
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"Invalid --tile-size '{args.tile_size}'. Expected 'HxW' or 'full'.") from exc

    shape = resolution
    results = [
        measure_once(
            shape,
            iterations=args.iterations,
            uv_mode=args.uv_mode,
            edge_gain_strength=args.edge_gain_strength,
            edge_gain_power=args.edge_gain_power,
            tile_shape=tile_shape,
            overlap=args.tiled_overlap,
            num_threads=args.tiled_processes,
        )
        for _ in range(repeat)
    ]
    avg_metrics: dict[str, float | bool] = {}
    for key, value in results[0].items():
        if isinstance(value, float):
            avg_metrics[key] = mean(result[key] for result in results)  # type: ignore[arg-type]
        elif isinstance(value, bool):
            avg_metrics[key] = all(result[key] for result in results)  # type: ignore[arg-type]
        else:
            # Skip non-metric keys (e.g., arrays) in the averaged report
            continue

    print_metrics(shape, avg_metrics)
    if repeat > 1:
        print(f"  (averaged over {repeat} runs)")
    print()

    # Capture arrays from the last timing run (highest quality) for saving.
    last_run = results[-1]
    last_baseline_arr = last_run["baseline"]  # type: ignore[index]
    last_experimental_arr = last_run["masked"]  # type: ignore[index]
    last_tiled_arr = last_run["tiled"]  # type: ignore[index]
    last_experimental_gain_arr = last_run["masked_gain"]  # type: ignore[index]
    last_shape = shape

    if (
        args.output_image
        and last_baseline_arr is not None
        and last_experimental_arr is not None
        and last_tiled_arr is not None
        and last_experimental_gain_arr is not None
    ):
        # Reuse the last computed arrays and lock normalization to the baseline panel.
        demo_mask = _circular_mask(last_shape)
        panels = _locked_display_panels(
            {
                "baseline": last_baseline_arr,
                "experimental": last_experimental_arr,
                "tiled": last_tiled_arr,
                "experimental_gain": last_experimental_gain_arr,
            },
            order=["baseline", "experimental", "tiled", "experimental_gain"],
            clip_percent=1.5,
            contrast=1.5,
            mask=demo_mask,
            edge_display_boost=args.edge_display_boost,
        )
        # Visualize the synthetic mask as interior fill for parity.
        for panel in panels:
            panel[demo_mask] = 0.0

        stacked = np.concatenate(panels, axis=1)
        png = (stacked * 255.0).clip(0, 255).astype(np.uint8)
        args.output_image.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(png, mode="L").save(args.output_image)
        print(f"Saved visual comparison to {args.output_image}")


if __name__ == "__main__":
    main()
