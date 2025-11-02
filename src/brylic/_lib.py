from __future__ import annotations

"""Python wrapper for the bryLIC experimental kernels."""

__all__ = ["convolve", "tiled_convolve"]

import sys
from typing import TYPE_CHECKING, cast

import numpy as np

from ._boundaries import BoundarySet
from ._core import convolve_f32, convolve_f64
from ._typing import Boundary, BoundaryDict, BoundaryPair, UVMode

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup  # pyright: ignore[reportUnreachable]

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeVar

    from numpy import dtype, ndarray
    from numpy import float32 as f32
    from numpy import float64 as f64

    F = TypeVar("F", f32, f64)

_KNOWN_UV_MODES = ["velocity", "polarization"]
_SUPPORTED_DTYPES: list[np.dtype[np.floating]] = [
    np.dtype("float32"),
    np.dtype("float64"),
]


def convolve(
    texture: ndarray[tuple[int, int], dtype[F]],
    /,
    u: ndarray[tuple[int, int], dtype[F]],
    v: ndarray[tuple[int, int], dtype[F]],
    *,
    kernel: ndarray[tuple[int], dtype[F]],
    uv_mode: UVMode = "velocity",
    boundaries: Boundary | BoundaryDict = "closed",
    iterations: int = 1,
    mask: ndarray[tuple[int, int], dtype[np.bool_]] | None = None,
    edge_gain_strength: float = 0.0,
    edge_gain_power: float = 2.0,
    tile_shape: tuple[int, int] | None = None,
    overlap: int | None = None,
    num_threads: int | None = None,
) -> ndarray[tuple[int, int], dtype[F]]:
    """2-dimensional line integral convolution.

    This is a direct copy of the production wrapper so experiments can start
    from known-good behavior before layering additional logic.
    """
    exceptions: list[Exception] = []
    if iterations < 0:
        exceptions.append(
            ValueError(
                f"Invalid number of iterations: {iterations}\n"
                "Expected a strictly positive integer."
            )
        )

    if uv_mode not in _KNOWN_UV_MODES:
        exceptions.append(
            ValueError(
                f"Invalid uv_mode {uv_mode!r}. Expected one of {_KNOWN_UV_MODES}"
            )
        )

    dtype_error_expectations = (
        f"Expected texture, u, v and kernel with identical dtype, from {_SUPPORTED_DTYPES}. "
        f"Got {texture.dtype=}, {u.dtype=}, {v.dtype=}, {kernel.dtype=}"
    )

    input_dtypes = {arr.dtype for arr in (texture, u, v, kernel)}
    if unsupported_dtypes := input_dtypes.difference(_SUPPORTED_DTYPES):
        exceptions.append(
            TypeError(
                f"Found unsupported data type(s): {list(unsupported_dtypes)}. "
                f"{dtype_error_expectations}"
            )
        )

    if len(input_dtypes) != 1:
        exceptions.append(TypeError(f"Data types mismatch. {dtype_error_expectations}"))

    if texture.ndim != 2:
        exceptions.append(
            ValueError(
                f"Expected a texture with exactly two dimensions. Got {texture.ndim=}"
            )
        )
    if np.any(texture < 0):
        exceptions.append(
            ValueError(
                "Found invalid texture element(s). Expected only positive values."
            )
        )
    if u.shape != texture.shape or v.shape != texture.shape:
        exceptions.append(
            ValueError(
                "Shape mismatch: expected texture, u and v with identical shapes. "
                f"Got {texture.shape=}, {u.shape=}, {v.shape=}"
            )
        )

    if kernel.ndim != 1:
        exceptions.append(
            ValueError(
                f"Expected a kernel with exactly one dimension. Got {kernel.ndim=}"
            )
        )
    if np.any(~np.isfinite(kernel)):
        exceptions.append(ValueError("Found non-finite value(s) in kernel."))

    if (bs := BoundarySet.from_user_input(boundaries)) is None:
        exceptions.append(TypeError(f"Invalid boundary specification {boundaries}"))
    else:
        exceptions.extend(bs.collect_exceptions())

    if len(exceptions) == 1:
        raise exceptions[0]
    elif exceptions:
        raise ExceptionGroup("Invalid inputs were received.", exceptions)

    bs = cast("BoundarySet", bs)
    if iterations == 0:
        return texture.copy()

    input_dtype = texture.dtype
    retf: Callable[
        [
            ndarray[tuple[int, int], dtype[F]],
            tuple[
                ndarray[tuple[int, int], dtype[F]],
                ndarray[tuple[int, int], dtype[F]],
                UVMode,
            ],
            ndarray[tuple[int], dtype[F]],
            tuple[BoundaryPair, BoundaryPair],
            int,
            ndarray[tuple[int, int], dtype[np.bool_]] | None,
            F,
            F,
            tuple[int, int] | None,
            int | None,
            int | None,
        ],
        ndarray[tuple[int, int], dtype[F]],
    ]
    if input_dtype == np.dtype("float32"):
        retf = convolve_f32  # type: ignore[assignment] # pyright: ignore[reportAssignmentType]
    elif input_dtype == np.dtype("float64"):
        retf = convolve_f64  # type: ignore[assignment] # pyright: ignore[reportAssignmentType]
    else:  # pragma: no cover - should be impossible due to validation above
        raise RuntimeError

    try:
        return retf(
            texture,
            (u, v, uv_mode),
            kernel,
            (bs.x, bs.y),
            iterations,
            mask,
            edge_gain_strength,  # coerced to F by pyo3
            edge_gain_power,
            tile_shape,
            overlap,
            num_threads,
        )
    except TypeError:
        # Older build without mask support: fall back to no-mask call.
        retf_nomask: Callable[
            [
                ndarray[tuple[int, int], dtype[F]],
                tuple[
                    ndarray[tuple[int, int], dtype[F]],
                    ndarray[tuple[int, int], dtype[F]],
                    UVMode,
                ],
            ndarray[tuple[int], dtype[F]],
            tuple[BoundaryPair, BoundaryPair],
            int,
        ],
        ndarray[tuple[int, int], dtype[F]],
    ] = retf  # type: ignore[assignment]
        return retf_nomask(texture, (u, v, uv_mode), kernel, (bs.x, bs.y), iterations)


def tiled_convolve(
    texture: ndarray[tuple[int, int], dtype[F]],
    /,
    u: ndarray[tuple[int, int], dtype[F]],
    v: ndarray[tuple[int, int], dtype[F]],
    *,
    kernel: ndarray[tuple[int], dtype[F]],
    uv_mode: UVMode = "velocity",
    boundaries: Boundary | BoundaryDict = "closed",
    iterations: int = 1,
    mask: ndarray[tuple[int, int], dtype[np.bool_]] | None = None,
    edge_gain_strength: float = 0.0,
    edge_gain_power: float = 2.0,
    tile_shape: tuple[int, int] | None = (512, 512),
    overlap: int | None = None,
    num_threads: int | None = None,
) -> ndarray[tuple[int, int], dtype[F]]:
    """Convenience wrapper that enables tiling parameters by default."""

    return convolve(
        texture,
        u,
        v,
        kernel=kernel,
        uv_mode=uv_mode,
        boundaries=boundaries,
        iterations=iterations,
        mask=mask,
        edge_gain_strength=edge_gain_strength,
        edge_gain_power=edge_gain_power,
        tile_shape=tile_shape,
        overlap=overlap,
        num_threads=num_threads,
    )
