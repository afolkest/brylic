from numpy import dtype, ndarray
from numpy import bool_ as bool_
from numpy import float32 as f32
from numpy import float64 as f64

from rlic._typing import BoundaryPair, UVMode

def convolve_f32(
    texture: ndarray[tuple[int, int], dtype[f32]],
    uv: tuple[
        ndarray[tuple[int, int], dtype[f32]],
        ndarray[tuple[int, int], dtype[f32]],
        UVMode,
    ],
    kernel: ndarray[tuple[int], dtype[f32]],
    boundaries: tuple[BoundaryPair, BoundaryPair],
    iterations: int = 1,
    blocked: ndarray[tuple[int, int], dtype[bool_]] | None = ...,
    edge_gain_strength: f32 = ...,
    edge_gain_power: f32 = ...,
    tile_shape: tuple[int, int] | None = ...,
    overlap: int | None = ...,
    num_threads: int | None = ...,
) -> ndarray[tuple[int, int], dtype[f32]]: ...
def convolve_f64(
    texture: ndarray[tuple[int, int], dtype[f64]],
    uv: tuple[
        ndarray[tuple[int, int], dtype[f64]],
        ndarray[tuple[int, int], dtype[f64]],
        UVMode,
    ],
    kernel: ndarray[tuple[int], dtype[f64]],
    boundaries: tuple[BoundaryPair, BoundaryPair],
    iterations: int = 1,
    blocked: ndarray[tuple[int, int], dtype[bool_]] | None = ...,
    edge_gain_strength: f64 = ...,
    edge_gain_power: f64 = ...,
    tile_shape: tuple[int, int] | None = ...,
    overlap: int | None = ...,
    num_threads: int | None = ...,
) -> ndarray[tuple[int, int], dtype[f64]]: ...
