use crate::boundaries::{Boundary, BoundaryPair, BoundarySet};
use either::Either;
use num_traits::{abs, signum, Float, Signed};
use numpy::borrow::{PyReadonlyArray1, PyReadonlyArray2};
use numpy::ndarray::{s, Array2, ArrayView1, ArrayView2};
use numpy::{PyArray2, ToPyArray};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use std::ops::{AddAssign, Mul, Neg};
use std::sync::Mutex;
use std::thread;

#[derive(Clone)]
enum UVMode {
    Velocity,
    Polarization,
}
impl UVMode {
    fn new(uv_mode: String) -> UVMode {
        match uv_mode.as_str() {
            "polarization" => UVMode::Polarization,
            "velocity" => UVMode::Velocity,
            _ => panic!("unknown uv_mode"),
        }
    }
}

#[derive(Clone)]
struct UVField<'a, T> {
    u: ArrayView2<'a, T>,
    v: ArrayView2<'a, T>,
    mode: UVMode,
}

struct PixelFraction<T> {
    x: T,
    y: T,
}

#[derive(Clone)]
struct ImageDimensions {
    x: usize,
    y: usize,
}

#[derive(Clone)]
struct PixelCoordinates {
    x: usize,
    y: usize,
    dimensions: ImageDimensions,
}
impl PixelCoordinates {
    fn apply_one_dir(c: &mut usize, image_size: usize, boundaries: &BoundaryPair) {
        if *c == usize::MAX {
            *c = match boundaries.left {
                Boundary::Closed => 0,
                Boundary::Periodic => image_size - 1,
            };
        } else if *c == image_size {
            *c = match boundaries.right {
                Boundary::Closed => image_size - 1,
                Boundary::Periodic => 0,
            };
        }
    }
    fn apply(&mut self, boundaries: &BoundarySet) {
        PixelCoordinates::apply_one_dir(&mut self.x, self.dimensions.x, &boundaries.x);
        PixelCoordinates::apply_one_dir(&mut self.y, self.dimensions.y, &boundaries.y);
    }
}
mod boundaries {
    #[derive(Clone, Copy)]
    pub enum Boundary {
        Closed,
        Periodic,
    }
    impl Boundary {
        fn new(boundary: String) -> Boundary {
            match boundary.as_str() {
                "closed" => Boundary::Closed,
                "periodic" => Boundary::Periodic,
                _ => panic!("unknown boundary"),
            }
        }
    }

    #[derive(Clone, Copy)]
    pub struct BoundaryPair {
        pub left: Boundary,
        pub right: Boundary,
    }
    impl BoundaryPair {
        fn new(pair: (String, String)) -> BoundaryPair {
            BoundaryPair {
                left: Boundary::new(pair.0),
                right: Boundary::new(pair.1),
            }
        }
    }

    #[derive(Clone, Copy)]
    pub struct BoundarySet {
        pub x: BoundaryPair,
        pub y: BoundaryPair,
    }
    impl BoundarySet {
        pub fn new(set: ((String, String), (String, String))) -> BoundarySet {
            BoundarySet {
                x: BoundaryPair::new(set.0),
                y: BoundaryPair::new(set.1),
            }
        }
    }
}

#[derive(Clone)]
struct UVPoint<T: Copy> {
    u: T,
    v: T,
}
impl<T: Neg<Output = T> + Copy> Neg for UVPoint<T> {
    type Output = UVPoint<T>;

    fn neg(self) -> Self::Output {
        UVPoint {
            u: -self.u,
            v: -self.v,
        }
    }
}

fn select_pixel<T: Copy>(arr: &ArrayView2<T>, coords: &PixelCoordinates) -> T {
    arr[[coords.y, coords.x]]
}

#[cfg(test)]
mod test_pixel_select {
    use numpy::ndarray::array;

    use crate::{select_pixel, ImageDimensions, PixelCoordinates};
    #[test]
    fn selection() {
        let arr = array![[1.0, 2.0], [3.0, 4.0]];
        let coords = PixelCoordinates {
            x: 1,
            y: 1,
            dimensions: ImageDimensions { x: 4, y: 4 },
        };
        let res = select_pixel(&arr.view(), &coords);
        assert_eq!(res, 4.0);
    }
}

trait AtLeastF32:
    Float + From<f32> + Signed + AddAssign<<Self as Mul>::Output> + Send + Sync
{
}
impl AtLeastF32 for f32 {}
impl AtLeastF32 for f64 {}

fn time_to_next_pixel<T: AtLeastF32>(velocity: T, current_frac: T) -> T {
    // this is the branchless version of
    // if velocity > 0.0 {
    //     (1.0 - current_frac) / velocity
    // } else {
    //     -(current_frac / velocity)
    // }
    let one: T = 1.0.into();
    let two: T = 2.0.into();
    let mtwo = -two;
    let d1 = current_frac;
    let remaining_frac = (one + signum(velocity)).mul_add((mtwo.mul_add(d1, one)) / two, d1);
    abs(remaining_frac / velocity)
}

#[cfg(test)]
mod test_time_to_next_pixel {
    use super::time_to_next_pixel;
    use std::assert_eq;
    #[test]
    fn positive_vel() {
        let res = time_to_next_pixel(1.0, 0.0);
        assert_eq!(res, 1.0);
    }
    #[test]
    fn negative_vel() {
        let res = time_to_next_pixel(-1.0, 1.0);
        assert_eq!(res, 1.0);
    }
    #[test]
    fn infinite_time_f32() {
        let res = time_to_next_pixel(0.0f32, 0.5f32);
        assert_eq!(res, f32::INFINITY);
    }
    #[test]
    fn infinite_time_f64() {
        let res = time_to_next_pixel(0.0, 0.5);
        assert_eq!(res, f64::INFINITY);
    }
}

#[inline(always)]
fn update_state<T: AtLeastF32>(
    velocity_parallel: &T,
    velocity_orthogonal: &T,
    coord_parallel: &mut usize,
    frac_parallel: &mut T,
    frac_orthogonal: &mut T,
    time_parallel: &T,
) {
    if *velocity_parallel >= 0.0.into() {
        *coord_parallel += 1;
        *frac_parallel = 0.0.into();
    } else {
        *coord_parallel = coord_parallel.wrapping_sub(1);
        *frac_parallel = 1.0.into();
    }
    *frac_orthogonal = (*time_parallel).mul_add(*velocity_orthogonal, *frac_orthogonal);
}

#[inline(always)]
fn advance<T: AtLeastF32>(
    uv: &UVPoint<T>,
    coords: &mut PixelCoordinates,
    pix_frac: &mut PixelFraction<T>,
    boundaries: &BoundarySet,
) {
    if uv.u == 0.0.into() && uv.v == 0.0.into() {
        return;
    }

    let tx = time_to_next_pixel(uv.u, pix_frac.x);
    let ty = time_to_next_pixel(uv.v, pix_frac.y);

    if tx < ty {
        // We reached the next pixel along x first.
        update_state(
            &uv.u,
            &uv.v,
            &mut coords.x,
            &mut pix_frac.x,
            &mut pix_frac.y,
            &tx,
        );
    } else {
        // We reached the next pixel along y first.
        update_state(
            &uv.v,
            &uv.u,
            &mut coords.y,
            &mut pix_frac.y,
            &mut pix_frac.x,
            &ty,
        );
    }
    // All boundary conditions must be applicable on each step.
    // This is done to allow for complex cases like shearing boxes.
    coords.apply(boundaries);
}

#[cfg(test)]
mod test_advance {
    use crate::{
        advance, Boundary, BoundaryPair, BoundarySet, ImageDimensions, PixelCoordinates,
        PixelFraction, UVPoint,
    };

    #[test]
    fn zero_vel() {
        let uv = UVPoint { u: 0.0, v: 0.0 };
        let mut coords = PixelCoordinates {
            x: 5,
            y: 5,
            dimensions: ImageDimensions { x: 10, y: 10 },
        };
        let mut pix_frac = PixelFraction { x: 0.5, y: 0.5 };
        let boundaries = BoundarySet {
            x: BoundaryPair {
                left: Boundary::Closed,
                right: Boundary::Closed,
            },
            y: BoundaryPair {
                left: Boundary::Closed,
                right: Boundary::Closed,
            },
        };
        advance(&uv, &mut coords, &mut pix_frac, &boundaries);
        assert_eq!(coords.x, 5);
        assert_eq!(coords.y, 5);
        assert_eq!(pix_frac.x, 0.5);
        assert_eq!(pix_frac.y, 0.5);
    }
}

enum Direction {
    Forward,
    Backward,
}

#[inline(always)]
fn accumulate_direction<T: AtLeastF32>(
    starting_point: &PixelCoordinates,
    uv: &UVField<T>,
    kernel: &ArrayView1<T>,
    input: &ArrayView2<T>,
    boundaries: &BoundarySet,
    direction: &Direction,
    blocked: Option<&ArrayView2<bool>>,
) -> (T, T) {
    let mut coords: PixelCoordinates = starting_point.clone();
    let mut pix_frac = PixelFraction {
        x: 0.5.into(),
        y: 0.5.into(),
    };

    let mut last_p: UVPoint<T> = UVPoint {
        u: 0.0.into(),
        v: 0.0.into(),
    };

    let kmid = kernel.len() / 2;
    let mut acc: T = 0.0.into();
    let mut used_sum: T = 0.0.into();
    let range = match direction {
        Direction::Forward => Either::Right((kmid + 1)..kernel.len()),
        Direction::Backward => Either::Left((0..kmid).rev()),
    };

    for k in range {
        let mut p = UVPoint {
            u: select_pixel(&uv.u, &coords),
            v: select_pixel(&uv.v, &coords),
        };
        if p.u.is_nan() || p.v.is_nan() {
            break;
        }
        match uv.mode {
            UVMode::Polarization => {
                if (p.u * last_p.u + p.v * last_p.v) < 0.0.into() {
                    p = -p;
                }
                last_p = p.clone();
            }
            UVMode::Velocity => {}
        };
        let mp = match direction {
            Direction::Forward => p.clone(),
            Direction::Backward => -p,
        };
        advance(&mp, &mut coords, &mut pix_frac, boundaries);
        // If a blocked mask is provided, stop when entering a blocked pixel.
        if let Some(b) = blocked {
            if b[[coords.y, coords.x]] {
                break;
            }
        }
        let w = kernel[[k]];
        acc = w.mul_add(select_pixel(input, &coords), acc);
        used_sum += w;
    }
    (acc, used_sum)
}

fn compute_pixel<T: AtLeastF32>(
    uv: &UVField<T>,
    kernel: &ArrayView1<T>,
    input: &ArrayView2<T>,
    boundaries: &BoundarySet,
    blocked: Option<&ArrayView2<bool>>,
    dims: &ImageDimensions,
    kmid: usize,
    full_sum: T,
    edge_gain_strength: T,
    edge_gain_power: T,
    y: usize,
    x: usize,
) -> T {
    if let Some(mask) = blocked {
        if mask[[y, x]] {
            let center = input[[y, x]];
            if full_sum > 0.0.into() {
                return full_sum * center;
            } else {
                return center;
            }
        }
    }
    let center_weight = kernel[[kmid]];
    let mut value = center_weight.mul_add(input[[y, x]], 0.0.into());
    let mut used_sum: T = center_weight;

    let starting_point = PixelCoordinates {
        x,
        y,
        dimensions: dims.clone(),
    };

    let (acc_fwd, used_fwd) = accumulate_direction(
        &starting_point,
        uv,
        kernel,
        input,
        boundaries,
        &Direction::Forward,
        blocked,
    );
    value += acc_fwd;
    used_sum += used_fwd;

    let (acc_bwd, used_bwd) = accumulate_direction(
        &starting_point,
        uv,
        kernel,
        input,
        boundaries,
        &Direction::Backward,
        blocked,
    );
    value += acc_bwd;
    used_sum += used_bwd;

    if let Some(mask) = blocked {
        if !mask[[y, x]] && used_sum > center_weight {
            let zero: T = 0.0.into();
            let one: T = 1.0.into();
            let denom = full_sum - center_weight;
            let mut support_factor = if denom > zero {
                (used_sum - center_weight) / denom
            } else {
                zero
            };
            if support_factor < zero {
                support_factor = zero;
            }
            if support_factor > one {
                support_factor = one;
            }

            if used_sum > zero && used_sum < full_sum {
                value = (full_sum / used_sum) * value;
            }
            if edge_gain_strength != zero && full_sum > zero && support_factor > zero {
                let mut t = (full_sum - used_sum) / full_sum;
                if t < zero {
                    t = zero;
                }
                if t > one {
                    t = one;
                }
                let gain =
                    one + edge_gain_strength * t.powf(edge_gain_power) * support_factor;
                value = gain * value;
            }
        }
    }

    value
}

#[derive(Clone, Copy)]
struct TileSpec {
    y0: usize,
    y1: usize,
    x0: usize,
    x1: usize,
}

fn build_tile_specs(
    height: usize,
    width: usize,
    tile_shape: Option<(usize, usize)>,
) -> Vec<TileSpec> {
    let (tile_h, tile_w) = tile_shape.unwrap_or((height, width));
    let tile_h = tile_h.max(1).min(height);
    let tile_w = tile_w.max(1).min(width);

    let mut tiles = Vec::new();
    let mut y0 = 0;
    while y0 < height {
        let y1 = (y0 + tile_h).min(height);
        let mut x0 = 0;
        while x0 < width {
            let x1 = (x0 + tile_w).min(width);
            tiles.push(TileSpec { y0, y1, x0, x1 });
            x0 = x1;
        }
        y0 = y1;
    }
    tiles
}

fn convolve_tiles<T: AtLeastF32>(
    uv: &UVField<T>,
    kernel: &ArrayView1<T>,
    boundaries: &BoundarySet,
    input: &ArrayView2<T>,
    blocked: Option<ArrayView2<bool>>,
    edge_gain_strength: T,
    edge_gain_power: T,
    tile_shape: Option<(usize, usize)>,
    num_threads: Option<usize>,
    output: &mut Array2<T>,
) {
    let dims = ImageDimensions {
        x: input.shape()[1],
        y: input.shape()[0],
    };
    let tiles = build_tile_specs(dims.y, dims.x, tile_shape);
    let kmid = kernel.len() / 2;
    let full_sum: T = kernel.iter().cloned().fold(0.0.into(), |acc, v| acc + v);

    let uv = uv.clone();
    let kernel = kernel.clone();
    let input = input.clone();
    let boundaries = *boundaries;
    let blocked_owned = blocked.map(|view| view.to_owned());

    let worker_count = num_threads
        .unwrap_or_else(|| thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
        .max(1);
    let worker_count = worker_count.min(tiles.len().max(1));
    let chunk_size = (tiles.len() + worker_count - 1) / worker_count;

    let results = Mutex::new(Vec::with_capacity(tiles.len()));

    thread::scope(|scope| {
        for chunk in tiles.chunks(chunk_size.max(1)) {
            let chunk = chunk;
            let uv_ref = &uv;
            let kernel_ref = &kernel;
            let input_ref = &input;
            let boundaries_ref = boundaries;
            let dims_ref = dims.clone();
            let blocked_ref = &blocked_owned;
            let results_ref = &results;
            scope.spawn(move || {
                let blocked_view = blocked_ref.as_ref().map(|arr| arr.view());
                let mut local = Vec::with_capacity(chunk.len());
                for tile in chunk.iter().copied() {
                    let mut tile_out = Array2::<T>::zeros((tile.y1 - tile.y0, tile.x1 - tile.x0));
                    for (local_y, global_y) in (tile.y0..tile.y1).enumerate() {
                        for (local_x, global_x) in (tile.x0..tile.x1).enumerate() {
                            tile_out[[local_y, local_x]] = compute_pixel(
                                uv_ref,
                                kernel_ref,
                                input_ref,
                                &boundaries_ref,
                                blocked_view.as_ref(),
                                &dims_ref,
                                kmid,
                                full_sum,
                                edge_gain_strength,
                                edge_gain_power,
                                global_y,
                                global_x,
                            );
                        }
                    }
                    local.push((tile, tile_out));
                }
                if !local.is_empty() {
                    let mut guard = results_ref.lock().unwrap();
                    guard.extend(local);
                }
            });
        }
    });

    let results = results.into_inner().unwrap();
    for (tile, tile_out) in results {
        output
            .slice_mut(s![tile.y0..tile.y1, tile.x0..tile.x1])
            .assign(&tile_out);
    }
}

fn convolve<'py, T: AtLeastF32>(
    uv: &UVField<'py, T>,
    kernel: ArrayView1<'py, T>,
    boundaries: &BoundarySet,
    input: ArrayView2<T>,
    output: &mut Array2<T>,
    blocked: Option<&ArrayView2<bool>>,
    edge_gain_strength: T,
    edge_gain_power: T,
) {
    let dims = ImageDimensions {
        x: uv.u.shape()[1],
        y: uv.u.shape()[0],
    };
    let uv = uv.clone();
    let kernel = kernel.clone();
    let input = input.clone();
    let blocked = blocked;
    let full_sum: T = kernel.iter().cloned().fold(0.0.into(), |acc, v| acc + v);
    let kmid = kernel.len() / 2;
    let boundaries = *boundaries;

    for i in 0..dims.y {
        for j in 0..dims.x {
            output[[i, j]] = compute_pixel(
                &uv,
                &kernel,
                &input,
                &boundaries,
                blocked,
                &dims,
                kmid,
                full_sum,
                edge_gain_strength,
                edge_gain_power,
                i,
                j,
            );
        }
    }
}

fn convolve_iteratively<'py, T: AtLeastF32 + numpy::Element>(
    py: Python<'py>,
    texture: PyReadonlyArray2<'py, T>,
    uv: (PyReadonlyArray2<'py, T>, PyReadonlyArray2<'py, T>, String),
    kernel: PyReadonlyArray1<'py, T>,
    boundaries: BoundarySet,
    iterations: i64,
    blocked: Option<PyReadonlyArray2<'py, bool>>,
    edge_gain_strength: T,
    edge_gain_power: T,
    tile_shape: Option<(usize, usize)>,
    _overlap: Option<usize>,
    num_threads: Option<usize>,
) -> Bound<'py, PyArray2<T>> {
    let uv = UVField {
        u: uv.0.as_array(),
        v: uv.1.as_array(),
        mode: UVMode::new(uv.2),
    };
    let kernel = kernel.as_array();
    let texture = texture.as_array();
    let mut input =
        Array2::from_shape_vec(texture.raw_dim(), texture.iter().cloned().collect()).unwrap();
    let mut output = Array2::<T>::zeros(texture.raw_dim());
    let blocked_owned = blocked.map(|b| b.as_array().to_owned());
    let dims = ImageDimensions {
        x: input.shape()[1],
        y: input.shape()[0],
    };
    let derived_tile = if let Some(shape) = tile_shape {
        Some(shape)
    } else if let Some(threads) = num_threads {
        if threads > 1 {
            let stripe_h = (dims.y + threads - 1) / threads;
            Some((stripe_h.max(1), dims.x))
        } else {
            None
        }
    } else {
        None
    };

    let mut it_count = 0;
    while it_count < iterations {
        let blocked_view = blocked_owned.as_ref().map(|arr| arr.view());

        if let Some(shape) = derived_tile {
            convolve_tiles(
                &uv,
                &kernel,
                &boundaries,
                &input.view(),
                blocked_view,
                edge_gain_strength,
                edge_gain_power,
                Some(shape),
                num_threads,
                &mut output,
            );
        } else {
            convolve(
                &uv,
                kernel.clone(),
                &boundaries,
                input.view(),
                &mut output,
                blocked_view.as_ref(),
                edge_gain_strength,
                edge_gain_power,
            );
        }
        it_count += 1;
        if it_count < iterations {
            input.assign(&output);
            output.fill(0.0.into());
        }
    }

    output.to_pyarray(py)
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule(gil_used = false)]
fn _core<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "convolve_f32")]
    #[pyo3(signature = (texture, uv, kernel, boundaries, iterations, blocked=None, edge_gain_strength=0.0, edge_gain_power=2.0, tile_shape=None, overlap=None, num_threads=None))]
    fn convolve_f32_py<'py>(
        py: Python<'py>,
        texture: PyReadonlyArray2<'py, f32>,
        uv: (
            PyReadonlyArray2<'py, f32>,
            PyReadonlyArray2<'py, f32>,
            String,
        ),
        kernel: PyReadonlyArray1<'py, f32>,
        boundaries: ((String, String), (String, String)),
        iterations: i64,
        blocked: Option<PyReadonlyArray2<'py, bool>>,
        edge_gain_strength: f32,
        edge_gain_power: f32,
        tile_shape: Option<(usize, usize)>,
        overlap: Option<usize>,
        num_threads: Option<usize>,
    ) -> Bound<'py, PyArray2<f32>> {
        let boundaries = BoundarySet::new(boundaries);
        convolve_iteratively(
            py,
            texture,
            uv,
            kernel,
            boundaries,
            iterations,
            blocked,
            edge_gain_strength,
            edge_gain_power,
            tile_shape,
            overlap,
            num_threads,
        )
    }

    #[pyfn(m)]
    #[pyo3(name = "convolve_f64")]
    #[pyo3(signature = (texture, uv, kernel, boundaries, iterations, blocked=None, edge_gain_strength=0.0, edge_gain_power=2.0, tile_shape=None, overlap=None, num_threads=None))]
    fn convolve_f64_py<'py>(
        py: Python<'py>,
        texture: PyReadonlyArray2<'py, f64>,
        uv: (
            PyReadonlyArray2<'py, f64>,
            PyReadonlyArray2<'py, f64>,
            String,
        ),
        kernel: PyReadonlyArray1<'py, f64>,
        boundaries: ((String, String), (String, String)),
        iterations: i64,
        blocked: Option<PyReadonlyArray2<'py, bool>>,
        edge_gain_strength: f64,
        edge_gain_power: f64,
        tile_shape: Option<(usize, usize)>,
        overlap: Option<usize>,
        num_threads: Option<usize>,
    ) -> Bound<'py, PyArray2<f64>> {
        let boundaries = BoundarySet::new(boundaries);
        convolve_iteratively(
            py,
            texture,
            uv,
            kernel,
            boundaries,
            iterations,
            blocked,
            edge_gain_strength,
            edge_gain_power,
            tile_shape,
            overlap,
            num_threads,
        )
    }
    Ok(())
}
