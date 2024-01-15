#![allow(unused)]

use crate::StrError;
use russell_lab::{generate2d, Matrix, Vector};
use russell_sparse::CooMatrix;
use std::collections::{HashMap, HashSet};

/// Specifies the (boundary) side of a rectangle
pub enum Side {
    Left,
    Right,
    Bottom,
    Top,
}

/// Implements the Finite Difference (FDM) Laplacian operator in 2D
///
/// ```text
///              ∂²u        ∂²u
///    L{u} = kx ———  +  ky ———
///              ∂x²        ∂y²
/// ```
///
/// **Notes:**
///
/// * The operator is built with a five-point stencil.
/// * The boundary nodes are 'mirrored' yielding a no-flux barrier.
pub struct DiscreteLaplacian2d {
    kx: f64,            // diffusion parameter x
    ky: f64,            // diffusion parameter y
    xx: Matrix,         // (ny * nx) matrix of coordinates
    yy: Matrix,         // (ny * nx) matrix of coordinates
    nx: usize,          // number of points along x (≥ 2)
    ny: usize,          // number of points along y (≥ 2)
    dx: f64,            // grid spacing along x
    dy: f64,            // grid spacing along y
    left: Vec<usize>,   // indices of nodes on the left edge
    right: Vec<usize>,  // indices of nodes on the right edge
    bottom: Vec<usize>, // indices of nodes on the bottom edge
    top: Vec<usize>,    // indices of nodes on the top edge

    /// Collects the essential boundary conditions
    /// Maps node => prescribed_value
    essential: HashMap<usize, f64>,
}

impl DiscreteLaplacian2d {
    /// Allocates a new instance
    ///
    /// # Input
    ///
    /// * `kx` -- diffusion parameter x
    /// * `ky` -- diffusion parameter y
    /// * `xmin`, `xmax` -- range along x
    /// * `ymin`, `ymax` -- range along y
    /// * `nx` -- number of points along x (≥ 2)
    /// * `ny` -- number of points along y (≥ 2)
    pub fn new(
        kx: f64,
        ky: f64,
        xmin: f64,
        xmax: f64,
        ymin: f64,
        ymax: f64,
        nx: usize,
        ny: usize,
    ) -> Result<Self, StrError> {
        if nx < 2 {
            return Err("nx must be ≥ 2");
        }
        if ny < 2 {
            return Err("ny must be ≥ 2");
        }
        let (xx, yy) = generate2d(xmin, xmax, ymin, ymax, nx, ny);
        let dx = xx.get(0, 1) - xx.get(0, 0);
        let dy = yy.get(1, 0) - yy.get(0, 0);
        let dim = nx * ny;
        let max_bandwidth = 5;
        let max_nnz = dim * max_bandwidth + dim; // the last +dim corresponds to the 1s put on the diagonal
        Ok(DiscreteLaplacian2d {
            kx,
            ky,
            xx,
            yy,
            nx,
            ny,
            dx,
            dy,
            left: (0..dim).step_by(nx).collect(),
            right: ((nx - 1)..dim).step_by(nx).collect(),
            bottom: (0..nx).collect(),
            top: ((dim - nx)..dim).collect(),
            essential: HashMap::new(),
        })
    }

    /// Sets essential (Dirichlet) boundary condition
    pub fn set_essential_boundary_condition(&mut self, side: Side, value: f64) {
        match side {
            Side::Left => self.left.iter().for_each(|n| {
                self.essential.insert(*n, value);
            }),
            Side::Right => self.right.iter().for_each(|n| {
                self.essential.insert(*n, value);
            }),
            Side::Bottom => self.bottom.iter().for_each(|n| {
                self.essential.insert(*n, value);
            }),
            Side::Top => self.top.iter().for_each(|n| {
                self.essential.insert(*n, value);
            }),
        };
    }

    /// Computes coefficient matrix 'A' of A ⋅ x = b
    ///
    /// **Note:** This function must be called after [DiscreteLaplacian2d::set_essential_boundary_condition]
    pub fn coefficient_matrix(&mut self) -> Result<CooMatrix, StrError> {
        // allocate 'A' matrix
        let dim = self.nx * self.ny;
        let max_bandwidth = 5;
        let max_nnz = dim * max_bandwidth + dim; // the last +dim corresponds to the 1s put on the diagonal
        let mut aa = CooMatrix::new(dim, dim, max_nnz, None, false)?;

        // auxiliary
        let dx2 = self.dx * self.dx;
        let dy2 = self.dy * self.dy;
        let alpha = -2.0 * (self.kx / dx2 + self.ky / dy2);
        let beta = self.kx / dx2;
        let gamma = self.ky / dy2;
        let molecule = [alpha, beta, beta, gamma, gamma];
        let mut jays = [0, 0, 0, 0, 0];

        // loop over all nx * ny equations
        for i in 0..dim {
            if !self.essential.contains_key(&i) {
                let col = i % self.nx; // grid column number
                let row = i / self.nx; // grid row number

                // j-index of grid nodes (mirror if needed)
                jays[0] = i; // current node
                jays[1] = if col == 0 { i + 1 } else { i - 1 }; // left node
                jays[2] = if col == self.nx - 1 { i - 1 } else { i + 1 }; // right node
                jays[3] = if row == 0 { i + self.nx } else { i - self.nx }; // bottom node
                jays[4] = if row == self.ny - 1 { i - self.nx } else { i + self.nx }; // top node

                // set 'A' matrix value
                for (k, j) in jays.iter().enumerate() {
                    if !self.essential.contains_key(j) {
                        aa.put(i, *j, molecule[k]).unwrap();
                    }
                }
            }
        }

        // put ones on the diagonal corresponding to essential boundary conditions
        for i in self.essential.keys() {
            aa.put(*i, *i, 1.0);
        }
        Ok((aa))
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::{DiscreteLaplacian2d, Side};
    use russell_lab::{mat_approx_eq, Matrix};
    use russell_sparse::CooMatrix;

    #[test]
    fn new_works() {
        let nx = 2;
        let ny = 3;
        let lap = DiscreteLaplacian2d::new(7.0, 8.0, -1.0, 1.0, -3.0, 3.0, nx, ny).unwrap();
        assert_eq!(lap.kx, 7.0);
        assert_eq!(lap.ky, 8.0);
        assert_eq!(lap.xx.dims(), (ny, nx));
        assert_eq!(lap.yy.dims(), (ny, nx));
        assert_eq!(lap.nx, nx);
        assert_eq!(lap.ny, ny);
        assert_eq!(lap.dx, 2.0);
        assert_eq!(lap.dy, 3.0);
        assert_eq!(lap.left, &[0, 2, 4]);
        assert_eq!(lap.right, &[1, 3, 5]);
        assert_eq!(lap.bottom, &[0, 1]);
        assert_eq!(lap.top, &[4, 5]);
        assert_eq!(
            format!("{}", lap.xx),
            "┌       ┐\n\
             │ -1  1 │\n\
             │ -1  1 │\n\
             │ -1  1 │\n\
             └       ┘"
        );
        assert_eq!(
            format!("{}", lap.yy),
            "┌       ┐\n\
             │ -3 -3 │\n\
             │  0  0 │\n\
             │  3  3 │\n\
             └       ┘"
        );
    }

    #[test]
    fn set_essential_boundary_condition_works() {
        let mut lap = DiscreteLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();
        const LEF: f64 = 1.0;
        const RIG: f64 = 2.0;
        const BOT: f64 = 3.0;
        const TOP: f64 = 4.0;
        lap.set_essential_boundary_condition(Side::Left, LEF); //    0*   4   8  12*
        lap.set_essential_boundary_condition(Side::Right, RIG); //   3*   7  11  15
        lap.set_essential_boundary_condition(Side::Bottom, BOT); //  0*   1   2   3
        lap.set_essential_boundary_condition(Side::Top, TOP); //    12*  13  14  15*  (corner*)
        assert_eq!(lap.left, &[0, 4, 8, 12]);
        assert_eq!(lap.right, &[3, 7, 11, 15]);
        assert_eq!(lap.bottom, &[0, 1, 2, 3]);
        assert_eq!(lap.top, &[12, 13, 14, 15]);
        let mut essential: Vec<_> = lap.essential.iter().map(|(k, v)| (*k, *v)).collect();
        essential.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        assert_eq!(
            essential,
            &[
                (0, BOT),  // bottom* and left  (wins*)
                (1, BOT),  // bottom
                (2, BOT),  // bottom
                (3, BOT),  // bottom* and right
                (4, LEF),  // left
                (7, RIG),  // right
                (8, LEF),  // left
                (11, RIG), // right
                (12, TOP), // top* and left
                (13, TOP), // top
                (14, TOP), // top
                (15, TOP), // top* and right
            ]
        );
    }

    #[test]
    fn coefficient_matrix_works() {
        let mut lap = DiscreteLaplacian2d::new(1.0, 1.0, 0.0, 2.0, 0.0, 2.0, 3, 3).unwrap();
        let aa = lap.coefficient_matrix().unwrap();
        let ___ = 0.0;
        #[rustfmt::skip]
        let aa_correct = Matrix::from(&[
            [-4.0,  2.0,  ___,  2.0,  ___,  ___,  ___,  ___,  ___],
            [ 1.0, -4.0,  1.0,  ___,  2.0,  ___,  ___,  ___,  ___],
            [ ___,  2.0, -4.0,  ___,  ___,  2.0,  ___,  ___,  ___],
            [ 1.0,  ___,  ___, -4.0,  2.0,  ___,  1.0,  ___,  ___],
            [ ___,  1.0,  ___,  1.0, -4.0,  1.0,  ___,  1.0,  ___],
            [ ___,  ___,  1.0,  ___,  2.0, -4.0,  ___,  ___,  1.0],
            [ ___,  ___,  ___,  2.0,  ___,  ___, -4.0,  2.0,  ___],
            [ ___,  ___,  ___,  ___,  2.0,  ___,  1.0, -4.0,  1.0],
            [ ___,  ___,  ___,  ___,  ___,  2.0,  ___,  2.0, -4.0],
        ]);
        mat_approx_eq(&aa.as_dense(), &aa_correct, 1e-15);
    }

    #[test]
    fn coefficient_matrix_with_essential_prescribed_works() {
        let mut lap = DiscreteLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();
        lap.set_essential_boundary_condition(Side::Left, 0.0);
        lap.set_essential_boundary_condition(Side::Right, 0.0);
        lap.set_essential_boundary_condition(Side::Bottom, 0.0);
        lap.set_essential_boundary_condition(Side::Top, 0.0);
        let aa = lap.coefficient_matrix().unwrap();
        let ___ = 0.0;
        #[rustfmt::skip]
        let aa_correct = Matrix::from(&[
             [1.0, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  0 prescribed
             [___, 1.0, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  1 prescribed
             [___, ___, 1.0, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  2 prescribed
             [___, ___, ___, 1.0, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  3 prescribed
             [___, ___, ___, ___, 1.0, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___], //  4 prescribed
             [___, ___, ___, ___, ___,-4.0, 1.0, ___, ___, 1.0, ___, ___, ___, ___, ___, ___], //  5
             [___, ___, ___, ___, ___, 1.0,-4.0, ___, ___, ___, 1.0, ___, ___, ___, ___, ___], //  6
             [___, ___, ___, ___, ___, ___, ___, 1.0, ___, ___, ___, ___, ___, ___, ___, ___], //  7 prescribed
             [___, ___, ___, ___, ___, ___, ___, ___, 1.0, ___, ___, ___, ___, ___, ___, ___], //  8 prescribed
             [___, ___, ___, ___, ___, 1.0, ___, ___, ___,-4.0, 1.0, ___, ___, ___, ___, ___], //  9
             [___, ___, ___, ___, ___, ___, 1.0, ___, ___, 1.0,-4.0, ___, ___, ___, ___, ___], // 10
             [___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, 1.0, ___, ___, ___, ___], // 11 prescribed
             [___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, 1.0, ___, ___, ___], // 12 prescribed
             [___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, 1.0, ___, ___], // 13 prescribed
             [___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, 1.0, ___], // 14 prescribed
             [___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, ___, 1.0], // 15 prescribed
         ]); //  0   1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
             //  p   p    p    p    p              p    p              p    p    p    p    p
        mat_approx_eq(&aa.as_dense(), &aa_correct, 1e-15);
    }
}
