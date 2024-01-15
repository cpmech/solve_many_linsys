#![allow(unused)]

use crate::StrError;
use russell_lab::{generate2d, Matrix};
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

    /// Assembles operator into A matrix of [A] ⋅ {x} = {b}
    pub fn assemble(&self, aa: &mut CooMatrix) {
        // reset A matrix
        aa.reset();

        // auxiliary
        let dx2 = self.dx * self.dx;
        let dy2 = self.dy * self.dy;
        let alpha = -2.0 * (self.kx / dx2 + self.ky / dy2);
        let beta = self.kx / dx2;
        let gamma = self.ky / dy2;
        let molecule = [alpha, beta, beta, gamma, gamma];
        let mut jays = [0, 0, 0, 0, 0];

        // loop over all nx * ny equations
        let dim = self.nx * self.ny;
        for i in 0..dim {
            let col = i % self.nx; // grid column number
            let row = i / self.nx; // grid row number

            // j-index of grid nodes (mirror if needed)
            jays[0] = i; // current node
            jays[1] = if col == 0 { i + 1 } else { i - 1 }; // left node
            jays[2] = if col == self.nx - 1 { i - 1 } else { i + 1 }; // right node
            jays[3] = if row == 0 { i + self.nx } else { i - self.nx }; // bottom node
            jays[4] = if row == self.ny - 1 { i - self.nx } else { i + self.nx }; // top node

            // assemble
            for (k, &j) in jays.iter().enumerate() {
                aa.put(i, j, molecule[k]).unwrap();
            }
        }
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
    fn assemble_works() {
        let lap = DiscreteLaplacian2d::new(1.0, 1.0, 0.0, 2.0, 0.0, 2.0, 3, 3).unwrap();
        let dim = lap.nx * lap.ny;
        let max_bandwidth = 5;
        let max_nnz = dim * max_bandwidth;
        let mut aa = CooMatrix::new(dim, dim, max_nnz, None, false).unwrap();
        lap.assemble(&mut aa);
        let aa_correct = Matrix::from(&[
            [-4.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, -4.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, -4.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, -4.0, 2.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 2.0, -4.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, -4.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, -4.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, -4.0],
        ]);
        mat_approx_eq(&aa.as_dense(), &aa_correct, 1e-15);
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
}
