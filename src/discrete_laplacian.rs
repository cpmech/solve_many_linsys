#![allow(unused)]

use crate::StrError;
use russell_lab::{generate2d, Matrix};
use russell_sparse::CooMatrix;

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
        let n = nx * ny;
        Ok(DiscreteLaplacian2d {
            kx,
            ky,
            xx,
            yy,
            nx,
            ny,
            dx,
            dy,
            left: (0..n).step_by(nx).collect(),
            right: ((nx - 1)..n).step_by(nx).collect(),
            bottom: (0..nx).collect(),
            top: ((n - nx)..n).collect(),
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
        for i in 0..(self.nx * self.ny) {
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::DiscreteLaplacian2d;
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
        let n = lap.nx * lap.ny;
        let max_bandwidth = 5;
        let max_nnz = n * max_bandwidth;
        let mut aa = CooMatrix::new(n, n, max_nnz, None, false).unwrap();
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
}
