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
    kx: f64,    // diffusion parameter x
    ky: f64,    // diffusion parameter y
    xx: Matrix, // (ny * nx) matrix of coordinates
    yy: Matrix, // (ny * nx) matrix of coordinates
    nx: usize,  // number of points along x (≥ 2)
    ny: usize,  // number of points along y (≥ 2)
    dx: f64,    // grid spacing along x
    dy: f64,    // grid spacing along y
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
        let (xx, yy) = generate2d(xmin, xmax, ymin, ymax, nx, ny);
        let dx = xx.get(0, 1) - xx.get(0, 0);
        let dy = yy.get(1, 0) - yy.get(0, 0);
        Ok(DiscreteLaplacian2d {
            kx,
            ky,
            xx,
            yy,
            nx,
            ny,
            dx,
            dy,
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
        let mut jays = vec![0; 5];

        // loop over all nx * ny equations
        for i in 0..(self.nx * self.ny) {
            let col = i % self.nx; // grid column number
            let row = i / self.nx; // grid row number
            jays[0] = i; // current node
            jays[1] = i - 1; // left node
            jays[2] = i + 1; // right node
            jays[3] = i - self.nx; // bottom node
            jays[4] = i + self.nx; // top node

            // 'mirror' boundaries
            if col == 0 {
                jays[1] = jays[2];
            }
            if col == self.nx - 1 {
                jays[2] = jays[1];
            }
            if row == 0 {
                jays[3] = jays[4];
            }
            if row == self.ny - 1 {
                jays[4] = jays[3];
            }

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

    #[test]
    fn new_works() {
        let lap = DiscreteLaplacian2d::new(1.0, 1.0, -1.0, 1.0, -3.0, 3.0, 2, 3).unwrap();
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
        assert_eq!(lap.dx, 2.0);
        assert_eq!(lap.dy, 3.0);
    }
}
