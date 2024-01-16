use russell_lab::{vec_approx_eq, StrError, Vector};
use russell_sparse::{Genie, LinSolver, SparseMatrix};
use solve_many_linsys::DiscreteLaplacian2d;

#[test]
fn main() -> Result<(), StrError> {
    // Approximate (with the Finite Differences Method, FDM) the solution of
    //
    //  ∂²ϕ     ∂²ϕ
    //  ———  +  ——— =  source(x, y)
    //  ∂x²     ∂y²
    //
    // on a (1.0 × 1.0) square with homogeneous essential boundary conditions
    //
    // The source term is given by (for a manufactured solution):
    //
    // source(x, y) = 14y³ - (16 - 12x) y² - (-42x² + 54x - 2) y + 4x³ - 16x² + 12x
    //
    // The analytical solution is:
    //
    // ϕ(x, y) = x (1 - x) y (1 - y) (1 + 2x + 7y)

    // allocate the Laplacian operator
    let (nx, ny) = (7, 7);
    let mut fdm = DiscreteLaplacian2d::new(1.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, ny).unwrap();

    // set zero essential boundary conditions
    fdm.set_homogeneous_boundary_conditions();

    // compute the augmented coefficient matrix
    let (dim, _, aa, _) = fdm.coefficient_matrix().unwrap();

    // allocate the left- and right-hand side vectors
    let mut phi = Vector::new(dim);
    let mut rhs = Vector::new(dim);

    // set the 'prescribed' part of the left-hand side vector with the essential values
    fdm.loop_over_prescribed_values(|i, value| {
        phi[i] = value; // xp := xp
    });

    // initialize the right-hand side vector with the correction
    // (this step is not needed with homogeneous boundary conditions)

    // set the right-hand side vector with the source term
    fdm.loop_over_grid_points(|i, x, y| {
        let (xx, yy) = (x * x, y * y);
        let (xxx, yyy) = (xx * x, yy * y);
        let source =
            14.0 * yyy - (16.0 - 12.0 * x) * yy - (-42.0 * xx + 54.0 * x - 2.0) * y + 4.0 * xxx - 16.0 * xx + 12.0 * x;
        rhs[i] = source;
    });

    // set the 'prescribed' part of the right-hand side vector with the essential values
    fdm.loop_over_prescribed_values(|i, value| {
        rhs[i] = value; // bp := xp
    });

    // solve the linear system
    let mut mat = SparseMatrix::from_coo(aa);
    let mut solver = LinSolver::new(Genie::Umfpack)?;
    solver.actual.factorize(&mut mat, None)?;
    solver.actual.solve(&mut phi, &mut mat, &rhs, false)?;

    // check
    let mut phi_correct = Vector::new(dim);
    fdm.loop_over_grid_points(|i, x, y| {
        let analytical = x * (1.0 - x) * y * (1.0 - y) * (1.0 + 2.0 * x + 7.0 * y);
        phi_correct[i] = analytical;
    });
    vec_approx_eq(phi.as_data(), phi_correct.as_data(), 1e-15);
    Ok(())
}
