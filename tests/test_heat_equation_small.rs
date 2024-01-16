use russell_lab::{vec_approx_eq, StrError, Vector};
use russell_sparse::{Genie, LinSolver, SparseMatrix};
use solve_many_linsys::{DiscreteLaplacian2d, Side};

#[test]
fn main() -> Result<(), StrError> {
    // Approximate (with the Finite Differences Method, FDM) the solution of
    //
    //  ∂²ϕ     ∂²ϕ
    //  ———  +  ——— = 0
    //  ∂x²     ∂y²
    //
    // on a (3.0 × 3.0) rectangle with the following
    // essential (Dirichlet) boundary conditions:
    //
    // left:    ϕ(0.0, y) = 1.0
    // right:   ϕ(3.0, y) = 2.0
    // bottom:  ϕ(x, 0.0) = 1.0
    // top:     ϕ(x, 3.0) = 2.0

    // allocate the Laplacian operator
    let mut fdm = DiscreteLaplacian2d::new(1.0, 1.0, 0.0, 3.0, 0.0, 3.0, 4, 4).unwrap();

    // set essential boundary conditions
    fdm.set_essential_boundary_condition(Side::Left, 1.0);
    fdm.set_essential_boundary_condition(Side::Right, 2.0);
    fdm.set_essential_boundary_condition(Side::Bottom, 1.0);
    fdm.set_essential_boundary_condition(Side::Top, 2.0);

    // compute the augmented coefficient matrix and the correction matrix
    //
    // ┌          ┐ ┌    ┐   ┌             ┐
    // │ Auu   0  │ │ xu │   │ bu - Aup⋅xp │
    // │          │ │    │ = │             │
    // │  0    1  │ │ xp │   │     xp      │
    // └          ┘ └    ┘   └             ┘
    // A := augmented(Auu)
    //
    // ┌          ┐ ┌    ┐   ┌        ┐
    // │  0   Aup │ │ .. │   │ Aup⋅xp │
    // │          │ │    │ = │        │
    // │  0    0  │ │ xp │   │   0    │
    // └          ┘ └    ┘   └        ┘
    // C := augmented(Aup)
    let (dim, _, aa, cc) = fdm.coefficient_matrix().unwrap();

    // allocate the left- and right-hand side vectors
    let mut x = Vector::new(dim);
    let mut b = Vector::new(dim);

    // set the 'prescribed' part of the left-hand side vector with the essential values
    fdm.loop_over_prescribed_values(|i, value| {
        x[i] = value; // xp := xp
    });

    // initialize the right-hand side vector with the correction
    cc.mat_vec_mul(&mut b, -1.0, &x)?; // bu := -Aup⋅xp

    // if there where natural (Neumann) boundary conditions,
    // we could set `bu := natural()` here

    // set the 'prescribed' part of the right-hand side vector with the essential values
    fdm.loop_over_prescribed_values(|i, value| {
        b[i] = value; // bp := xp
    });

    // solve the linear system
    let mut mat = SparseMatrix::from_coo(aa);
    let mut solver = LinSolver::new(Genie::Umfpack)?;
    solver.actual.factorize(&mut mat, None)?;
    solver.actual.solve(&mut x, &mut mat, &b, false)?;

    // check
    let x_correct = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.25, 1.5, 2.0, 1.0, 1.5, 1.75, 2.0, 2.0, 2.0, 2.0, 2.0,
    ];
    vec_approx_eq(&x.as_data(), &x_correct, 1e-15);
    Ok(())
}
