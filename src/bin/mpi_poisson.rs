use msgpass::{mpi_finalize, mpi_init_thread, Communicator, MpiThread};
use russell_lab::{Stopwatch, StrError, Vector};
use russell_sparse::{Genie, LinSolver, SparseMatrix};
use solve_many_linsys::DiscreteLaplacian2d;
use std::fmt;
use structopt::StructOpt;

// Approximate (with the Finite Differences Method, FDM) the solution of
//
// ∂²ϕ   ∂²ϕ
// ——— + ——— = multiplier * 2 x (y - 1) (y - 2 x + x y + 2) exp(x - y)
// ∂x²   ∂y²
//
// on a (1.0 × 1.0) square with the homogeneous boundary conditions.
//
// The analytical solution is:
//
// ϕ(x, y) = multiplier * x y (x - 1) (y - 1) exp(x - y)

fn create_discrete_laplacian(nx: usize, ny: usize, one_based: bool) -> (DiscreteLaplacian2d, SparseMatrix) {
    let mut fdm = DiscreteLaplacian2d::new(1.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx, ny).unwrap();
    fdm.set_homogeneous_boundary_conditions();
    let (aa, _) = fdm.coefficient_matrix(one_based).unwrap();
    let mat = SparseMatrix::from_coo(aa);
    (fdm, mat)
}

fn populate_rhs_vector(fdm: &DiscreteLaplacian2d, multiplier: f64) -> Vector {
    let dim = fdm.dim();
    let mut rhs = Vector::new(dim);
    fdm.loop_over_grid_points(|i, x, y| {
        rhs[i] = multiplier * 2.0 * x * (y - 1.0) * (y - 2.0 * x + x * y + 2.0) * f64::exp(x - y);
    });
    fdm.loop_over_prescribed_values(|i, value| {
        rhs[i] = value;
    });
    rhs
}

fn compare_analytical_solution(fdm: &DiscreteLaplacian2d, num: &Vector, multiplier: f64) -> f64 {
    let mut err_max = 0.0;
    fdm.loop_over_grid_points(|i, x, y| {
        let ana = multiplier * x * y * (x - 1.0) * (y - 1.0) * f64::exp(x - y);
        let err = f64::abs(num[i] - ana);
        if err > err_max {
            err_max = err
        }
    });
    err_max
}

#[derive(StructOpt)]
struct Options {
    #[structopt(default_value = "21")]
    nx: usize,

    #[structopt(default_value = "Umfpack")]
    genie: String,
}

const ROOT: usize = 0;

fn main() -> Result<(), StrError> {
    // initialize the MPI engine
    mpi_init_thread(MpiThread::Serialized)?;

    // allocate MPI communicator and determine this processor's rank
    let mut comm = Communicator::new()?;
    let rank = comm.rank()?;
    let size = comm.size()?;

    // parse command line arguments
    let opt = Options::from_args();
    let genie = Genie::from(&opt.genie);
    let one_based = if genie == Genie::Mumps { true } else { false };

    // start stopwatch
    let mut stopwatch = Stopwatch::new("");

    // create coefficient matrix
    let (fdm, mut mat) = create_discrete_laplacian(opt.nx, opt.nx, one_based);

    // message
    let dim = fdm.dim();
    if rank == ROOT {
        println!("size = {}, nx = {}, dim = {}, genie = {:?}", size, opt.nx, dim, genie);
    }

    // allocate linear solver
    let mut solver = LinSolver::new(genie)?;

    // perform the factorization
    solver.actual.factorize(&mut mat, None)?;

    // allocate the solution vector
    let mut x = Vector::new(dim);

    // solve many times with increasing multipliers
    const MULTIPLIERS: &[f64] = &[1.0, 2.0, 5.0, 10.0, 100.0];
    for multiplier in MULTIPLIERS {
        // perform the solution
        let b = populate_rhs_vector(&fdm, *multiplier);
        solver.actual.solve(&mut x, &mat, &b, false)?;

        // compare the results
        let err_max = compare_analytical_solution(&fdm, &x, *multiplier);

        // gather the results
        if rank == ROOT {
            let mut all_err_max = vec![0.0; size];
            comm.gather_f64(ROOT, Some(&mut all_err_max), &[err_max])?;
            println!("multiplier = {:>3}, errors ={}", multiplier, P(all_err_max));
        } else {
            comm.gather_f64(ROOT, None, &[err_max])?;
        }
    }

    // message
    if rank == ROOT {
        stopwatch.stop();
        println!("elapsed time = {}", stopwatch);
    }

    // finalize the MPI engine
    mpi_finalize()?;
    Ok(())
}

struct P(Vec<f64>);

impl fmt::Display for P {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for v in &self.0 {
            write!(f, "{:5.0e}", v)?;
        }
        Ok(())
    }
}
