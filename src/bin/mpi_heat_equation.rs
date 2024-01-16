use msgpass::{mpi_finalize, mpi_init_thread, Communicator, MpiOp, MpiThread};
use russell_lab::{Stopwatch, StrError, Vector};
use russell_sparse::{Genie, LinSolver, SparseMatrix};
use solve_many_linsys::DiscreteLaplacian2d;
use structopt::StructOpt;

const ROOT: usize = 0;
const TOLERANCE: f64 = 1e-10;

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
        let (xx, yy) = (x * x, y * y);
        let (xxx, yyy) = (xx * x, yy * y);
        let source =
            14.0 * yyy - (16.0 - 12.0 * x) * yy - (-42.0 * xx + 54.0 * x - 2.0) * y + 4.0 * xxx - 16.0 * xx + 12.0 * x;
        rhs[i] = multiplier * source;
    });
    fdm.loop_over_prescribed_values(|i, value| {
        rhs[i] = value;
    });
    rhs
}

fn match_analytical_solution(fdm: &DiscreteLaplacian2d, num: &Vector, multiplier: f64) -> u32 {
    let mut ok = 1;
    fdm.loop_over_grid_points(|i, x, y| {
        let ana = multiplier * x * (1.0 - x) * y * (1.0 - y) * (1.0 + 2.0 * x + 7.0 * y);
        if f64::abs(num[i] - ana) > TOLERANCE {
            ok = 0;
        }
    });
    ok
}

#[derive(StructOpt)]
struct Options {
    #[structopt(default_value = "21")]
    nx: usize,

    #[structopt(default_value = "Umfpack")]
    genie: String,
}

fn main() -> Result<(), StrError> {
    // initialize the MPI engine
    mpi_init_thread(MpiThread::Serialized)?;

    // start stopwatch
    let mut stopwatch = Stopwatch::new("");

    // allocate MPI communicator and determine this processor's rank
    let mut comm = Communicator::new()?;
    let rank = comm.rank()?;
    let size = comm.size()?;

    // parse command line arguments
    let opt = Options::from_args();
    let genie = Genie::from(&opt.genie);
    let one_based = if genie == Genie::Mumps { true } else { false };

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

        // message
        if rank == ROOT {
            print!("done with multiplier = {:>3}", multiplier);
        }

        // check the results
        let ok = match_analytical_solution(&fdm, &x, *multiplier);

        // share the results among all processors
        let mut all_ok = [0];
        comm.allreduce_u32(&mut all_ok, &[ok], MpiOp::Land)?;

        // message
        if all_ok[0] == 1 {
            if rank == ROOT {
                println!(" (success)")
            }
        } else {
            if rank == ROOT {
                println!(" (failed)")
            }
        }
    }

    // finalize the MPI engine
    mpi_finalize()?;
    if rank == ROOT {
        stopwatch.stop();
        println!("elapsed time = {}", stopwatch);
    }
    Ok(())
}
