use msgpass::{mpi_finalize, mpi_init_thread, Communicator, MpiThread};
use russell_lab::{StrError, Vector};
use russell_sparse::prelude::*;

/// Solves A * x = b
pub struct LinearSystem<'a> {
    pub aa: SparseMatrix,
    pub x: Vector,
    pub solver: LinSolver<'a>,
}

impl<'a> LinearSystem<'a> {
    /// Allocates a new instance
    pub fn new(ndim: usize) -> Result<Self, StrError> {
        let nnz = ndim;
        let symmetry = None;
        let one_based = false;
        let mut coo = CooMatrix::new(ndim, ndim, nnz, symmetry, one_based)?;
        for i in 0..ndim {
            coo.put(i, i, 0.5)?; // will double 'b'
        }
        Ok(LinearSystem {
            aa: SparseMatrix::from_coo(coo),
            x: Vector::new(ndim),
            solver: LinSolver::new(Genie::Umfpack)?,
        })
    }

    /// Factorizes the coefficient matrix
    pub fn factorize(&mut self) -> Result<(), StrError> {
        self.solver.actual.factorize(&mut self.aa, None)
    }

    /// Solves the linear system
    pub fn solve(&mut self, b: &Vector) -> Result<(), StrError> {
        self.solver.actual.solve(&mut self.x, &self.aa, &b, false)
    }
}

fn main() -> Result<(), StrError> {
    mpi_init_thread(MpiThread::Serialized)?;

    let mut comm = Communicator::new()?;
    let rank = comm.rank()?;
    // let size = comm.size()?;

    let mut ls = LinearSystem::new(100 * (rank + 1))?;
    ls.factorize()?;

    mpi_finalize()?;

    if rank == 0 {
        println!("... success ...");
    }
    Ok(())
}
