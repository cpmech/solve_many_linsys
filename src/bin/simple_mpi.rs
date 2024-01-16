use msgpass::{mpi_finalize, mpi_init_thread, Communicator, MpiThread};
use russell_lab::{Stopwatch, StrError};
use std::thread;
use std::time::Duration;

fn main() -> Result<(), StrError> {
    mpi_init_thread(MpiThread::Serialized)?;
    let mut comm = Communicator::new()?;
    let mut stopwatch = Stopwatch::new("");
    thread::sleep(Duration::from_secs(1));
    comm.barrier()?;
    if comm.rank()? == 0 {
        stopwatch.stop();
        println!("elapsed time = {}", stopwatch);
    }
    mpi_finalize()?;
    Ok(())
}
