use msgpass::*;

fn main() -> Result<(), StrError> {
    mpi_init()?;
    println!("Hello, world!");
    mpi_finalize()
}
