/// Defines a type alias for the error type as a static string
pub type StrError = &'static str;

mod discrete_laplacian;
pub use crate::discrete_laplacian::*;
