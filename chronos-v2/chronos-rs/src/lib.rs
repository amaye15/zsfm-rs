pub mod config;
pub mod infer;

#[cfg(feature = "python")]
mod py;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn chronos_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    py::register(m)
}
