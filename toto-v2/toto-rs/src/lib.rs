pub mod config;
pub mod infer;

#[cfg(feature = "python")]
mod py;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn toto_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    py::register(m)
}
