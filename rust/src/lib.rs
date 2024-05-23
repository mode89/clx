use pyo3::prelude::*;

#[pyfunction]
fn hello(name: &str) -> String {
    format!("Hello, {}!", name)
}

#[pymodule]
fn clx_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    Ok(())
}
