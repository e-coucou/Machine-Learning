use pyo3::prelude::*;

/// Une fonction Rust que nous allons appeler depuis Python
#[pyfunction]
fn addition_rust(a: usize, b: usize) -> PyResult<usize> {
    Ok(a + b)
}

/// Définition du module Python
#[pymodule]
fn test_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(addition_rust, m)?)?;
    Ok(())
}