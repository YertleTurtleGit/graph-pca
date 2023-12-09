use pyo3::prelude::*;
mod graph_pca_lib;

#[pyfunction]
fn calculate(vectors: Vec<Vec<f64>>, radius: f64, edge_length: f64) -> PyResult<Vec<Vec<f64>>> {
    Ok(graph_pca_lib::perform_pca(&vectors, radius, edge_length))
}

#[pymodule]
fn graph_pca(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(calculate, module)?)?;

    Ok(())
}
