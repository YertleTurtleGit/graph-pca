use graph_pca_lib::Feature;
use pyo3::prelude::*;
mod graph_pca_lib;

#[pymodule]
fn graph_pca(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(calculate_features, module)?)?;
    module.add_class::<Feature>()?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (vectors, features, radius, max_edge_length=None))]
fn calculate_features(
    vectors: Vec<Vec<f64>>,
    features: Vec<Feature>,
    radius: f64,
    max_edge_length: Option<f64>,
) -> PyResult<Vec<Vec<Vec<f64>>>> {
    Ok(graph_pca_lib::calculate_features(
        &vectors,
        &features,
        radius,
        max_edge_length.unwrap_or(radius),
    ))
}
