use kdtree::{distance::squared_euclidean, KdTree};
use nalgebra::{DMatrix, SymmetricEigen};
use petgraph::algo::dijkstra;
use petgraph::graph::{NodeIndex, UnGraph};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::vec::Vec;

fn covariance_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let num_samples = data.len();
    let num_features = data[0].len();

    let mean: Vec<f64> = (0..num_features)
        .map(|j| data.iter().map(|row| row[j]).sum::<f64>() / num_samples as f64)
        .collect();

    (0..num_features)
        .map(|i| {
            let mean_i = mean[i];
            (0..num_features)
                .map(|j| {
                    let mean_j = mean[j];
                    let sum = data
                        .iter()
                        .map(|row| (row[i] - mean_i) * (row[j] - mean_j))
                        .sum::<f64>();
                    sum / (num_samples as f64 - 1.0)
                })
                .collect()
        })
        .collect()
}

fn euclidean_distance(v1: &[f64], v2: &[f64]) -> f64 {
    v1.iter()
        .zip(v2.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn filter_vectors_with_graph(
    vectors: &[Vec<f64>],
    pivot_vector: &[f64],
    max_edge_length: f64,
    radius: f64,
) -> Vec<Vec<f64>> {
    if max_edge_length >= radius {
        return vectors.to_vec();
    }

    let mut graph = UnGraph::<(), f64>::new_undirected();

    let nodes: Vec<NodeIndex> = vectors.iter().map(|_| graph.add_node(())).collect();
    let mut pivot_node_index = 0;

    for (i, &node1) in nodes.iter().enumerate() {
        if vectors[i] == pivot_vector {
            pivot_node_index = i;
        }

        for (j, &node2) in nodes.iter().enumerate() {
            if i < j {
                let distance = euclidean_distance(&vectors[i], &vectors[j]);

                if distance <= max_edge_length {
                    graph.add_edge(node1, node2, distance);
                }
            }
        }
    }

    let pivot_node = NodeIndex::new(pivot_node_index);
    let mut distances = dijkstra(&graph, pivot_node, None, |_| 1.0);
    distances.retain(|_, distance| *distance >= radius); // TODO ???

    distances
        .into_keys()
        .map(|node_index| vectors[node_index.index()].clone())
        .collect()
}

fn get_neighbor_vectors(
    kdtree: &KdTree<f64, usize, &Vec<f64>>,
    vectors: &[Vec<f64>],
    pivot_vector: &[f64],
    radius: f64,
    max_edge_length: f64,
) -> Vec<Vec<f64>> {
    let neighbors: Vec<Vec<f64>> = kdtree
        .within(pivot_vector, radius.powi(2), &squared_euclidean)
        .unwrap()
        .iter()
        .map(|&(_, neighborhood_id)| vectors[*neighborhood_id].clone())
        .collect();

    filter_vectors_with_graph(&neighbors, pivot_vector, max_edge_length, radius)
}

fn get_eigenvalues(covariance_matrix: &Vec<Vec<f64>>) -> Vec<f64> {
    let num_features = covariance_matrix.len();
    let d_matrix: DMatrix<f64> =
        DMatrix::from_vec(num_features, num_features, covariance_matrix.concat());

    let eigen = SymmetricEigen::new(d_matrix);

    eigen.eigenvalues.data.as_vec().to_vec()
}

fn build_kdtree(vectors: &[Vec<f64>]) -> kdtree::KdTree<f64, usize, &std::vec::Vec<f64>> {
    let mut kdtree = KdTree::with_capacity(vectors[0].len(), vectors.len());
    for (index, vector) in vectors.iter().enumerate() {
        kdtree.add(vector, index).unwrap();
    }
    kdtree
}

fn get_pca(eigenvalues: &[f64]) -> Vec<f64> {
    let sum: f64 = eigenvalues.iter().sum();

    let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
    indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());

    indices
        .iter()
        .map(|&index| eigenvalues[index] / sum)
        .collect()
}

#[pyclass]
#[derive(Clone, PartialEq)]
pub enum Feature {
    Eigenvalues,
    PrincipalComponentValues,
}

pub fn calculate_features(
    vectors: &Vec<Vec<f64>>,
    features: &[Feature],
    radius: f64,
    max_edge_length: f64,
) -> Vec<Vec<Vec<f64>>> {
    let input_feature_count = vectors[0].len();
    let output_feature_length = features.len();
    let vector_length = vectors.len();

    let mut vectors_per_feature =
        vec![vec![vec![f64::NAN; input_feature_count]; vector_length]; output_feature_length];

    if features.is_empty() {
        return vectors_per_feature;
    }

    let kdtree = build_kdtree(vectors);

    let eigenvalue_index = features
        .iter()
        .position(|feature| feature == &Feature::Eigenvalues);

    let pca_index = features
        .iter()
        .position(|feature| feature == &Feature::PrincipalComponentValues);

    let features_per_vector: Vec<Vec<Vec<f64>>> = vectors
        .par_iter()
        .map(|vector| {
            let mut feature_vector = vec![vec![f64::NAN; vectors[0].len()]; features.len()];

            let neighborhood_vectors =
                get_neighbor_vectors(&kdtree, vectors, vector, radius, max_edge_length);

            if neighborhood_vectors.len() <= 1 {
                return feature_vector;
            }

            let covariance_matrix = covariance_matrix(&neighborhood_vectors);
            let eigenvalues = get_eigenvalues(&covariance_matrix);

            if let Some(feature_index) = pca_index {
                feature_vector[feature_index] = get_pca(&eigenvalues);
            }
            if let Some(feature_index) = eigenvalue_index {
                feature_vector[feature_index] = eigenvalues;
            }

            feature_vector
        })
        .collect();

    for output_feature_index in 0..output_feature_length {
        for vector_index in 0..vector_length {
            for input_feature_index in 0..input_feature_count {
                vectors_per_feature[output_feature_index][vector_index][input_feature_index] =
                    features_per_vector[vector_index][output_feature_index][input_feature_index];
            }
        }
    }

    vectors_per_feature
}
