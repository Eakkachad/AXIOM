//! TDA Mapper Algorithm Implementation.
//!
//! Builds a Reeb graph from a point cloud of hypervectors by:
//! 1. Applying a filter function to get scalar values
//! 2. Creating overlapping intervals (covering)
//! 3. Clustering within each interval
//! 4. Connecting clusters that share points in overlapping regions

use tle_vsa::HyperVector;
use crate::filter::FilterFunction;
use crate::reeb_graph::{NodeType, ReebGraph, ReebNode};

/// Configuration for the Mapper algorithm.
#[derive(Clone, Debug)]
pub struct MapperConfig {
    /// Number of intervals in the covering.
    pub num_intervals: usize,
    /// Overlap percentage between adjacent intervals (0.0 to 1.0).
    pub overlap_fraction: f32,
    /// Maximum distance for clustering within an interval.
    pub cluster_radius: f32,
    /// Filter function to use.
    pub filter: FilterFunction,
}

impl Default for MapperConfig {
    fn default() -> Self {
        Self {
            num_intervals: 5,
            overlap_fraction: 0.3,
            cluster_radius: 50.0, // Suitable for D=10240 bipolar vectors
            filter: FilterFunction::Norm,
        }
    }
}

/// Build a Reeb graph from a set of hypervectors using the Mapper algorithm.
///
/// This is the core function that creates the topological routing structure.
/// It is fully deterministic: same inputs always produce same graph.
pub fn build_reeb_graph(
    vectors: &[HyperVector],
    config: &MapperConfig,
) -> ReebGraph {
    if vectors.is_empty() {
        return ReebGraph::new();
    }

    // Step 1: Apply filter function
    let refs: Vec<&HyperVector> = vectors.iter().collect();
    let filter_values = config.filter.evaluate_batch(&refs);

    // Step 2: Determine covering intervals
    let min_val = filter_values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = filter_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;

    if range < 1e-10 {
        // All vectors have same filter value → single node
        let centroid = compute_centroid(vectors);
        let mut graph = ReebGraph::new();
        graph.add_node(ReebNode {
            id: 0,
            centroid,
            filter_range: (min_val, max_val),
            member_count: vectors.len(),
            node_type: classify_node_type(0, config.num_intervals),
        });
        return graph;
    }

    let interval_width = range / (config.num_intervals as f32 - config.overlap_fraction * (config.num_intervals as f32 - 1.0)).max(1.0);
    let step = interval_width * (1.0 - config.overlap_fraction);

    // Step 3: For each interval, find points and cluster them
    let mut graph = ReebGraph::new();
    let mut interval_clusters: Vec<Vec<usize>> = Vec::new(); // node_ids per interval

    for interval_idx in 0..config.num_intervals {
        let interval_start = min_val + step * interval_idx as f32;
        let interval_end = interval_start + interval_width;

        // Find vectors in this interval
        let members: Vec<usize> = filter_values
            .iter()
            .enumerate()
            .filter(|(_, &v)| v >= interval_start && v <= interval_end)
            .map(|(i, _)| i)
            .collect();

        if members.is_empty() {
            interval_clusters.push(Vec::new());
            continue;
        }

        // Cluster within this interval using simple single-linkage
        let clusters = single_linkage_cluster(&members, vectors, config.cluster_radius);

        let mut cluster_node_ids = Vec::new();
        for cluster in &clusters {
            let cluster_vectors: Vec<&HyperVector> = cluster.iter().map(|&i| &vectors[i]).collect();
            let centroid = compute_centroid_refs(&cluster_vectors);

            let node_id = graph.add_node(ReebNode {
                id: graph.node_count(),
                centroid,
                filter_range: (interval_start, interval_end),
                member_count: cluster.len(),
                node_type: classify_node_type(interval_idx, config.num_intervals),
            });
            cluster_node_ids.push(node_id);
        }

        interval_clusters.push(cluster_node_ids);
    }

    // Step 4: Connect clusters in adjacent overlapping intervals
    for i in 0..interval_clusters.len().saturating_sub(1) {
        for &node_a in &interval_clusters[i] {
            for &node_b in &interval_clusters[i + 1] {
                // Check if clusters share conceptual overlap
                // (simplified: connect if centroids have positive similarity)
                let sim = tle_vsa::cosine_similarity(
                    &graph.nodes[node_a].centroid,
                    &graph.nodes[node_b].centroid,
                );
                if sim > -0.5 {
                    // Liberal connection for graph connectivity
                    graph.add_edge(node_a, node_b, sim.max(0.01));
                }
            }
        }
    }

    graph
}

/// Simple single-linkage clustering by distance threshold.
fn single_linkage_cluster(
    indices: &[usize],
    vectors: &[HyperVector],
    radius: f32,
) -> Vec<Vec<usize>> {
    let n = indices.len();
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut assigned = vec![false; n];

    for i in 0..n {
        if assigned[i] {
            continue;
        }

        let mut cluster = vec![indices[i]];
        assigned[i] = true;

        // BFS to find all connected points within radius
        let mut queue = vec![i];
        while let Some(current) = queue.pop() {
            for j in 0..n {
                if assigned[j] {
                    continue;
                }
                let dist = vector_distance(&vectors[indices[current]], &vectors[indices[j]]);
                if dist < radius {
                    assigned[j] = true;
                    cluster.push(indices[j]);
                    queue.push(j);
                }
            }
        }

        clusters.push(cluster);
    }

    clusters
}

/// Euclidean distance between two hypervectors.
fn vector_distance(a: &HyperVector, b: &HyperVector) -> f32 {
    let diff = a.sub(b);
    diff.norm()
}

/// Compute centroid of a set of vectors.
fn compute_centroid(vectors: &[HyperVector]) -> HyperVector {
    let refs: Vec<&HyperVector> = vectors.iter().collect();
    compute_centroid_refs(&refs)
}

/// Compute centroid from references.
fn compute_centroid_refs(vectors: &[&HyperVector]) -> HyperVector {
    if vectors.is_empty() {
        return HyperVector::zeros(1);
    }

    let dim = vectors[0].dim();
    let mut sum = vec![0.0f32; dim];
    for v in vectors {
        for (i, &x) in v.as_slice().iter().enumerate() {
            sum[i] += x;
        }
    }

    let n = vectors.len() as f32;
    let avg: Vec<f32> = sum.iter().map(|&s| s / n).collect();
    HyperVector::new(avg)
}

/// Classify node type based on position in the filter range.
/// Maps interval position to expert type for the MoE analogy.
fn classify_node_type(interval_idx: usize, total_intervals: usize) -> NodeType {
    if total_intervals <= 1 {
        return NodeType::General;
    }

    let position = interval_idx as f32 / (total_intervals - 1) as f32;

    if position < 0.2 {
        NodeType::Syntax
    } else if position < 0.4 {
        NodeType::Semantic
    } else if position < 0.6 {
        NodeType::Pragmatic
    } else if position < 0.8 {
        NodeType::Memory
    } else {
        NodeType::Generation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tle_vsa::DEFAULT_DIM;

    #[test]
    fn test_build_reeb_graph_basic() {
        let vectors: Vec<HyperVector> = (0..20)
            .map(|i| HyperVector::random_bipolar(DEFAULT_DIM, i * 100))
            .collect();

        let config = MapperConfig::default();
        let graph = build_reeb_graph(&vectors, &config);

        assert!(graph.node_count() > 0, "Should have at least one node");
        println!(
            "Graph: {} nodes, {} edges",
            graph.node_count(),
            graph.edge_count()
        );
    }

    #[test]
    fn test_deterministic_graph() {
        let vectors: Vec<HyperVector> = (0..10)
            .map(|i| HyperVector::random_bipolar(DEFAULT_DIM, i * 50))
            .collect();

        let config = MapperConfig::default();
        let graph1 = build_reeb_graph(&vectors, &config);
        let graph2 = build_reeb_graph(&vectors, &config);

        assert_eq!(graph1.node_count(), graph2.node_count());
        assert_eq!(graph1.edge_count(), graph2.edge_count());
    }

    #[test]
    fn test_empty_input() {
        let vectors: Vec<HyperVector> = Vec::new();
        let config = MapperConfig::default();
        let graph = build_reeb_graph(&vectors, &config);
        assert_eq!(graph.node_count(), 0);
    }
}
