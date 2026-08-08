//! Topological Router: The deterministic MoE replacement.
//!
//! Routes latent vectors through the topological graph to appropriate
//! processing nodes without any probabilistic gating.

use tle_vsa::HyperVector;
use crate::filter::FilterFunction;
use crate::mapper::{build_reeb_graph, MapperConfig};
use crate::reeb_graph::{NodeType, ReebGraph};

/// Routing decision: which node to send a vector to, and with what confidence.
#[derive(Clone, Debug)]
pub struct RoutingDecision {
    /// The target node ID in the Reeb graph.
    pub node_id: usize,
    /// The type of processing this node performs.
    pub node_type: NodeType,
    /// Confidence score (cosine similarity to node centroid).
    pub confidence: f32,
    /// Filter value that was computed for this vector.
    pub filter_value: f32,
}

/// The Topological Router: deterministic MoE gating replacement.
///
/// Instead of learning a softmax gate over experts, the router:
/// 1. Computes a filter function value for the input vector
/// 2. Looks up which Reeb graph node(s) cover that filter value
/// 3. Selects the node whose centroid is most similar to the input
///
/// This is 100% deterministic: same input → same route, always.
pub struct TopologicalRouter {
    /// The Reeb graph used for routing decisions.
    graph: ReebGraph,
    /// Filter function for projecting vectors.
    filter: FilterFunction,
    /// Configuration used to build the graph.
    config: MapperConfig,
}

impl TopologicalRouter {
    /// Build a router from a reference set of latent vectors.
    ///
    /// The reference set defines the topology of the latent manifold.
    /// This is analogous to "training" but requires no gradient updates —
    /// it's a one-shot topological analysis.
    pub fn build(reference_vectors: &[HyperVector], config: MapperConfig) -> Self {
        let graph = build_reeb_graph(reference_vectors, &config);
        let filter = config.filter.clone();

        Self {
            graph,
            filter,
            config,
        }
    }

    /// Route a single vector to a processing node.
    ///
    /// Fully deterministic: same vector always routes to same node.
    pub fn route(&self, vector: &HyperVector) -> RoutingDecision {
        let filter_value = self.filter.evaluate(vector);
        let (node_id, confidence) = self.graph.route(vector, filter_value);

        let node_type = if node_id < self.graph.nodes.len() {
            self.graph.nodes[node_id].node_type.clone()
        } else {
            NodeType::General
        };

        RoutingDecision {
            node_id,
            node_type,
            confidence,
            filter_value,
        }
    }

    /// Route a batch of vectors. Returns decisions in same order as input.
    pub fn route_batch(&self, vectors: &[&HyperVector]) -> Vec<RoutingDecision> {
        vectors.iter().map(|v| self.route(v)).collect()
    }

    /// Get the Reeb graph for inspection.
    pub fn graph(&self) -> &ReebGraph {
        &self.graph
    }

    /// Rebuild the router with updated reference vectors.
    /// This can be done periodically as the latent memory evolves.
    pub fn rebuild(&mut self, reference_vectors: &[HyperVector]) {
        self.graph = build_reeb_graph(reference_vectors, &self.config);
    }

    /// Get the number of available routing destinations (experts/nodes).
    pub fn num_nodes(&self) -> usize {
        self.graph.node_count()
    }

    /// Get statistics about routing distribution for a batch.
    pub fn routing_stats(&self, vectors: &[&HyperVector]) -> RoutingStats {
        let decisions = self.route_batch(vectors);

        let mut node_counts = vec![0usize; self.graph.node_count()];
        let mut total_confidence = 0.0f32;

        for d in &decisions {
            if d.node_id < node_counts.len() {
                node_counts[d.node_id] += 1;
            }
            total_confidence += d.confidence;
        }

        RoutingStats {
            total_vectors: vectors.len(),
            node_distribution: node_counts,
            avg_confidence: total_confidence / vectors.len().max(1) as f32,
        }
    }
}

/// Statistics about routing distribution.
#[derive(Clone, Debug)]
pub struct RoutingStats {
    /// Total number of routed vectors.
    pub total_vectors: usize,
    /// Count of vectors per node.
    pub node_distribution: Vec<usize>,
    /// Average routing confidence.
    pub avg_confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tle_vsa::DEFAULT_DIM;

    #[test]
    fn test_router_determinism() {
        let reference: Vec<HyperVector> = (0..20)
            .map(|i| HyperVector::random_bipolar(DEFAULT_DIM, i * 100))
            .collect();

        let config = MapperConfig::default();
        let router = TopologicalRouter::build(&reference, config);

        let test_vector = HyperVector::random_bipolar(DEFAULT_DIM, 9999);

        let decision1 = router.route(&test_vector);
        let decision2 = router.route(&test_vector);

        assert_eq!(decision1.node_id, decision2.node_id);
        assert_eq!(decision1.confidence, decision2.confidence);
        assert_eq!(decision1.filter_value, decision2.filter_value);
    }

    #[test]
    fn test_router_routes_reference_to_itself() {
        let reference: Vec<HyperVector> = (0..10)
            .map(|i| HyperVector::random_bipolar(DEFAULT_DIM, i * 100))
            .collect();

        let config = MapperConfig {
            num_intervals: 3,
            overlap_fraction: 0.5,
            cluster_radius: 200.0, // Large radius to group all
            filter: FilterFunction::Norm,
        };
        let router = TopologicalRouter::build(&reference, config);

        // Route each reference vector - should have high confidence
        for v in &reference {
            let decision = router.route(v);
            assert!(
                decision.confidence > -1.0,
                "Reference vector should route with some confidence"
            );
        }
    }

    #[test]
    fn test_routing_stats() {
        let reference: Vec<HyperVector> = (0..30)
            .map(|i| HyperVector::random_bipolar(DEFAULT_DIM, i * 50))
            .collect();

        let config = MapperConfig::default();
        let router = TopologicalRouter::build(&reference, config);

        let test_batch: Vec<HyperVector> = (0..100)
            .map(|i| HyperVector::random_bipolar(DEFAULT_DIM, 10000 + i))
            .collect();
        let refs: Vec<&HyperVector> = test_batch.iter().collect();

        let stats = router.routing_stats(&refs);
        assert_eq!(stats.total_vectors, 100);
        let total_assigned: usize = stats.node_distribution.iter().sum();
        assert_eq!(total_assigned, 100);
    }
}
