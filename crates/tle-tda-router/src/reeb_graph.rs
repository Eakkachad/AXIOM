//! Reeb Graph: The topological structure used for routing decisions.
//!
//! A Reeb graph captures the connected components of level sets
//! of a filter function. Nodes represent clusters, edges represent
//! topological transitions between them.

use tle_vsa::HyperVector;

/// A node in the Reeb graph representing a cluster of latent vectors.
#[derive(Clone, Debug)]
pub struct ReebNode {
    /// Unique identifier for this node.
    pub id: usize,
    /// The centroid of vectors assigned to this node.
    pub centroid: HyperVector,
    /// Filter function range [min, max] for this node.
    pub filter_range: (f32, f32),
    /// Number of vectors currently assigned to this node.
    pub member_count: usize,
    /// Node type/label (maps to expert type in MoE analogy).
    pub node_type: NodeType,
}

/// Classification of Reeb graph nodes (analogous to "experts" in MoE).
#[derive(Clone, Debug, PartialEq)]
pub enum NodeType {
    /// Syntactic processing node (grammar, structure)
    Syntax,
    /// Semantic processing node (meaning, reference)
    Semantic,
    /// Pragmatic processing node (context, intent)
    Pragmatic,
    /// Memory access node (retrieval, storage)
    Memory,
    /// Generation node (output formation)
    Generation,
    /// Unspecialized (default)
    General,
}

/// An edge in the Reeb graph connecting two nodes.
#[derive(Clone, Debug)]
pub struct ReebEdge {
    /// Source node ID.
    pub from: usize,
    /// Target node ID.
    pub to: usize,
    /// Edge weight (transition strength based on overlap).
    pub weight: f32,
}

/// The Reeb graph: a topological summary of the latent manifold.
///
/// This structure is the deterministic routing table for the MoE system.
/// Vectors are assigned to nodes based on their filter values and
/// proximity to node centroids.
#[derive(Clone, Debug)]
pub struct ReebGraph {
    /// All nodes in the graph.
    pub nodes: Vec<ReebNode>,
    /// All edges (adjacency list representation).
    pub edges: Vec<ReebEdge>,
}

impl ReebGraph {
    /// Create an empty Reeb graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: ReebNode) -> usize {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }

    /// Add an edge between two nodes.
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f32) {
        self.edges.push(ReebEdge { from, to, weight });
    }

    /// Route a vector to its nearest node in the graph.
    ///
    /// Routing is deterministic: uses filter value to narrow candidates,
    /// then cosine similarity to select the best node.
    ///
    /// Returns (node_id, confidence).
    pub fn route(&self, hv: &HyperVector, filter_value: f32) -> (usize, f32) {
        if self.nodes.is_empty() {
            return (0, 0.0);
        }

        // Phase 1: Find nodes whose filter range contains this value
        let mut candidates: Vec<(usize, f32)> = Vec::new();

        for (i, node) in self.nodes.iter().enumerate() {
            if filter_value >= node.filter_range.0 && filter_value <= node.filter_range.1 {
                let sim = tle_vsa::cosine_similarity(hv, &node.centroid);
                candidates.push((i, sim));
            }
        }

        // If no filter-range match, use all nodes
        if candidates.is_empty() {
            for (i, node) in self.nodes.iter().enumerate() {
                let sim = tle_vsa::cosine_similarity(hv, &node.centroid);
                candidates.push((i, sim));
            }
        }

        // Phase 2: Select highest similarity
        candidates
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .copied()
            .unwrap_or((0, 0.0))
    }

    /// Get all nodes adjacent to a given node.
    pub fn neighbors(&self, node_id: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter_map(|e| {
                if e.from == node_id {
                    Some(e.to)
                } else if e.to == node_id {
                    Some(e.from)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get the node type for a given node ID.
    pub fn node_type(&self, id: usize) -> &NodeType {
        &self.nodes[id].node_type
    }
}

impl Default for ReebGraph {
    fn default() -> Self {
        Self::new()
    }
}
