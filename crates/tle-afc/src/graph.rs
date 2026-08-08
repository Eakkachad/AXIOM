//! Flow graph — sequential DAG composition of `FlowNode`s.
//!
//! The `FlowGraph` executes nodes in order, threading `FlowState`
//! through each transformation.

use crate::node::{FlowNode, FlowState};

/// A directed acyclic graph of flow nodes executed sequentially.
///
/// Nodes are applied in insertion order. Each node receives the
/// output state of the previous node.
pub struct FlowGraph {
    nodes: Vec<Box<dyn FlowNode>>,
}

impl FlowGraph {
    /// Create an empty flow graph.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Add a node to the end of the execution pipeline.
    pub fn add_node(&mut self, node: Box<dyn FlowNode>) {
        self.nodes.push(node);
    }

    /// Execute all nodes sequentially, threading state through each.
    pub fn execute(&self, state: FlowState) -> FlowState {
        self.nodes.iter().fold(state, |s, node| node.transform(s))
    }

    /// Returns the number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl Default for FlowGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nodes::{ContextAccumNode, TransitionScoreNode};
    use tle_vsa::{bind, HyperVector};

    #[test]
    fn test_flow_graph_sequential() {
        let dim = 256;
        let current = HyperVector::random_bipolar(dim, 1);
        let candidate = HyperVector::random_bipolar(dim, 2);
        let tm = bind(&current.permute(1), &candidate);

        let mut graph = FlowGraph::new();
        graph.add_node(Box::new(TransitionScoreNode::new(tm, 1.0)));
        graph.add_node(Box::new(ContextAccumNode::new(0.5, 0.5)));

        let mut state = FlowState::new(dim);
        state.current = current;
        state.candidates = vec![candidate];
        state.scores = vec![0.0];

        let result = graph.execute(state);
        assert!(result.scores[0] > 0.0);
    }

    #[test]
    fn test_empty_graph() {
        let graph = FlowGraph::new();
        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);

        let state = FlowState::new(64);
        let result = graph.execute(state);
        assert_eq!(result.scores.len(), 0);
    }
}
