//! Flow combinators — higher-order compositions of `FlowNode`s.
//!
//! These combinators allow building complex pipelines from simpler nodes:
//! - `SequentialFlow`: chains nodes in series
//! - `ParallelFlow`: runs nodes independently and averages scores
//! - `ConditionalFlow`: branches based on a score threshold

use crate::node::{FlowNode, FlowState};

/// Chains multiple `FlowNode`s in sequence (same as FlowGraph but implements FlowNode).
///
/// This allows nesting: a SequentialFlow can be used as a single node
/// inside another FlowGraph or combinator.
pub struct SequentialFlow {
    nodes: Vec<Box<dyn FlowNode>>,
}

impl SequentialFlow {
    /// Create a new empty sequential flow.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Add a node to the sequence.
    pub fn add_node(&mut self, node: Box<dyn FlowNode>) {
        self.nodes.push(node);
    }

    /// Create from a vector of nodes.
    pub fn from_nodes(nodes: Vec<Box<dyn FlowNode>>) -> Self {
        Self { nodes }
    }
}

impl Default for SequentialFlow {
    fn default() -> Self {
        Self::new()
    }
}

impl FlowNode for SequentialFlow {
    fn transform(&self, state: FlowState) -> FlowState {
        self.nodes.iter().fold(state, |s, node| node.transform(s))
    }
}

/// Runs multiple nodes independently (each on a clone of the input state)
/// and averages the resulting scores.
///
/// The output state's non-score fields (current, context, history) come from
/// the original input state (unmodified). Only scores are aggregated.
pub struct ParallelFlow {
    nodes: Vec<Box<dyn FlowNode>>,
}

impl ParallelFlow {
    /// Create a new empty parallel flow.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Add a node to the parallel set.
    pub fn add_node(&mut self, node: Box<dyn FlowNode>) {
        self.nodes.push(node);
    }

    /// Create from a vector of nodes.
    pub fn from_nodes(nodes: Vec<Box<dyn FlowNode>>) -> Self {
        Self { nodes }
    }
}

impl Default for ParallelFlow {
    fn default() -> Self {
        Self::new()
    }
}

impl FlowNode for ParallelFlow {
    fn transform(&self, state: FlowState) -> FlowState {
        if self.nodes.is_empty() {
            return state;
        }

        let num_nodes = self.nodes.len() as f32;
        let num_candidates = state.scores.len();

        // Run each node on a clone and collect scores
        let mut aggregated_scores = vec![0.0f32; num_candidates];

        for node in &self.nodes {
            let result = node.transform(state.clone());
            for (i, &score) in result.scores.iter().enumerate() {
                if i < num_candidates {
                    aggregated_scores[i] += score;
                }
            }
        }

        // Average the scores
        for score in &mut aggregated_scores {
            *score /= num_nodes;
        }

        let mut output = state;
        output.scores = aggregated_scores;
        output
    }
}

/// Branches execution based on whether the maximum current score
/// exceeds a threshold.
///
/// If max(scores) >= threshold, executes `then_node`.
/// Otherwise, executes `else_node`.
pub struct ConditionalFlow {
    /// Score threshold for branching.
    pub threshold: f32,
    /// Node to execute if condition is met.
    then_node: Box<dyn FlowNode>,
    /// Node to execute if condition is not met.
    else_node: Box<dyn FlowNode>,
}

impl ConditionalFlow {
    /// Create a new conditional flow.
    pub fn new(
        threshold: f32,
        then_node: Box<dyn FlowNode>,
        else_node: Box<dyn FlowNode>,
    ) -> Self {
        Self {
            threshold,
            then_node,
            else_node,
        }
    }
}

impl FlowNode for ConditionalFlow {
    fn transform(&self, state: FlowState) -> FlowState {
        let max_score = state
            .scores
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        if max_score >= self.threshold {
            self.then_node.transform(state)
        } else {
            self.else_node.transform(state)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nodes::{ContextAccumNode, TransitionScoreNode};
    use tle_vsa::{bind, HyperVector};

    #[test]
    fn test_sequential_flow() {
        let dim = 256;
        let current = HyperVector::random_bipolar(dim, 1);
        let candidate = HyperVector::random_bipolar(dim, 2);
        let tm = bind(&current.permute(1), &candidate);

        let mut seq = SequentialFlow::new();
        seq.add_node(Box::new(TransitionScoreNode::new(tm, 1.0)));
        seq.add_node(Box::new(ContextAccumNode::new(0.5, 0.5)));

        let mut state = FlowState::new(dim);
        state.current = current;
        state.candidates = vec![candidate];
        state.scores = vec![0.0];

        let result = seq.transform(state);
        assert!(result.scores[0] > 0.0);
    }

    #[test]
    fn test_parallel_flow() {
        let dim = 256;
        let current = HyperVector::random_bipolar(dim, 1);
        let candidate = current.clone();

        let mut par = ParallelFlow::new();
        par.add_node(Box::new(ContextAccumNode::new(0.0, 1.0)));
        par.add_node(Box::new(ContextAccumNode::new(0.0, 1.0)));

        let mut state = FlowState::new(dim);
        state.current = current;
        state.candidates = vec![candidate];
        state.scores = vec![0.0];

        let result = par.transform(state);
        // Both nodes would give ~1.0, averaged = ~1.0
        assert!(result.scores[0] > 0.8);
    }

    #[test]
    fn test_conditional_flow() {
        let dim = 64;

        let then_node = Box::new(ContextAccumNode::new(0.0, 10.0));
        let else_node = Box::new(ContextAccumNode::new(0.0, 0.1));

        let cond = ConditionalFlow::new(0.5, then_node, else_node);

        // High score → then branch
        let current = HyperVector::random_bipolar(dim, 1);
        let mut state = FlowState::new(dim);
        state.current = current.clone();
        state.candidates = vec![current.clone()];
        state.scores = vec![1.0]; // above threshold

        let result = cond.transform(state);
        // then_node (weight=10.0) was used
        assert!(result.scores[0] > 5.0);

        // Low score → else branch
        let mut state2 = FlowState::new(dim);
        state2.current = current.clone();
        state2.candidates = vec![current];
        state2.scores = vec![0.0]; // below threshold

        let result2 = cond.transform(state2);
        // else_node (weight=0.1) was used
        assert!(result2.scores[0] < 1.0);
    }
}
