//! Transition score node — scores candidates against transition memory.
//!
//! Implements the core bigram energy: E_transition = cos(pi(current) * TM, candidate)

use tle_vsa::{bind, cosine_similarity, HyperVector};

use crate::node::{FlowNode, FlowState};

/// Scores each candidate by how well it follows from the current token
/// according to learned transition memory.
///
/// The transition memory (TM) is a bundled superposition of
/// pi(token_i) * token_{i+1} pairs from the training corpus.
/// To predict what follows `current`, we compute pi(current) * TM
/// and measure similarity to each candidate.
pub struct TransitionScoreNode {
    /// The transition memory hypervector (bundled bigram associations).
    pub transition_memory: HyperVector,
    /// Weight multiplier for this node's contribution to total score.
    pub weight: f32,
}

impl TransitionScoreNode {
    /// Create a new transition score node.
    pub fn new(transition_memory: HyperVector, weight: f32) -> Self {
        Self {
            transition_memory,
            weight,
        }
    }
}

impl FlowNode for TransitionScoreNode {
    fn transform(&self, mut state: FlowState) -> FlowState {
        // pi(current) — permute current by 1 position (standard bigram shift)
        let permuted = state.current.permute(1);
        // pi(current) * TM — unbind to get predicted-next vector
        let predicted = bind(&permuted, &self.transition_memory);

        // Score each candidate by similarity to prediction
        for (i, candidate) in state.candidates.iter().enumerate() {
            let sim = cosine_similarity(&predicted, candidate);
            state.scores[i] += self.weight * sim;
        }

        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_score_node() {
        let dim = 1024;
        let current = HyperVector::random_bipolar(dim, 1);
        let candidate_a = HyperVector::random_bipolar(dim, 2);
        let candidate_b = HyperVector::random_bipolar(dim, 3);

        // Build a TM that encodes current -> candidate_a
        let tm = bind(&current.permute(1), &candidate_a);

        let node = TransitionScoreNode::new(tm, 1.0);

        let mut state = FlowState::new(dim);
        state.current = current;
        state.candidates = vec![candidate_a.clone(), candidate_b.clone()];
        state.scores = vec![0.0, 0.0];

        let result = node.transform(state);
        assert!(
            result.scores[0] > result.scores[1],
            "Expected candidate_a to score higher: {} vs {}",
            result.scores[0],
            result.scores[1]
        );
    }
}
