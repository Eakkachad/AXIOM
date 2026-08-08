//! Context accumulation node — exponentially-decayed context (JEPA-inspired).
//!
//! Maintains a running context vector that blends all previously seen tokens
//! with exponential decay.

use tle_vsa::cosine_similarity;

use crate::node::{FlowNode, FlowState};

/// Accumulates context via exponentially-decayed bundling and scores
/// candidates by similarity to the accumulated context.
///
/// Update rule: ctx' = decay * ctx + (1 - decay) * current
/// Score contribution: weight * cos(ctx', candidate_i)
pub struct ContextAccumNode {
    /// Exponential decay factor in (0, 1). Higher = more memory of past.
    pub decay: f32,
    /// Weight multiplier for this node's score contribution.
    pub weight: f32,
}

impl ContextAccumNode {
    /// Create a new context accumulation node.
    pub fn new(decay: f32, weight: f32) -> Self {
        Self { decay, weight }
    }
}

impl FlowNode for ContextAccumNode {
    fn transform(&self, mut state: FlowState) -> FlowState {
        // Update context: ctx' = decay * ctx + (1 - decay) * current
        let decayed_ctx = state.context.scale(self.decay);
        let fresh = state.current.scale(1.0 - self.decay);
        state.context = decayed_ctx.add(&fresh);

        // Score each candidate by similarity to updated context
        for (i, candidate) in state.candidates.iter().enumerate() {
            let sim = cosine_similarity(&state.context, candidate);
            state.scores[i] += self.weight * sim;
        }

        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tle_vsa::HyperVector;

    #[test]
    fn test_context_accum_node() {
        let dim = 1024;
        let current = HyperVector::random_bipolar(dim, 10);
        let candidate_a = current.clone();
        let candidate_b = HyperVector::random_bipolar(dim, 20);

        let node = ContextAccumNode::new(0.0, 1.0);

        let mut state = FlowState::new(dim);
        state.current = current;
        state.candidates = vec![candidate_a, candidate_b];
        state.scores = vec![0.0, 0.0];

        let result = node.transform(state);
        assert!(result.scores[0] > 0.9);
        assert!(result.scores[1].abs() < 0.1);
    }
}
