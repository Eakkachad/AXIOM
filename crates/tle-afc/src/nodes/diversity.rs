//! Diversity penalty node — frequency-based penalty (VICReg-inspired).
//!
//! Penalizes candidates proportionally to how often they have appeared
//! in the full generation history, encouraging vocabulary diversity.

use crate::node::{FlowNode, FlowState};

/// Penalizes candidates by `weight * ln(1 + count)` where `count` is
/// the number of times that candidate index appears in the full history.
///
/// Inspired by VICReg's variance term: prevents mode collapse onto
/// a small subset of the vocabulary.
pub struct DiversityPenaltyNode {
    /// Weight multiplier for the penalty.
    pub weight: f32,
}

impl DiversityPenaltyNode {
    /// Create a new diversity penalty node.
    pub fn new(weight: f32) -> Self {
        Self { weight }
    }
}

impl FlowNode for DiversityPenaltyNode {
    fn transform(&self, mut state: FlowState) -> FlowState {
        let num_candidates = state.candidates.len();

        for idx in 0..num_candidates {
            let count = state.history.iter().filter(|&&h| h == idx).count();
            if count > 0 {
                let penalty = (1.0 + count as f32).ln();
                state.scores[idx] -= self.weight * penalty;
            }
        }

        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tle_vsa::HyperVector;

    #[test]
    fn test_diversity_penalty() {
        let dim = 64;
        let node = DiversityPenaltyNode::new(1.0);

        let mut state = FlowState::new(dim);
        state.candidates = vec![
            HyperVector::zeros(dim),
            HyperVector::zeros(dim),
            HyperVector::zeros(dim),
        ];
        state.scores = vec![0.0, 0.0, 0.0];
        state.history = vec![0, 0, 0, 1];

        let result = node.transform(state);
        // Index 0: penalty = ln(1 + 3) = ln(4)
        assert!((result.scores[0] - (-(4.0f32).ln())).abs() < 1e-5);
        // Index 1: penalty = ln(1 + 1) = ln(2)
        assert!((result.scores[1] - (-(2.0f32).ln())).abs() < 1e-5);
        // Index 2: no penalty
        assert!((result.scores[2]).abs() < 1e-5);
    }
}
