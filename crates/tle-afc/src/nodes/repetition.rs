//! Repetition penalty node — penalizes recently-used candidates.
//!
//! Prevents degenerate loops by subtracting a penalty from candidates
//! that appear in the recent history window.

use crate::node::{FlowNode, FlowState};

/// Penalizes candidates whose indices appear in the recent generation history.
///
/// For each candidate index that matches a token in the last `window` steps,
/// subtracts `weight * base_penalty` from that candidate's score.
pub struct RepetitionPenaltyNode {
    /// How many recent history steps to look back.
    pub window: usize,
    /// Base penalty value to subtract per occurrence.
    pub base_penalty: f32,
    /// Weight multiplier (typically 1.0).
    pub weight: f32,
}

impl RepetitionPenaltyNode {
    /// Create a new repetition penalty node.
    pub fn new(window: usize, base_penalty: f32, weight: f32) -> Self {
        Self {
            window,
            base_penalty,
            weight,
        }
    }
}

impl FlowNode for RepetitionPenaltyNode {
    fn transform(&self, mut state: FlowState) -> FlowState {
        let history_len = state.history.len();
        let start = history_len.saturating_sub(self.window);
        let recent = &state.history[start..];

        let num_candidates = state.candidates.len();
        for idx in 0..num_candidates {
            if recent.contains(&idx) {
                state.scores[idx] -= self.weight * self.base_penalty;
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
    fn test_repetition_penalty() {
        let dim = 64;
        let node = RepetitionPenaltyNode::new(3, 0.5, 1.0);

        let mut state = FlowState::new(dim);
        state.candidates = vec![
            HyperVector::zeros(dim),
            HyperVector::zeros(dim),
            HyperVector::zeros(dim),
        ];
        state.scores = vec![1.0, 1.0, 1.0];
        state.history = vec![0, 2, 1, 0];

        let result = node.transform(state);
        // Index 0 is in last 3 [2,1,0] -> penalized
        assert!((result.scores[0] - 0.5).abs() < 1e-6);
        // Index 1 is in last 3 -> penalized
        assert!((result.scores[1] - 0.5).abs() < 1e-6);
        // Index 2 is in last 3 -> penalized
        assert!((result.scores[2] - 0.5).abs() < 1e-6);
    }
}
