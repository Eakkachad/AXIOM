//! Permute node — applies circular permutation to the current vector.
//!
//! Used for positional encoding in sequence generation.

use crate::node::{FlowNode, FlowState};

/// Applies a circular permutation of `amount` positions to the current vector.
///
/// This is useful as a preprocessing step before transition scoring,
/// or for encoding relative position in multi-step lookahead.
pub struct PermuteNode {
    /// Number of positions to shift (positive = right, negative = left).
    pub amount: i32,
}

impl PermuteNode {
    /// Create a new permute node with the given shift amount.
    pub fn new(amount: i32) -> Self {
        Self { amount }
    }
}

impl FlowNode for PermuteNode {
    fn transform(&self, mut state: FlowState) -> FlowState {
        state.current = state.current.permute(self.amount);
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tle_vsa::HyperVector;

    #[test]
    fn test_permute_node() {
        let dim = 128;
        let original = HyperVector::random_bipolar(dim, 42);

        let node = PermuteNode::new(5);

        let mut state = FlowState::new(dim);
        state.current = original.clone();

        let result = node.transform(state);
        let expected = original.permute(5);
        assert_eq!(result.current, expected);
    }
}
