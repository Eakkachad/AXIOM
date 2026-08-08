//! Bind node — binds the current vector with a stored reference vector.
//!
//! Creates an association between the current token and a role/context vector.

use tle_vsa::{bind, HyperVector};

use crate::node::{FlowNode, FlowState};

/// Binds the current vector with a stored reference vector using
/// Hadamard product (element-wise multiplication).
///
/// Result: current' = current * reference
///
/// Useful for encoding role-filler bindings in the generation pipeline.
pub struct BindNode {
    /// The reference vector to bind with current.
    pub reference: HyperVector,
}

impl BindNode {
    /// Create a new bind node with the given reference vector.
    pub fn new(reference: HyperVector) -> Self {
        Self { reference }
    }
}

impl FlowNode for BindNode {
    fn transform(&self, mut state: FlowState) -> FlowState {
        state.current = bind(&state.current, &self.reference);
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bind_node() {
        let dim = 128;
        let current = HyperVector::random_bipolar(dim, 1);
        let reference = HyperVector::random_bipolar(dim, 2);

        let node = BindNode::new(reference.clone());

        let mut state = FlowState::new(dim);
        state.current = current.clone();

        let result = node.transform(state);
        let expected = bind(&current, &reference);
        assert_eq!(result.current, expected);
    }
}
