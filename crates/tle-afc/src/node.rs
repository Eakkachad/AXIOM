//! Core abstractions: `FlowState` and the `FlowNode` trait.

use tle_vsa::HyperVector;

/// The mutable state that flows through the node graph.
///
/// Each `FlowNode` receives a `FlowState`, transforms it, and returns
/// the updated state. Scores accumulate additively across nodes so that
/// the final selection can be made by argmax.
#[derive(Clone)]
pub struct FlowState {
    /// The current token's hypervector representation.
    pub current: HyperVector,
    /// Accumulated context vector (exponentially-decayed history).
    pub context: HyperVector,
    /// Candidate hypervectors (codebook subset or full vocabulary).
    pub candidates: Vec<HyperVector>,
    /// Accumulated scores for each candidate (same length as `candidates`).
    pub scores: Vec<f32>,
    /// History of selected token indices.
    pub history: Vec<usize>,
    /// Current generation step (0-indexed).
    pub step: usize,
    /// Dimensionality of the hypervectors.
    pub dim: usize,
}

impl FlowState {
    /// Create a new `FlowState` with zero vectors and empty collections.
    pub fn new(dim: usize) -> Self {
        Self {
            current: HyperVector::zeros(dim),
            context: HyperVector::zeros(dim),
            candidates: Vec::new(),
            scores: Vec::new(),
            history: Vec::new(),
            step: 0,
            dim,
        }
    }
}

/// A single deterministic transformation node in the flow graph.
///
/// Implementors must be fully deterministic: given the same `FlowState`,
/// `transform` must always return the same result.
pub trait FlowNode {
    /// Transform the state. Typically modifies `scores`, `context`, or `current`.
    fn transform(&self, state: FlowState) -> FlowState;
}
