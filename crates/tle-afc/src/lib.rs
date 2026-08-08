//! # Algebraic Flow Composition (AFC)
//!
//! Composable, deterministic generation pipelines for the topological-latent-engine.
//!
//! AFC models token generation as a directed acyclic graph of `FlowNode`
//! transformations over a `FlowState`. Each node reads and modifies the state
//! (scores, context, current vector) without any sampling or randomness.
//!
//! ## Architecture
//!
//! ```text
//! FlowState --> [TransitionScore] --> [ContextAccum] --> [RepetitionPenalty]
//!                                                              |
//!                                                              v
//!                                       [DiversityPenalty] <-- FlowState
//! ```
//!
//! ## Design Principles
//!
//! - **Deterministic**: No sampling, no randomness. Given the same input state,
//!   every node produces the same output.
//! - **Composable**: Nodes can be freely combined via `FlowGraph`, `SequentialFlow`,
//!   `ParallelFlow`, and `ConditionalFlow` combinators.
//! - **Energy-based**: The final selection is a deterministic argmax over
//!   accumulated scores (composite energy).

pub mod combinators;
pub mod energy;
pub mod graph;
pub mod node;
pub mod nodes;

pub use combinators::{ConditionalFlow, ParallelFlow, SequentialFlow};
pub use energy::{CompositeEnergyFlow, EnergyConfig};
pub use graph::FlowGraph;
pub use node::{FlowNode, FlowState};
pub use nodes::{
    BindNode, ContextAccumNode, DiversityPenaltyNode, MultiHopNode, PermuteNode,
    RepetitionPenaltyNode, TransitionScoreNode, compose_transition_chain,
};
