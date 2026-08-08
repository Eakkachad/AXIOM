//! # Resonator Networks for VSA Cleanup
//!
//! Implements non-linear iterative cleanup to suppress crosstalk noise
//! that accumulates during VSA unbinding operations.
//!
//! ## The Crosstalk Problem
//!
//! When k items are superimposed in a bundle S = Σ(R_i ⊗ F_i),
//! unbinding R_j from S yields: F_j + noise(k-1 cross-terms).
//! The noise magnitude grows as √(k-1)/√D, degrading retrieval.
//!
//! ## Resonator Solution
//!
//! A resonator network iteratively applies non-linear cleanup rules
//! to project noisy estimates back toward valid codebook entries:
//!
//! 1. Start with initial estimate x₀ = unbind(role, composite)
//! 2. Apply cleanup: x_{t+1} = cleanup(C^T · softmax(C · x_t))
//! 3. Converge when ||x_{t+1} - x_t|| < ε
//!
//! Where C is the codebook matrix and cleanup is sign/threshold/softmax.

pub mod cleanup;
pub mod network;

pub use cleanup::CleanupRule;
pub use network::{ResonatorNetwork, ResonatorConfig, ResonatorResult};
