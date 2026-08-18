//! # Vector Symbolic Architecture (VSA) Core
//!
//! Implements Hyperdimensional Computing (HDC) operations for the
//! Model-less Non-Parametric Multi-Node Latent Execution Engine.
//!
//! ## Mathematical Foundation
//!
//! Concepts are represented as hypervectors in ℝ^D where D > 10,000.
//! Three core operations form a complete algebraic system:
//!
//! - **Binding** (⊗): Circular convolution or Hadamard product.
//!   Creates associations: V_bound = V_role ⊗ V_filler
//! - **Unbinding** (⊗⁻¹): Inverse binding for retrieval.
//! - **Bundling** (+): Element-wise addition for superposition.
//!   Combines multiple bound pairs into a single composite vector.
//!
//! ## Properties
//!
//! - Bound vectors are quasi-orthogonal to their components
//! - Bundled vectors are similar to all their constituents
//! - Operations are fully deterministic (zero-sampling)
//! - SNR degrades as 1/√k for k superimposed items

pub mod clifford;
pub mod codebook;
pub mod gf2;
pub mod hypervector;
pub mod ops;
pub mod phasor;
pub mod similarity;

pub use clifford::{Clifford3D, SyntacticRotorCodebook};
pub use codebook::Codebook;
pub use gf2::{factorize_bundle, Gf2Mat, LinearCode};
pub use hypervector::HyperVector;
pub use ops::{bind, bundle, unbind};
pub use phasor::PhasorVector;
pub use similarity::{cosine_similarity, dot_product};

/// Default dimensionality for hypervectors.
/// Must be > 10,000 for reliable VSA operations.
/// We use 10,240 (10 * 1024) for SIMD-friendly alignment.
pub const DEFAULT_DIM: usize = 10_240;

/// Threshold for cosine similarity to consider a match.
/// Based on theoretical bounds: cos_sim > 1/√D implies likely match.
pub const SIMILARITY_THRESHOLD: f32 = 0.1;
