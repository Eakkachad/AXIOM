//! # Engram — Multi-Head N-gram Hash Table for O(1) Factual Retrieval
//!
//! The Engram is the "fast memory" layer of the Deep Man architecture.
//! It stores N-gram patterns (bigram through 5-gram) in a frozen hash table,
//! enabling O(1) lookup of likely next tokens given a context window.
//!
//! ## Architecture
//!
//! ```text
//! Context: ["the", "cat", "sat", "on"]
//!            │        │         │      │
//!            ▼        ▼         ▼      ▼
//!     ┌─────────┐ ┌────────┐ ┌─────┐ ┌───┐
//!     │ 4-gram  │ │ 3-gram │ │2-gram│ │1-g│  ← Hash Heads
//!     │  hash   │ │  hash  │ │ hash │ │   │
//!     └────┬────┘ └───┬────┘ └──┬───┘ └─┬─┘
//!          │          │         │        │
//!          ▼          ▼         ▼        ▼
//!     ┌─────────────────────────────────────┐
//!     │     Sigmoid-Gated Fusion Layer      │  ← Confidence-weighted merge
//!     └────────────────┬────────────────────┘
//!                      │
//!                      ▼
//!              [Candidate Scores]
//! ```
//!
//! ## Design Principles
//!
//! - **O(1) lookup**: FxHash-based table, no iteration over entries
//! - **Multi-head**: Different context lengths capture different patterns
//! - **Sigmoid fusion**: Confident heads override uncertain ones
//! - **Frozen after build**: Immutable at inference time → deterministic
//! - **Incremental ingest**: Can add new corpus data to expand coverage

pub mod builder;
pub mod fusion;
pub mod hash;
pub mod node;
pub mod table;

pub use builder::EngramBuilder;
pub use fusion::SigmoidFusion;
pub use hash::NgramHash;
pub use node::EngramNode;
pub use table::{EngramEntry, EngramTable};
