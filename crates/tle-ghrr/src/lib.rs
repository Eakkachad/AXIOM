//! GHRR block-unitary binding + relation-schema path retrieval (PathHD,
//! arXiv:2512.09369), deterministic + CPU-only.
//!
//! Provides: order-sensitive relation-path binding (blockwise O(4) product),
//! blockwise-cosine similarity, a seeded relation codebook, and a calibrated
//! path-scoring layer (`calibrated_score`, IDF over evidence-schema frequency,
//! length penalty `β·λ^|z|`).

pub mod block;
pub mod codebook;
pub mod retrieval;
pub mod vector;

pub use block::{D_BLOCKS, DIM, M};
pub use codebook::GhrrCodebook;
pub use retrieval::{RelationSchemaIndex, calibrated_score, path_scores_to_entity};
pub use vector::GhrrVector;
