//! # Topological Latent Execution Pipeline
//!
//! The end-to-end deterministic language processing pipeline.
//!
//! ## Processing Flow
//!
//! ```text
//! English Input
//!     → Encode (Codebook lookup)
//!     → Bind roles (VSA binding)
//!     → Route (TDA Mapper)
//!     → Process by node type:
//!         - Syntax: Clifford algebra transformations
//!         - Semantic: Memory weaver read/write
//!         - Generation: Decode to output
//!     → Cleanup (Resonator network)
//!     → Decode (Codebook nearest-neighbor)
//! English Output
//! ```
//!
//! The entire pipeline is deterministic: same input → same output, always.

pub mod engine;

pub use engine::LatentEngine;
