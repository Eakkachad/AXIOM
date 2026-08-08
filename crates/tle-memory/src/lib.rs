//! # Latent Memory Weaver
//!
//! Persistent memory bank using VSA-bundled hypervectors.
//! Replaces text-based RAG with pure latent-space memory operations.
//!
//! ## Architecture
//!
//! The memory weaver operates as a dynamic state machine:
//! - **Write**: New facts are bound (role ⊗ filler) and added to the bundle
//! - **Read**: Roles are used to unbind and retrieve fillers
//! - **Forget**: Anti-vectors subtract old bindings from the bundle
//! - **Consolidate**: Resonator cleanup compresses and stabilizes memory
//!
//! This provides an effectively infinite context window compressed
//! into a fixed-size hypervector structure.

pub mod bank;
pub mod operations;

pub use bank::MemoryBank;
pub use operations::MemoryOp;
