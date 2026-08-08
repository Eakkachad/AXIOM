//! # Compressed Knowledge Representation (CKR)
//!
//! Hierarchical VSA-based knowledge storage that scales to 200K+ facts
//! in ~16MB memory. Uses tiered bundles with automatic splitting when
//! SNR degrades.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │         Category Index (Bloom Filter)        │  ← "Do I know about X?" O(1)
//! ├─────────────────────────────────────────────┤
//! │  Category: "animals"    │  Category: "geo"  │  ← Topic clusters
//! │  ┌─────────────────┐    │  ┌──────────────┐ │
//! │  │ Bundle 0 (≤200) │    │  │ Bundle 0     │ │  ← VSA vectors (D=4096)
//! │  │ Bundle 1 (≤200) │    │  │ Bundle 1     │ │
//! │  │ ...             │    │  │ ...          │ │
//! │  └─────────────────┘    │  └──────────────┘ │
//! ├─────────────────────────────────────────────┤
//! │         Exact Store (overflow/recent)        │  ← HashMap for precise recall
//! └─────────────────────────────────────────────┘
//! ```
//!
//! ## Key Properties
//! - O(√N) memory: bundles of fixed size, only number of bundles grows
//! - O(1) "do I know this?" check via Bloom filter
//! - Auto-split bundles when SNR drops below threshold
//! - Exact store for high-priority/recent facts (HashMap)
//! - Fully deterministic

pub mod bloom;
pub mod bundle;
pub mod category;
pub mod store;

pub use bloom::BloomFilter;
pub use bundle::KnowledgeBundle;
pub use category::CategoryIndex;
pub use store::CompressedKnowledgeStore;
