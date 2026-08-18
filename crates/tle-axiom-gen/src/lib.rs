//! # AXIOM-Gen: Compositional Text Generation via Energy-Guided Knowledge Graph Traversal
//!
//! This crate implements the AXIOM-Gen algorithm, which generates compositional
//! sentences by traversing a knowledge graph using energy-based beam search,
//! guided by hyperdimensional vector representations.
//!
//! ## Architecture
//!
//! 1. **Knowledge Graph** (`graph`): Stores factual triples (subject, relation, object)
//! 2. **Energy Function** (`energy`): Scores candidate paths using VSA similarity
//! 3. **Beam Search** (`search`): Explores the graph with energy-guided pruning
//! 4. **Linearization** (`linearize`): Converts graph paths to natural language
//! 5. **Engine** (`engine`): Orchestrates the full generation pipeline

pub mod answer_type;
pub mod decompose;
pub mod energy;
pub mod engine;
pub mod flash_hopfield;
pub mod graph;
pub mod hopfield;
pub mod inference;
pub mod linearize;
pub mod mdl;
pub mod ring_buffer;
pub mod search;
pub mod semantic;
pub mod sheaf;
pub mod sheaf_layer;
pub mod simd_ops;
pub mod templates;

pub use decompose::{DecomposedFact, decompose_sentence, extract_sentence_entities, is_fact_worthy, query_relations};
pub use engine::{AxiomGen, GenerationResult};
pub use flash_hopfield::{FlashHopfieldConfig, FlashHopfieldLayer};
pub use graph::{KnowledgeGraph, Triple};
pub use energy::EnergyConfig;
pub use ring_buffer::{CachePadded, SpscRingBuffer};
pub use search::{SearchConfig, ScoredPath};
pub use sheaf_layer::{RotorType, SheafConfig, SheafContextLayer};
pub use simd_ops::{AlignedBuffer64, fast_exp_f32, simd_dot_f32};
pub use linearize::Intent;
