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
pub mod graph;
pub mod hopfield;
pub mod inference;
pub mod linearize;
pub mod mdl;
pub mod search;
pub mod semantic;
pub mod sheaf;
pub mod templates;

pub use decompose::{DecomposedFact, decompose_sentence, extract_sentence_entities, is_fact_worthy, query_relations};
pub use engine::{AxiomGen, GenerationResult};
pub use graph::{KnowledgeGraph, Triple};
pub use energy::EnergyConfig;
pub use search::{SearchConfig, ScoredPath};
pub use linearize::Intent;
