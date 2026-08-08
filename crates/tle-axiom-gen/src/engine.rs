//! AXIOM-Gen Engine: orchestrates the full compositional generation pipeline.
//!
//! The engine ties together:
//! 1. Knowledge Graph storage
//! 2. VSA codebook for symbol→vector mapping
//! 3. Energy-guided beam search
//! 4. Linearization to natural language
//!
//! ## Usage
//!
//! ```rust
//! use tle_axiom_gen::AxiomGen;
//!
//! let mut gen = AxiomGen::new(2048);
//! gen.add_fact("sky", "is", "blue");
//! gen.add_fact("blue_light", "has", "short_wavelength");
//!
//! let result = gen.generate("why is the sky blue?");
//! println!("{}", result.sentence);
//! ```

use tle_vsa::{Codebook, HyperVector};

use crate::energy::EnergyConfig;
use crate::graph::KnowledgeGraph;
use crate::linearize::{classify_intent, linearize};
use crate::search::{beam_search, SearchConfig};

/// The result of a generation query.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// The generated natural language sentence.
    pub sentence: String,
    /// Reasoning trace: descriptions of each step in the path.
    pub reasoning: Vec<String>,
    /// Energy score of the best path found.
    pub energy: f32,
    /// Number of triples in the selected path.
    pub path_length: usize,
}

/// The main AXIOM-Gen engine for compositional text generation.
pub struct AxiomGen {
    /// The knowledge graph storing all facts.
    pub graph: KnowledgeGraph,
    /// VSA codebook mapping symbols to hypervectors.
    pub codebook: Codebook,
    /// Energy function configuration.
    pub energy_config: EnergyConfig,
    /// Beam search configuration.
    pub search_config: SearchConfig,
}

impl AxiomGen {
    /// Create a new AXIOM-Gen engine with the given vector dimensionality.
    pub fn new(dim: usize) -> Self {
        Self {
            graph: KnowledgeGraph::new(),
            codebook: Codebook::new(dim, 0xA10A_CAFE_BEAD_0001),
            energy_config: EnergyConfig::default(),
            search_config: SearchConfig::default(),
        }
    }

    /// Create a new engine with custom configuration.
    pub fn with_config(
        dim: usize,
        energy_config: EnergyConfig,
        search_config: SearchConfig,
    ) -> Self {
        Self {
            graph: KnowledgeGraph::new(),
            codebook: Codebook::new(dim, 0xA10A_CAFE_BEAD_0001),
            energy_config,
            search_config,
        }
    }

    /// Add a fact (triple) to the knowledge graph.
    ///
    /// Automatically registers entities and relations in the codebook.
    pub fn add_fact(&mut self, subject: &str, relation: &str, object: &str) {
        // Add to knowledge graph
        self.graph.add_triple(subject, relation, object);

        // Pre-populate codebook with all symbols
        self.codebook.get_or_insert(subject);
        self.codebook.get_or_insert(relation);
        self.codebook.get_or_insert(object);
    }

    /// Generate a compositional sentence answering the query.
    ///
    /// Pipeline:
    /// 1. Parse query → extract entities + classify intent
    /// 2. BFS subgraph extraction from relevant entities
    /// 3. Energy-guided beam search for best path
    /// 4. Linearize best path into natural language
    /// 5. Return result with reasoning trace
    pub fn generate(&mut self, query: &str) -> GenerationResult {
        // Step 1: Parse query — extract known entities and classify intent
        let intent = classify_intent(query);
        let query_entities = self.extract_query_entities(query);

        if query_entities.is_empty() {
            return GenerationResult {
                sentence: String::new(),
                reasoning: vec!["No known entities found in query.".to_string()],
                energy: 0.0,
                path_length: 0,
            };
        }

        // Step 2: Build query vector by bundling entity vectors
        let query_vector = self.build_query_vector(query);

        // Step 3: BFS subgraph from query entities
        let _subgraph = self.graph.bfs_subgraph(&query_entities, self.search_config.max_hops);

        // Step 4: Beam search for best paths
        let results = beam_search(
            &self.graph,
            &query_entities,
            &query_vector,
            &mut self.codebook,
            &self.energy_config,
            &self.search_config,
        );

        if results.is_empty() {
            return GenerationResult {
                sentence: String::new(),
                reasoning: vec!["No paths found in knowledge graph.".to_string()],
                energy: 0.0,
                path_length: 0,
            };
        }

        // Step 5: Select best path — prefer paths meeting target length, then by energy
        let target_len = self.energy_config.target_length;
        let best = results
            .iter()
            .filter(|p| p.path.len() >= target_len)
            .max_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap_or(std::cmp::Ordering::Equal))
            .or_else(|| {
                // Fallback: pick longest path available
                results.iter().max_by_key(|p| p.path.len())
            })
            .unwrap();

        let path_triples: Vec<_> = best.path.iter().map(|&i| self.graph.triples[i]).collect();

        // Build reasoning trace
        let reasoning: Vec<String> = path_triples
            .iter()
            .map(|t| {
                format!(
                    "{} {} {}",
                    self.graph.entity_name(t.subject_id),
                    self.graph.relation_name(t.relation_id),
                    self.graph.entity_name(t.object_id),
                )
            })
            .collect();

        // Linearize path to natural language
        let sentence = linearize(
            &path_triples,
            &self.graph.entities,
            &self.graph.relations,
            intent,
        );

        GenerationResult {
            sentence,
            reasoning,
            energy: best.energy,
            path_length: path_triples.len(),
        }
    }

    /// Extract entity IDs mentioned in the query that exist in the knowledge graph.
    fn extract_query_entities(&self, query: &str) -> Vec<usize> {
        let lower = query.to_lowercase();
        // Remove punctuation for matching
        let cleaned: String = lower
            .chars()
            .map(|c| if c.is_alphanumeric() || c == ' ' || c == '_' { c } else { ' ' })
            .collect();
        let words: Vec<&str> = cleaned.split_whitespace().collect();

        let mut found_entities: Vec<usize> = Vec::new();

        // Try to match entities in the graph
        for (name, &id) in &self.graph.entity_index {
            let entity_lower = name.to_lowercase();
            let entity_words: Vec<&str> = entity_lower.split('_').collect();

            // Check if any entity word appears in the query
            let matches = entity_words.iter().any(|ew| words.contains(ew));
            if matches && !found_entities.contains(&id) {
                found_entities.push(id);
            }
        }

        found_entities
    }

    /// Build a query vector by bundling vectors of all words in the query.
    fn build_query_vector(&mut self, query: &str) -> HyperVector {
        let dim = self.codebook.dim();
        let lower = query.to_lowercase();
        let words: Vec<&str> = lower
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|w| !w.is_empty())
            .collect();

        if words.is_empty() {
            return HyperVector::zeros(dim);
        }

        let mut result = HyperVector::zeros(dim);
        for word in &words {
            let vec = self.codebook.get_or_insert(word).clone();
            result = result.add(&vec);
        }

        result.normalize()
    }
}

// Base seed for AXIOM-Gen codebook
const _AXIOM_GEN_SEED: u64 = 0xAA10_CAFE_BEAD_0001;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axiom_gen_new() {
        let gen = AxiomGen::new(2048);
        assert_eq!(gen.graph.entities.len(), 0);
        assert_eq!(gen.graph.triples.len(), 0);
    }

    #[test]
    fn test_add_fact() {
        let mut gen = AxiomGen::new(2048);
        gen.add_fact("sky", "is", "blue");

        assert_eq!(gen.graph.entities.len(), 2);
        assert_eq!(gen.graph.relations.len(), 1);
        assert_eq!(gen.graph.triples.len(), 1);
        // Codebook should have entries for all symbols
        assert!(gen.codebook.get("sky").is_some());
        assert!(gen.codebook.get("is").is_some());
        assert!(gen.codebook.get("blue").is_some());
    }

    #[test]
    fn test_extract_query_entities() {
        let mut gen = AxiomGen::new(2048);
        gen.add_fact("sky", "is", "blue");
        gen.add_fact("blue_light", "has", "short_wavelength");

        let entities = gen.extract_query_entities("why is the sky blue?");
        assert!(!entities.is_empty());
        // Should find "sky" and/or "blue"
        let sky_id = gen.graph.entity_id("sky").unwrap();
        let blue_id = gen.graph.entity_id("blue").unwrap();
        assert!(
            entities.contains(&sky_id) || entities.contains(&blue_id),
            "Should find sky or blue in query"
        );
    }

    #[test]
    fn test_generate_simple() {
        let mut gen = AxiomGen::new(2048);
        gen.add_fact("sky", "is", "blue");

        let result = gen.generate("what is the sky?");
        assert!(!result.sentence.is_empty());
        assert!(result.path_length >= 1);
        assert!(!result.reasoning.is_empty());
    }

    #[test]
    fn test_generate_novel_sentence() {
        let mut gen = AxiomGen::new(2048);
        // Use a connected chain: sky -> blue -> short_wavelength -> more_in_atmosphere
        gen.add_fact("sky", "is", "blue");
        gen.add_fact("blue", "has", "short_wavelength");
        gen.add_fact("short_wavelength", "scatters", "more_in_atmosphere");

        // Adjust config to prefer longer multi-hop paths
        gen.energy_config.lambda_simplicity = 0.0;
        gen.energy_config.lambda_length = 0.5;
        gen.energy_config.target_length = 3;

        let result = gen.generate("why is the sky blue?");
        assert!(!result.sentence.is_empty());
        assert!(
            result.sentence.to_lowercase().contains("blue"),
            "Sentence should mention blue: {}",
            result.sentence
        );
        assert!(
            result.reasoning.len() >= 2,
            "Should have multi-hop reasoning, got {}: {:?}",
            result.reasoning.len(),
            result.reasoning
        );
    }

    #[test]
    fn test_generate_no_entities() {
        let mut gen = AxiomGen::new(2048);
        gen.add_fact("sky", "is", "blue");

        let result = gen.generate("what is xyz123?");
        // Should gracefully handle unknown entities
        assert_eq!(result.path_length, 0);
    }

    #[test]
    fn test_generate_energy_finite() {
        let mut gen = AxiomGen::new(2048);
        gen.add_fact("water", "is", "liquid");
        gen.add_fact("liquid", "has", "flow");

        let result = gen.generate("what is water?");
        assert!(result.energy.is_finite());
    }

    #[test]
    fn test_generate_with_reasoning_trace() {
        let mut gen = AxiomGen::new(2048);
        gen.add_fact("sun", "causes", "heat");
        gen.add_fact("heat", "causes", "evaporation");
        gen.add_fact("evaporation", "causes", "clouds");

        let result = gen.generate("why do we have clouds?");
        // Reasoning should describe the path
        for step in &result.reasoning {
            assert!(!step.is_empty());
        }
    }
}
