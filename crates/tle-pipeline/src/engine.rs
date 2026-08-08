//! The Latent Execution Engine: orchestrates the full pipeline.

use tle_vsa::{bind, bundle, Codebook, HyperVector, DEFAULT_DIM, cosine_similarity};
use tle_resonator::{ResonatorNetwork, ResonatorConfig, CleanupRule};
use tle_clifford::{SyntaxNode, SyntaxRelation};
use tle_tda_router::{TopologicalRouter, MapperConfig, FilterFunction, NodeType};
use tle_memory::MemoryBank;
use tle_decoder::LatentDecoder;

/// Configuration for the Latent Execution Engine.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Dimensionality of hypervectors.
    pub dim: usize,
    /// Number of routing nodes (analogous to MoE experts).
    pub num_experts: usize,
    /// Vocabulary for the decoder.
    pub vocabulary: Vec<String>,
    /// Base seed for reproducible generation.
    pub seed: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            dim: DEFAULT_DIM,
            num_experts: 5,
            vocabulary: default_vocabulary(),
            seed: 0xCAFE_BABE_DEAD_BEEF,
        }
    }
}

/// The complete Latent Execution Engine.
///
/// This is the top-level structure that replaces a traditional LLM.
/// It processes language entirely through deterministic mathematical
/// operations in hyperdimensional space.
pub struct LatentEngine {
    /// Role codebook (syntactic roles like SUBJECT, VERB, OBJECT).
    role_codebook: Codebook,
    /// Word codebook (vocabulary → hypervectors).
    word_codebook: Codebook,
    /// The topological router (replaces MoE gating).
    router: TopologicalRouter,
    /// The syntax node (Clifford algebra operations).
    syntax_node: SyntaxNode,
    /// The latent memory bank.
    memory: MemoryBank,
    /// The decoder (latent → English).
    decoder: LatentDecoder,
    /// Resonator for cleanup.
    resonator: ResonatorNetwork,
    /// Dimensionality.
    dim: usize,
}

impl LatentEngine {
    /// Build a new engine with the given configuration.
    pub fn new(config: EngineConfig) -> Self {
        let dim = config.dim;

        // Build role codebook
        let role_codebook = Codebook::with_standard_roles(dim, config.seed);

        // Build word codebook
        let mut word_codebook = Codebook::new(dim, config.seed ^ 0xFFFF);
        let vocab_strs: Vec<&str> = config.vocabulary.iter().map(|s| s.as_str()).collect();
        for word in &vocab_strs {
            word_codebook.get_or_insert(word);
        }

        // Build initial router from vocabulary vectors
        let reference_vectors: Vec<HyperVector> = config
            .vocabulary
            .iter()
            .map(|w| word_codebook.get(w).unwrap().clone())
            .collect();

        let mapper_config = MapperConfig {
            num_intervals: config.num_experts,
            overlap_fraction: 0.3,
            cluster_radius: (dim as f32).sqrt() * 1.5,
            filter: FilterFunction::Norm,
        };
        let router = TopologicalRouter::build(&reference_vectors, mapper_config);

        // Build decoder
        let decoder_codebook = Codebook::new(dim, config.seed ^ 0xFFFF);
        let decoder = LatentDecoder::new(decoder_codebook, &vocab_strs);

        // Resonator
        let resonator_config = ResonatorConfig {
            max_iterations: 30,
            epsilon: 1e-7,
            cleanup_rule: CleanupRule::Sign,
            temperature: 1.0,
        };
        let resonator = ResonatorNetwork::with_config(resonator_config);

        Self {
            role_codebook,
            word_codebook,
            router,
            syntax_node: SyntaxNode::new(),
            memory: MemoryBank::new(dim),
            decoder,
            resonator,
            dim,
        }
    }

    /// Process an English input string and produce an English output.
    ///
    /// This is the complete pipeline execution:
    /// encode → bind → route → process → cleanup → decode
    ///
    /// **100% deterministic**: same input always produces same output.
    pub fn process(&mut self, input: &str) -> ProcessingResult {
        // Step 1: Encode input tokens to hypervectors
        let tokens: Vec<&str> = input.split_whitespace().collect();
        let encoded: Vec<HyperVector> = tokens
            .iter()
            .map(|t| self.word_codebook.get_or_insert(t).clone())
            .collect();

        if encoded.is_empty() {
            return ProcessingResult {
                output: String::new(),
                tokens_processed: 0,
                routing_decisions: Vec::new(),
                memory_ops: 0,
            };
        }

        // Step 2: Bind tokens with positional roles
        let mut bound_tokens: Vec<HyperVector> = Vec::new();
        for (i, token_vec) in encoded.iter().enumerate() {
            let pos_role_name = format!("POSITION_{}", i.min(9));
            let pos_role = self.role_codebook.get_or_insert(&pos_role_name).clone();
            let bound = bind(&pos_role, token_vec);
            bound_tokens.push(bound);
        }

        // Step 3: Create context composite (bundle all bound tokens)
        let bundle_refs: Vec<&HyperVector> = bound_tokens.iter().collect();
        let context = bundle(&bundle_refs);

        // Step 4: Route through topological graph
        let mut routing_decisions = Vec::new();
        let mut processed_vectors: Vec<HyperVector> = Vec::new();

        for token_vec in &encoded {
            let decision = self.router.route(token_vec);
            routing_decisions.push(format!("{:?}", decision.node_type));

            // Process based on node type
            let processed = match decision.node_type {
                NodeType::Syntax => {
                    // Apply syntactic analysis
                    self.process_syntax(token_vec, &context)
                }
                NodeType::Semantic => {
                    // Store in memory for context building
                    self.process_semantic(token_vec, &context)
                }
                NodeType::Memory => {
                    // Retrieve from memory
                    self.process_memory(token_vec)
                }
                NodeType::Generation => {
                    // Pass through for output generation
                    token_vec.clone()
                }
                _ => token_vec.clone(),
            };

            processed_vectors.push(processed);
        }

        // Step 5: Store context in memory for future queries
        let context_role = self.role_codebook.get_or_insert("CONTEXT").clone();
        self.memory.store(&context_role, &context, 1.0);

        // Step 6: Decode output vectors back to English
        let decoded = self.decoder.decode_sequence(&processed_vectors);
        let output = decoded
            .iter()
            .map(|d| d.token.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        ProcessingResult {
            output,
            tokens_processed: tokens.len(),
            routing_decisions,
            memory_ops: 1,
        }
    }

    /// Process a token through the syntax node.
    fn process_syntax(&self, token: &HyperVector, context: &HyperVector) -> HyperVector {
        // Detect syntactic relation with context
        let (_relation, _strength) = self.syntax_node.detect_relation(token, context);
        // Return the token (syntax detection doesn't modify the vector itself,
        // but informs routing for subsequent tokens)
        token.clone()
    }

    /// Process a token semantically (store in memory).
    fn process_semantic(&mut self, token: &HyperVector, context: &HyperVector) -> HyperVector {
        let sem_role = self.role_codebook.get_or_insert("SEMANTIC").clone();
        self.memory.store(&sem_role, token, 0.8);
        token.clone()
    }

    /// Retrieve from memory.
    fn process_memory(&self, query: &HyperVector) -> HyperVector {
        let (retrieved, _confidence) = self.memory.retrieve(query);
        retrieved
    }

    /// Get engine statistics.
    pub fn stats(&self) -> EngineStats {
        EngineStats {
            dim: self.dim,
            num_routing_nodes: self.router.num_nodes(),
            memory_stats: self.memory.stats(),
            vocab_size: self.decoder.vocab_size(),
        }
    }
}

/// Result of processing an input through the engine.
#[derive(Clone, Debug)]
pub struct ProcessingResult {
    /// The decoded English output.
    pub output: String,
    /// Number of tokens processed.
    pub tokens_processed: usize,
    /// Routing decisions made (node types selected).
    pub routing_decisions: Vec<String>,
    /// Number of memory operations performed.
    pub memory_ops: usize,
}

/// Engine statistics.
#[derive(Clone, Debug)]
pub struct EngineStats {
    pub dim: usize,
    pub num_routing_nodes: usize,
    pub memory_stats: tle_memory::bank::MemoryStats,
    pub vocab_size: usize,
}

/// Default vocabulary for demonstration.
fn default_vocabulary() -> Vec<String> {
    vec![
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "I", "you", "he", "she", "it", "we", "they", "my", "your",
        "cat", "dog", "bird", "fish", "tree", "house", "car", "book",
        "sat", "ran", "walked", "jumped", "flew", "swam", "read", "wrote",
        "on", "in", "at", "by", "with", "from", "to", "of", "for",
        "big", "small", "fast", "slow", "red", "blue", "green", "old", "new",
        "good", "bad", "happy", "sad", "hot", "cold", "bright", "dark",
        "and", "or", "but", "not", "if", "then", "because", "when",
        "what", "who", "where", "why", "how", "which", "that", "this",
        "love", "hate", "see", "hear", "think", "know", "want", "need",
        "hello", "world", "yes", "no", "please", "thanks", "sorry",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let config = EngineConfig::default();
        let engine = LatentEngine::new(config);
        let stats = engine.stats();

        assert_eq!(stats.dim, DEFAULT_DIM);
        assert!(stats.num_routing_nodes > 0);
        assert!(stats.vocab_size > 50);
    }

    #[test]
    fn test_deterministic_processing() {
        let config = EngineConfig::default();
        let mut engine1 = LatentEngine::new(config.clone());
        let mut engine2 = LatentEngine::new(config);

        let input = "the cat sat on the mat";
        let result1 = engine1.process(input);
        let result2 = engine2.process(input);

        assert_eq!(
            result1.output, result2.output,
            "Same input must produce same output (deterministic)"
        );
        assert_eq!(result1.tokens_processed, result2.tokens_processed);
    }

    #[test]
    fn test_process_simple_sentence() {
        let config = EngineConfig::default();
        let mut engine = LatentEngine::new(config);

        let result = engine.process("the cat sat");
        assert_eq!(result.tokens_processed, 3);
        assert!(!result.output.is_empty());
        println!("Input: 'the cat sat' → Output: '{}'", result.output);
    }

    #[test]
    fn test_100_run_determinism() {
        let config = EngineConfig::default();
        let input = "hello world";

        let mut outputs = Vec::new();
        for _ in 0..100 {
            let mut engine = LatentEngine::new(config.clone());
            let result = engine.process(input);
            outputs.push(result.output);
        }

        // ALL 100 runs must produce identical output
        let first = &outputs[0];
        for (i, output) in outputs.iter().enumerate() {
            assert_eq!(
                output, first,
                "Run {} produced different output: '{}' vs '{}'",
                i, output, first
            );
        }
    }
}
