//! Composite Energy Flow — the full Algorithm 1 generation pipeline.
//!
//! Implements deterministic token generation by composing transition scoring,
//! context accumulation, repetition penalty, and diversity penalty into a
//! single energy-based selection loop.

use tle_vsa::{cosine_similarity, HyperVector};

use crate::graph::FlowGraph;
use crate::node::FlowState;
use crate::nodes::{
    ContextAccumNode, DiversityPenaltyNode, RepetitionPenaltyNode, TransitionScoreNode,
};

/// Configuration for the composite energy flow pipeline.
#[derive(Clone, Debug)]
pub struct EnergyConfig {
    /// Weight for transition memory scoring.
    pub alpha: f32,
    /// Weight for context accumulation scoring.
    pub beta: f32,
    /// Weight for repetition penalty.
    pub gamma: f32,
    /// Weight for diversity penalty.
    pub delta: f32,
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Window size for repetition penalty.
    pub context_window: usize,
    /// Decay factor for context accumulation.
    pub context_decay: f32,
    /// Minimum confidence (max score) to continue generation.
    pub confidence_threshold: f32,
}

impl Default for EnergyConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 0.5,
            gamma: 0.3,
            delta: 0.1,
            max_tokens: 128,
            context_window: 5,
            context_decay: 0.9,
            confidence_threshold: 0.01,
        }
    }
}

/// The full deterministic generation pipeline (Algorithm 1).
///
/// Composes all flow nodes into a single generation loop that:
/// 1. Initializes state from the prompt
/// 2. At each step, scores all candidates via the flow graph
/// 3. Selects the argmax candidate (deterministic)
/// 4. Updates state and repeats until max_tokens or confidence drops
pub struct CompositeEnergyFlow {
    /// Transition memory hypervector.
    pub transition_memory: HyperVector,
    /// Codebook vectors (one per vocabulary token).
    pub codebook_vectors: Vec<HyperVector>,
    /// Vocabulary tokens (parallel to codebook_vectors).
    pub vocab: Vec<String>,
    /// Pipeline configuration.
    pub config: EnergyConfig,
}

impl CompositeEnergyFlow {
    /// Create a new composite energy flow.
    pub fn new(
        transition_memory: HyperVector,
        codebook_vectors: Vec<HyperVector>,
        vocab: Vec<String>,
        config: EnergyConfig,
    ) -> Self {
        assert_eq!(
            codebook_vectors.len(),
            vocab.len(),
            "Codebook and vocab must have same length"
        );
        Self {
            transition_memory,
            codebook_vectors,
            vocab,
            config,
        }
    }

    /// Build the flow graph for a single generation step.
    fn build_graph(&self) -> FlowGraph {
        let mut graph = FlowGraph::new();

        graph.add_node(Box::new(TransitionScoreNode::new(
            self.transition_memory.clone(),
            self.config.alpha,
        )));
        graph.add_node(Box::new(ContextAccumNode::new(
            self.config.context_decay,
            self.config.beta,
        )));
        graph.add_node(Box::new(RepetitionPenaltyNode::new(
            self.config.context_window,
            self.config.gamma,
            1.0,
        )));
        graph.add_node(Box::new(DiversityPenaltyNode::new(self.config.delta)));

        graph
    }

    /// Generate a sequence of token indices from a prompt.
    ///
    /// The prompt is a slice of token indices that seed the generation.
    /// Returns the full generated sequence (excluding the prompt).
    pub fn generate(&self, prompt: &[usize]) -> Vec<usize> {
        if self.codebook_vectors.is_empty() {
            return Vec::new();
        }

        let dim = self.codebook_vectors[0].dim();
        let graph = self.build_graph();
        let mut output = Vec::new();

        // Initialize state from prompt
        let mut state = FlowState::new(dim);
        state.candidates = self.codebook_vectors.clone();

        // Seed context and current from prompt
        for &idx in prompt {
            if idx < self.codebook_vectors.len() {
                let token_vec = &self.codebook_vectors[idx];
                // Update context with prompt tokens
                let decayed = state.context.scale(self.config.context_decay);
                let fresh = token_vec.scale(1.0 - self.config.context_decay);
                state.context = decayed.add(&fresh);
                state.current = token_vec.clone();
                state.history.push(idx);
            }
        }

        // Generation loop
        for step in 0..self.config.max_tokens {
            state.step = step;
            state.scores = vec![0.0; self.codebook_vectors.len()];

            // Run the flow graph
            state = graph.execute(state);

            // Deterministic selection: argmax
            let (best_idx, best_score) = state
                .scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, &s)| (i, s))
                .unwrap_or((0, 0.0));

            // Confidence check
            if best_score < self.config.confidence_threshold && step > 0 {
                break;
            }

            // Update state for next step
            state.current = self.codebook_vectors[best_idx].clone();
            state.history.push(best_idx);
            output.push(best_idx);
        }

        output
    }

    /// Decode a sequence of token indices into a string.
    pub fn decode(&self, indices: &[usize]) -> String {
        indices
            .iter()
            .filter_map(|&i| self.vocab.get(i))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Compute the confidence (cosine similarity) between predicted and actual.
    pub fn confidence_at(&self, state: &FlowState, candidate_idx: usize) -> f32 {
        if candidate_idx >= self.codebook_vectors.len() {
            return 0.0;
        }
        cosine_similarity(&state.current, &self.codebook_vectors[candidate_idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tle_vsa::{bind, HyperVector};

    #[test]
    fn test_energy_flow_generation() {
        let dim = 512;
        // Create a small vocabulary
        let vocab: Vec<String> = vec!["the", "cat", "sat", "on", "mat"]
            .into_iter()
            .map(String::from)
            .collect();
        let codebook: Vec<HyperVector> = (0..5)
            .map(|i| HyperVector::random_bipolar(dim, i * 100))
            .collect();

        // Build TM: encode "the" -> "cat" and "cat" -> "sat"
        let tm_the_cat = bind(&codebook[0].permute(1), &codebook[1]);
        let tm_cat_sat = bind(&codebook[1].permute(1), &codebook[2]);
        let tm = tm_the_cat.add(&tm_cat_sat);

        let config = EnergyConfig {
            max_tokens: 3,
            confidence_threshold: -10.0, // don't stop early
            ..Default::default()
        };

        let flow = CompositeEnergyFlow::new(tm, codebook, vocab, config);
        let result = flow.generate(&[0]); // prompt = "the"

        // Should generate something (non-empty)
        assert!(!result.is_empty());
        // First generated token should be "cat" (index 1) since TM encodes the->cat
        assert_eq!(result[0], 1);
    }

    #[test]
    fn test_decode() {
        let dim = 64;
        let vocab = vec!["hello".to_string(), "world".to_string()];
        let codebook = vec![
            HyperVector::random_bipolar(dim, 1),
            HyperVector::random_bipolar(dim, 2),
        ];
        let tm = HyperVector::zeros(dim);

        let flow = CompositeEnergyFlow::new(tm, codebook, vocab, EnergyConfig::default());
        assert_eq!(flow.decode(&[0, 1]), "hello world");
    }
}
