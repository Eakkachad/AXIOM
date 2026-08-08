//! EngramNode — AFC FlowNode integration.
//!
//! Wraps the Engram table + fusion layer as a FlowNode that can be composed
//! into any AFC generation pipeline. Queries the Engram for the current
//! context and adds scores to FlowState candidates.

use tle_afc::node::{FlowNode, FlowState};

use crate::builder::BuiltEngram;
use crate::fusion::{FusionConfig, SigmoidFusion};

/// A FlowNode that queries the Engram multi-head hash table and adds
/// scored results to the FlowState.
///
/// This is the bridge between Engram (fast hash lookup) and AFC (composable flows).
pub struct EngramNode {
    /// The built Engram (tables + vocab + hasher).
    engram: BuiltEngram,
    /// Fusion layer for combining multi-head results.
    fusion: SigmoidFusion,
    /// Weight multiplier for this node's contribution.
    pub weight: f32,
    /// Context history (token IDs) maintained across steps.
    /// Updated externally or via history_from_flow_state.
    context_buffer: Vec<u16>,
}

impl EngramNode {
    /// Create an EngramNode from a built Engram.
    pub fn new(engram: BuiltEngram, weight: f32) -> Self {
        let vocab_size = engram.vocab.len();
        Self {
            engram,
            fusion: SigmoidFusion::new(vocab_size),
            weight,
            context_buffer: Vec::new(),
        }
    }

    /// Create with custom fusion configuration.
    pub fn with_fusion_config(engram: BuiltEngram, weight: f32, fusion_config: FusionConfig) -> Self {
        let vocab_size = engram.vocab.len();
        Self {
            engram,
            fusion: SigmoidFusion::with_config(vocab_size, fusion_config),
            weight,
            context_buffer: Vec::new(),
        }
    }

    /// Set the context buffer directly (for testing or manual control).
    pub fn set_context(&mut self, context: Vec<u16>) {
        self.context_buffer = context;
    }

    /// Push a token ID to the context buffer.
    pub fn push_context(&mut self, token_id: u16) {
        self.context_buffer.push(token_id);
    }

    /// Clear the context buffer.
    pub fn clear_context(&mut self) {
        self.context_buffer.clear();
    }

    /// Get the vocabulary from the engram.
    pub fn vocab(&self) -> &crate::builder::Vocab {
        &self.engram.vocab
    }

    /// Query the engram with current context and return fused scores.
    pub fn query_scores(&self) -> Vec<f32> {
        let results = self.engram.query(&self.context_buffer);
        self.fusion.fuse(
            &results
                .iter()
                .map(|(order, conf, entry)| (*order, *conf, *entry))
                .collect::<Vec<_>>(),
        )
    }

    /// Get statistics about the engram.
    pub fn stats(&self) {
        self.engram.print_stats();
    }
}

impl FlowNode for EngramNode {
    fn transform(&self, mut state: FlowState) -> FlowState {
        // Use history from FlowState as context (token IDs are indices)
        let context: Vec<u16> = state.history.iter().map(|&idx| idx as u16).collect();

        if context.is_empty() {
            return state;
        }

        // Query engram with the context
        let results = self.engram.query(&context);
        if results.is_empty() {
            return state;
        }

        // Fuse multi-head results
        let fused = self.fusion.fuse(
            &results
                .iter()
                .map(|(order, conf, entry)| (*order, *conf, *entry))
                .collect::<Vec<_>>(),
        );

        // Add fused scores to state.scores (weighted)
        // Map engram token IDs to candidate indices
        for (candidate_idx, score) in state.scores.iter_mut().enumerate() {
            if candidate_idx < fused.len() {
                *score += self.weight * fused[candidate_idx];
            }
        }

        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::EngramBuilder;
    use tle_vsa::HyperVector;

    #[test]
    fn test_engram_node_basic() {
        let mut builder = EngramBuilder::new();
        for _ in 0..20 {
            builder.ingest_line("the cat sat on the mat");
        }
        let engram = builder.build();
        let vocab_size = engram.vocab.len();

        let node = EngramNode::new(engram, 1.0);

        // Create a FlowState with history = [the_id, cat_id]
        let mut state = FlowState::new(64);
        state.candidates = (0..vocab_size)
            .map(|_| HyperVector::zeros(64))
            .collect();
        state.scores = vec![0.0; vocab_size];
        // History: "the" = 0, "cat" = 1 (first two tokens ingested)
        state.history = vec![0, 1]; // "the", "cat"

        let result = node.transform(state);

        // Should have non-zero scores for at least some candidates
        let non_zero: usize = result.scores.iter().filter(|&&s| s > 0.0).count();
        assert!(non_zero > 0, "Engram should score at least one candidate");
    }

    #[test]
    fn test_engram_node_predicts_correctly() {
        let mut builder = EngramBuilder::new();
        for _ in 0..50 {
            builder.ingest_line("a b c d e");
        }
        let engram = builder.build();
        let vocab_size = engram.vocab.len();

        let b_id = engram.vocab.get_id("b").unwrap();
        let c_id = engram.vocab.get_id("c").unwrap();

        let node = EngramNode::new(engram, 1.0);

        let mut state = FlowState::new(64);
        state.candidates = (0..vocab_size)
            .map(|_| HyperVector::zeros(64))
            .collect();
        state.scores = vec![0.0; vocab_size];
        // Context: "a", "b" → should predict "c"
        state.history = vec![0, b_id as usize];

        let result = node.transform(state);

        // "c" should have the highest score
        let best_idx = result
            .scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(best_idx, c_id as usize, "Should predict 'c' after 'a b'");
    }
}
