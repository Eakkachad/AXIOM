//! Sigmoid-gated fusion of multi-head Engram results.
//!
//! When multiple N-gram heads return results (e.g., unigram, bigram, trigram
//! all match), the fusion layer combines them using confidence-weighted
//! sigmoid gating. Longer matches (higher-order N-grams) are given higher
//! confidence, but only if they have sufficient observation count.

use crate::table::EngramEntry;

/// Configuration for sigmoid fusion.
#[derive(Clone, Debug)]
pub struct FusionConfig {
    /// Confidence threshold for sigmoid (controls sharpness).
    pub confidence_threshold: f32,
    /// Weight boost per N-gram order (higher order = more trusted).
    pub order_boost: f32,
    /// Temperature for softmax-like normalization of fused scores.
    pub temperature: f32,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 3.0,
            order_boost: 0.5,
            temperature: 1.0,
        }
    }
}

/// Sigmoid-gated fusion layer for multi-head Engram results.
///
/// Combines results from different N-gram heads (1-gram through 5-gram)
/// into a single probability distribution over the vocabulary.
///
/// The key insight (from katgpt-rs Engram design):
/// - Higher-order matches are more specific → higher confidence
/// - But they may have lower count → need enough evidence
/// - Sigmoid gates each head's contribution by its confidence
pub struct SigmoidFusion {
    /// Configuration.
    pub config: FusionConfig,
    /// Vocabulary size (for output distribution).
    pub vocab_size: usize,
}

impl SigmoidFusion {
    /// Create a new fusion layer.
    pub fn new(vocab_size: usize) -> Self {
        Self {
            config: FusionConfig::default(),
            vocab_size,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(vocab_size: usize, config: FusionConfig) -> Self {
        Self { config, vocab_size }
    }

    /// Fuse results from multiple heads into a score distribution.
    ///
    /// Input: Vec of (order, confidence, &EngramEntry) from each head.
    /// Output: Vec<f32> of scores for each token in vocabulary (length = vocab_size).
    ///
    /// Tokens not mentioned by any head get score 0.0.
    pub fn fuse(&self, head_results: &[(usize, f32, &EngramEntry)]) -> Vec<f32> {
        let mut scores = vec![0.0f32; self.vocab_size];

        if head_results.is_empty() {
            return scores;
        }

        for &(order, confidence, entry) in head_results {
            // Sigmoid gate: confidence × order_boost
            let gate = self.sigmoid_gate(order, confidence);

            // Add gated scores for each candidate
            for (i, &token_id) in entry.candidates.iter().enumerate() {
                if (token_id as usize) < self.vocab_size {
                    // Score = gate * log_probability (from entry)
                    // Higher gate = more influence from this head
                    let log_prob = entry.scores.get(i).copied().unwrap_or(-10.0);
                    // Convert log-prob to a positive score contribution
                    let score_contribution = gate * log_prob.exp();
                    scores[token_id as usize] += score_contribution;
                }
            }
        }

        // Normalize by temperature
        if self.config.temperature != 1.0 {
            let inv_temp = 1.0 / self.config.temperature;
            for s in &mut scores {
                *s *= inv_temp;
            }
        }

        scores
    }

    /// Compute sigmoid gate value for a head.
    ///
    /// gate = sigmoid(confidence + order_boost * (order - 1))
    ///
    /// Higher order + higher confidence → gate closer to 1.0
    #[inline]
    fn sigmoid_gate(&self, order: usize, confidence: f32) -> f32 {
        let x = confidence + self.config.order_boost * (order as f32 - 1.0);
        sigmoid(x)
    }

    /// Get the top-k token IDs from fused scores.
    pub fn top_k_from_scores(scores: &[f32], k: usize) -> Vec<(u16, f32)> {
        let mut indexed: Vec<(u16, f32)> = scores
            .iter()
            .enumerate()
            .filter(|(_, &s)| s > 0.0)
            .map(|(i, &s)| (i as u16, s))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }

    /// Select the best token deterministically (argmax).
    pub fn select_best(scores: &[f32]) -> Option<u16> {
        scores
            .iter()
            .enumerate()
            .filter(|(_, &s)| s > 0.0)
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u16)
    }
}

/// Fast sigmoid: 1 / (1 + exp(-x))
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x > 15.0 {
        return 1.0;
    }
    if x < -15.0 {
        return 0.0;
    }
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::EngramEntry;

    #[test]
    fn test_sigmoid_gate() {
        let fusion = SigmoidFusion::new(100);

        // High confidence + high order → gate close to 1
        let gate_high = fusion.sigmoid_gate(5, 0.9);
        assert!(gate_high > 0.8);

        // Low confidence + low order → gate close to 0.5 or below
        let gate_low = fusion.sigmoid_gate(1, 0.1);
        assert!(gate_low < gate_high);
    }

    #[test]
    fn test_fuse_single_head() {
        let fusion = SigmoidFusion::new(10);
        let entry = EngramEntry::from_counts(&[(3u16, 10u32), (5, 5), (7, 2)]);

        let results = vec![(2, 0.8, &entry)]; // bigram head, high confidence
        let scores = fusion.fuse(&results);

        // Token 3 should have highest score (most frequent)
        assert!(scores[3] > scores[5]);
        assert!(scores[5] > scores[7]);
        // Tokens not in entry should be 0
        assert_eq!(scores[0], 0.0);
    }

    #[test]
    fn test_fuse_multiple_heads() {
        let fusion = SigmoidFusion::new(10);

        // Bigram says token 3
        let entry_bi = EngramEntry::from_counts(&[(3u16, 10u32)]);
        // Trigram says token 5
        let entry_tri = EngramEntry::from_counts(&[(5u16, 10u32)]);

        let results = vec![
            (2, 0.5, &entry_bi),  // bigram, moderate confidence
            (3, 0.9, &entry_tri), // trigram, high confidence
        ];
        let scores = fusion.fuse(&results);

        // Trigram has higher order + higher confidence → token 5 should win
        assert!(scores[5] > scores[3]);
    }

    #[test]
    fn test_select_best() {
        let scores = vec![0.0, 0.3, 0.1, 0.8, 0.2];
        assert_eq!(SigmoidFusion::select_best(&scores), Some(3));
    }

    #[test]
    fn test_empty_results() {
        let fusion = SigmoidFusion::new(10);
        let scores = fusion.fuse(&[]);
        assert!(scores.iter().all(|&s| s == 0.0));
    }
}
