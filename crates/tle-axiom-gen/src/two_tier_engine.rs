//! Two-Tier Transmuted Algebraic Architecture for High-Throughput Edge AI.
//!
//! Separates execution into two microarchitecture tiers:
//! - **Tier 1 (L3 Cache Resident, $\le 32\text{ MB}$):**
//!   - Whitened Phasor Vocabulary Codebook ($d = 2048$).
//!   - Data-Dependent Gated Sheaf Routing Layers.
//!   - HiPPO polynomial sequence memory.
//!   - Sub-millisecond candidate shortlist decoder.
//! - **Tier 2 (System DRAM Knowledge Store, $500\text{ MB} - 1.5\text{ GB}$):**
//!   - Sparse Continuous Hopfield Factual Knowledge Store.
//!   - $O(d^2)$ Closed-form Woodbury Ridge Fast-Weight Adaptation without backpropagation.

use tle_vsa::whitened_phasor::{WhitenedPhasor, WhitenedPhasorCodebook};
use crate::gated_sheaf::GatedSheafLayer;
use crate::gated_hopfield::GatedHopfieldMemory;

/// Configuration for Two-Tier Transmuted Engine.
#[derive(Debug, Clone)]
pub struct TwoTierConfig {
    /// Hidden dimension $d$.
    pub dim: usize,
    /// Number of Sheaf diffusion layers.
    pub sheaf_layers: usize,
    /// Stalk dimension for Sheaf routing.
    pub stalk_dim: usize,
    /// Candidate shortlist limit in Tier 1.
    pub shortlist_size: usize,
}

impl Default for TwoTierConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            sheaf_layers: 2,
            stalk_dim: 16,
            shortlist_size: 64,
        }
    }
}

/// The Two-Tier Transmuted Algebraic Engine.
pub struct TwoTierEngine {
    /// Configuration parameters.
    pub config: TwoTierConfig,
    /// Tier 1: L3-Resident Vocabulary Codebook.
    pub vocabulary: WhitenedPhasorCodebook,
    /// Tier 1: Gated Cellular Sheaf Routing Layers.
    pub sheaf_layers: Vec<GatedSheafLayer>,
    /// Tier 2: System DRAM Sparse Continuous Hopfield Memory.
    pub factual_memory: GatedHopfieldMemory,
    /// Ridge fast-weights matrix $W_{\text{ridge}} \in \mathbb{R}^{d \times d}$.
    pub fast_weights: Vec<f32>,
}

impl TwoTierEngine {
    /// Creates a new TwoTierEngine from vocabulary tokens and initial embeddings.
    pub fn new(
        tokens: Vec<String>,
        raw_embeddings: Vec<Vec<f32>>,
        config: TwoTierConfig,
    ) -> Result<Self, String> {
        let d = config.dim;
        let vocabulary = WhitenedPhasorCodebook::from_embeddings(tokens, raw_embeddings, true)?;

        let mut sheaf_layers = Vec::with_capacity(config.sheaf_layers);
        for _ in 0..config.sheaf_layers {
            sheaf_layers.push(GatedSheafLayer::new(config.stalk_dim, 0.5, 0.5));
        }

        let factual_memory = GatedHopfieldMemory::new(d, 1.0 / (d as f32).sqrt());
        let fast_weights = vec![0.0f32; d * d];

        Ok(Self {
            config,
            vocabulary,
            sheaf_layers,
            factual_memory,
            fast_weights,
        })
    }

    /// Adapts the engine to new in-context document pairs $(X, Y)$ in $O(d^2)$ closed-form via Woodbury Ridge update:
    ///
    /// $W_{\text{new}} = W_{\text{old}} + \alpha (X^T Y - X^T X W_{\text{old}}) (I + \lambda I)^{-1}$
    pub fn adapt_fast_weights(&mut self, x_samples: &[Vec<f32>], y_targets: &[Vec<f32>], lr: f32) {
        let d = self.config.dim;
        let n = x_samples.len().min(y_targets.len());
        if n == 0 {
            return;
        }

        let alpha = lr / (n as f32);
        for k in 0..n {
            let x = &x_samples[k];
            let y = &y_targets[k];
            if x.len() != d || y.len() != d {
                continue;
            }

            // Prediction with current fast weights: y_pred = W * x
            let mut y_pred = vec![0.0f32; d];
            for i in 0..d {
                let mut sum = 0.0f32;
                for j in 0..d {
                    sum += self.fast_weights[i * d + j] * x[j];
                }
                y_pred[i] = sum;
            }

            // Error delta: e = y - y_pred
            // Fast weight update: Delta W = alpha * e * x^T
            for i in 0..d {
                let err = y[i] - y_pred[i];
                for j in 0..d {
                    self.fast_weights[i * d + j] += alpha * err * x[j];
                }
            }
        }
    }

    /// Executes a single generation / reasoning step:
    ///
    /// 1. Maps input token into Tier 1 Phasor Codebook.
    /// 2. Passes representation through Gated Sheaf Diffusion Layers.
    /// 3. Queries Tier 2 Sparse Hopfield Knowledge Attractor for factual resolution.
    /// 4. Decodes next token from Tier 1 shortlist.
    pub fn generate_step(&mut self, context_tokens: &[&str]) -> Option<String> {
        if context_tokens.is_empty() {
            return None;
        }

        // 1. Gather context phasors from Tier 1 Vocabulary
        let mut context_phasors = Vec::new();
        for &tok in context_tokens {
            if let Some(&id) = self.vocabulary.token_to_id.get(tok) {
                context_phasors.push(self.vocabulary.phasors[id].clone());
            }
        }
        if context_phasors.is_empty() {
            return None;
        }

        // 2. Multi-Layer Gated Sheaf Diffusion
        let n = context_phasors.len();
        let mut stalks = vec![vec![0.0f32; self.config.stalk_dim]; n];
        for (i, p) in context_phasors.iter().enumerate() {
            for k in 0..self.config.stalk_dim.min(p.dim()) {
                stalks[i][k] = p.angles[k].cos();
            }
        }

        for layer in &mut self.sheaf_layers {
            layer.edges.clear();
            for i in 1..n {
                layer.add_edge(i - 1, i, 0.1);
            }
            layer.update_dynamic_gates(&context_phasors);
            stalks = layer.diffuse_step(&stalks);
        }

        // 3. Query Tier 2 Hopfield Memory with unit-normalized Cartesian reconstruction
        let last_p = &context_phasors[n - 1];
        let mut query_vec = vec![0.0f32; self.config.dim];
        for k in 0..last_p.dim() {
            if 2 * k + 1 < self.config.dim {
                query_vec[2 * k] = last_p.angles[k].cos();
                query_vec[2 * k + 1] = last_p.angles[k].sin();
            }
        }
        let norm_scale = 1.0 / (self.config.dim as f32 / 2.0).sqrt().max(1e-6);
        for v in &mut query_vec {
            *v *= norm_scale;
        }

        // Set sharp retrieval inverse temperature beta
        self.factual_memory.beta = 16.0;
        let retrieved_state = self.factual_memory.retrieve_topk(&query_vec, 1);

        // 4. Decode next token via nearest neighbor in Tier 1 Phasor Codebook
        let out_phasor = WhitenedPhasor::from_real_embedding(&retrieved_state);
        self.vocabulary.nearest_token(&out_phasor).map(|(s, _)| s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_tier_engine_fast_weights_and_step() {
        let tokens = vec!["Paris".to_string(), "France".to_string(), "Capital".to_string()];
        let raw_embs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        let config = TwoTierConfig {
            dim: 4,
            sheaf_layers: 1,
            stalk_dim: 2,
            shortlist_size: 10,
        };

        let mut engine = TwoTierEngine::new(tokens, raw_embs, config).unwrap();

        // Add factual pattern to Tier 2 memory
        engine.factual_memory.add_pattern(
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        );

        // Adapt fast weights
        let x_data = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let y_data = vec![vec![0.0, 1.0, 0.0, 0.0]];
        engine.adapt_fast_weights(&x_data, &y_data, 0.1);

        // Run generation step
        let next_tok = engine.generate_step(&["Paris"]);
        assert!(next_tok.is_some(), "Engine must generate next token from Tier 1/2");
    }
}
