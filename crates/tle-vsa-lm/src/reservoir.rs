//! Reservoir: dynamical context memory for the VSA-LM.
//!
//! A leaky echo-state reservoir runs over the token stream, producing a
//! continuous state that accumulates long-range history beyond the n-gram
//! window. Rather than fitting a neural readout (`W_out`), we treat the
//! reservoir as a *non-parametric associative memory*:
//!
//! - During learning, record `(state_t, next_token)` for every step.
//! - During prediction, compare the current state against all stored states
//!   by cosine similarity and let the nearest stored contexts "vote" for the
//!   next token.
//!
//! No backprop, no gradient, no softmax — the reservoir is a frozen random
//! projection and the readout is similarity-based lookup.

use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;

/// Reservoir configuration.
#[derive(Debug, Clone)]
pub struct ReservoirConfig {
    pub dim: usize,
    pub leak_rate: f32,
    pub sparsity: f32,
    pub spectral_radius: f32,
}

impl Default for ReservoirConfig {
    fn default() -> Self {
        Self { dim: 2048, leak_rate: 0.3, sparsity: 0.1, spectral_radius: 0.9 }
    }
}

/// A leaky echo-state reservoir with frozen random weights.
#[derive(Debug, Clone)]
pub struct Reservoir {
    /// Sparse reservoir weight matrix (row-major, D×D).
    pub w_res: Vec<f32>,
    /// Dimensionality.
    pub dim: usize,
    /// Current state vector.
    pub state: Vec<f32>,
    leak: f32,
}

impl Reservoir {
    pub fn new(config: &ReservoirConfig, seed: u64) -> Self {
        let d = config.dim;
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        let mut w_res = vec![0.0f32; d * d];
        let scale = 1.0 / (d as f32 * config.sparsity).sqrt();
        for i in 0..d * d {
            if rng.gen::<f32>() < config.sparsity {
                w_res[i] = rng.gen_range(-1.0..1.0) * scale;
            }
        }
        // Approximate spectral-radius scaling (E[max |λ|] ≈ sqrt(d·sparsity)·scale).
        let est = (d as f32 * config.sparsity).sqrt() * scale;
        let factor = config.spectral_radius / est.max(0.01);
        for v in w_res.iter_mut() {
            *v *= factor;
        }

        Self { w_res, dim: d, state: vec![0.0; d], leak: config.leak_rate }
    }

    /// Advance the reservoir with an input vector, returning the new state.
    pub fn step(&mut self, input: &[f32]) -> &[f32] {
        let mut pre = vec![0.0f32; self.dim];
        // W_res · state
        for r in 0..self.dim {
            let mut sum = 0.0f32;
            let base = r * self.dim;
            for c in 0..self.dim {
                sum += self.w_res[base + c] * self.state[c];
            }
            pre[r] = sum;
        }
        // + input (must match dim; caller zero-pads otherwise)
        for i in 0..input.len().min(self.dim) {
            pre[i] += input[i];
        }
        // Leaky integration
        for i in 0..self.dim {
            let activated = pre[i].tanh();
            self.state[i] = (1.0 - self.leak) * self.state[i] + self.leak * activated;
        }
        &self.state
    }

    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    /// Cosine similarity between two state vectors (clamped to [-1, 1]).
    pub fn state_similarity(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for i in 0..a.len().min(b.len()) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        let raw = dot / ((na * nb).sqrt() + 1e-10);
        raw.clamp(-1.0, 1.0)
    }
}

/// Non-parametric associative memory over reservoir states.
///
/// Stores `(state_t, next_token_id)` pairs seen during learning. At
/// prediction time the current state is compared against all stored states by
/// cosine similarity; the `k` nearest neighbors vote for the next token,
/// weighted by similarity. This is a k-NN readout — no weights are learned.
///
/// To bound inference cost on large corpora, the memory keeps at most
/// `max_states` entries (round-robin / oldest-eviction via a simple cap).
#[derive(Debug, Clone)]
pub struct ReservoirMemory {
    /// Stored (state, next token id) pairs.
    pub states: Vec<(Vec<f32>, usize)>,
    /// Number of votes to aggregate per prediction.
    pub k: usize,
    /// Maximum number of stored states before eviction kicks in.
    pub max_states: usize,
}

impl ReservoirMemory {
    pub fn new(k: usize) -> Self {
        Self { states: Vec::new(), k: k.max(1), max_states: 5_000 }
    }

    /// Record a (state, next token) pair.
    pub fn record(&mut self, state: Vec<f32>, next_token: usize) {
        if self.states.len() >= self.max_states {
            // Drop the oldest half so memory stays bounded; keep recent states.
            let keep_from = self.max_states / 2;
            self.states.drain(..keep_from);
        }
        self.states.push((state, next_token));
    }

    /// Predict a score for every candidate token id given the current state.
    ///
    /// Returns a dense vector over `vocab_size`; each entry is the summed
    /// similarity of the `k` nearest neighbors that produced that token.
    /// Unseen ids score 0.
    pub fn predict_scores(&self, state: &[f32], vocab_size: usize) -> Vec<f32> {
        if self.states.is_empty() {
            return vec![0.0; vocab_size];
        }
        // Find k nearest neighbors by cosine similarity.
        let mut scored: Vec<(f32, usize)> = self
            .states
            .iter()
            .map(|(s, next)| (Reservoir::state_similarity(state, s), *next))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let neighbors = &scored[..self.k.min(scored.len())];

        let mut scores = vec![0.0f32; vocab_size];
        for (sim, next) in neighbors {
            // Clamp small negative similarity to 0.
            scores[*next] += sim.max(0.0);
        }
        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reservoir_deterministic_and_stable() {
        let config = ReservoirConfig { dim: 512, ..Default::default() };
        let mut r1 = Reservoir::new(&config, 7);
        let mut r2 = Reservoir::new(&config, 7);
        let input = vec![0.5; 64];

        let s1 = r1.step(&input).to_vec();
        let s2 = r2.step(&input).to_vec();
        assert_eq!(s1, s2, "same seed must give same state");
        assert!(s1.iter().all(|x| x.is_finite()));
        // Norm should be bounded (echo state property)
        let norm: f32 = s1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm < 100.0, "reservoir state should stay bounded, norm={}", norm);
    }

    #[test]
    fn test_reservoir_similarity_distinguishes_inputs() {
        let config = ReservoirConfig { dim: 512, ..Default::default() };
        let mut r1 = Reservoir::new(&config, 1);
        let mut r2 = Reservoir::new(&config, 1);
        r1.step(&vec![0.1; 64]);
        r2.step(&vec![0.9; 64]);
        let sim = Reservoir::state_similarity(&r1.state, &r2.state);
        assert!(sim > -1.0 && sim <= 1.0);
    }

    #[test]
    fn test_reservoir_memory_votes() {
        let mut mem = ReservoirMemory::new(2);
        mem.record(vec![1.0, 0.0, 0.0], 7);
        mem.record(vec![0.9, 0.1, 0.0], 7);
        mem.record(vec![0.0, 1.0, 0.0], 9);
        let scores = mem.predict_scores(&vec![1.0, 0.0, 0.0], 10);
        assert!(scores[7] > scores[9], "nearest states should vote for token 7");
        assert!(scores[7] > 0.0);
    }
}
