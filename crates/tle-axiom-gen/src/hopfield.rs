//! Modern Continuous Hopfield Networks & Hetero-Associative Attractor Memory
//! Implements Log-Sum-Exp energy, CCCP fixed-point convergence, and 1-step attractor recovery
//! based on Ramsauer et al. (ICLR 2021) and Hoover et al. (NeurIPS 2023).

use std::collections::HashMap;

/// Modern Continuous Hopfield Network.
#[derive(Debug, Clone)]
pub struct ContinuousHopfield {
    pub dim: usize,
    pub num_patterns: usize,
    pub beta: f64,
    /// Column-major storage: shape [dim, num_patterns]
    pub memory_matrix: Vec<f64>,
}

impl ContinuousHopfield {
    pub fn new(patterns: &[Vec<f64>], beta: f64) -> Self {
        assert!(!patterns.is_empty(), "Patterns cannot be empty");
        let dim = patterns[0].len();
        let num_patterns = patterns.len();
        let mut memory_matrix = Vec::with_capacity(dim * num_patterns);

        for p in patterns {
            assert_eq!(p.len(), dim, "All patterns must have matching dimension");
            let norm = (p.iter().map(|v| v * v).sum::<f64>()).sqrt();
            let inv_norm = if norm > 1e-12 { 1.0 / norm } else { 1.0 };
            for val in p {
                memory_matrix.push(val * inv_norm);
            }
        }

        Self {
            dim,
            num_patterns,
            beta,
            memory_matrix,
        }
    }

    /// Exact Continuous Hopfield Energy:
    /// E(x) = - (1 / beta) * ln( sum_{i=1}^N exp(beta * <x, mu_i>) ) + 0.5 * ||x||^2
    pub fn energy(&self, state: &[f64]) -> f64 {
        assert_eq!(state.len(), self.dim);

        let mut dots = vec![0.0; self.num_patterns];
        for j in 0..self.num_patterns {
            let offset = j * self.dim;
            let mut dot = 0.0;
            for i in 0..self.dim {
                dot += state[i] * self.memory_matrix[offset + i];
            }
            dots[j] = dot;
        }

        let max_dot = dots.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = dots.iter().map(|&d| (self.beta * (d - max_dot)).exp()).sum();
        let lse = max_dot + (sum_exp.ln() / self.beta);

        let norm_sq: f64 = state.iter().map(|v| v * v).sum();
        -lse + 0.5 * norm_sq
    }

    /// Single-step discrete CCCP / Attention update:
    /// x^{t+1} = M * softmax(beta * M^T * x^t)
    pub fn update_step(&self, state: &[f64]) -> Vec<f64> {
        assert_eq!(state.len(), self.dim);

        let mut logits = vec![0.0; self.num_patterns];
        for j in 0..self.num_patterns {
            let offset = j * self.dim;
            let mut dot = 0.0;
            for i in 0..self.dim {
                dot += state[i] * self.memory_matrix[offset + i];
            }
            logits[j] = self.beta * dot;
        }

        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut weights: Vec<f64> = logits.iter().map(|&z| (z - max_logit).exp()).collect();
        let sum_weights: f64 = weights.iter().sum();
        for w in weights.iter_mut() {
            *w /= sum_weights;
        }

        let mut new_state = vec![0.0; self.dim];
        for j in 0..self.num_patterns {
            let w = weights[j];
            let offset = j * self.dim;
            for i in 0..self.dim {
                new_state[i] += self.memory_matrix[offset + i] * w;
            }
        }

        new_state
    }
}

/// Hetero-Associative Modern Hopfield Knowledge Memory.
/// Stores (head ⊛ relation) -> tail associations with 1-step attractor retrieval.
#[derive(Debug, Clone)]
pub struct HeteroHopfieldMemory {
    pub dim: usize,
    pub beta: f64,
    pub num_entries: usize,
    pub key_memory: Vec<f64>,
    pub val_memory: Vec<f64>,
    pub entity_vectors: HashMap<String, Vec<f64>>,
}

impl HeteroHopfieldMemory {
    /// Bind two vectors via circular convolution: (a ⊛ b)_k = sum_j a_j * b_{(k-j) mod d}
    pub fn circular_convolution(a: &[f64], b: &[f64]) -> Vec<f64> {
        let n = a.len();
        let mut result = vec![0.0; n];
        for k in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                let b_idx = (k + n - j) % n;
                sum += a[j] * b[b_idx];
            }
            result[k] = sum;
        }
        let norm = result.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for v in result.iter_mut() {
                *v /= norm;
            }
        }
        result
    }

    pub fn new(dim: usize, beta: f64) -> Self {
        Self {
            dim,
            beta,
            num_entries: 0,
            key_memory: Vec::new(),
            val_memory: Vec::new(),
            entity_vectors: HashMap::new(),
        }
    }

    pub fn add_association(&mut self, key: &[f64], value: &[f64]) {
        assert_eq!(key.len(), self.dim);
        assert_eq!(value.len(), self.dim);

        let k_norm = key.iter().map(|v| v * v).sum::<f64>().sqrt();
        let v_norm = value.iter().map(|v| v * v).sum::<f64>().sqrt();

        for x in key {
            self.key_memory.push(if k_norm > 1e-12 { x / k_norm } else { *x });
        }
        for x in value {
            self.val_memory.push(if v_norm > 1e-12 { x / v_norm } else { *x });
        }
        self.num_entries += 1;
    }

    /// Single-step hetero-associative retrieval: target = V_mem * softmax(beta * K_mem^T * query)
    pub fn retrieve_step(&self, query: &[f64]) -> Vec<f64> {
        if self.num_entries == 0 {
            return vec![0.0; self.dim];
        }

        let mut logits = vec![0.0; self.num_entries];
        for j in 0..self.num_entries {
            let offset = j * self.dim;
            let mut dot = 0.0;
            for i in 0..self.dim {
                dot += query[i] * self.key_memory[offset + i];
            }
            logits[j] = self.beta * dot;
        }

        let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut weights: Vec<f64> = logits.iter().map(|&l| (l - max_l).exp()).collect();
        let sum_w: f64 = weights.iter().sum();
        for w in weights.iter_mut() {
            *w /= sum_w;
        }

        let mut retrieved = vec![0.0; self.dim];
        for j in 0..self.num_entries {
            let w = weights[j];
            let offset = j * self.dim;
            for i in 0..self.dim {
                retrieved[i] += self.val_memory[offset + i] * w;
            }
        }

        retrieved
    }
}

/// Constructs a Continuous Hopfield Network from HyperVector patterns.
pub fn build_hopfield_from_hypervectors(vectors: &[&tle_vsa::HyperVector], beta: f64) -> ContinuousHopfield {
    let patterns: Vec<Vec<f64>> = vectors
        .iter()
        .map(|hv| hv.as_slice().iter().map(|&v| v as f64).collect())
        .collect();
    ContinuousHopfield::new(&patterns, beta)
}

/// Snaps a query hypervector to its nearest memory pattern attractor via 1-step CCCP update.
pub fn snap_to_attractor(query: &tle_vsa::HyperVector, hopfield: &ContinuousHopfield) -> tle_vsa::HyperVector {
    let q_vec: Vec<f64> = query.as_slice().iter().map(|&v| v as f64).collect();
    let retrieved = hopfield.update_step(&q_vec);
    let snapped_values: Vec<f32> = retrieved.into_iter().map(|v| v as f32).collect();
    tle_vsa::HyperVector::new(snapped_values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuous_hopfield_1step_attractor_snapping() {
        let dim = 32;
        let beta = 30.0;

        let mut p1 = vec![0.0; dim];
        p1[0] = 1.0;
        let mut p2 = vec![0.0; dim];
        p2[1] = 1.0;

        let hopfield = ContinuousHopfield::new(&[p1.clone(), p2.clone()], beta);

        // Noisy query near p1
        let mut noisy = p1.clone();
        noisy[5] = 0.2;
        let norm = noisy.iter().map(|v| v * v).sum::<f64>().sqrt();
        for x in noisy.iter_mut() { *x /= norm; }

        let retrieved = hopfield.update_step(&noisy);
        let sim_p1: f64 = retrieved.iter().zip(p1.iter()).map(|(a, b)| a * b).sum();

        assert!(sim_p1 > 0.99, "1-step retrieval must snap to p1 attractor, got sim {}", sim_p1);
    }

    #[test]
    fn test_snap_to_attractor_hypervector() {
        let hv1 = tle_vsa::HyperVector::new(vec![1.0, 0.0, 0.0, 0.0]);
        let hv2 = tle_vsa::HyperVector::new(vec![0.0, 1.0, 0.0, 0.0]);

        let hopfield = build_hopfield_from_hypervectors(&[&hv1, &hv2], 25.0);

        let noisy_q = tle_vsa::HyperVector::new(vec![0.9, 0.1, 0.05, 0.0]);
        let snapped = snap_to_attractor(&noisy_q, &hopfield);

        let sim = tle_vsa::cosine_similarity(&snapped, &hv1);
        assert!(sim > 0.98, "HyperVector must snap to hv1, got cosine {}", sim);
    }
}
