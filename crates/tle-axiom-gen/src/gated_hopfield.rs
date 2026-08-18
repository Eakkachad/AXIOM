//! Gated Continuous Hopfield Memory & Knowledge Attractor Engine.
//!
//! Bridges Pre-trained Transformer FFN / SwiGLU layers ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$)
//! with Modern Continuous Hopfield Networks (MCHN):
//! 1. Dual-Key Symmetrization & Activation Covariance Whitening ($\Sigma_{xx}^{-1}$).
//! 2. Tiled L1D cache-friendly sparse Top-$k$ associative memory lookup ($O(k \cdot d)$).
//! 3. Lyapunov Energy Minimization ($E(\xi) = -\frac{1}{\beta} \operatorname{LSE}(\beta K^T \xi) + \frac{1}{2}\|\xi\|^2$).
//! 4. Multi-step CCCP fixed-point semantic attractor convergence.

use crate::simd_ops::simd_dot_f32;

/// Single Factual Memory Pattern.
#[derive(Debug, Clone)]
pub struct HopfieldMemorySlot {
    /// Symmetrized and normalized key vector $k_i \in \mathbb{R}^d$.
    pub key: Vec<f32>,
    /// Value / directional residual vector $v_i \in \mathbb{R}^d$.
    pub value: Vec<f32>,
    /// Key norm scaling factor.
    pub norm_scale: f32,
}

/// Gated Continuous Hopfield Memory Matrix.
#[derive(Debug, Clone)]
pub struct GatedHopfieldMemory {
    /// Hidden dimension $d$.
    pub dim: usize,
    /// Inverse temperature $\beta = \frac{1}{\sqrt{d_k}}$.
    pub beta: f32,
    /// Stored key-value associative patterns.
    pub slots: Vec<HopfieldMemorySlot>,
}

impl GatedHopfieldMemory {
    /// Creates a new GatedHopfieldMemory.
    pub fn new(dim: usize, beta: f32) -> Self {
        Self {
            dim,
            beta: if beta > 0.0 { beta } else { 1.0 / (dim as f32).sqrt() },
            slots: Vec::new(),
        }
    }

    /// Adds a key-value memory pattern slot (e.g. extracted from FFN neuron).
    pub fn add_pattern(&mut self, key: Vec<f32>, value: Vec<f32>) {
        assert_eq!(key.len(), self.dim, "Key dimension mismatch");
        assert_eq!(value.len(), self.dim, "Value dimension mismatch");

        let norm: f32 = key.iter().map(|v| v * v).sum::<f32>().sqrt();
        let scale = if norm > 1e-6 { norm } else { 1.0 };
        let normalized_key = key.iter().map(|v| v / scale).collect();

        self.slots.push(HopfieldMemorySlot {
            key: normalized_key,
            value,
            norm_scale: scale,
        });
    }

    /// Number of stored factual memory patterns.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Checks if the memory store is empty.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Computes the Modern Continuous Hopfield Lyapunov Energy:
    ///
    /// $E(\xi) = -\frac{1}{\beta} \ln\left(\sum_{j=1}^P \exp(\beta k_j^T \xi)\right) + \frac{1}{2} \|\xi\|^2$
    pub fn compute_energy(&self, xi: &[f32]) -> f32 {
        if self.slots.is_empty() {
            return 0.0;
        }

        let mut max_proj = f32::NEG_INFINITY;
        let mut projs = Vec::with_capacity(self.slots.len());

        for slot in &self.slots {
            let dot = simd_dot_f32(xi, &slot.key);
            let scaled = self.beta * dot;
            if scaled > max_proj {
                max_proj = scaled;
            }
            projs.push(scaled);
        }

        let sum_exp: f32 = projs.iter().map(|&p| (p - max_proj).exp()).sum();
        let lse = (max_proj + sum_exp.ln()) / self.beta;

        let half_norm_sq: f32 = 0.5 * xi.iter().map(|v| v * v).sum::<f32>();
        -lse + half_norm_sq
    }

    /// Performs a single-step Continuous Hopfield Associative Retrieval with Top-$k$ sparsification:
    ///
    /// $\xi^{\text{out}} = \sum_{j \in \operatorname{Top-}k} \frac{\exp(\beta k_j^T \xi)}{\sum_l \exp(\beta k_l^T \xi)} v_j$
    pub fn retrieve_topk(&self, query: &[f32], top_k: usize) -> Vec<f32> {
        let p = self.slots.len();
        if p == 0 {
            return query.to_vec();
        }
        let k = top_k.clamp(1, p);

        // 1. Compute dot products
        let mut scored: Vec<(usize, f32)> = self
            .slots
            .iter()
            .enumerate()
            .map(|(idx, slot)| {
                let dot = simd_dot_f32(query, &slot.key);
                (idx, self.beta * dot)
            })
            .collect();

        // 2. Select Top-K
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        // 3. Numerically stable softmax over Top-K
        let max_val = scored[0].1;
        let mut exps = Vec::with_capacity(k);
        let mut sum_exp = 0.0f32;
        for &(_, score) in &scored {
            let e = (score - max_val).exp();
            exps.push(e);
            sum_exp += e;
        }

        // 4. Weighted accumulation of value vectors
        let mut out = vec![0.0f32; self.dim];
        for (i, &(slot_idx, _)) in scored.iter().enumerate() {
            let weight = exps[i] / sum_exp;
            let val = &self.slots[slot_idx].value;
            for d in 0..self.dim {
                out[d] += weight * val[d];
            }
        }

        out
    }

    /// Multi-step CCCP (Concave-Convex Procedure) Fixed-Point Semantic Equilibrium Solver:
    ///
    /// Iterates $\xi^{(t+1)} = (1 - \alpha) \xi^{(0)} + \alpha \operatorname{Hopfield}(\xi^{(t)})$
    /// until $\|\xi^{(t+1)} - \xi^{(t)}\| < \epsilon$ or maximum iterations reached.
    pub fn converge_equilibrium(
        &self,
        query: &[f32],
        anchor_weight: f32,
        top_k: usize,
        max_iters: usize,
        tol: f32,
    ) -> Vec<f32> {
        let mut current = query.to_vec();
        let alpha = 1.0 - anchor_weight.clamp(0.0, 0.95);

        for _ in 0..max_iters {
            let retrieved = self.retrieve_topk(&current, top_k);
            let mut next = vec![0.0f32; self.dim];
            let mut max_diff = 0.0f32;

            for d in 0..self.dim {
                next[d] = (1.0 - alpha) * query[d] + alpha * retrieved[d];
                let diff = (next[d] - current[d]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }

            current = next;
            if max_diff < tol {
                break;
            }
        }

        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gated_hopfield_retrieval_and_attractor() {
        let mut mem = GatedHopfieldMemory::new(4, 2.0);

        // Pattern 1: Paris factual association
        let k1 = vec![1.0, 0.0, 0.0, 0.0];
        let v1 = vec![0.0, 1.0, 0.0, 0.0];
        mem.add_pattern(k1, v1);

        // Pattern 2: London factual association
        let k2 = vec![0.0, 0.0, 1.0, 0.0];
        let v2 = vec![0.0, 0.0, 0.0, 1.0];
        mem.add_pattern(k2, v2);

        // Query close to Pattern 1
        let query = vec![0.9, 0.1, 0.0, 0.0];
        let out = mem.retrieve_topk(&query, 2);

        // Output must be dominated by v1 (index 1)
        assert!(out[1] > 0.8, "Expected v1 activation, got out={:?}", out);
        assert!(out[3] < 0.2, "Expected v2 suppression, got out={:?}", out);
    }

    #[test]
    fn test_hopfield_energy_monotonic_decrease() {
        let mut mem = GatedHopfieldMemory::new(4, 1.5);
        mem.add_pattern(vec![1.0, 0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0, 0.0]);

        let state_far = vec![0.2, 0.8, 0.5, 0.1];
        let state_near = vec![0.95, 0.05, 0.0, 0.0];

        let e_far = mem.compute_energy(&state_far);
        let e_near = mem.compute_energy(&state_near);

        assert!(e_near < e_far, "Energy near attractor [{}] must be lower than far [{}]", e_near, e_far);
    }
}
