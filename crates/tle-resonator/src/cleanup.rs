//! Non-linear cleanup rules for resonator networks.
//!
//! These rules project noisy vectors toward the nearest valid codebook entry.
//! Each rule represents a different trade-off between convergence speed and accuracy.
//!
//! ## Available Rules (from literature: Kent et al., 2024)
//!
//! 1. **Sign**: Hardest cleanup. Maps each dimension to ±1.
//!    - Fastest convergence, but can get stuck in local minima.
//! 2. **Threshold**: Soft sign with dead zone.
//!    - Good balance of speed and accuracy.
//! 3. **Softmax Projection**: Projects onto codebook similarity simplex.
//!    - Best accuracy for small codebooks, O(N·D) per iteration.
//! 4. **Polynomial**: x^p normalization (p=3 or p=5).
//!    - Continuous approximation to sign, good for gradient-free optimization.

use tle_vsa::HyperVector;

/// Enumeration of available cleanup rules.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CleanupRule {
    /// Hard sign function: x_i → sgn(x_i)
    Sign,
    /// Threshold with dead zone: x_i → sgn(x_i) if |x_i| > τ, else 0
    Threshold(f32),
    /// Polynomial sharpening: x_i → x_i^p / ||x^p||
    Polynomial(u32),
    /// Softmax projection onto codebook
    SoftmaxProjection,
}

impl CleanupRule {
    /// Apply the cleanup rule to a hypervector.
    ///
    /// For codebook-based rules (SoftmaxProjection), the codebook
    /// must be provided separately via `apply_with_codebook`.
    pub fn apply(&self, v: &HyperVector) -> HyperVector {
        match self {
            CleanupRule::Sign => v.sign(),

            CleanupRule::Threshold(tau) => {
                let tau = *tau;
                HyperVector::new(
                    v.data
                        .iter()
                        .map(|&x| {
                            if x > tau {
                                1.0
                            } else if x < -tau {
                                -1.0
                            } else {
                                0.0
                            }
                        })
                        .collect(),
                )
            }

            CleanupRule::Polynomial(p) => {
                let p = *p;
                let powered: Vec<f32> = v.data.iter().map(|&x| x.powi(p as i32)).collect();
                let norm: f32 = powered.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm < 1e-10 {
                    return HyperVector::zeros(v.dim());
                }
                let inv_norm = 1.0 / norm;
                HyperVector::new(powered.iter().map(|&x| x * inv_norm).collect())
            }

            CleanupRule::SoftmaxProjection => {
                // Without codebook, fall back to sign
                v.sign()
            }
        }
    }

    /// Apply codebook-based softmax projection cleanup.
    ///
    /// Computes: x' = C^T · softmax(C · x / temperature)
    ///
    /// Where C is the codebook matrix (each row is a codebook vector),
    /// and softmax selects the most similar codebook entries.
    pub fn apply_with_codebook(
        &self,
        v: &HyperVector,
        codebook: &[HyperVector],
        temperature: f32,
    ) -> HyperVector {
        match self {
            CleanupRule::SoftmaxProjection => {
                if codebook.is_empty() {
                    return v.clone();
                }

                // Compute similarities: s_i = v · c_i
                let similarities: Vec<f32> = codebook
                    .iter()
                    .map(|c| v.dot(c) / temperature)
                    .collect();

                // Softmax normalization
                let max_sim = similarities.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sims: Vec<f32> = similarities.iter().map(|&s| (s - max_sim).exp()).collect();
                let sum_exp: f32 = exp_sims.iter().sum();

                if sum_exp < 1e-10 {
                    return v.clone();
                }

                let weights: Vec<f32> = exp_sims.iter().map(|&e| e / sum_exp).collect();

                // Reconstruct: x' = Σ w_i · c_i
                let dim = v.dim();
                let mut result = vec![0.0f32; dim];
                for (w, c) in weights.iter().zip(codebook.iter()) {
                    if *w > 1e-8 {
                        for j in 0..dim {
                            result[j] += w * c.data[j];
                        }
                    }
                }

                HyperVector::new(result)
            }
            _ => self.apply(v),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tle_vsa::DEFAULT_DIM;

    #[test]
    fn test_sign_cleanup() {
        let v = HyperVector::new(vec![0.3, -0.7, 0.1, -0.9, 0.5]);
        let cleaned = CleanupRule::Sign.apply(&v);
        assert_eq!(cleaned.data, vec![1.0, -1.0, 1.0, -1.0, 1.0]);
    }

    #[test]
    fn test_threshold_cleanup() {
        let v = HyperVector::new(vec![0.3, -0.7, 0.1, -0.9, 0.05]);
        let cleaned = CleanupRule::Threshold(0.2).apply(&v);
        assert_eq!(cleaned.data, vec![1.0, -1.0, 0.0, -1.0, 0.0]);
    }

    #[test]
    fn test_polynomial_cleanup() {
        let v = HyperVector::new(vec![0.5, -0.8, 0.2, -0.9, 0.1]);
        let cleaned = CleanupRule::Polynomial(3).apply(&v);
        // Cubing amplifies large values, suppresses small ones
        assert!(cleaned.data[1].abs() > cleaned.data[0].abs());
        assert!(cleaned.data[3].abs() > cleaned.data[2].abs());
    }

    #[test]
    fn test_softmax_projection_exact_match() {
        let dim = 100; // Small for testing
        let target = HyperVector::random_bipolar(dim, 42);
        let noise = HyperVector::random_bipolar(dim, 99);

        // Create codebook with target
        let codebook = vec![
            HyperVector::random_bipolar(dim, 1),
            HyperVector::random_bipolar(dim, 2),
            target.clone(),
            HyperVector::random_bipolar(dim, 4),
        ];

        // Noisy version of target (add some noise)
        let noisy = target.add(&noise.scale(0.3));

        let cleaned = CleanupRule::SoftmaxProjection.apply_with_codebook(&noisy, &codebook, 1.0);

        // Should be most similar to target
        let sim_target = tle_vsa::cosine_similarity(&cleaned, &target);
        let sim_other = tle_vsa::cosine_similarity(&cleaned, &codebook[0]);
        assert!(sim_target > sim_other);
    }
}
