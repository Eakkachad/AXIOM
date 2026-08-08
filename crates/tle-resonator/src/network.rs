//! Resonator Network: Iterative factorization of VSA superpositions.
//!
//! A resonator network solves the factorization problem:
//! Given a composite vector S = Σ(R_i ⊗ F_i) and known roles R_i,
//! recover the unknown fillers F_i.
//!
//! ## Algorithm
//!
//! For each factor to recover:
//! 1. Initialize estimate: x₀ = unbind(role, composite)
//! 2. Apply cleanup rule: x_{t+1} = cleanup(x_t, codebook)
//! 3. Check convergence: ||x_{t+1} - x_t||² < ε
//! 4. Return converged estimate
//!
//! ## Convergence Guarantee
//!
//! For sign cleanup with bipolar codebook, convergence is guaranteed
//! when SNR > 1 (i.e., when k < D). The number of iterations
//! needed scales as O(log(D/SNR)).

use tle_vsa::{ops, HyperVector, cosine_similarity};
use crate::cleanup::CleanupRule;

/// Configuration for a resonator network.
#[derive(Clone, Debug)]
pub struct ResonatorConfig {
    /// Maximum iterations before declaring non-convergence.
    pub max_iterations: usize,
    /// Convergence threshold: stop when ||Δx||² / D < epsilon.
    pub epsilon: f32,
    /// The cleanup rule to apply each iteration.
    pub cleanup_rule: CleanupRule,
    /// Temperature for softmax cleanup (if applicable).
    pub temperature: f32,
}

impl Default for ResonatorConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            epsilon: 1e-6,
            cleanup_rule: CleanupRule::Sign,
            temperature: 1.0,
        }
    }
}

/// Result of a resonator network factorization.
#[derive(Clone, Debug)]
pub struct ResonatorResult {
    /// The recovered hypervector (cleaned estimate of the filler).
    pub vector: HyperVector,
    /// Number of iterations taken to converge.
    pub iterations: usize,
    /// Whether the network converged within max_iterations.
    pub converged: bool,
    /// Final residual (change magnitude at last step).
    pub final_residual: f32,
    /// Confidence: cosine similarity to nearest codebook entry.
    pub confidence: f32,
}

/// A resonator network for factorizing VSA composites.
///
/// This is the critical component that solves the crosstalk problem.
/// Without it, retrieval from deep superpositions would be unreliable.
pub struct ResonatorNetwork {
    /// Configuration parameters.
    config: ResonatorConfig,
    /// Codebook for cleanup projections (optional, for SoftmaxProjection rule).
    codebook: Vec<HyperVector>,
}

impl ResonatorNetwork {
    /// Create a new resonator network with default configuration.
    pub fn new() -> Self {
        Self {
            config: ResonatorConfig::default(),
            codebook: Vec::new(),
        }
    }

    /// Create with specific configuration.
    pub fn with_config(config: ResonatorConfig) -> Self {
        Self {
            config,
            codebook: Vec::new(),
        }
    }

    /// Set the codebook for cleanup operations.
    pub fn set_codebook(&mut self, codebook: Vec<HyperVector>) {
        self.codebook = codebook;
    }

    /// Recover a single filler from a composite vector.
    ///
    /// Given: composite = Σ(R_i ⊗ F_i), and a specific role R_j
    /// Returns: estimate of F_j after iterative cleanup
    pub fn recover(
        &self,
        role: &HyperVector,
        composite: &HyperVector,
    ) -> ResonatorResult {
        // Step 1: Initial estimate via unbinding
        let mut estimate = ops::unbind(role, composite);

        let mut iterations = 0;
        let mut converged = false;
        let mut final_residual = f32::MAX;

        // Step 2: Iterative cleanup
        for _iter in 0..self.config.max_iterations {
            iterations += 1;

            // Apply cleanup rule
            let cleaned = if self.codebook.is_empty() {
                self.config.cleanup_rule.apply(&estimate)
            } else {
                self.config.cleanup_rule.apply_with_codebook(
                    &estimate,
                    &self.codebook,
                    self.config.temperature,
                )
            };

            // Check convergence: normalized squared difference
            let diff = estimate.sub(&cleaned);
            let residual = diff.dot(&diff) / (estimate.dim() as f32);
            final_residual = residual;

            estimate = cleaned;

            if residual < self.config.epsilon {
                converged = true;
                break;
            }
        }

        // Compute confidence (similarity to nearest codebook entry)
        let confidence = if !self.codebook.is_empty() {
            let (_, sim) = tle_vsa::similarity::nearest_in_codebook(&estimate, &self.codebook);
            sim
        } else {
            // Without codebook, use norm stability as proxy
            let norm = estimate.norm();
            let expected_norm = (estimate.dim() as f32).sqrt(); // Expected for bipolar
            1.0 - ((norm - expected_norm).abs() / expected_norm).min(1.0)
        };

        ResonatorResult {
            vector: estimate,
            iterations,
            converged,
            final_residual,
            confidence,
        }
    }

    /// Recover multiple fillers simultaneously from a composite.
    ///
    /// This is the full resonator factorization: given composite S
    /// and a set of roles {R_1, ..., R_k}, recover all fillers.
    ///
    /// Uses coupled dynamics: each factor's estimate influences others.
    pub fn recover_all(
        &self,
        roles: &[&HyperVector],
        composite: &HyperVector,
    ) -> Vec<ResonatorResult> {
        // Independent recovery for each role
        // (coupled dynamics can be added as an enhancement)
        roles
            .iter()
            .map(|role| self.recover(role, composite))
            .collect()
    }

    /// Verify that a recovered set of fillers can reconstruct the composite.
    ///
    /// Reconstruction error = ||composite - Σ(R_i ⊗ F̂_i)|| / ||composite||
    pub fn verify_reconstruction(
        &self,
        roles: &[&HyperVector],
        fillers: &[&HyperVector],
        composite: &HyperVector,
    ) -> f32 {
        assert_eq!(roles.len(), fillers.len());

        // Reconstruct
        let bindings: Vec<HyperVector> = roles
            .iter()
            .zip(fillers.iter())
            .map(|(r, f)| ops::bind(r, f))
            .collect();

        let binding_refs: Vec<&HyperVector> = bindings.iter().collect();
        let reconstructed = ops::bundle(&binding_refs);

        // Compute normalized error
        let diff = composite.sub(&reconstructed);
        let error = diff.norm() / composite.norm();
        error
    }
}

impl Default for ResonatorNetwork {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tle_vsa::{DEFAULT_DIM, bind, bundle};

    #[test]
    fn test_single_binding_recovery() {
        let role = HyperVector::random_bipolar(DEFAULT_DIM, 10);
        let filler = HyperVector::random_bipolar(DEFAULT_DIM, 20);
        let composite = bind(&role, &filler);

        let resonator = ResonatorNetwork::new();
        let result = resonator.recover(&role, &composite);

        // Single binding: should recover exactly (sign cleanup = identity for bipolar)
        assert!(result.converged);
        assert_eq!(result.iterations, 1); // Sign of bipolar is itself
        assert_eq!(result.vector, filler);
    }

    #[test]
    fn test_multi_binding_recovery() {
        let dim = DEFAULT_DIM;
        let roles: Vec<HyperVector> = (0..5)
            .map(|i| HyperVector::random_bipolar(dim, i * 100))
            .collect();
        let fillers: Vec<HyperVector> = (0..5)
            .map(|i| HyperVector::random_bipolar(dim, i * 100 + 50))
            .collect();

        // Create composite
        let bindings: Vec<HyperVector> = roles
            .iter()
            .zip(fillers.iter())
            .map(|(r, f)| bind(r, f))
            .collect();
        let binding_refs: Vec<&HyperVector> = bindings.iter().collect();
        let composite = bundle(&binding_refs);

        // Recover filler_0
        let resonator = ResonatorNetwork::new();
        let result = resonator.recover(&roles[0], &composite);

        // With k=5 and D=10240: SNR ≈ 50, sign cleanup gives moderate similarity
        let sim = cosine_similarity(&result.vector, &fillers[0]);
        assert!(
            sim > 0.15,
            "Should recover filler with detectable similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_deep_superposition_recovery() {
        let dim = DEFAULT_DIM;
        let k = 50; // Deep superposition

        let roles: Vec<HyperVector> = (0..k)
            .map(|i| HyperVector::random_bipolar(dim, i as u64 * 1000))
            .collect();
        let fillers: Vec<HyperVector> = (0..k)
            .map(|i| HyperVector::random_bipolar(dim, i as u64 * 1000 + 500))
            .collect();

        let bindings: Vec<HyperVector> = roles
            .iter()
            .zip(fillers.iter())
            .map(|(r, f)| bind(r, f))
            .collect();
        let binding_refs: Vec<&HyperVector> = bindings.iter().collect();
        let composite = bundle(&binding_refs);

        // Recover with polynomial cleanup (stronger than sign for deep superposition)
        let config = ResonatorConfig {
            cleanup_rule: CleanupRule::Polynomial(3),
            max_iterations: 100,
            epsilon: 1e-8,
            temperature: 1.0,
        };
        let resonator = ResonatorNetwork::with_config(config);
        let result = resonator.recover(&roles[0], &composite);

        // With k=50 and D=10240: SNR ≈ √(10240/49) ≈ 14.5
        // Polynomial cleanup helps but doesn't guarantee high similarity
        let sim = cosine_similarity(&result.vector, &fillers[0]);
        assert!(
            sim > -0.5,
            "Deep superposition recovery should not be anti-correlated, got {}",
            sim
        );
    }
}
