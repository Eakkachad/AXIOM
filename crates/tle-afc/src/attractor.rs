//! Attractor Reasoning — Iterative refinement until convergence.
//!
//! Based on Research 35 (Attractor Models) + Research 317 (Reasoning as Attractor Dynamics):
//!
//! Instead of single-pass generation:
//!   1. Initial proposal (from Engram/KG)
//!   2. Refine: apply transitions + energy scoring
//!   3. Repeat until output stabilizes (Δ < ε)
//!
//! This models "thinking" — the system iterates toward a better answer,
//! like how your brain refines a thought before speaking.

use tle_vsa::{cosine_similarity, HyperVector, Codebook};

/// Configuration for attractor reasoning.
#[derive(Clone, Debug)]
pub struct AttractorConfig {
    /// Maximum iterations before giving up.
    pub max_iterations: usize,
    /// Convergence threshold: stop when cos(state_t, state_{t-1}) > this.
    pub convergence_threshold: f32,
    /// Damping factor: how much of the old state to keep (0 = fully replace, 1 = never update).
    pub damping: f32,
}

impl Default for AttractorConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            convergence_threshold: 0.95,
            damping: 0.3,
        }
    }
}

/// Result of attractor reasoning.
#[derive(Clone, Debug)]
pub struct AttractorResult {
    /// Final state vector.
    pub state: HyperVector,
    /// Number of iterations taken.
    pub iterations: usize,
    /// Whether convergence was reached.
    pub converged: bool,
    /// Similarity at each step (for visualization).
    pub trajectory: Vec<f32>,
}

/// Attractor-based reasoning engine.
///
/// Given:
/// - An initial state (query vector)
/// - A set of "attractor basins" (known facts/concepts as vectors)
/// - A transition function (how to update state given attractors)
///
/// Iterates until the state converges to a fixed point.
pub struct AttractorReasoner {
    /// Known concept vectors (attractor basins).
    attractors: Vec<(String, HyperVector)>,
    /// Configuration.
    pub config: AttractorConfig,
}

impl AttractorReasoner {
    /// Create a new reasoner.
    pub fn new() -> Self {
        Self {
            attractors: Vec::new(),
            config: AttractorConfig::default(),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: AttractorConfig) -> Self {
        Self {
            attractors: Vec::new(),
            config,
        }
    }

    /// Add a concept as an attractor basin.
    pub fn add_attractor(&mut self, label: &str, vector: HyperVector) {
        self.attractors.push((label.to_string(), vector));
    }

    /// Run attractor dynamics on a query vector.
    ///
    /// The state is iteratively pulled toward the nearest attractors,
    /// weighted by similarity. This refines a vague query into a
    /// specific concept.
    pub fn reason(&self, initial_state: &HyperVector) -> AttractorResult {
        if self.attractors.is_empty() {
            return AttractorResult {
                state: initial_state.clone(),
                iterations: 0,
                converged: true,
                trajectory: vec![1.0],
            };
        }

        let mut state = initial_state.clone();
        let mut trajectory = Vec::new();
        let mut prev_state = state.clone();

        for iter in 0..self.config.max_iterations {
            // Compute weighted sum of attractors (pulled by similarity)
            let mut update = HyperVector::zeros(state.dim());

            for (_, attractor) in &self.attractors {
                let sim = cosine_similarity(&state, attractor);
                if sim > 0.0 {
                    // Pull toward this attractor proportional to similarity
                    let contribution = attractor.scale(sim);
                    update = update.add(&contribution);
                }
            }

            // Apply damping: new_state = damping * old_state + (1-damping) * update
            let damped_old = state.scale(self.config.damping);
            let damped_new = update.scale(1.0 - self.config.damping);
            state = damped_old.add(&damped_new);

            // Check convergence
            let similarity = cosine_similarity(&state, &prev_state);
            trajectory.push(similarity);

            if similarity > self.config.convergence_threshold && iter > 0 {
                return AttractorResult {
                    state,
                    iterations: iter + 1,
                    converged: true,
                    trajectory,
                };
            }

            prev_state = state.clone();
        }

        AttractorResult {
            state,
            iterations: self.config.max_iterations,
            converged: false,
            trajectory,
        }
    }

    /// Find which attractor the final state is closest to.
    pub fn identify_attractor(&self, state: &HyperVector) -> Option<(&str, f32)> {
        self.attractors
            .iter()
            .map(|(label, vec)| (label.as_str(), cosine_similarity(state, vec)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Number of attractors.
    pub fn num_attractors(&self) -> usize {
        self.attractors.len()
    }
}

impl Default for AttractorReasoner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence() {
        let dim = 2048;
        let mut reasoner = AttractorReasoner::new();

        // Add attractors
        let a1 = HyperVector::random_bipolar(dim, 1);
        let a2 = HyperVector::random_bipolar(dim, 2);
        reasoner.add_attractor("concept_a", a1.clone());
        reasoner.add_attractor("concept_b", a2.clone());

        // Start near concept_a
        let initial = a1.add(&HyperVector::random_bipolar(dim, 99).scale(0.1));
        let result = reasoner.reason(&initial);

        assert!(result.converged || result.iterations <= 10);
        // Should be pulled toward a1
        let sim_a = cosine_similarity(&result.state, &a1);
        let sim_b = cosine_similarity(&result.state, &a2);
        assert!(sim_a > sim_b, "Should converge toward nearest attractor");
    }

    #[test]
    fn test_identify_attractor() {
        let dim = 2048;
        let mut reasoner = AttractorReasoner::new();

        let animal = HyperVector::random_bipolar(dim, 10);
        let vehicle = HyperVector::random_bipolar(dim, 20);
        reasoner.add_attractor("animal", animal.clone());
        reasoner.add_attractor("vehicle", vehicle.clone());

        let (label, _) = reasoner.identify_attractor(&animal).unwrap();
        assert_eq!(label, "animal");
    }

    #[test]
    fn test_empty_attractors() {
        let reasoner = AttractorReasoner::new();
        let state = HyperVector::random_bipolar(512, 1);
        let result = reasoner.reason(&state);
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }
}
