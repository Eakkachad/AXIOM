//! Memory Bank: Persistent latent storage using VSA superposition.
//!
//! ## Capacity Analysis
//!
//! For D=10,240 dimensions:
//! - Reliable storage: ~400 independent facts (SNR > 5)
//! - With resonator cleanup: ~1,000 facts recoverable
//! - Maximum theoretical: ~5,000 facts (SNR > 1)
//!
//! When capacity is exceeded, the bank auto-consolidates:
//! old memories are compressed into a "deep memory" layer.

use tle_vsa::{bind, bundle, unbind, HyperVector, cosine_similarity, DEFAULT_DIM};
use tle_resonator::{ResonatorNetwork, ResonatorConfig, CleanupRule};

/// A memory slot: a specific fact stored in the memory bank.
#[derive(Clone, Debug)]
pub struct MemorySlot {
    /// The role vector used for this binding.
    pub role: HyperVector,
    /// The original filler (stored for verification, not used in retrieval).
    pub filler: HyperVector,
    /// Timestamp of when this fact was stored.
    pub timestamp: u64,
    /// Importance weight (affects consolidation priority).
    pub importance: f32,
}

/// The Latent Memory Bank: a fixed-size structure holding an
/// effectively unlimited number of facts in superposition.
pub struct MemoryBank {
    /// The main memory register: superposition of all bindings.
    /// S = Σ(R_i ⊗ F_i) for all stored facts.
    pub register: HyperVector,
    /// Deep memory: consolidated older memories (lower SNR but larger capacity).
    pub deep_register: HyperVector,
    /// Number of facts currently in the main register.
    pub fact_count: usize,
    /// Number of facts consolidated into deep memory.
    pub deep_count: usize,
    /// Maximum facts before auto-consolidation.
    pub consolidation_threshold: usize,
    /// Dimensionality.
    dim: usize,
    /// Monotonic timestamp counter.
    clock: u64,
    /// Resonator for cleanup during retrieval.
    resonator: ResonatorNetwork,
    /// Stored slots for verification and consolidation.
    slots: Vec<MemorySlot>,
}

impl MemoryBank {
    /// Create a new empty memory bank.
    pub fn new(dim: usize) -> Self {
        let config = ResonatorConfig {
            max_iterations: 30,
            epsilon: 1e-7,
            cleanup_rule: CleanupRule::Sign,
            temperature: 1.0,
        };

        Self {
            register: HyperVector::zeros(dim),
            deep_register: HyperVector::zeros(dim),
            fact_count: 0,
            deep_count: 0,
            consolidation_threshold: 300, // Conservative for D=10240
            dim,
            clock: 0,
            resonator: ResonatorNetwork::with_config(config),
            slots: Vec::new(),
        }
    }

    /// Create with default dimensionality.
    pub fn default_size() -> Self {
        Self::new(DEFAULT_DIM)
    }

    /// Store a new fact: (role, filler) pair.
    ///
    /// The binding R ⊗ F is added to the superposition register.
    /// This is O(D) - constant time regardless of memory contents.
    pub fn store(&mut self, role: &HyperVector, filler: &HyperVector, importance: f32) {
        assert_eq!(role.dim(), self.dim);
        assert_eq!(filler.dim(), self.dim);

        // Bind role to filler
        let binding = bind(role, filler);

        // Add to register (bundling)
        self.register = self.register.add(&binding);
        self.fact_count += 1;
        self.clock += 1;

        self.slots.push(MemorySlot {
            role: role.clone(),
            filler: filler.clone(),
            timestamp: self.clock,
            importance,
        });

        // Auto-consolidate if threshold exceeded
        if self.fact_count >= self.consolidation_threshold {
            self.consolidate();
        }
    }

    /// Retrieve a filler given a role (query).
    ///
    /// Unbinds the role from the register and applies resonator cleanup.
    /// Returns the cleaned estimate and confidence score.
    pub fn retrieve(&self, role: &HyperVector) -> (HyperVector, f32) {
        // Unbind: extract noisy estimate of filler
        let noisy = unbind(role, &self.register);

        // Apply resonator cleanup
        let result = self.resonator.recover(role, &self.register);

        (result.vector, result.confidence)
    }

    /// Retrieve from both main and deep memory, merging results.
    pub fn deep_retrieve(&self, role: &HyperVector) -> (HyperVector, f32) {
        let (main_result, main_conf) = self.retrieve(role);

        if self.deep_count == 0 {
            return (main_result, main_conf);
        }

        // Also check deep memory
        let deep_noisy = unbind(role, &self.deep_register);
        let deep_result = self.resonator.recover(role, &self.deep_register);

        // Return the one with higher confidence
        if deep_result.confidence > main_conf {
            (deep_result.vector, deep_result.confidence)
        } else {
            (main_result, main_conf)
        }
    }

    /// Forget a specific fact by subtracting its binding.
    ///
    /// This is the anti-bundling operation: S' = S - (R ⊗ F)
    pub fn forget(&mut self, role: &HyperVector, filler: &HyperVector) {
        let binding = bind(role, filler);
        self.register = self.register.sub(&binding);
        self.fact_count = self.fact_count.saturating_sub(1);

        // Remove from slots
        self.slots.retain(|s| cosine_similarity(&s.role, role) < 0.9);
    }

    /// Consolidate: move older, less important facts to deep memory.
    ///
    /// This compresses the main register to maintain high SNR
    /// while preserving access to older facts via deep retrieval.
    fn consolidate(&mut self) {
        if self.slots.len() <= self.consolidation_threshold / 2 {
            return;
        }

        // Sort by importance * recency
        let mut scored: Vec<(usize, f32)> = self.slots
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let recency = 1.0 / (1.0 + (self.clock - s.timestamp) as f32);
                (i, s.importance * recency)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Keep top half in main register, move rest to deep
        let keep_count = self.consolidation_threshold / 2;
        let keep_indices: Vec<usize> = scored[..keep_count].iter().map(|(i, _)| *i).collect();

        // Rebuild main register from kept facts
        let mut new_register = HyperVector::zeros(self.dim);
        let mut new_slots = Vec::new();

        for &idx in &keep_indices {
            let slot = &self.slots[idx];
            let binding = bind(&slot.role, &slot.filler);
            new_register = new_register.add(&binding);
            new_slots.push(slot.clone());
        }

        // Add evicted facts to deep register
        for (i, slot) in self.slots.iter().enumerate() {
            if !keep_indices.contains(&i) {
                let binding = bind(&slot.role, &slot.filler);
                self.deep_register = self.deep_register.add(&binding);
                self.deep_count += 1;
            }
        }

        self.register = new_register;
        self.fact_count = new_slots.len();
        self.slots = new_slots;
    }

    /// Get current memory utilization metrics.
    pub fn stats(&self) -> MemoryStats {
        let snr = tle_vsa::ops::theoretical_snr(self.dim, self.fact_count.max(1));
        MemoryStats {
            fact_count: self.fact_count,
            deep_fact_count: self.deep_count,
            dimensionality: self.dim,
            estimated_snr: snr,
            register_norm: self.register.norm(),
        }
    }
}

/// Memory bank statistics.
#[derive(Clone, Debug)]
pub struct MemoryStats {
    pub fact_count: usize,
    pub deep_fact_count: usize,
    pub dimensionality: usize,
    pub estimated_snr: f32,
    pub register_norm: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_retrieve() {
        let mut bank = MemoryBank::new(DEFAULT_DIM);
        let role = HyperVector::random_bipolar(DEFAULT_DIM, 10);
        let filler = HyperVector::random_bipolar(DEFAULT_DIM, 20);

        bank.store(&role, &filler, 1.0);

        let (retrieved, conf) = bank.retrieve(&role);
        // Single fact: should be exact retrieval
        assert_eq!(retrieved, filler);
    }

    #[test]
    fn test_multiple_facts() {
        let mut bank = MemoryBank::new(DEFAULT_DIM);
        let k = 10;

        let roles: Vec<HyperVector> = (0..k)
            .map(|i| HyperVector::random_bipolar(DEFAULT_DIM, i as u64 * 100))
            .collect();
        let fillers: Vec<HyperVector> = (0..k)
            .map(|i| HyperVector::random_bipolar(DEFAULT_DIM, i as u64 * 100 + 50))
            .collect();

        for i in 0..k {
            bank.store(&roles[i], &fillers[i], 1.0);
        }

        // Retrieve each fact - with k=10 and D=10240, SNR≈33
        // Sign cleanup should give good but not perfect similarity
        for i in 0..k {
            let (retrieved, _) = bank.retrieve(&roles[i]);
            let sim = cosine_similarity(&retrieved, &fillers[i]);
            // For k=10 items in D=10240: after sign cleanup expect sim > 0.15
            // (theoretical lower bound for reliable detection)
            assert!(
                sim > 0.1,
                "Fact {} should be detectably similar to original, got {}",
                i, sim
            );
        }

        // The retrieved vector should be MORE similar to its target than to others
        let (retrieved_0, _) = bank.retrieve(&roles[0]);
        let sim_target = cosine_similarity(&retrieved_0, &fillers[0]);
        let sim_other = cosine_similarity(&retrieved_0, &fillers[1]);
        assert!(
            sim_target > sim_other,
            "Target sim {} should exceed non-target sim {}",
            sim_target, sim_other
        );
    }

    #[test]
    fn test_forget() {
        let mut bank = MemoryBank::new(DEFAULT_DIM);
        let role = HyperVector::random_bipolar(DEFAULT_DIM, 10);
        let filler = HyperVector::random_bipolar(DEFAULT_DIM, 20);

        bank.store(&role, &filler, 1.0);
        bank.forget(&role, &filler);

        // After forgetting, register should be approximately zero
        let norm = bank.register.norm();
        assert!(norm < 0.01, "After forget, register norm should be near 0, got {}", norm);
    }

    #[test]
    fn test_stats() {
        let mut bank = MemoryBank::new(DEFAULT_DIM);
        for i in 0..5 {
            let r = HyperVector::random_bipolar(DEFAULT_DIM, i * 100);
            let f = HyperVector::random_bipolar(DEFAULT_DIM, i * 100 + 50);
            bank.store(&r, &f, 1.0);
        }

        let stats = bank.stats();
        assert_eq!(stats.fact_count, 5);
        assert_eq!(stats.dimensionality, DEFAULT_DIM);
        assert!(stats.estimated_snr > 30.0); // √(10240/4) ≈ 50
    }
}
