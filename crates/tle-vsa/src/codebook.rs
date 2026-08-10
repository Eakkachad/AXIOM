//! Codebook: Maps between discrete symbols and their hypervector representations.
//!
//! A codebook is the deterministic mapping layer that replaces learned embeddings.
//! Each symbol (word, role, concept) gets a unique, reproducibly-generated hypervector.
//!
//! ## Design Principle
//! Unlike neural embeddings which require backpropagation to learn,
//! codebook vectors are generated from deterministic hash-based seeds.
//! The high-dimensional space guarantees quasi-orthogonality between
//! any two randomly generated vectors.

use std::collections::HashMap;

use crate::hypervector::HyperVector;
use crate::DEFAULT_DIM;

/// A deterministic mapping from string symbols to hypervectors.
///
/// Supports two types of entries:
/// - **Atomic symbols**: Words/tokens with unique random vectors
/// - **Role vectors**: Syntactic roles (Subject, Verb, Object, etc.)
///
/// All vectors are generated reproducibly from symbol names via hashing.
#[derive(Clone)]
pub struct Codebook {
    /// Symbol name → hypervector mapping
    entries: HashMap<String, HyperVector>,
    /// Dimensionality of all vectors in this codebook
    dim: usize,
    /// Base seed for reproducible generation
    base_seed: u64,
}

impl Codebook {
    /// Create a new empty codebook.
    pub fn new(dim: usize, base_seed: u64) -> Self {
        Self {
            entries: HashMap::new(),
            dim,
            base_seed,
        }
    }

    /// Create a codebook with default parameters.
    pub fn default_params() -> Self {
        Self::new(DEFAULT_DIM, 0xDEAD_BEEF_CAFE_1234)
    }

    /// Get or generate a hypervector for a symbol.
    /// First lookup returns a generated vector; subsequent lookups return the same vector.
    /// This is the core "embedding-without-training" mechanism.
    pub fn get_or_insert(&mut self, symbol: &str) -> &HyperVector {
        if !self.entries.contains_key(symbol) {
            let seed = self.symbol_seed(symbol);
            let hv = HyperVector::random_bipolar(self.dim, seed);
            // Bipolar vectors are already compressed at creation (data is empty).
            self.entries.insert(symbol.to_string(), hv);
        }
        &self.entries[symbol]
    }

    /// Get a vector for a symbol if it exists.
    pub fn get(&self, symbol: &str) -> Option<&HyperVector> {
        self.entries.get(symbol)
    }

    /// Insert a specific vector for a symbol (useful for learned/pre-defined roles).
    pub fn insert(&mut self, symbol: &str, vector: HyperVector) {
        assert_eq!(vector.dim(), self.dim, "Dimension mismatch");
        self.entries.insert(symbol.to_string(), vector);
    }

    /// Get all entries as a slice for nearest-neighbor search.
    pub fn all_entries(&self) -> Vec<(&str, &HyperVector)> {
        self.entries
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect()
    }

    /// Get all vectors for cleanup memory lookup.
    pub fn all_vectors(&self) -> Vec<&HyperVector> {
        self.entries.values().collect()
    }

    /// Number of symbols in the codebook.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the codebook is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Dimensionality of vectors in this codebook.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Generate a deterministic seed from a symbol name.
    /// Uses FNV-1a hash combined with the base seed for reproducibility.
    fn symbol_seed(&self, symbol: &str) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
        for byte in symbol.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3); // FNV prime
        }
        hash ^ self.base_seed
    }

    /// Create a standard linguistic role codebook.
    /// Pre-generates role vectors for common syntactic roles.
    pub fn with_standard_roles(dim: usize, base_seed: u64) -> Self {
        let mut cb = Self::new(dim, base_seed);
        // Standard grammatical roles
        let roles = [
            "SUBJECT", "VERB", "OBJECT", "INDIRECT_OBJECT",
            "ADJECTIVE", "ADVERB", "PREPOSITION", "CONJUNCTION",
            "DETERMINER", "PRONOUN", "NOUN", "AUX_VERB",
            "COMPLEMENT", "SPECIFIER", "POSITION_0", "POSITION_1",
            "POSITION_2", "POSITION_3", "POSITION_4", "POSITION_5",
            "POSITION_6", "POSITION_7", "POSITION_8", "POSITION_9",
            "QUERY", "KEY", "VALUE", "CONTEXT",
            "CAUSE", "EFFECT", "AGENT", "PATIENT",
            "THEME", "EXPERIENCER", "INSTRUMENT", "LOCATION",
            "TIME", "MANNER", "PURPOSE", "NEGATION",
        ];
        for role in &roles {
            cb.get_or_insert(role);
        }
        cb
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::similarity::cosine_similarity;

    #[test]
    fn test_deterministic_codebook() {
        let mut cb1 = Codebook::default_params();
        let mut cb2 = Codebook::default_params();

        let v1 = cb1.get_or_insert("hello").clone();
        let v2 = cb2.get_or_insert("hello").clone();

        assert_eq!(v1, v2, "Same symbol must produce same vector");
    }

    #[test]
    fn test_different_symbols_orthogonal() {
        let mut cb = Codebook::default_params();
        let v1 = cb.get_or_insert("cat").clone();
        let v2 = cb.get_or_insert("dog").clone();

        let sim = cosine_similarity(&v1, &v2);
        assert!(sim.abs() < 0.05, "Different symbols should be quasi-orthogonal");
    }

    #[test]
    fn test_role_codebook() {
        let cb = Codebook::with_standard_roles(DEFAULT_DIM, 42);
        assert!(cb.len() >= 30); // Should have all standard roles
        assert!(cb.get("SUBJECT").is_some());
        assert!(cb.get("VERB").is_some());
        assert!(cb.get("OBJECT").is_some());
    }
}
