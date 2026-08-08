//! Knowledge Bundle — a single VSA vector storing up to MAX_ITEMS facts.
//!
//! Each bundle is a D-dimensional vector that is the superposition (sum)
//! of encoded fact triples. SNR = √(D/(k-1)) where k = number of items.
//!
//! For D=4096 and k=200: SNR ≈ 4.5 (still retrievable with cleanup)

use tle_vsa::{cosine_similarity, HyperVector, Codebook};

/// Maximum items per bundle before SNR degrades too much.
pub const MAX_BUNDLE_SIZE: usize = 200;

/// A single knowledge bundle — stores facts as VSA superposition.
#[derive(Clone)]
pub struct KnowledgeBundle {
    /// The bundled vector (sum of encoded triples).
    pub vector: HyperVector,
    /// Number of facts stored in this bundle.
    pub count: usize,
    /// Dimensionality.
    dim: usize,
}

impl KnowledgeBundle {
    /// Create a new empty bundle.
    pub fn new(dim: usize) -> Self {
        Self {
            vector: HyperVector::zeros(dim),
            count: 0,
            dim,
        }
    }

    /// Add a fact (encoded as a hypervector) to this bundle.
    ///
    /// Returns false if bundle is full (should split).
    pub fn add(&mut self, encoded_fact: &HyperVector) -> bool {
        if self.count >= MAX_BUNDLE_SIZE {
            return false;
        }
        self.vector = self.vector.add(encoded_fact);
        self.count += 1;
        true
    }

    /// Query: how similar is a query vector to this bundle's content?
    ///
    /// Higher similarity = bundle likely contains relevant facts.
    pub fn similarity(&self, query: &HyperVector) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        cosine_similarity(&self.vector, query)
    }

    /// Check if bundle is full.
    pub fn is_full(&self) -> bool {
        self.count >= MAX_BUNDLE_SIZE
    }

    /// Current Signal-to-Noise Ratio.
    pub fn snr(&self) -> f32 {
        if self.count <= 1 {
            return f32::INFINITY;
        }
        (self.dim as f32 / (self.count - 1) as f32).sqrt()
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.dim * 4 // f32 per dimension
    }

    /// Is the bundle empty?
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Encode a fact triple (subject, relation, object) as a hypervector.
///
/// Formula: HDV(s,r,o) = C(s) ⊙ C(r) ⊙ ρ(C(o))
/// The permutation on object breaks symmetry (subject ≠ object).
pub fn encode_fact(
    subject: &str,
    relation: &str,
    object: &str,
    codebook: &mut Codebook,
) -> HyperVector {
    let s_vec = codebook.get_or_insert(subject).clone();
    let r_vec = codebook.get_or_insert(relation).clone();
    let o_vec = codebook.get_or_insert(object).clone();

    // C(s) ⊙ C(r) ⊙ ρ(C(o))
    let o_shifted = o_vec.permute(1);
    s_vec.hadamard(&r_vec).hadamard(&o_shifted)
}

/// Query a bundle for facts about a subject with a given relation.
///
/// Unbind: result = C(subject) ⊙ C(relation) ⊙ bundle_vector
/// The result should be similar to ρ(C(object)) if the fact exists.
pub fn query_bundle(
    subject: &str,
    relation: &str,
    bundle: &KnowledgeBundle,
    codebook: &Codebook,
) -> Option<HyperVector> {
    let s_vec = codebook.get(subject)?;
    let r_vec = codebook.get(relation)?;

    // Unbind: s ⊙ r ⊙ bundle → should recover ρ(object)
    let query = s_vec.hadamard(r_vec);
    let result = query.hadamard(&bundle.vector);
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bundle_add() {
        let dim = 4096;
        let mut bundle = KnowledgeBundle::new(dim);
        let fact = HyperVector::random_bipolar(dim, 42);

        assert!(bundle.add(&fact));
        assert_eq!(bundle.count, 1);
        assert!(!bundle.is_full());
    }

    #[test]
    fn test_bundle_full() {
        let dim = 256;
        let mut bundle = KnowledgeBundle::new(dim);
        for i in 0..MAX_BUNDLE_SIZE {
            let fact = HyperVector::random_bipolar(dim, i as u64);
            assert!(bundle.add(&fact));
        }
        assert!(bundle.is_full());

        let overflow = HyperVector::random_bipolar(dim, 999);
        assert!(!bundle.add(&overflow)); // rejected — full
    }

    #[test]
    fn test_encode_and_query() {
        let dim = 4096;
        let mut codebook = Codebook::new(dim, 42);

        let encoded = encode_fact("cat", "is", "animal", &mut codebook);
        let mut bundle = KnowledgeBundle::new(dim);
        bundle.add(&encoded);

        // Query: cat is ? → should recover something close to ρ(animal)
        let result = query_bundle("cat", "is", &bundle, &codebook).unwrap();
        let expected = codebook.get("animal").unwrap().permute(1);

        let sim = cosine_similarity(&result, &expected);
        assert!(sim > 0.9, "Should recover object with high similarity, got {}", sim);
    }

    #[test]
    fn test_snr() {
        let dim = 4096;
        let bundle = KnowledgeBundle { vector: HyperVector::zeros(dim), count: 100, dim };
        // SNR = √(4096/99) ≈ 6.4
        assert!((bundle.snr() - 6.4).abs() < 0.5);
    }
}
