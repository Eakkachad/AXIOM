//! Energy function for scoring knowledge graph paths.
//!
//! The energy function combines multiple objectives:
//! - **Relevance**: How well the path answers the query (cosine similarity)
//! - **Coherence**: How smoothly consecutive triples connect (semantic flow)
//! - **Length penalty**: Prefer paths near a target length
//! - **Simplicity**: Occam's razor — shorter explanations preferred
//!
//! Each path is encoded as a hyperdimensional vector via binding and bundling,
//! enabling efficient similarity computation in VSA space.

use tle_vsa::{cosine_similarity, Codebook, HyperVector};

use crate::graph::Triple;

/// Configuration for the energy function weights.
#[derive(Debug, Clone)]
pub struct EnergyConfig {
    /// Weight for query relevance (cosine similarity to query).
    pub lambda_relevance: f32,
    /// Weight for grammatical coherence (reserved for future use).
    pub lambda_grammar: f32,
    /// Weight for consecutive triple coherence.
    pub lambda_coherence: f32,
    /// Weight for length penalty (deviation from target).
    pub lambda_length: f32,
    /// Weight for simplicity (Occam's razor).
    pub lambda_simplicity: f32,
    /// Target path length for length penalty.
    pub target_length: usize,
}

impl Default for EnergyConfig {
    fn default() -> Self {
        Self {
            lambda_relevance: 1.0,
            lambda_grammar: 0.5,
            lambda_coherence: 0.8,
            lambda_length: 0.3,
            lambda_simplicity: 0.2,
            target_length: 3,
        }
    }
}

/// Compute the total energy score for a path.
///
/// Higher energy = better path. The energy is a weighted sum of:
/// - Relevance: cosine similarity between path vector and query vector
/// - Coherence: average pairwise similarity of consecutive triples
/// - Length penalty: penalizes deviation from target length
/// - Simplicity: rewards shorter paths (Occam's razor)
pub fn compute_energy(
    path_triples: &[Triple],
    query_vector: &HyperVector,
    config: &EnergyConfig,
    entities: &[String],
    relations: &[String],
    codebook: &mut Codebook,
) -> f32 {
    if path_triples.is_empty() {
        return 0.0;
    }

    let path_hdv = encode_path(path_triples, entities, relations, codebook);
    let relevance = compute_relevance(&path_hdv, query_vector);
    let coherence = compute_coherence(path_triples, entities, relations, codebook);
    let length_penalty = compute_length_penalty(path_triples.len(), config.target_length);
    let simplicity = compute_simplicity(path_triples.len());

    config.lambda_relevance * relevance
        + config.lambda_coherence * coherence
        + config.lambda_length * length_penalty
        + config.lambda_simplicity * simplicity
}

/// Compute relevance: cosine similarity between path HDV and query vector.
pub fn compute_relevance(path_hdv: &HyperVector, query_vector: &HyperVector) -> f32 {
    cosine_similarity(path_hdv, query_vector)
}

/// Compute coherence: average cosine similarity between consecutive triple encodings.
///
/// This measures how smoothly the path flows from one fact to the next.
/// Adjacent triples that share entities will naturally have higher similarity.
pub fn compute_coherence(
    path_triples: &[Triple],
    entities: &[String],
    relations: &[String],
    codebook: &mut Codebook,
) -> f32 {
    if path_triples.len() < 2 {
        return 1.0; // Single triple is maximally coherent with itself
    }

    let encoded: Vec<HyperVector> = path_triples
        .iter()
        .map(|t| encode_triple(t, entities, relations, codebook))
        .collect();

    let mut total_sim = 0.0;
    let pairs = encoded.len() - 1;
    for i in 0..pairs {
        total_sim += cosine_similarity(&encoded[i], &encoded[i + 1]);
    }

    total_sim / pairs as f32
}

/// Compute length penalty: penalizes paths that deviate from the target length.
///
/// Returns a value in [0, 1] where 1.0 means exactly target length.
pub fn compute_length_penalty(path_len: usize, target_len: usize) -> f32 {
    let diff = (path_len as f32 - target_len as f32).abs();
    1.0 / (1.0 + diff)
}

/// Compute simplicity score: Occam's razor — shorter paths are better.
///
/// Returns a value in (0, 1] where shorter paths score higher.
pub fn compute_simplicity(path_len: usize) -> f32 {
    1.0 / (path_len as f32)
}

/// Encode a single triple as a hypervector using VSA binding.
///
/// Formula: C(subject) ⊙ C(relation) ⊙ ρ(C(object))
///
/// Where:
/// - C(x) = codebook lookup for symbol x
/// - ⊙ = Hadamard product (binding)
/// - ρ = permutation (shifts the object vector to distinguish subject/object roles)
pub fn encode_triple(
    triple: &Triple,
    entities: &[String],
    relations: &[String],
    codebook: &mut Codebook,
) -> HyperVector {
    let subject_name = &entities[triple.subject_id];
    let relation_name = &relations[triple.relation_id];
    let object_name = &entities[triple.object_id];

    let s_vec = codebook.get_or_insert(subject_name).clone();
    let r_vec = codebook.get_or_insert(relation_name).clone();
    let o_vec = codebook.get_or_insert(object_name).clone();

    // C(s) ⊙ C(r) ⊙ ρ(C(o))
    let o_permuted = o_vec.permute(1);
    s_vec.hadamard(&r_vec).hadamard(&o_permuted)
}

/// Encode a path (sequence of triples) as a single hypervector.
///
/// Each triple is encoded and permuted by its position index,
/// then all are bundled (summed) into a single composite vector.
/// This preserves both content and ordering information.
pub fn encode_path(
    path_triples: &[Triple],
    entities: &[String],
    relations: &[String],
    codebook: &mut Codebook,
) -> HyperVector {
    if path_triples.is_empty() {
        return HyperVector::zeros(codebook.dim());
    }

    let dim = codebook.dim();
    let mut result = HyperVector::zeros(dim);

    for (i, triple) in path_triples.iter().enumerate() {
        let triple_vec = encode_triple(triple, entities, relations, codebook);
        // Permute by position to encode sequence order
        let positioned = triple_vec.permute(i as i32 * 2);
        result = result.add(&positioned);
    }

    result.normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::KnowledgeGraph;

    fn setup_kg() -> KnowledgeGraph {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple("sky", "is", "blue");
        kg.add_triple("blue", "has", "short_wavelength");
        kg
    }

    #[test]
    fn test_encode_triple() {
        let kg = setup_kg();
        let mut codebook = Codebook::new(2048, 42);
        let triple = &kg.triples[0];
        let encoded = encode_triple(triple, &kg.entities, &kg.relations, &mut codebook);
        assert_eq!(encoded.dim(), 2048);
        // Encoded triple should not be zero
        assert!(encoded.norm() > 0.0);
    }

    #[test]
    fn test_encode_path() {
        let kg = setup_kg();
        let mut codebook = Codebook::new(2048, 42);
        let path_hdv = encode_path(&kg.triples, &kg.entities, &kg.relations, &mut codebook);
        assert_eq!(path_hdv.dim(), 2048);
        // Path vector should be normalized
        assert!((path_hdv.norm() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_relevance_self() {
        let kg = setup_kg();
        let mut codebook = Codebook::new(2048, 42);
        let path_hdv = encode_path(&kg.triples, &kg.entities, &kg.relations, &mut codebook);
        let relevance = compute_relevance(&path_hdv, &path_hdv);
        assert!((relevance - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_coherence() {
        let kg = setup_kg();
        let mut codebook = Codebook::new(2048, 42);
        let coherence = compute_coherence(&kg.triples, &kg.entities, &kg.relations, &mut codebook);
        // Connected triples should have some coherence (shared entity "blue")
        assert!(coherence > -1.0);
        assert!(coherence <= 1.0);
    }

    #[test]
    fn test_length_penalty() {
        assert!((compute_length_penalty(3, 3) - 1.0).abs() < 0.01);
        assert!(compute_length_penalty(1, 3) < compute_length_penalty(3, 3));
        assert!(compute_length_penalty(5, 3) < compute_length_penalty(3, 3));
    }

    #[test]
    fn test_simplicity() {
        assert!(compute_simplicity(1) > compute_simplicity(2));
        assert!(compute_simplicity(2) > compute_simplicity(3));
        assert!((compute_simplicity(1) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_energy() {
        let kg = setup_kg();
        let mut codebook = Codebook::new(2048, 42);
        let config = EnergyConfig::default();
        let query_vec = codebook.get_or_insert("sky").clone();

        let energy = compute_energy(
            &kg.triples,
            &query_vec,
            &config,
            &kg.entities,
            &kg.relations,
            &mut codebook,
        );
        // Energy should be finite and non-zero for a valid path
        assert!(energy.is_finite());
    }
}
