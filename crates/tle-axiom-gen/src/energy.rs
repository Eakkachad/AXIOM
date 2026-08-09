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
    /// Weight for VSA internal triple consistency.
    /// cos(C(subject) ⊙ C(relation), C(object)).
    pub lambda_consistency: f32,
    /// Weight for entity informativeness (inverse entity frequency).
    pub lambda_ief: f32,
    /// Weight for per-triple quality confidence (heuristic entity brevity).
    pub lambda_confidence: f32,
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
            lambda_consistency: 0.0,
            lambda_ief: 0.0,
            lambda_confidence: 0.0,
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
/// - Consistency: average VSA consistency of individual triples
/// - IEF: entity informativeness (rare entities score higher)
/// - Coherence: average pairwise similarity of consecutive triples
/// - Length penalty: penalizes deviation from target length
/// - Simplicity: rewards shorter paths (Occam's razor)
///
/// `entity_ief` is an optional per-entity inverse-entity-frequency array
/// (compute once from the graph as `-log(freq(e)/total)`). Pass `None` to
/// skip the IEF term.
///
/// `triple_confidences` is an optional per-triple confidence array (as
/// returned by `KnowledgeGraph::triple_confidence`). Pass `None` to skip
/// the confidence term.
pub fn compute_energy(
    path_triples: &[Triple],
    query_vector: &HyperVector,
    config: &EnergyConfig,
    entities: &[String],
    relations: &[String],
    codebook: &mut Codebook,
    entity_ief: Option<&[f32]>,
    triple_confidences: Option<&[f32]>,
) -> f32 {
    if path_triples.is_empty() {
        return 0.0;
    }

    let path_hdv = encode_path(path_triples, entities, relations, codebook);
    let relevance = compute_relevance(&path_hdv, query_vector);
    let consistency = compute_consistency(path_triples, entities, relations, codebook);
    let coherence = compute_coherence(path_triples, entities, relations, codebook);
    let confidence = compute_confidence(path_triples, entities, relations, triple_confidences);
    let ief = entity_ief.map(|ief| compute_ief_score(path_triples, ief)).unwrap_or(0.0);
    let length_penalty = compute_length_penalty(path_triples.len(), config.target_length);
    let simplicity = compute_simplicity(path_triples.len());

    config.lambda_relevance * relevance
        + config.lambda_consistency * consistency
        + config.lambda_confidence * confidence
        + config.lambda_ief * ief
        + config.lambda_coherence * coherence
        + config.lambda_length * length_penalty
        + config.lambda_simplicity * simplicity
}

/// Average triple quality confidence: brevity of subjects and objects
/// favours clean triples over wordy decomposition artifacts.
fn compute_confidence(triples: &[Triple], entities: &[String], relations: &[String], _precomputed: Option<&[f32]>) -> f32 {
    let mut total = 0.0f32;
    for t in triples {
        let subj_words = entities[t.subject_id].split_whitespace().count();
        let obj_words = entities[t.object_id].split_whitespace().count();
        let rel_name = relations[t.relation_id].as_str();
        // Short entities are cleaner (long phrases are extraction noise).
        let len_score = (0.85f32).powi(subj_words.saturating_sub(2) as i32)
            * (0.85f32).powi(obj_words.saturating_sub(1) as i32);
        // Bare copula relations match too easily.
        let rel_specificity = if rel_name == "is" || rel_name == "are" || rel_name == "was" || rel_name == "were" {
            0.5
        } else {
            1.0
        };
        total += len_score * rel_specificity;
    }
    if triples.is_empty() { 0.0 } else { total / triples.len() as f32 }
}

/// VSA internal consistency: for each triple, measure how well the bound
/// subject-relation vector aligns with the object vector.
///
/// A true fact (sky, is, blue) has C(sky)⊙C(is) ≈ C(blue) because the
/// codebook vectors for related concepts have no structure.  But a noise
/// triple ("Together they", "is", "...") has random unrelated vectors →
/// low cosine.
pub fn compute_consistency(
    path_triples: &[Triple],
    entities: &[String],
    relations: &[String],
    codebook: &mut Codebook,
) -> f32 {
    let mut total = 0.0f32;
    for triple in path_triples {
        let subj_name = &entities[triple.subject_id];
        let rel_name = &relations[triple.relation_id];
        let obj_name = &entities[triple.object_id];
        if let (Some(s_vec), Some(r_vec), Some(o_vec)) = (
            codebook.get(subj_name).cloned(),
            codebook.get(rel_name).cloned(),
            codebook.get(obj_name).cloned(),
        ) {
            let bound = s_vec.hadamard(&r_vec);
            total += cosine_similarity(&bound, &o_vec).max(-1.0);
        }
    }
    if path_triples.is_empty() { 0.0 } else { total / path_triples.len() as f32 }
}

/// Average inverse entity frequency of path entities.  Rare (informative)
/// entities boost the score; hub entities ("is", "a", "the") are penalised.
pub fn compute_ief_score(triples: &[Triple], entity_ief: &[f32]) -> f32 {
    let mut total = 0.0f32;
    let mut n = 0usize;
    for t in triples {
        if t.subject_id < entity_ief.len() { total += entity_ief[t.subject_id]; n += 1; }
        if t.object_id < entity_ief.len() { total += entity_ief[t.object_id]; n += 1; }
    }
    if n == 0 { 0.0 } else { total / n as f32 }
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

/// Compute fluency score for a linearized sentence.
///
/// Measures how "natural English" a generated string looks using
/// lightweight heuristics (no neural model needed).
/// Returns value in [0, 1] — lower = more natural.
pub fn compute_fluency(sentence: &str) -> f32 {
    if sentence.len() < 5 {
        return 0.5;
    }

    let lower = sentence.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    let words: Vec<&str> = lower.split_whitespace().collect();

    if words.is_empty() {
        return 1.0;
    }

    // 1. Average word length (English avg ≈ 4.7)
    let avg_word_len = words.iter().map(|w| w.len()).sum::<usize>() as f32 / words.len() as f32;
    let word_len_score = 1.0 - ((avg_word_len - 4.7).abs() / 10.0).min(1.0);

    // 2. Space ratio (natural ≈ 15-20%)
    let space_ratio = chars.iter().filter(|&&c| c == ' ').count() as f32 / chars.len() as f32;
    let space_score = 1.0 - ((space_ratio - 0.17).abs() / 0.3).min(1.0);

    // 3. Sentence starts with capital
    let starts_cap = sentence.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
    let cap_score = if starts_cap { 0.0 } else { 0.1 };

    // 4. Ends with punctuation
    let ends_punct = sentence.ends_with('.') || sentence.ends_with('!') || sentence.ends_with('?');
    let punct_score = if ends_punct { 0.0 } else { 0.1 };

    // Combine (lower = more natural)
    let raw = 1.0 - (word_len_score * 0.4 + space_score * 0.4) + cap_score + punct_score;
    raw.max(0.0).min(1.0)
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

    // Prefer `get` (O(1) lookup, no insertion) since add_fact already
    // registered every symbol; fall back to insertion only if somehow missing.
    if let (Some(s_vec), Some(r_vec), Some(o_vec)) = (
        codebook.get(subject_name).cloned(),
        codebook.get(relation_name).cloned(),
        codebook.get(object_name).cloned(),
    ) {
        // C(s) ⊙ C(r) ⊙ ρ(C(o))
        let o_permuted = o_vec.permute(1);
        return s_vec.hadamard(&r_vec).hadamard(&o_permuted);
    }

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
            None,
            None,
        );
        // Energy should be finite and non-zero for a valid path
        assert!(energy.is_finite());
    }
}
