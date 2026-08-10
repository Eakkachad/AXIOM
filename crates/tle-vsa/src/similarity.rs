//! Similarity measures for hypervector comparison.
//!
//! In VSA, similarity is used to:
//! 1. Determine if a retrieved vector matches a codebook entry (cleanup)
//! 2. Measure quality of unbinding (signal vs noise)
//! 3. Route vectors to appropriate processing nodes (TDA router input)

use crate::HyperVector;

/// Compute cosine similarity between two hypervectors.
///
/// cos(a, b) = (a · b) / (||a|| · ||b||)
///
/// Range: [-1, 1]
/// - 1.0 = identical direction
/// - 0.0 = orthogonal (unrelated)
/// - -1.0 = opposite direction
///
/// Fast path: when both vectors are bit-packed bipolar (±1), cosine
/// is computed via XOR+popcount — 10-20× faster than f32 multiply-accumulate.
#[inline]
pub fn cosine_similarity(a: &HyperVector, b: &HyperVector) -> f32 {
    // Fast path: bit-packed bipolar vectors.
    if let (Some(ap), Some(bp)) = (&a.packed, &b.packed) {
        let n = ap.len().min(bp.len());
        let dim = a.dim();
        let mut matches: u32 = 0;
        for i in 0..n {
            matches += (!(ap[i] ^ bp[i])).count_ones();
        }
        return 2.0 * matches as f32 / dim as f32 - 1.0;
    }
    let dot = a.dot(b);
    let na = a.norm();
    let nb = b.norm();
    if na < 1e-10 || nb < 1e-10 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Compute raw dot product (unnormalized similarity).
#[inline]
pub fn dot_product(a: &HyperVector, b: &HyperVector) -> f32 {
    a.dot(b)
}

/// Hamming distance for bipolar vectors.
/// Counts the number of positions where signs differ.
///
/// For bipolar vectors: hamming = (D - dot(a,b)) / 2
pub fn hamming_distance(a: &HyperVector, b: &HyperVector) -> usize {
    debug_assert_eq!(a.dim(), b.dim());
    let da = a.as_slice();
    let db = b.as_slice();
    da.iter().zip(db.iter()).filter(|(&x, &y)| x.signum() != y.signum()).count()
}

/// Find the index of the most similar vector in a codebook.
/// Returns (index, similarity_score).
///
/// This is the core "cleanup memory" lookup operation.
/// Deterministic: always returns the same result for the same input.
pub fn nearest_in_codebook(query: &HyperVector, codebook: &[HyperVector]) -> (usize, f32) {
    assert!(!codebook.is_empty(), "Codebook must not be empty");

    let mut best_idx = 0;
    let mut best_sim = f32::NEG_INFINITY;

    for (i, entry) in codebook.iter().enumerate() {
        let sim = cosine_similarity(query, entry);
        if sim > best_sim {
            best_sim = sim;
            best_idx = i;
        }
    }

    (best_idx, best_sim)
}

/// Find top-k nearest vectors in a codebook.
/// Returns Vec<(index, similarity)> sorted by descending similarity.
pub fn top_k_nearest(query: &HyperVector, codebook: &[HyperVector], k: usize) -> Vec<(usize, f32)> {
    let mut scores: Vec<(usize, f32)> = codebook
        .iter()
        .enumerate()
        .map(|(i, entry)| (i, cosine_similarity(query, entry)))
        .collect();

    // Partial sort: we only need top-k
    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(k);
    scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_DIM;

    #[test]
    fn test_self_similarity() {
        let v = HyperVector::random_bipolar(DEFAULT_DIM, 42);
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_orthogonal_similarity() {
        let v1 = HyperVector::random_bipolar(DEFAULT_DIM, 1);
        let v2 = HyperVector::random_bipolar(DEFAULT_DIM, 2);
        let sim = cosine_similarity(&v1, &v2);
        // For D=10240: std of cos_sim ≈ 1/√D ≈ 0.01
        assert!(sim.abs() < 0.05);
    }

    #[test]
    fn test_nearest_codebook() {
        let target = HyperVector::random_bipolar(DEFAULT_DIM, 42);
        let codebook: Vec<HyperVector> = (0..10)
            .map(|i| HyperVector::random_bipolar(DEFAULT_DIM, i * 100))
            .collect();

        // Add target to codebook at position 5
        let mut codebook_with_target = codebook.clone();
        codebook_with_target.push(target.clone());

        let (idx, sim) = nearest_in_codebook(&target, &codebook_with_target);
        assert_eq!(idx, 10); // Last position (where we added the target)
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_hamming_distance() {
        let v = HyperVector::random_bipolar(DEFAULT_DIM, 42);
        assert_eq!(hamming_distance(&v, &v), 0);

        let v2 = HyperVector::random_bipolar(DEFAULT_DIM, 43);
        let dist = hamming_distance(&v, &v2);
        // Expected: ~D/2 for random bipolar
        let expected = DEFAULT_DIM / 2;
        assert!(
            (dist as i64 - expected as i64).unsigned_abs() < (DEFAULT_DIM / 10) as u64,
            "Hamming distance {} should be near {}", dist, expected
        );
    }
}
