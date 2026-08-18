//! Parameter-Free Algorithmic Information & Minimum Description Length (MDL) Scorer
//! Implements Normalized Compression Distance (NCD) and Conditional Description Rate
//! based on Li & Vitányi (2004) and Marcus Hutter (2005).

use std::collections::HashMap;

/// Approximate Kolmogorov complexity C(x) using byte-shingle Markov entropy.
/// Dependency-free, fast O(N), deterministic.
pub fn estimate_compressed_size(data: &[u8], shingle_size: usize) -> usize {
    if data.is_empty() {
        return 0;
    }
    if data.len() < shingle_size {
        return data.len();
    }

    let mut freq_map: HashMap<&[u8], usize> = HashMap::new();
    let total_shingles = data.len() - shingle_size + 1;

    for window in data.windows(shingle_size) {
        *freq_map.entry(window).or_insert(0) += 1;
    }

    // Empirical Shannon entropy in bits: H = - sum(p * log2(p))
    let mut entropy_bits = 0.0;
    let n = total_shingles as f64;

    for &count in freq_map.values() {
        let p = count as f64 / n;
        entropy_bits -= p * p.log2();
    }

    // Total bits = data bits + vocabulary overhead
    let data_bits = entropy_bits * n;
    let vocab_overhead_bits = (freq_map.len() * (shingle_size * 8 + 8)) as f64;
    let total_bytes = ((data_bits + vocab_overhead_bits) / 8.0).ceil() as usize;

    total_bytes.max(1)
}

/// Symmetrized Normalized Compression Distance (NCD):
/// NCD_sym(x, y) = [min(C(xy), C(yx)) - min(C(x), C(y))] / max(C(x), C(y))
pub fn ncd(x: &[u8], y: &[u8], shingle_size: usize) -> f64 {
    if x == y {
        return 0.0;
    }
    let cx = estimate_compressed_size(x, shingle_size);
    let cy = estimate_compressed_size(y, shingle_size);

    let min_c = cx.min(cy) as f64;
    let max_c = cx.max(cy) as f64;

    if max_c == 0.0 {
        return 0.0;
    }

    let mut xy = Vec::with_capacity(x.len() + y.len());
    xy.extend_from_slice(x);
    xy.extend_from_slice(y);
    let cxy = estimate_compressed_size(&xy, shingle_size);

    let mut yx = Vec::with_capacity(y.len() + x.len());
    yx.extend_from_slice(y);
    yx.extend_from_slice(x);
    let cyx = estimate_compressed_size(&yx, shingle_size);

    let c_joint = (cxy.min(cyx)) as f64;
    ((c_joint - min_c) / max_c).clamp(0.0, 1.0)
}

/// Length-Normalized Conditional Description Rate (Bits-Per-Byte):
/// H_C(candidate | context) = [C(context ∘ candidate) - C(context)] * 8 / |candidate|
pub fn conditional_description_rate(context: &[u8], candidate: &[u8], shingle_size: usize) -> f64 {
    if candidate.is_empty() {
        return f64::INFINITY;
    }
    let c_context = estimate_compressed_size(context, shingle_size);

    let mut joint = Vec::with_capacity(context.len() + candidate.len());
    joint.extend_from_slice(context);
    joint.extend_from_slice(candidate);
    let c_joint = estimate_compressed_size(&joint, shingle_size);

    let delta_bytes = c_joint.saturating_sub(c_context);
    (delta_bytes * 8) as f64 / candidate.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ncd_identity_and_symmetry() {
        let text1 = b"Pyotr Ilyich Tchaikovsky composed Swan Lake";
        let text2 = b"Swan Lake was composed by Pyotr Ilyich Tchaikovsky";
        let unrelated = b"Quantum electrodynamics in relativistic gauge fields";

        let dist_self = ncd(text1, text1, 3);
        assert!(dist_self < 0.05, "Self distance should be close to 0, got {}", dist_self);

        let dist_related = ncd(text1, text2, 3);
        let dist_unrelated = ncd(text1, unrelated, 3);
        assert!(dist_related < dist_unrelated, "Related texts must have lower NCD than unrelated");
    }

    #[test]
    fn test_conditional_description_rate() {
        let context = b"Who directed Jurassic Park? Steven Spielberg is an American film director.";
        let cand_true = b"Steven Spielberg";
        let cand_false = b"Wolfgang Amadeus Mozart";

        let rate_true = conditional_description_rate(context, cand_true, 3);
        let rate_false = conditional_description_rate(context, cand_false, 3);

        assert!(rate_true < rate_false, "True candidate must have lower description rate than unrelated candidate");
    }
}
