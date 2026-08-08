//! Multi-head N-gram hash encoder.
//!
//! Hashes context windows of varying lengths (1-gram through 5-gram) into
//! table keys. Uses FxHash (Fibonacci hashing) for speed — no cryptographic
//! security needed, just low collision rate and fast computation.

/// FxHash constant (Fibonacci hashing golden ratio for u64).
const FX_SEED: u64 = 0x517cc1b727220a95;

/// Maximum supported N-gram order.
pub const MAX_ORDER: usize = 5;

/// Multi-head N-gram hasher.
///
/// Each "head" operates on a different context window length:
/// - Head 0: unigram (1 token)
/// - Head 1: bigram (2 tokens)
/// - Head 2: trigram (3 tokens)
/// - Head 3: 4-gram (4 tokens)
/// - Head 4: 5-gram (5 tokens)
///
/// Longer context = more specific match = higher confidence.
pub struct NgramHash {
    /// Number of active heads (1..=MAX_ORDER).
    pub num_heads: usize,
}

impl NgramHash {
    /// Create a new multi-head hasher with the given number of heads.
    ///
    /// `num_heads` is clamped to 1..=MAX_ORDER.
    pub fn new(num_heads: usize) -> Self {
        Self {
            num_heads: num_heads.clamp(1, MAX_ORDER),
        }
    }

    /// Hash a context window for a specific head (N-gram order).
    ///
    /// `context` is the sequence of token IDs preceding the position to predict.
    /// `order` is 1-indexed (1 = unigram using last token, 2 = bigram using last 2, etc.)
    ///
    /// Returns None if context is shorter than the requested order.
    #[inline]
    pub fn hash_head(&self, context: &[u16], order: usize) -> Option<u64> {
        if context.len() < order || order == 0 || order > MAX_ORDER {
            return None;
        }

        let start = context.len() - order;
        let window = &context[start..];
        Some(Self::fx_hash_window(window))
    }

    /// Hash all heads that have sufficient context.
    ///
    /// Returns a Vec of (head_index, hash_key) pairs for heads that matched.
    /// Higher head index = longer context = more specific.
    pub fn hash_all_heads(&self, context: &[u16]) -> Vec<(usize, u64)> {
        let mut results = Vec::with_capacity(self.num_heads);
        for order in 1..=self.num_heads {
            if let Some(key) = self.hash_head(context, order) {
                results.push((order - 1, key));
            }
        }
        results
    }

    /// Core FxHash function for a window of token IDs.
    ///
    /// Combines token IDs with position-dependent mixing to ensure
    /// different orderings produce different hashes.
    #[inline]
    pub fn fx_hash_window(tokens: &[u16]) -> u64 {
        let mut hash: u64 = 0;
        for (i, &token) in tokens.iter().enumerate() {
            // Position-dependent mixing: rotate hash then XOR with token
            hash = hash.rotate_left(5);
            hash ^= (token as u64).wrapping_mul(FX_SEED);
            hash = hash.wrapping_add(i as u64);
        }
        // Final avalanche
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xff51afd7ed558ccd);
        hash ^= hash >> 33;
        hash
    }
}

impl Default for NgramHash {
    fn default() -> Self {
        Self::new(MAX_ORDER)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_deterministic() {
        let hasher = NgramHash::new(5);
        let ctx: Vec<u16> = vec![10, 20, 30, 40, 50];

        let h1 = hasher.hash_head(&ctx, 3);
        let h2 = hasher.hash_head(&ctx, 3);
        assert_eq!(h1, h2, "Same input must produce same hash");
    }

    #[test]
    fn test_hash_order_sensitivity() {
        let hasher = NgramHash::new(5);
        let ctx_a: Vec<u16> = vec![10, 20, 30];
        let ctx_b: Vec<u16> = vec![30, 20, 10]; // reversed

        let ha = hasher.hash_head(&ctx_a, 3).unwrap();
        let hb = hasher.hash_head(&ctx_b, 3).unwrap();
        assert_ne!(ha, hb, "Different orderings must produce different hashes");
    }

    #[test]
    fn test_hash_different_lengths() {
        let hasher = NgramHash::new(5);
        let ctx: Vec<u16> = vec![10, 20, 30, 40, 50];

        let h2 = hasher.hash_head(&ctx, 2).unwrap(); // last 2: [40, 50]
        let h3 = hasher.hash_head(&ctx, 3).unwrap(); // last 3: [30, 40, 50]
        assert_ne!(h2, h3, "Different context lengths must produce different hashes");
    }

    #[test]
    fn test_hash_all_heads() {
        let hasher = NgramHash::new(3);
        let ctx: Vec<u16> = vec![1, 2, 3];

        let results = hasher.hash_all_heads(&ctx);
        assert_eq!(results.len(), 3); // unigram, bigram, trigram all available
    }

    #[test]
    fn test_insufficient_context() {
        let hasher = NgramHash::new(5);
        let ctx: Vec<u16> = vec![10, 20]; // only 2 tokens

        assert!(hasher.hash_head(&ctx, 3).is_none()); // need 3, have 2
        assert!(hasher.hash_head(&ctx, 2).is_some()); // need 2, have 2
        assert!(hasher.hash_head(&ctx, 1).is_some()); // need 1, have 2
    }
}
