//! Bloom Filter for O(1) "do I know about this topic?" checks.
//!
//! Before doing expensive VSA cosine search, check the Bloom filter first.
//! False positives are OK (we'll do full search and find nothing).
//! False negatives are NOT OK (we'd miss knowledge we have).

/// Simple Bloom filter using 3 hash functions.
pub struct BloomFilter {
    bits: Vec<u64>,
    num_bits: usize,
    num_hashes: usize,
}

impl BloomFilter {
    /// Create a new Bloom filter with approximate capacity.
    ///
    /// For 100K items with 1% false positive rate: ~960K bits = 120KB
    pub fn new(expected_items: usize) -> Self {
        // Optimal: m = -n*ln(p) / (ln2)^2, k = (m/n)*ln2
        // For p=0.01: m ≈ 9.6 * n, k ≈ 7
        let num_bits = (expected_items * 10).max(1024);
        let num_words = (num_bits + 63) / 64;
        Self {
            bits: vec![0u64; num_words],
            num_bits,
            num_hashes: 3, // simple: 3 hashes is good enough for our use
        }
    }

    /// Insert a key into the filter.
    pub fn insert(&mut self, key: &str) {
        for i in 0..self.num_hashes {
            let bit = self.hash(key, i) % self.num_bits;
            self.bits[bit / 64] |= 1u64 << (bit % 64);
        }
    }

    /// Check if a key might be in the filter.
    ///
    /// Returns true if POSSIBLY present (may be false positive).
    /// Returns false if DEFINITELY not present (no false negatives).
    pub fn maybe_contains(&self, key: &str) -> bool {
        for i in 0..self.num_hashes {
            let bit = self.hash(key, i) % self.num_bits;
            if self.bits[bit / 64] & (1u64 << (bit % 64)) == 0 {
                return false;
            }
        }
        true
    }

    /// Approximate number of items inserted (from bit density).
    pub fn estimated_count(&self) -> usize {
        let set_bits: usize = self.bits.iter().map(|w| w.count_ones() as usize).sum();
        let m = self.num_bits as f64;
        let k = self.num_hashes as f64;
        // n ≈ -(m/k) * ln(1 - X/m) where X = set bits
        let ratio = 1.0 - (set_bits as f64 / m);
        if ratio <= 0.0 { return self.num_bits; }
        (-(m / k) * ratio.ln()) as usize
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.bits.len() * 8
    }

    /// FNV-1a inspired hash with seed.
    fn hash(&self, key: &str, seed: usize) -> usize {
        let mut h: u64 = 0xcbf29ce484222325 ^ (seed as u64 * 0x517cc1b727220a95);
        for byte in key.bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_check() {
        let mut bf = BloomFilter::new(1000);
        bf.insert("cat");
        bf.insert("dog");
        bf.insert("bird");

        assert!(bf.maybe_contains("cat"));
        assert!(bf.maybe_contains("dog"));
        assert!(bf.maybe_contains("bird"));
        // "elephant" was never inserted — might still be false positive but unlikely
    }

    #[test]
    fn test_definite_negative() {
        let mut bf = BloomFilter::new(100);
        bf.insert("hello");

        // With a fresh filter and only 1 item, most queries should be negative
        let mut false_positives = 0;
        for i in 0..100 {
            if bf.maybe_contains(&format!("test_{}", i)) {
                false_positives += 1;
            }
        }
        // Should have very few false positives with 100-item capacity and 1 insert
        assert!(false_positives < 10, "Too many false positives: {}", false_positives);
    }

    #[test]
    fn test_memory() {
        let bf = BloomFilter::new(100_000);
        // Should be ~120KB for 100K items
        assert!(bf.memory_bytes() < 200_000);
    }
}
