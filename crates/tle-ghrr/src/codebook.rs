//! GHRR deterministic relation codebook.
//!
//! Maps a relation name (or symbol) to a fixed GhrrVector, constructed from a
//! per-(symbol, block) seed so the whole codebook is reproducible across runs.

use std::collections::HashMap;

use crate::block::{D_BLOCKS, UnitaryBlock, random_orthogonal_block};
use crate::vector::GhrrVector;

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn fnv1a(s: &str) -> u64 {
    let mut h = FNV_OFFSET;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Deterministic relation/symbol codebook.
#[derive(Debug, Clone)]
pub struct GhrrCodebook {
    base_seed: u64,
    entries: HashMap<String, GhrrVector>,
}

impl GhrrCodebook {
    pub fn new(base_seed: u64) -> Self {
        Self { base_seed, entries: HashMap::new() }
    }

    /// Get or deterministically create the vector for a symbol. Same symbol +
    /// same base_seed ⇒ same vector (across runs and across codebooks).
    pub fn get_or_insert(&mut self, symbol: &str) -> GhrrVector {
        if let Some(v) = self.entries.get(symbol) {
            return v.clone();
        }
        let sym_seed = fnv1a(symbol) ^ self.base_seed;
        let blocks: Vec<UnitaryBlock> = (0..D_BLOCKS)
            .map(|j| random_orthogonal_block(sym_seed ^ (j as u64).wrapping_mul(0x9E3779B97F4A7C15)))
            .collect();
        let v = GhrrVector::from_blocks(blocks);
        self.entries.insert(symbol.to_string(), v.clone());
        v
    }

    /// Number of cached symbols.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_deterministic_and_seed_sensitive() {
        let mut c1 = GhrrCodebook::new(7);
        let mut c2 = GhrrCodebook::new(7);
        let mut c3 = GhrrCodebook::new(9);
        let a1 = c1.get_or_insert("located_in");
        let a2 = c2.get_or_insert("located_in");
        let a3 = c3.get_or_insert("located_in");
        assert_eq!(a1, a2, "same base_seed must give identical vectors");
        assert_ne!(a1, a3, "different base_seed must differ");
    }

    #[test]
    fn distinct_symbols_quasi_orthogonal() {
        let mut cb = GhrrCodebook::new(1);
        let a = cb.get_or_insert("capital_of");
        let b = cb.get_or_insert("located_in");
        assert!(a.blockwise_cosine(&b).abs() < 0.15);
    }

    #[test]
    fn symbols_cache() {
        let mut cb = GhrrCodebook::new(1);
        cb.get_or_insert("won");
        assert_eq!(cb.len(), 1);
        cb.get_or_insert("won");
        assert_eq!(cb.len(), 1);
    }
}
