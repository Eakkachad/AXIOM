//! EngramTable — the core frozen hash table for N-gram pattern storage.
//!
//! Stores N-gram context → candidate next-tokens with probability scores.
//! Once built, the table is immutable (frozen) for deterministic inference.

use std::collections::HashMap;

/// A single entry in the Engram table: candidates for a given context hash.
#[derive(Clone, Debug)]
pub struct EngramEntry {
    /// Candidate token IDs sorted by probability (highest first).
    pub candidates: Vec<u16>,
    /// Log-probability scores (parallel to candidates).
    pub scores: Vec<f32>,
    /// Total observation count for this context (used for confidence).
    pub count: u32,
}

impl EngramEntry {
    /// Create a new entry from candidates and their counts.
    pub fn from_counts(token_counts: &[(u16, u32)]) -> Self {
        let total: u32 = token_counts.iter().map(|(_, c)| c).sum();
        if total == 0 {
            return Self {
                candidates: Vec::new(),
                scores: Vec::new(),
                count: 0,
            };
        }

        // Sort by count descending
        let mut sorted: Vec<(u16, u32)> = token_counts.to_vec();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        let candidates: Vec<u16> = sorted.iter().map(|(id, _)| *id).collect();
        let scores: Vec<f32> = sorted
            .iter()
            .map(|(_, c)| (*c as f32 / total as f32).ln())
            .collect();

        Self {
            candidates,
            scores,
            count: total,
        }
    }

    /// Get the top-k candidates with their log-probabilities.
    pub fn top_k(&self, k: usize) -> &[u16] {
        &self.candidates[..k.min(self.candidates.len())]
    }

    /// Get the confidence of this entry (based on observation count).
    /// Higher count = more confident. Returns sigmoid(ln(count) - threshold).
    pub fn confidence(&self, threshold: f32) -> f32 {
        let x = (self.count as f32).ln() - threshold;
        1.0 / (1.0 + (-x).exp())
    }

    /// Number of unique candidates.
    pub fn num_candidates(&self) -> usize {
        self.candidates.len()
    }
}

/// The frozen Engram table: one per N-gram head.
///
/// Maps context hashes to EngramEntry (candidate distributions).
/// Immutable after construction for deterministic inference.
#[derive(Clone)]
pub struct EngramTable {
    /// The underlying hash map: context_hash → entry.
    entries: HashMap<u64, EngramEntry>,
    /// N-gram order this table represents (1=unigram, 2=bigram, etc.).
    pub order: usize,
    /// Total number of entries stored.
    num_entries: usize,
}

impl EngramTable {
    /// Create a new empty table for the given N-gram order.
    pub fn new(order: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order,
            num_entries: 0,
        }
    }

    /// Create a table with pre-allocated capacity.
    pub fn with_capacity(order: usize, capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            order,
            num_entries: 0,
        }
    }

    /// Insert an entry for a context hash.
    pub fn insert(&mut self, hash: u64, entry: EngramEntry) {
        self.entries.insert(hash, entry);
        self.num_entries = self.entries.len();
    }

    /// Look up candidates for a context hash.
    ///
    /// Returns None if the context was never seen during building.
    #[inline]
    pub fn lookup(&self, hash: u64) -> Option<&EngramEntry> {
        self.entries.get(&hash)
    }

    /// Number of distinct contexts stored.
    pub fn len(&self) -> usize {
        self.num_entries
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.num_entries == 0
    }

    /// Memory usage estimate in bytes.
    pub fn memory_bytes(&self) -> usize {
        let entry_overhead = std::mem::size_of::<(u64, EngramEntry)>();
        let avg_candidates: usize = if self.num_entries > 0 {
            self.entries.values().map(|e| e.candidates.len()).sum::<usize>() / self.num_entries
        } else {
            0
        };
        let per_entry = entry_overhead + avg_candidates * (2 + 4); // u16 + f32
        self.num_entries * per_entry
    }

    /// Statistics about the table.
    pub fn stats(&self) -> TableStats {
        if self.entries.is_empty() {
            return TableStats {
                order: self.order,
                num_contexts: 0,
                total_candidates: 0,
                avg_candidates_per_context: 0.0,
                max_candidates: 0,
                memory_kb: 0.0,
            };
        }

        let total_candidates: usize = self.entries.values().map(|e| e.candidates.len()).sum();
        let max_candidates = self.entries.values().map(|e| e.candidates.len()).max().unwrap_or(0);

        TableStats {
            order: self.order,
            num_contexts: self.num_entries,
            total_candidates,
            avg_candidates_per_context: total_candidates as f64 / self.num_entries as f64,
            max_candidates,
            memory_kb: self.memory_bytes() as f64 / 1024.0,
        }
    }
}

/// Statistics about an EngramTable.
#[derive(Clone, Debug)]
pub struct TableStats {
    pub order: usize,
    pub num_contexts: usize,
    pub total_candidates: usize,
    pub avg_candidates_per_context: f64,
    pub max_candidates: usize,
    pub memory_kb: f64,
}

impl std::fmt::Display for TableStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Head {} ({}-gram): {} contexts, avg {:.1} candidates, {:.1} KB",
            self.order,
            self.order,
            self.num_contexts,
            self.avg_candidates_per_context,
            self.memory_kb
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entry_from_counts() {
        let counts = vec![(5u16, 10u32), (3, 5), (7, 2)];
        let entry = EngramEntry::from_counts(&counts);

        assert_eq!(entry.candidates[0], 5); // highest count first
        assert_eq!(entry.candidates[1], 3);
        assert_eq!(entry.candidates[2], 7);
        assert_eq!(entry.count, 17);
        assert!(entry.scores[0] > entry.scores[1]); // higher prob = less negative log
    }

    #[test]
    fn test_table_lookup() {
        let mut table = EngramTable::new(2);
        let entry = EngramEntry::from_counts(&[(1, 10), (2, 5)]);
        table.insert(12345, entry);

        assert!(table.lookup(12345).is_some());
        assert!(table.lookup(99999).is_none());
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn test_confidence() {
        let entry = EngramEntry {
            candidates: vec![1],
            scores: vec![0.0],
            count: 100,
        };
        // ln(100) ≈ 4.6, with threshold=3 → sigmoid(1.6) ≈ 0.83
        let conf = entry.confidence(3.0);
        assert!(conf > 0.7 && conf < 0.9);

        let low_entry = EngramEntry {
            candidates: vec![1],
            scores: vec![0.0],
            count: 2,
        };
        // ln(2) ≈ 0.69, with threshold=3 → sigmoid(-2.3) ≈ 0.09
        let low_conf = low_entry.confidence(3.0);
        assert!(low_conf < 0.2);
    }
}
