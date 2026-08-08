//! CompressedKnowledgeStore — the main interface for storing and querying knowledge.
//!
//! Combines: Bloom filter (fast check) + CategoryIndex (VSA bundles) + Exact store (HashMap)

use std::collections::HashMap;
use tle_vsa::{cosine_similarity, Codebook, HyperVector};

use crate::bloom::BloomFilter;
use crate::bundle::{encode_fact, query_bundle};
use crate::category::CategoryIndex;

/// Configuration for the knowledge store.
#[derive(Clone, Debug)]
pub struct StoreConfig {
    /// VSA dimensionality.
    pub dim: usize,
    /// Codebook seed.
    pub seed: u64,
    /// Expected maximum number of facts.
    pub expected_facts: usize,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            dim: 4096,
            seed: 0xAA10_CAFE_DEAD_BEEF,
            expected_facts: 200_000,
        }
    }
}

/// The main compressed knowledge store.
///
/// Three tiers:
/// 1. Bloom filter: O(1) "do I know about X?" (no false negatives)
/// 2. Exact store: HashMap for precise taught facts (always correct)
/// 3. Category VSA: hierarchical bundles for semantic retrieval
pub struct CompressedKnowledgeStore {
    /// Bloom filter for fast existence check.
    bloom: BloomFilter,
    /// Exact fact store: subject → [(relation, object)]
    exact: HashMap<String, Vec<(String, String)>>,
    /// Category-indexed VSA bundles.
    categories: CategoryIndex,
    /// Codebook for VSA encoding.
    pub codebook: Codebook,
    /// Configuration.
    config: StoreConfig,
    /// Statistics.
    pub stats: StoreStats,
}

/// Statistics about the knowledge store.
#[derive(Clone, Debug, Default)]
pub struct StoreStats {
    pub total_facts: usize,
    pub exact_facts: usize,
    pub vsa_facts: usize,
    pub num_categories: usize,
    pub bloom_checks: usize,
    pub bloom_hits: usize,
}

impl CompressedKnowledgeStore {
    /// Create a new knowledge store with default config.
    pub fn new() -> Self {
        Self::with_config(StoreConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: StoreConfig) -> Self {
        Self {
            bloom: BloomFilter::new(config.expected_facts),
            exact: HashMap::new(),
            categories: CategoryIndex::new(config.dim),
            codebook: Codebook::new(config.dim, config.seed),
            config,
            stats: StoreStats::default(),
        }
    }

    /// Store a fact. Goes into all three tiers.
    pub fn store_fact(&mut self, subject: &str, relation: &str, object: &str) {
        let subj_lower = subject.to_lowercase();
        let rel_lower = relation.to_lowercase();
        let obj_lower = object.to_lowercase();

        // Tier 1: Bloom filter
        self.bloom.insert(&subj_lower);

        // Tier 2: Exact store
        self.exact
            .entry(subj_lower.clone())
            .or_default()
            .push((rel_lower.clone(), obj_lower.clone()));
        self.stats.exact_facts += 1;

        // Tier 3: VSA category bundles
        self.categories.add_fact(&subj_lower, &rel_lower, &obj_lower, &mut self.codebook);
        self.stats.vsa_facts += 1;

        self.stats.total_facts += 1;
        self.stats.num_categories = self.categories.num_categories();
    }

    /// Query: get all known facts about a subject.
    ///
    /// Returns: Vec<(relation, object)> — exact matches first.
    pub fn query_subject(&mut self, subject: &str) -> Vec<(String, String)> {
        let subj_lower = subject.to_lowercase();

        // Fast check: do we know anything about this?
        self.stats.bloom_checks += 1;
        if !self.bloom.maybe_contains(&subj_lower) {
            return Vec::new(); // definitely don't know
        }
        self.stats.bloom_hits += 1;

        // Tier 2: Exact recall
        if let Some(facts) = self.exact.get(&subj_lower) {
            return facts.clone();
        }

        Vec::new()
    }

    /// Query: get a specific fact (subject, relation) → object.
    pub fn query_fact(&mut self, subject: &str, relation: &str) -> Option<String> {
        let subj_lower = subject.to_lowercase();
        let rel_lower = relation.to_lowercase();

        // Bloom check
        self.stats.bloom_checks += 1;
        if !self.bloom.maybe_contains(&subj_lower) {
            return None;
        }
        self.stats.bloom_hits += 1;

        // Exact store
        if let Some(facts) = self.exact.get(&subj_lower) {
            for (rel, obj) in facts {
                if *rel == rel_lower {
                    return Some(obj.clone());
                }
            }
        }

        // VSA retrieval (fuzzy)
        self.query_vsa(&subj_lower, &rel_lower)
    }

    /// VSA-based fuzzy retrieval.
    fn query_vsa(&self, subject: &str, relation: &str) -> Option<String> {
        let subject_vec = self.codebook.get(subject)?;

        // Find relevant categories
        let relevant = self.categories.find_relevant(subject_vec, 3);

        // Search bundles in relevant categories
        for (cat_idx, _sim) in relevant {
            let cat = &self.categories.categories[cat_idx];
            for bundle in &cat.bundles {
                if let Some(result) = query_bundle(subject, relation, bundle, &self.codebook) {
                    // Find nearest entity in codebook to the result
                    // (result should be ≈ ρ(object), so inv_permute first)
                    let unshifted = result.inv_permute(1);

                    // TODO: search codebook for nearest match
                    // For now, return None (exact store handles most cases)
                    let _ = unshifted;
                }
            }
        }

        None
    }

    /// Check if we might know about a subject (Bloom filter).
    pub fn might_know(&self, subject: &str) -> bool {
        self.bloom.maybe_contains(&subject.to_lowercase())
    }

    /// Total memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.bloom.memory_bytes()
        + self.categories.memory_bytes()
        + self.exact.len() * 100 // rough estimate for HashMap
    }

    /// Memory usage in KB.
    pub fn memory_kb(&self) -> f64 {
        self.memory_bytes() as f64 / 1024.0
    }

    /// Print stats.
    pub fn print_stats(&self) {
        println!("  Knowledge Store:");
        println!("    Total facts: {}", self.stats.total_facts);
        println!("    Categories: {}", self.stats.num_categories);
        println!("    Memory: {:.1} KB", self.memory_kb());
        println!("    Bloom hit rate: {:.1}%",
            if self.stats.bloom_checks > 0 {
                self.stats.bloom_hits as f64 / self.stats.bloom_checks as f64 * 100.0
            } else { 0.0 });
    }
}

impl Default for CompressedKnowledgeStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_query() {
        let mut store = CompressedKnowledgeStore::new();
        store.store_fact("cat", "is", "animal");
        store.store_fact("cat", "has", "four legs");
        store.store_fact("dog", "is", "animal");

        let cat_facts = store.query_subject("cat");
        assert_eq!(cat_facts.len(), 2);
        assert_eq!(cat_facts[0], ("is".to_string(), "animal".to_string()));

        let result = store.query_fact("cat", "is");
        assert_eq!(result, Some("animal".to_string()));
    }

    #[test]
    fn test_bloom_negative() {
        let mut store = CompressedKnowledgeStore::new();
        store.store_fact("cat", "is", "animal");

        // "elephant" was never stored — Bloom should catch it
        let facts = store.query_subject("xyz_unknown_entity");
        assert!(facts.is_empty());
    }

    #[test]
    fn test_memory_scaling() {
        let config = StoreConfig {
            dim: 512, // small for test speed
            seed: 42,
            expected_facts: 200_000,
        };
        let mut store = CompressedKnowledgeStore::with_config(config);

        // Add 200 facts (fast test)
        for i in 0..200 {
            store.store_fact(
                &format!("entity_{}", i),
                "has",
                &format!("property_{}", i % 20),
            );
        }

        let memory_mb = store.memory_bytes() as f64 / (1024.0 * 1024.0);
        println!("200 facts → {:.2} MB", memory_mb);
        assert!(memory_mb < 16.0, "Memory too high: {:.2} MB", memory_mb);
    }

    #[test]
    fn test_might_know() {
        let mut store = CompressedKnowledgeStore::new();
        store.store_fact("tokyo", "is", "capital of japan");

        assert!(store.might_know("tokyo"));
        // Unknown entities should mostly return false
        // (some false positives possible with Bloom)
    }
}
