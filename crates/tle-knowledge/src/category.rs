//! Category Index — groups facts by topic for efficient retrieval.
//!
//! Facts are automatically categorized by subject similarity.
//! Each category has its own set of bundles.

use std::collections::HashMap;
use tle_vsa::{cosine_similarity, HyperVector, Codebook};

use crate::bundle::{KnowledgeBundle, encode_fact};

/// A category groups related facts together.
pub struct Category {
    /// Category name (e.g., "animals", "geography", "science").
    pub name: String,
    /// Prototype vector — average of all subjects in this category.
    pub prototype: HyperVector,
    /// Bundles of facts in this category.
    pub bundles: Vec<KnowledgeBundle>,
    /// Number of facts in this category.
    pub fact_count: usize,
    /// Dimensionality.
    dim: usize,
}

impl Category {
    /// Create a new category.
    pub fn new(name: &str, dim: usize) -> Self {
        Self {
            name: name.to_string(),
            prototype: HyperVector::zeros(dim),
            bundles: vec![KnowledgeBundle::new(dim)],
            fact_count: 0,
            dim,
        }
    }

    /// Add a fact to this category.
    pub fn add_fact(&mut self, encoded: &HyperVector, subject_vec: &HyperVector) {
        // Update prototype (running average)
        let n = (self.fact_count + 1) as f32;
        let old_weight = (n - 1.0) / n;
        let new_weight = 1.0 / n;
        self.prototype = self.prototype.scale(old_weight).add(&subject_vec.scale(new_weight));

        // Add to current bundle (or create new one if full)
        let last = self.bundles.last_mut().unwrap();
        if !last.add(encoded) {
            // Bundle full — create new one
            let mut new_bundle = KnowledgeBundle::new(self.dim);
            new_bundle.add(encoded);
            self.bundles.push(new_bundle);
        }

        self.fact_count += 1;
    }

    /// How similar is a query to this category?
    pub fn relevance(&self, query: &HyperVector) -> f32 {
        cosine_similarity(&self.prototype, query)
    }

    /// Total memory usage.
    pub fn memory_bytes(&self) -> usize {
        self.dim * 4 + // prototype
        self.bundles.iter().map(|b| b.memory_bytes()).sum::<usize>()
    }
}

/// Category Index — routes facts to appropriate categories.
pub struct CategoryIndex {
    /// All categories.
    pub categories: Vec<Category>,
    /// Threshold for matching to existing category.
    pub similarity_threshold: f32,
    /// Dimensionality.
    dim: usize,
    /// Maximum categories before forced merge.
    max_categories: usize,
}

impl CategoryIndex {
    /// Create a new category index.
    pub fn new(dim: usize) -> Self {
        Self {
            categories: Vec::new(),
            similarity_threshold: 0.15,
            dim,
            max_categories: 500,
        }
    }

    /// Find the best matching category for a subject vector.
    ///
    /// Returns category index, or None if no match (should create new).
    pub fn find_category(&self, subject_vec: &HyperVector) -> Option<usize> {
        let mut best_idx = None;
        let mut best_sim = self.similarity_threshold;

        for (i, cat) in self.categories.iter().enumerate() {
            let sim = cat.relevance(subject_vec);
            if sim > best_sim {
                best_sim = sim;
                best_idx = Some(i);
            }
        }

        best_idx
    }

    /// Add a fact — automatically categorize.
    pub fn add_fact(
        &mut self,
        subject: &str,
        relation: &str,
        object: &str,
        codebook: &mut Codebook,
    ) {
        let subject_vec = codebook.get_or_insert(subject).clone();
        let encoded = encode_fact(subject, relation, object, codebook);

        // Find or create category
        match self.find_category(&subject_vec) {
            Some(idx) => {
                self.categories[idx].add_fact(&encoded, &subject_vec);
            }
            None => {
                // Create new category named after the subject
                if self.categories.len() < self.max_categories {
                    let mut cat = Category::new(subject, self.dim);
                    cat.add_fact(&encoded, &subject_vec);
                    self.categories.push(cat);
                } else {
                    // Forced into least-full category
                    let min_idx = self.categories.iter().enumerate()
                        .min_by_key(|(_, c)| c.fact_count)
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    self.categories[min_idx].add_fact(&encoded, &subject_vec);
                }
            }
        }
    }

    /// Query: find relevant categories for a query.
    pub fn find_relevant(&self, query_vec: &HyperVector, top_k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = self.categories.iter().enumerate()
            .map(|(i, cat)| (i, cat.relevance(query_vec)))
            .filter(|(_, sim)| *sim > 0.05)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// Total facts stored.
    pub fn total_facts(&self) -> usize {
        self.categories.iter().map(|c| c.fact_count).sum()
    }

    /// Total memory usage.
    pub fn memory_bytes(&self) -> usize {
        self.categories.iter().map(|c| c.memory_bytes()).sum()
    }

    /// Number of categories.
    pub fn num_categories(&self) -> usize {
        self.categories.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_categorize() {
        let dim = 2048;
        let mut codebook = Codebook::new(dim, 42);
        let mut index = CategoryIndex::new(dim);

        index.add_fact("cat", "is", "animal", &mut codebook);
        index.add_fact("dog", "is", "animal", &mut codebook);
        index.add_fact("paris", "is", "city", &mut codebook);

        assert!(index.total_facts() == 3);
        assert!(index.num_categories() >= 1);
    }

    #[test]
    fn test_memory_efficiency() {
        let dim = 512; // small for test speed
        let mut codebook = Codebook::new(dim, 42);
        let mut index = CategoryIndex::new(dim);

        // Add 100 facts (fast test)
        for i in 0..100 {
            index.add_fact(
                &format!("entity_{}", i),
                "has",
                &format!("property_{}", i % 10),
                &mut codebook,
            );
        }

        let memory = index.memory_bytes();
        let memory_kb = memory / 1024;
        assert!(memory_kb < 5000, "Memory too high: {} KB for 100 facts", memory_kb);
    }
}
