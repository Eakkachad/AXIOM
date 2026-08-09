//! Incremental Learning — add data on-the-fly, get smarter immediately.
//!
//! This is the key differentiator from LLMs: no retraining needed.
//! User feeds text → Engram + TBA update in real-time → better generation instantly.
//!
//! ## How It Works
//!
//! ```text
//! User: "Bangkok is the capital of Thailand"
//!   │
//!   ├─→ Engram: adds N-gram entries (bangkok→is, is→the, the→capital, ...)
//!   ├─→ TBA: adds transitions to Transition Memory (π(bangkok)⊗is + π(is)⊗the + ...)
//!   └─→ KG: encodes structured triple (bangkok, capital_of, thailand)
//!
//! Next query: "what is the capital of thailand?"
//!   → Engram hits "capital of thailand" → "bangkok" ✓
//! ```

use std::collections::{HashMap, HashSet};
use tle_vsa::{bind, HyperVector, Codebook};

/// Incremental knowledge store that updates both Engram and TBA on-the-fly.
pub struct IncrementalStore {
    /// Live N-gram counts: hash → (token_id → count)
    ngram_counts: Vec<HashMap<u64, HashMap<u16, u32>>>,
    /// Live Transition Memory (VSA vector, updated incrementally)
    pub transition_memory: HyperVector,
    /// Knowledge Graph triples: (subject, relation, object) as VSA bindings
    pub kg_memory: HyperVector,
    /// Codebook for VSA encoding
    pub codebook: Codebook,
    /// Vocabulary
    pub vocab: IncrVocab,
    /// Exact sentence memory: keyword → full sentences containing that keyword
    sentence_memory: HashMap<String, Vec<String>>,
    /// Fact store: subject → list of (relation, object) pairs
    fact_store: HashMap<String, Vec<(String, String)>>,
    /// Insertion order used to retain the newest facts during compaction.
    fact_log: Vec<(String, String, String)>,
    /// Configuration
    config: IncrConfig,
    /// Statistics
    pub stats: IncrStats,
}

/// Configuration for incremental learning.
#[derive(Clone, Debug)]
pub struct IncrConfig {
    /// VSA dimensionality.
    pub dim: usize,
    /// Maximum N-gram order.
    pub max_order: usize,
    /// Codebook seed.
    pub seed: u64,
    /// Run compaction after this many learned facts. Zero disables auto-compaction.
    pub compaction_interval: usize,
    /// Keep at most this many newest facts for each subject.
    pub max_facts_per_subject: usize,
    /// Merge multiple values for the same subject and relation.
    pub merge_same_relation: bool,
}

impl Default for IncrConfig {
    fn default() -> Self {
        Self {
            dim: 2048,
            max_order: 5,
            seed: 0xAB10_CAFE_1234_5678,
            compaction_interval: 10_000,
            max_facts_per_subject: 100,
            merge_same_relation: true,
        }
    }
}

/// Statistics for incremental learning.
#[derive(Clone, Debug, Default)]
pub struct IncrStats {
    pub facts_added: usize,
    pub tokens_ingested: usize,
    pub transitions_added: usize,
    pub vocab_size: usize,
    pub compactions: usize,
    pub facts_pruned: usize,
    pub facts_merged: usize,
}

/// Result of a production knowledge compaction pass.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CompactionReport {
    pub facts_before: usize,
    pub facts_after: usize,
    pub duplicates_removed: usize,
    pub facts_pruned: usize,
    pub facts_merged: usize,
}

/// Simple vocabulary for incremental store.
#[derive(Clone, Debug)]
pub struct IncrVocab {
    pub token_to_id: HashMap<String, u16>,
    pub id_to_token: Vec<String>,
}

impl IncrVocab {
    pub fn new() -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
        }
    }

    pub fn get_or_insert(&mut self, token: &str) -> u16 {
        if let Some(&id) = self.token_to_id.get(token) {
            return id;
        }
        let id = self.id_to_token.len() as u16;
        self.id_to_token.push(token.to_string());
        self.token_to_id.insert(token.to_string(), id);
        id
    }

    pub fn get_id(&self, token: &str) -> Option<u16> {
        self.token_to_id.get(token).copied()
    }

    pub fn get_token(&self, id: u16) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }
}

impl Default for IncrVocab {
    fn default() -> Self {
        Self::new()
    }
}

impl IncrementalStore {
    /// Create a new empty incremental store.
    pub fn new() -> Self {
        Self::with_config(IncrConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: IncrConfig) -> Self {
        let dim = config.dim;
        let codebook = Codebook::new(dim, config.seed);
        Self {
            ngram_counts: (0..config.max_order).map(|_| HashMap::new()).collect(),
            transition_memory: HyperVector::zeros(dim),
            kg_memory: HyperVector::zeros(dim),
            codebook,
            vocab: IncrVocab::new(),
            sentence_memory: HashMap::new(),
            fact_store: HashMap::new(),
            fact_log: Vec::new(),
            config,
            stats: IncrStats::default(),
        }
    }

    /// Learn from a text sentence (updates Engram N-grams + TBA transitions).
    ///
    /// This is the core "get smarter" operation.
    /// Call this with any new text and the system immediately knows about it.
    pub fn learn_text(&mut self, text: &str) {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        if tokens.len() < 2 {
            return;
        }

        // Store full sentence indexed by each keyword (for exact recall)
        let text_lower = text.to_lowercase();
        for token in &tokens {
            let key = token.to_lowercase();
            if key.len() > 2 {
                self.sentence_memory
                    .entry(key)
                    .or_default()
                    .push(text_lower.clone());
                // Keep max 10 sentences per keyword
                let entries = self.sentence_memory.get_mut(&token.to_lowercase()).unwrap();
                if entries.len() > 10 {
                    entries.remove(0);
                }
            }
        }

        // Tokenize
        let ids: Vec<u16> = tokens
            .iter()
            .map(|t| {
                let lower = t.to_lowercase();
                let id = self.vocab.get_or_insert(&lower);
                // Ensure codebook has this token
                self.codebook.get_or_insert(&lower);
                id
            })
            .collect();

        self.stats.tokens_ingested += ids.len();
        self.stats.vocab_size = self.vocab.len();

        // Update N-gram counts
        for pos in 1..ids.len() {
            let next_token = ids[pos];
            let context = &ids[..pos];

            for order in 1..=self.config.max_order {
                if context.len() >= order {
                    let hash = fx_hash_window(&context[context.len() - order..]);
                    *self.ngram_counts[order - 1]
                        .entry(hash)
                        .or_default()
                        .entry(next_token)
                        .or_insert(0) += 1;
                }
            }
        }

        // Update Transition Memory: TM += Σ π(w_i) ⊗ w_{i+1}
        for window in tokens.windows(2) {
            let from_lower = window[0].to_lowercase();
            let to_lower = window[1].to_lowercase();

            if let (Some(from_vec), Some(to_vec)) = (
                self.codebook.get(&from_lower).cloned(),
                self.codebook.get(&to_lower).cloned(),
            ) {
                let shifted = from_vec.permute(1);
                let transition = shifted.hadamard(&to_vec);
                self.transition_memory = self.transition_memory.add(&transition);
                self.stats.transitions_added += 1;
            }
        }
    }

    /// Learn a structured fact triple: (subject, relation, object).
    ///
    /// Encodes as: KG += π²(subject) ⊗ π(relation) ⊗ object
    ///
    /// This allows retrieval:
    ///   query(subject, relation) → object
    ///   query(subject, ?) → lists relations
    pub fn learn_fact(&mut self, subject: &str, relation: &str, object: &str) {
        let subj_lower = subject.to_lowercase();
        let rel_lower = relation.to_lowercase();
        let obj_lower = object.to_lowercase();

        // Store in explicit fact store (for exact retrieval)
        self.fact_store
            .entry(subj_lower.clone())
            .or_default()
            .push((rel_lower.clone(), obj_lower.clone()));
        self.fact_log.push((subj_lower.clone(), rel_lower.clone(), obj_lower.clone()));

        // Ensure all are in codebook
        self.codebook.get_or_insert(&subj_lower);
        self.codebook.get_or_insert(&rel_lower);
        self.codebook.get_or_insert(&obj_lower);

        if let (Some(s_vec), Some(r_vec), Some(o_vec)) = (
            self.codebook.get(&subj_lower).cloned(),
            self.codebook.get(&rel_lower).cloned(),
            self.codebook.get(&obj_lower).cloned(),
        ) {
            // Encode: π²(S) ⊗ π(R) ⊗ O
            let s_shifted = s_vec.permute(2);
            let r_shifted = r_vec.permute(1);
            let binding = s_shifted.hadamard(&r_shifted).hadamard(&o_vec);
            self.kg_memory = self.kg_memory.add(&binding);
            self.stats.facts_added += 1;
        }

        // Also add as text transition for generation
        let sentence = format!("{} {} {}", subject, relation, object);
        self.learn_text(&sentence);

        if self.config.compaction_interval > 0
            && self.stats.facts_added % self.config.compaction_interval == 0
        {
            self.compact_knowledge();
        }
    }

    /// Compact production facts and rebuild the bundled KG memory.
    ///
    /// Exact duplicate facts and older per-subject facts are removed. The KG must
    /// be rebuilt because its VSA representation is a superposition and cannot
    /// subtract individual stale bindings safely.
    pub fn compact_knowledge(&mut self) -> CompactionReport {
        let facts_before = self.fact_log.len();
        if facts_before == 0 {
            return CompactionReport::default();
        }

        let mut seen = HashSet::new();
        let mut subject_counts: HashMap<&str, usize> = HashMap::new();
        let mut retained = Vec::with_capacity(facts_before);
        for fact in self.fact_log.iter().rev() {
            if !seen.insert(fact.clone()) {
                continue;
            }
            let count = subject_counts.entry(fact.0.as_str()).or_insert(0);
            if *count < self.config.max_facts_per_subject.max(1) {
                retained.push(fact.clone());
                *count += 1;
            }
        }
        retained.reverse();

        let facts_before_merge = retained.len();
        if self.config.merge_same_relation {
            let mut positions: HashMap<(String, String), usize> = HashMap::new();
            let mut merged: Vec<(String, String, String)> = Vec::with_capacity(retained.len());
            for (subject, relation, object) in retained {
                let key = (subject.clone(), relation.clone());
                if let Some(&position) = positions.get(&key) {
                    let existing = &mut merged[position].2;
                    if !existing.split("; ").any(|value| value == object) {
                        existing.push_str("; ");
                        existing.push_str(&object);
                    }
                } else {
                    positions.insert(key, merged.len());
                    merged.push((subject, relation, object));
                }
            }
            retained = merged;
        }

        let facts_after = retained.len();
        let report = CompactionReport {
            facts_before,
            facts_after,
            duplicates_removed: facts_before - seen.len(),
            facts_pruned: seen.len() - facts_after,
            facts_merged: facts_before_merge - facts_after,
        };

        self.fact_log = retained;
        self.fact_store.clear();
        self.kg_memory = HyperVector::zeros(self.config.dim);
        let facts = self.fact_log.clone();
        for (subject, relation, object) in facts {
            self.fact_store
                .entry(subject.clone())
                .or_default()
                .push((relation.clone(), object.clone()));
            self.codebook.get_or_insert(&subject);
            self.codebook.get_or_insert(&relation);
            self.codebook.get_or_insert(&object);
            self.encode_fact_into_kg(&subject, &relation, &object);
        }

        self.stats.compactions += 1;
        self.stats.facts_pruned += report.duplicates_removed + report.facts_pruned;
        self.stats.facts_merged += report.facts_merged;
        report
    }

    fn encode_fact_into_kg(&mut self, subject: &str, relation: &str, object: &str) {
        let (Some(s_vec), Some(r_vec), Some(o_vec)) = (
            self.codebook.get(subject).cloned(),
            self.codebook.get(relation).cloned(),
            self.codebook.get(object).cloned(),
        ) else {
            return;
        };
        let binding = s_vec.permute(2).hadamard(&r_vec.permute(1)).hadamard(&o_vec);
        self.kg_memory = self.kg_memory.add(&binding);
    }

    /// Retrieve exact facts about a subject from the fact store.
    ///
    /// Returns all (relation, object) pairs for the given subject.
    /// This is O(1) HashMap lookup — always correct if taught.
    pub fn get_facts(&self, subject: &str) -> Vec<(&str, &str)> {
        match self.fact_store.get(&subject.to_lowercase()) {
            Some(facts) => facts.iter().map(|(r, o)| (r.as_str(), o.as_str())).collect(),
            None => Vec::new(),
        }
    }

    /// Retrieve sentences containing a keyword.
    ///
    /// Returns stored sentences that mention this keyword.
    pub fn get_sentences(&self, keyword: &str) -> Vec<&str> {
        match self.sentence_memory.get(&keyword.to_lowercase()) {
            Some(sentences) => sentences.iter().map(|s| s.as_str()).collect(),
            None => Vec::new(),
        }
    }

    /// Query the knowledge graph: given (subject, relation), retrieve object.
    ///
    /// Unbinds: result = π²(subject) ⊗ π(relation) ⊗ KG
    /// Then finds nearest neighbor in codebook.
    pub fn query_fact(&self, subject: &str, relation: &str) -> Option<(String, f32)> {
        let subj_lower = subject.to_lowercase();
        let rel_lower = relation.to_lowercase();

        let s_vec = self.codebook.get(&subj_lower)?;
        let r_vec = self.codebook.get(&rel_lower)?;

        // Unbind: π²(S) ⊗ π(R) ⊗ KG → should recover O
        let s_shifted = s_vec.permute(2);
        let r_shifted = r_vec.permute(1);
        let query = s_shifted.hadamard(&r_shifted);
        let result = query.hadamard(&self.kg_memory);

        // Find nearest neighbor in codebook
        let mut best_sim = f32::NEG_INFINITY;
        let mut best_token = String::new();

        for i in 0..self.vocab.len() {
            if let Some(token) = self.vocab.get_token(i as u16) {
                // Skip subject and relation themselves
                if token == subj_lower || token == rel_lower {
                    continue;
                }
                if let Some(vec) = self.codebook.get(token) {
                    let sim = tle_vsa::cosine_similarity(&result, vec);
                    if sim > best_sim {
                        best_sim = sim;
                        best_token = token.to_string();
                    }
                }
            }
        }

        if best_sim > 0.0 {
            Some((best_token, best_sim))
        } else {
            None
        }
    }

    /// Query N-gram prediction: given context tokens, what comes next?
    pub fn predict_next(&self, context: &[&str]) -> Vec<(String, f32)> {
        let ids: Vec<u16> = context
            .iter()
            .filter_map(|t| self.vocab.get_id(&t.to_lowercase()))
            .collect();

        if ids.is_empty() {
            return Vec::new();
        }

        // Query from highest order to lowest
        let mut results: HashMap<u16, f32> = HashMap::new();

        for order in (1..=self.config.max_order).rev() {
            if ids.len() >= order {
                let hash = fx_hash_window(&ids[ids.len() - order..]);
                if let Some(counts) = self.ngram_counts[order - 1].get(&hash) {
                    let total: u32 = counts.values().sum();
                    let weight = order as f32; // higher order = more weight
                    for (&token_id, &count) in counts {
                        let prob = count as f32 / total as f32;
                        *results.entry(token_id).or_insert(0.0) += weight * prob;
                    }
                }
            }
        }

        // Sort by score
        let mut sorted: Vec<(String, f32)> = results
            .into_iter()
            .filter_map(|(id, score)| {
                self.vocab.get_token(id).map(|t| (t.to_string(), score))
            })
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(10);
        sorted
    }

    /// Get statistics.
    pub fn stats(&self) -> &IncrStats {
        &self.stats
    }

    /// Get reference to the fact store (for AnalogicalEngine).
    pub fn fact_store(&self) -> &HashMap<String, Vec<(String, String)>> {
        &self.fact_store
    }

    /// Get reference to the codebook (for AnalogicalEngine).
    pub fn codebook(&self) -> &Codebook {
        &self.codebook
    }
}

impl Default for IncrementalStore {
    fn default() -> Self {
        Self::new()
    }
}

/// FxHash for token window (same as in tle-engram).
fn fx_hash_window(tokens: &[u16]) -> u64 {
    const FX_SEED: u64 = 0x517cc1b727220a95;
    let mut hash: u64 = 0;
    for (i, &token) in tokens.iter().enumerate() {
        hash = hash.rotate_left(5);
        hash ^= (token as u64).wrapping_mul(FX_SEED);
        hash = hash.wrapping_add(i as u64);
    }
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51afd7ed558ccd);
    hash ^= hash >> 33;
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learn_and_predict() {
        let mut store = IncrementalStore::new();

        // Teach it
        store.learn_text("the cat sat on the mat");
        store.learn_text("the cat sat on the mat");
        store.learn_text("the cat sat on the mat");

        // Should predict "sat" after "the cat"
        let predictions = store.predict_next(&["the", "cat"]);
        assert!(!predictions.is_empty());
        assert_eq!(predictions[0].0, "sat");
    }

    #[test]
    fn test_learn_fact_and_query() {
        let mut store = IncrementalStore::new();

        store.learn_fact("bangkok", "capital_of", "thailand");
        store.learn_fact("tokyo", "capital_of", "japan");
        store.learn_fact("paris", "capital_of", "france");

        // Query: bangkok capital_of ? → thailand
        let result = store.query_fact("bangkok", "capital_of");
        assert!(result.is_some());
        let (answer, confidence) = result.unwrap();
        assert_eq!(answer, "thailand");
        assert!(confidence > 0.0);
    }

    #[test]
    fn test_incremental_gets_smarter() {
        let mut store = IncrementalStore::new();

        // Initially knows nothing
        let pred0 = store.predict_next(&["the", "sky"]);
        assert!(pred0.is_empty());

        // Learn one thing
        store.learn_text("the sky is blue");
        let pred1 = store.predict_next(&["the", "sky"]);
        assert!(!pred1.is_empty());
        assert_eq!(pred1[0].0, "is");

        // Learn more → gets more options
        store.learn_text("the sky turns red at sunset");
        let pred2 = store.predict_next(&["the", "sky"]);
        assert!(pred2.len() >= 2); // now knows "is" and "turns"
    }

    #[test]
    fn test_stats() {
        let mut store = IncrementalStore::new();
        store.learn_text("hello world foo bar");
        store.learn_fact("rust", "is_a", "language");

        assert_eq!(store.stats.tokens_ingested, 4 + 3); // text + fact-as-text
        assert_eq!(store.stats.facts_added, 1);
        assert!(store.stats.transitions_added > 0);
    }

    #[test]
    fn test_compaction_rebuilds_production_knowledge() {
        let mut store = IncrementalStore::with_config(IncrConfig {
            dim: 128,
            max_order: 3,
            seed: 42,
            compaction_interval: 3,
            max_facts_per_subject: 2,
            merge_same_relation: true,
        });

        store.learn_fact("cat", "has", "four legs");
        store.learn_fact("cat", "likes", "sun");
        store.learn_fact("cat", "says", "meow");

        let facts = store.get_facts("cat");
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0], ("likes", "sun"));
        assert_eq!(facts[1], ("says", "meow"));
        assert_eq!(store.stats.compactions, 1);
        assert_eq!(store.stats.facts_pruned, 1);
        assert!(store.query_fact("cat", "likes").is_some());
    }

    #[test]
    fn test_compaction_merges_values_for_same_relation() {
        let mut store = IncrementalStore::with_config(IncrConfig {
            dim: 128,
            max_order: 3,
            seed: 42,
            compaction_interval: 0,
            max_facts_per_subject: 10,
            merge_same_relation: true,
        });

        store.learn_fact("rust", "supports", "ownership");
        store.learn_fact("rust", "supports", "zero cost abstractions");
        let report = store.compact_knowledge();

        assert_eq!(report.facts_merged, 1);
        assert_eq!(store.get_facts("rust"), vec![("supports", "ownership; zero cost abstractions")]);
        assert!(store.query_fact("rust", "supports").is_some());
    }
}
