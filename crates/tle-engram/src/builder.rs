//! EngramBuilder — corpus ingestion pipeline.
//!
//! Takes raw text, tokenizes, extracts all N-gram patterns with counts,
//! and builds frozen EngramTable instances for each head.

use std::collections::HashMap;

use crate::hash::NgramHash;
use crate::table::{EngramEntry, EngramTable};

/// Vocabulary: maps between tokens (strings) and IDs (u16).
#[derive(Clone, Debug)]
pub struct Vocab {
    /// Token string → ID.
    pub token_to_id: HashMap<String, u16>,
    /// ID → token string.
    pub id_to_token: Vec<String>,
}

impl Vocab {
    /// Create a new empty vocabulary.
    pub fn new() -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
        }
    }

    /// Get or insert a token, returning its ID.
    pub fn get_or_insert(&mut self, token: &str) -> u16 {
        if let Some(&id) = self.token_to_id.get(token) {
            return id;
        }
        let id = self.id_to_token.len() as u16;
        self.id_to_token.push(token.to_string());
        self.token_to_id.insert(token.to_string(), id);
        id
    }

    /// Look up a token's ID.
    pub fn get_id(&self, token: &str) -> Option<u16> {
        self.token_to_id.get(token).copied()
    }

    /// Look up a token by ID.
    pub fn get_token(&self, id: u16) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    /// Vocabulary size.
    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    /// Whether vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for EngramBuilder.
#[derive(Clone, Debug)]
pub struct BuilderConfig {
    /// Maximum N-gram order to extract (1..=5).
    pub max_order: usize,
    /// Minimum count threshold: ignore N-grams seen fewer times.
    pub min_count: u32,
    /// Maximum vocabulary size (0 = unlimited).
    pub max_vocab: usize,
    /// Maximum candidates per entry (prune long tails).
    pub max_candidates_per_entry: usize,
}

impl Default for BuilderConfig {
    fn default() -> Self {
        Self {
            max_order: 5,
            min_count: 1,
            max_vocab: 0, // unlimited
            max_candidates_per_entry: 50,
        }
    }
}

/// Builder that ingests corpus text and produces frozen EngramTables.
pub struct EngramBuilder {
    /// Configuration.
    config: BuilderConfig,
    /// Vocabulary being built.
    pub vocab: Vocab,
    /// Raw counts: counts[order-1][context_hash][next_token] = count.
    counts: Vec<HashMap<u64, HashMap<u16, u32>>>,
    /// The hasher.
    hasher: NgramHash,
    /// Total tokens processed.
    pub total_tokens: usize,
}

impl EngramBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self::with_config(BuilderConfig::default())
    }

    /// Create a new builder with custom configuration.
    pub fn with_config(config: BuilderConfig) -> Self {
        let num_heads = config.max_order;
        let hasher = NgramHash::new(num_heads);
        Self {
            config,
            vocab: Vocab::new(),
            counts: (0..num_heads).map(|_| HashMap::new()).collect(),
            hasher,
            total_tokens: 0,
        }
    }

    /// Ingest a line of text (tokenized by whitespace).
    pub fn ingest_line(&mut self, line: &str) {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 2 {
            return;
        }

        // Convert to IDs
        let ids: Vec<u16> = tokens.iter().map(|t| self.vocab.get_or_insert(t)).collect();
        self.ingest_ids(&ids);
    }

    /// Ingest a sequence of token IDs.
    pub fn ingest_ids(&mut self, ids: &[u16]) {
        if ids.len() < 2 {
            return;
        }

        self.total_tokens += ids.len();

        // For each position, extract N-gram context → next token
        for pos in 1..ids.len() {
            let next_token = ids[pos];
            let context = &ids[..pos];

            // For each head (order), hash the context and record the next token
            for order in 1..=self.config.max_order {
                if context.len() >= order {
                    if let Some(hash) = self.hasher.hash_head(context, order) {
                        *self.counts[order - 1]
                            .entry(hash)
                            .or_default()
                            .entry(next_token)
                            .or_insert(0) += 1;
                    }
                }
            }
        }
    }

    /// Ingest a full text corpus (multi-line).
    pub fn ingest_corpus(&mut self, text: &str) {
        for line in text.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                self.ingest_line(trimmed);
            }
        }
    }

    /// Build frozen EngramTables from accumulated counts.
    ///
    /// Returns one table per head (order), plus the vocabulary.
    pub fn build(self) -> BuiltEngram {
        let mut tables = Vec::with_capacity(self.config.max_order);

        for (order_idx, head_counts) in self.counts.into_iter().enumerate() {
            let order = order_idx + 1;
            let mut table = EngramTable::with_capacity(order, head_counts.len());

            for (hash, token_counts) in head_counts {
                // Filter by min_count
                let filtered: Vec<(u16, u32)> = token_counts
                    .into_iter()
                    .filter(|(_, count)| *count >= self.config.min_count)
                    .collect();

                if filtered.is_empty() {
                    continue;
                }

                // Build entry (auto-sorts by count, truncates)
                let mut entry = EngramEntry::from_counts(&filtered);

                // Truncate to max candidates
                if entry.candidates.len() > self.config.max_candidates_per_entry {
                    entry.candidates.truncate(self.config.max_candidates_per_entry);
                    entry.scores.truncate(self.config.max_candidates_per_entry);
                }

                table.insert(hash, entry);
            }

            tables.push(table);
        }

        BuiltEngram {
            tables,
            vocab: self.vocab,
            total_tokens: self.total_tokens,
            hasher: self.hasher,
        }
    }
}

impl Default for EngramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// The result of building: frozen tables + vocabulary + hasher.
pub struct BuiltEngram {
    /// One frozen table per N-gram head (index 0 = unigram, etc.).
    pub tables: Vec<EngramTable>,
    /// The vocabulary built during ingestion.
    pub vocab: Vocab,
    /// Total tokens ingested.
    pub total_tokens: usize,
    /// The hasher (for querying).
    pub hasher: NgramHash,
}

impl BuiltEngram {
    /// Query all heads for a given context, returning raw results per head.
    ///
    /// Returns: Vec<(order, confidence, &EngramEntry)> for each head that matched.
    pub fn query(&self, context: &[u16]) -> Vec<(usize, f32, &EngramEntry)> {
        let mut results = Vec::new();

        for (head_idx, key) in self.hasher.hash_all_heads(context) {
            let order = head_idx + 1;
            if let Some(entry) = self.tables[head_idx].lookup(key) {
                let confidence = entry.confidence(3.0);
                results.push((order, confidence, entry));
            }
        }

        results
    }

    /// Print statistics for all tables.
    pub fn print_stats(&self) {
        println!("=== Engram Statistics ===");
        println!("Vocabulary size: {}", self.vocab.len());
        println!("Total tokens ingested: {}", self.total_tokens);
        for table in &self.tables {
            println!("  {}", table.stats());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_basic() {
        let mut builder = EngramBuilder::new();
        builder.ingest_line("the cat sat on the mat");
        builder.ingest_line("the cat ran to the house");

        let engram = builder.build();

        assert!(engram.vocab.len() >= 7); // the, cat, sat, on, mat, ran, to, house
        assert!(!engram.tables[0].is_empty()); // unigram table should have entries
        assert!(!engram.tables[1].is_empty()); // bigram table should have entries
    }

    #[test]
    fn test_builder_query() {
        let mut builder = EngramBuilder::new();
        // Repeat to build strong signal
        for _ in 0..10 {
            builder.ingest_line("the cat sat on the mat");
        }

        let engram = builder.build();

        // Query with context "the cat" → should predict "sat"
        let the_id = engram.vocab.get_id("the").unwrap();
        let cat_id = engram.vocab.get_id("cat").unwrap();
        let sat_id = engram.vocab.get_id("sat").unwrap();

        let context = vec![the_id, cat_id];
        let results = engram.query(&context);

        // Should have at least bigram match
        assert!(!results.is_empty());

        // Find the bigram result and check it predicts "sat"
        let bigram_result = results.iter().find(|(order, _, _)| *order == 2);
        assert!(bigram_result.is_some());
        let (_, _, entry) = bigram_result.unwrap();
        assert_eq!(entry.candidates[0], sat_id);
    }

    #[test]
    fn test_vocab() {
        let mut vocab = Vocab::new();
        let id1 = vocab.get_or_insert("hello");
        let id2 = vocab.get_or_insert("world");
        let id1_again = vocab.get_or_insert("hello");

        assert_eq!(id1, id1_again);
        assert_ne!(id1, id2);
        assert_eq!(vocab.get_token(id1), Some("hello"));
        assert_eq!(vocab.len(), 2);
    }

    #[test]
    fn test_corpus_ingestion() {
        let corpus = "the sun is bright\nthe moon is dark\nthe star is far";
        let mut builder = EngramBuilder::new();
        builder.ingest_corpus(corpus);

        let engram = builder.build();
        assert!(engram.vocab.len() >= 8);
        assert!(engram.total_tokens >= 12);
    }
}
