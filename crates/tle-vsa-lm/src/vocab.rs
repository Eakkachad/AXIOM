//! Vocabulary: deterministic word ↔ id mapping backed by a VSA codebook.
//!
//! Each word receives a fixed bipolar hypervector via `tle_vsa::Codebook`,
//! so "embedding" is pure hashing — no training, fully reproducible.

use std::collections::HashMap;

use tle_vsa::{Codebook, HyperVector};

/// Bidirectional word vocabulary with deterministic VSA vectors.
#[derive(Clone)]
pub struct Vocab {
    word_to_id: HashMap<String, usize>,
    id_to_word: Vec<String>,
    codebook: Codebook,
}

impl std::fmt::Debug for Vocab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vocab")
            .field("words", &self.id_to_word.len())
            .field("dim", &self.codebook.dim())
            .finish()
    }
}

impl Vocab {
    /// Create an empty vocabulary with the given vector dimensionality.
    pub fn new(dim: usize, seed: u64) -> Self {
        Self {
            word_to_id: HashMap::new(),
            id_to_word: Vec::new(),
            codebook: Codebook::new(dim, seed),
        }
    }

    /// Add a word if new, returning its stable id.
    pub fn get_or_add(&mut self, word: &str) -> usize {
        if let Some(&id) = self.word_to_id.get(word) {
            return id;
        }
        self.codebook.get_or_insert(word);
        let id = self.id_to_word.len();
        self.id_to_word.push(word.to_string());
        self.word_to_id.insert(word.to_string(), id);
        id
    }

    /// Id of a word if it exists.
    pub fn id(&self, word: &str) -> Option<usize> {
        self.word_to_id.get(word).copied()
    }

    /// Word for an id.
    pub fn word(&self, id: usize) -> &str {
        &self.id_to_word[id]
    }

    /// VSA vector for a word (must be registered).
    pub fn vector(&self, word: &str) -> Option<&HyperVector> {
        self.codebook.get(word)
    }

    /// VSA vector for an id.
    pub fn vector_by_id(&self, id: usize) -> Option<&HyperVector> {
        self.codebook.get(&self.id_to_word[id])
    }

    /// Number of registered words.
    pub fn len(&self) -> usize {
        self.id_to_word.len()
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_word.is_empty()
    }

    /// Iterate over (id, word) pairs in registration order (deterministic).
    pub fn iter(&self) -> impl Iterator<Item = (usize, &str)> {
        self.id_to_word.iter().enumerate().map(|(i, w)| (i, w.as_str()))
    }

    /// Dimensionality of the underlying vectors.
    pub fn dim(&self) -> usize {
        self.codebook.dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_roundtrip() {
        let mut v = Vocab::new(1024, 42);
        let cat = v.get_or_add("cat");
        let dog = v.get_or_add("dog");
        assert_eq!(v.word(cat), "cat");
        assert_eq!(v.word(dog), "dog");
        assert_eq!(v.id("cat"), Some(cat));
        assert_eq!(v.id("unknown"), None);
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_vocab_vectors_are_deterministic_and_orthogonal() {
        let mut v1 = Vocab::new(1024, 7);
        let mut v2 = Vocab::new(1024, 7);
        let a1 = v1.get_or_add("sky");
        let a2 = v2.get_or_add("sky");
        assert_eq!(v1.vector_by_id(a1), v2.vector_by_id(a2));

        let b = v1.get_or_add("blue");
        let sim = tle_vsa::cosine_similarity(v1.vector_by_id(a1).unwrap(), v1.vector_by_id(b).unwrap());
        assert!(sim.abs() < 0.1, "different words should be quasi-orthogonal, got {}", sim);
    }
}
