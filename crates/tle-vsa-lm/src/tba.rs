//! Transition Binding Algebra (TBA): VSA transition memory for next-token
//! prediction without any neural readout.
//!
//! The transition memory is a bundled superposition of bigram bindings:
//!
//! ```text
//! TM = Σ_i  ρ(C(w_i)) ⊙ C(w_{i+1})
//! ```
//!
//! To predict the next token after `current`, unbind the shifted current
//! vector from the memory and measure similarity against every candidate:
//!
//! ```text
//! pred = ρ(C(current)) ⊙ TM
//! score(next) = cos(pred, C(next))
//! ```
//!
//! This is the "algebraic softmax": the decoder is a similarity lookup over
//! the codebook, not a learned projection matrix.

use std::collections::HashMap;

use tle_vsa::{cosine_similarity, HyperVector};

/// A bundled bigram transition memory for one direction of a token stream.
///
/// Per-source-word vectors eliminate crosstalk: TM["the"] bundles all next
/// tokens after "the", TM["cat"] bundles all next tokens after "cat".  The
/// prediction is a direct lookup — no global unbinding needed.
#[derive(Debug, Clone)]
pub struct TransitionMemory {
    per_word: HashMap<usize, HyperVector>,
    dim: usize,
    pub transitions: u64,
}

impl TransitionMemory {
    pub fn new(dim: usize) -> Self {
        Self { per_word: HashMap::new(), dim, transitions: 0 }
    }

    /// Add a transition: current → next.
    pub fn learn(&mut self, current_id: usize, next: &HyperVector) {
        let entry = self
            .per_word
            .entry(current_id)
            .or_insert_with(|| HyperVector::zeros(self.dim));
        *entry = entry.add(next);
        self.transitions += 1;
    }

    /// Learn all adjacent pairs in a token-id sequence.
    pub fn learn_ids(&mut self, ids: &[usize], _vocab: &crate::vocab::Vocab) {
        for window in ids.windows(2) {
            if let Some(next) = _vocab.vector_by_id(window[1]) {
                self.learn(window[0], next);
            }
        }
    }

    /// Predict the next-token bundle after `current_id`.
    /// Returns None if the word was never seen as a transition source.
    pub fn predict(&self, current_id: usize) -> Option<HyperVector> {
        self.per_word.get(&current_id).cloned()
    }

    /// Score a candidate as the next token after `current_id`.
    pub fn score(&self, current_id: usize, candidate: &HyperVector) -> f32 {
        match self.predict(current_id) {
            Some(bundle) => cosine_similarity(&bundle, candidate),
            None => 0.0,
        }
    }
}

/// Trigram Transition Memory: per-(prev,current) pair vectors.
///
/// Each unique (prev_id, current_id) pair stores the bundled next-token
/// vectors seen after that context.  No crosstalk between different source
/// pairs.
#[derive(Debug, Clone)]
pub struct TrigramMemory {
    per_pair: HashMap<(usize, usize), HyperVector>,
    dim: usize,
    pub transitions: u64,
}

impl TrigramMemory {
    pub fn new(dim: usize) -> Self {
        Self { per_pair: HashMap::new(), dim, transitions: 0 }
    }

    pub fn learn(&mut self, prev_id: usize, current_id: usize, next: &HyperVector) {
        let entry = self
            .per_pair
            .entry((prev_id, current_id))
            .or_insert_with(|| HyperVector::zeros(self.dim));
        *entry = entry.add(next);
        self.transitions += 1;
    }

    pub fn learn_ids(&mut self, ids: &[usize], _vocab: &crate::vocab::Vocab) {
        for w in ids.windows(3) {
            if let Some(next) = _vocab.vector_by_id(w[2]) {
                self.learn(w[0], w[1], next);
            }
        }
    }

    pub fn predict(&self, prev_id: usize, current_id: usize) -> Option<HyperVector> {
        self.per_pair.get(&(prev_id, current_id)).cloned()
    }

    pub fn score(&self, prev_id: usize, current_id: usize, candidate: &HyperVector) -> f32 {
        match self.predict(prev_id, current_id) {
            Some(bundle) => cosine_similarity(&bundle, candidate),
            None => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;    use crate::vocab::Vocab;

    #[test]
    fn test_tba_recovers_exact_transition() {
        let dim = 2048;
        let mut vocab = Vocab::new(dim, 42);
        let the = vocab.get_or_add("the");
        let cat = vocab.get_or_add("cat");
        let dog = vocab.get_or_add("dog");

        let mut tm = TransitionMemory::new(dim);
        tm.learn_ids(&[the, cat], &vocab);

        let cat_vec = vocab.vector_by_id(cat).unwrap();
        let dog_vec = vocab.vector_by_id(dog).unwrap();

        let score_cat = tm.score(the, cat_vec);
        let score_dog = tm.score(the, dog_vec);
        assert!(
            score_cat > score_dog,
            "Recovered transition should score higher: cat={}, dog={}",
            score_cat,
            score_dog
        );
    }

    #[test]
    fn test_tba_frequency_weighting() {
        let dim = 2048;
        let mut vocab = Vocab::new(dim, 43);
        let a = vocab.get_or_add("a");
        let b = vocab.get_or_add("b");
        let c = vocab.get_or_add("c");

        let mut tm = TransitionMemory::new(dim);
        for _ in 0..3 { tm.learn_ids(&[a, b], &vocab); }
        tm.learn_ids(&[a, c], &vocab);

        let score_b = tm.score(a, vocab.vector_by_id(b).unwrap());
        let score_c = tm.score(a, vocab.vector_by_id(c).unwrap());
        assert!(score_b > score_c, "More frequent next token should win");
    }
}
