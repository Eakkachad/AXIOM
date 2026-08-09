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

use tle_vsa::{bind, cosine_similarity, HyperVector};

use crate::vocab::Vocab;

/// A bundled bigram transition memory for one direction of a token stream.
#[derive(Debug, Clone)]
pub struct TransitionMemory {
    /// Superposed bigram bindings: ρ(C(w_i)) ⊙ C(w_{i+1}).
    pub memory: HyperVector,
    /// Number of transitions accumulated (for capacity / SNR diagnostics).
    pub transitions: u64,
}

impl TransitionMemory {
    pub fn new(dim: usize) -> Self {
        Self { memory: HyperVector::zeros(dim), transitions: 0 }
    }

    /// Add a transition binding current -> next.
    pub fn learn(&mut self, current: &HyperVector, next: &HyperVector) {
        let shifted = current.permute(1);
        let binding = bind(&shifted, next);
        self.memory = self.memory.add(&binding);
        self.transitions += 1;
    }

    /// Learn all adjacent pairs in a token-id sequence.
    pub fn learn_ids(&mut self, ids: &[usize], vocab: &Vocab) {
        for window in ids.windows(2) {
            if let (Some(cur), Some(next)) =
                (vocab.vector_by_id(window[0]), vocab.vector_by_id(window[1]))
            {
                self.learn(cur, next);
            }
        }
    }

    /// Predict the next-token vector after `current`.
    ///
    /// The unbinding retrieves a superposition of all next tokens that ever
    /// followed `current`, weighted by frequency, plus crosstalk noise.
    pub fn predict(&self, current: &HyperVector) -> HyperVector {
        let shifted = current.permute(1);
        bind(&shifted, &self.memory)
    }

    /// Score a candidate as the next token after `current`.
    pub fn score(&self, current: &HyperVector, candidate: &HyperVector) -> f32 {
        let pred = self.predict(current);
        cosine_similarity(&pred, candidate)
    }
}

/// Trigram Transition Memory: encodes (w_{i-1}, w_i) → w_{i+1} patterns.
///
/// ```text
/// TM = Σ ρ²(C(w_{i-1})) ⊙ ρ(C(w_i)) ⊙ C(w_{i+1})
/// ```
///
/// Prediction for context (prev, current):
/// ```text
/// pred = ρ²(C(prev)) ⊙ ρ(C(current)) ⊙ TM
/// ```
///
/// Because fewer trigrams share the same (prev, current) pair than bigrams
/// share a single current, the trigram signal-to-noise ratio is inherently
/// better — this is the algebraic analog of "higher-order n-gram smoothing."
#[derive(Debug, Clone)]
pub struct TrigramMemory {
    pub memory: HyperVector,
    pub transitions: u64,
}

impl TrigramMemory {
    pub fn new(dim: usize) -> Self {
        Self { memory: HyperVector::zeros(dim), transitions: 0 }
    }

    pub fn learn(&mut self, prev: &HyperVector, current: &HyperVector, next: &HyperVector) {
        let p2 = prev.permute(2);
        let p1 = current.permute(1);
        let binding = bind(&p2, &bind(&p1, next));
        self.memory = self.memory.add(&binding);
        self.transitions += 1;
    }

    pub fn learn_ids(&mut self, ids: &[usize], vocab: &Vocab) {
        for w in ids.windows(3) {
            if let (Some(p), Some(c), Some(n)) = (
                vocab.vector_by_id(w[0]),
                vocab.vector_by_id(w[1]),
                vocab.vector_by_id(w[2]),
            ) {
                self.learn(p, c, n);
            }
        }
    }

    pub fn predict(&self, prev: &HyperVector, current: &HyperVector) -> HyperVector {
        let p2 = prev.permute(2);
        let p1 = current.permute(1);
        bind(&p2, &bind(&p1, &self.memory))
    }

    pub fn score(&self, prev: &HyperVector, current: &HyperVector, candidate: &HyperVector) -> f32 {
        let pred = self.predict(prev, current);
        cosine_similarity(&pred, candidate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tba_recovers_exact_transition() {
        let dim = 2048;
        let mut vocab = Vocab::new(dim, 42);
        let the = vocab.get_or_add("the");
        let cat = vocab.get_or_add("cat");
        let dog = vocab.get_or_add("dog");

        let mut tm = TransitionMemory::new(dim);
        tm.learn_ids(&[the, cat], &vocab);

        let the_vec = vocab.vector_by_id(the).unwrap();
        let cat_vec = vocab.vector_by_id(cat).unwrap();
        let dog_vec = vocab.vector_by_id(dog).unwrap();

        let score_cat = tm.score(the_vec, cat_vec);
        let score_dog = tm.score(the_vec, dog_vec);
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
        // a->b 3 times, a->c 1 time
        for _ in 0..3 {
            tm.learn_ids(&[a, b], &vocab);
        }
        tm.learn_ids(&[a, c], &vocab);

        let a_vec = vocab.vector_by_id(a).unwrap();
        let score_b = tm.score(a_vec, vocab.vector_by_id(b).unwrap());
        let score_c = tm.score(a_vec, vocab.vector_by_id(c).unwrap());
        assert!(score_b > score_c, "More frequent next token should win");
    }
}
