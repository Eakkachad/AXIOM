//! Co-occurrence semantic layer.
//!
//! Gives the VSA codebook distributional structure so that words which
//! appear near each other in a corpus acquire positive cosine similarity:
//!   C_sem(w) = normalize( Σ_{c: c co-occurs with w} count(w, c) · C_base(c) )
//!
//! This is a deterministic, single-pass, no-training semantic layer on top of
//! the random bipolar codebook.  It does NOT replace the base codebook (which
//! keeps exact orthogonality for distinct symbols) — it only enriches the
//! `semantic_vector()` used for fuzzy relevance scoring.

use std::collections::HashMap;

use tle_vsa::{Codebook, HyperVector};

/// A window-based co-occurrence semantic layer.
#[derive(Clone, Default)]
pub struct SemanticLayer {
    /// cooccur[word][context] = weighted count (word appears near context).
    cooccur: HashMap<String, HashMap<String, f32>>,
    /// Precomputed semantic vectors: word → C_sem(word).
    vectors: HashMap<String, HyperVector>,
    /// Whether the layer has been built (has data).
    built: bool,
    /// Window radius (±N words).
    window: usize,
}

impl SemanticLayer {
    pub fn new() -> Self {
        Self { cooccur: HashMap::new(), vectors: HashMap::new(), built: false, window: 3 }
    }

    /// Count co-occurrences within a ±window around each content word.
    /// Lowercases and strips punctuation.  Skips common stopwords.
    pub fn ingest_text(&mut self, text: &str) {
        let tokens: Vec<String> = text.split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|w| w.len() >= 3 && !is_stopword(w))
            .collect();
        for i in 0..tokens.len() {
            let lo = i.saturating_sub(self.window);
            let hi = (i + self.window + 1).min(tokens.len());
            for j in lo..hi {
                if i == j { continue; }
                let w = tokens[i].clone();
                let c = tokens[j].clone();
                let entry = self.cooccur.entry(w).or_default();
                *entry.entry(c).or_insert(0.0) += 1.0;
            }
        }
        self.built = false; // vectors need rebuild
    }

    /// Build semantic vectors from accumulated co-occurrence counts.
    /// C_sem(w) = normalize( Σ_c count(w,c) · C_base(c) ).
    pub fn build(&mut self, codebook: &mut Codebook) {
        self.vectors.clear();
        let dim = codebook.dim();
        let mut total: f32 = 0.0;
        for counts in self.cooccur.values() {
            total += counts.values().sum::<f32>();
        }
        if total < 1.0 { self.built = true; return; }
        let total = total;
        for (word, counts) in &self.cooccur {
            let mut vec = HyperVector::zeros(dim);
            let mut weight_sum: f32 = 0.0;
            for (context, count) in counts {
                // PMI-ish weighting: normalize by global frequency pressure.
                let w = *count / (1.0 + total * 1e-5);
                // get_or_insert so context words enter the codebook.
                let cvec = codebook.get_or_insert(context);
                vec = vec.add(&cvec.scale(w));
                weight_sum += w;
            }
            if weight_sum > 0.0 {
                let v = vec.normalize();
                self.vectors.insert(word.clone(), v);
            }
        }
        self.built = true;
    }

    /// Semantic vector for a word.  Falls back to None if no co-occurrence data.
    pub fn vector(&self, word: &str) -> Option<&HyperVector> {
        self.vectors.get(&word.to_lowercase())
    }

    /// Whether the layer has useful semantic data.
    pub fn is_built(&self) -> bool {
        self.built && !self.vectors.is_empty()
    }

    /// Number of words with semantic vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Number of co-occurrence entries tracked.
    pub fn cooccur_pairs(&self) -> usize {
        self.cooccur.values().map(|m| m.len()).sum()
    }
}

fn is_stopword(w: &str) -> bool {
    matches!(w, "the" | "and" | "for" | "are" | "but" | "not" | "you" | "all"
        | "was" | "with" | "from" | "that" | "this" | "have" | "has" | "had"
        | "were" | "been" | "being" | "its" | "his" | "her" | "they" | "them"
        | "their" | "will" | "would" | "could" | "should" | "may" | "might"
        | "must" | "shall" | "there" | "here" | "what" | "which" | "when"
        | "where" | "who" | "whom" | "whose" | "how" | "why" | "because"
        | "while" | "after" | "before" | "during" | "about" | "into" | "onto"
        | "upon" | "also" | "just" | "then" | "than" | "more" | "most" | "some"
        | "such" | "only" | "very" | "can" | "into" | "over" | "under")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cooccurrence_gives_paris_france_similarity() {
        let mut cb = Codebook::new(2048, 0xA10A_CAFE_BEAD_0001);
        let mut layer = SemanticLayer::new();
        let corpus = "Paris is the capital of France. The president of France lives in Paris. France has many cities.";
        layer.ingest_text(corpus);
        eprintln!("cooccur pairs: {}", layer.cooccur_pairs());
        layer.build(&mut cb);
        eprintln!("semantic words: {}", layer.len());
        assert!(layer.len() > 0);
        let paris = layer.vector("paris").expect("paris vector");
        let france = layer.vector("france").expect("france vector");
        let cos = tle_vsa::cosine_similarity(paris, france);
        assert!(cos > 0.0, "paris and france should be positively similar, got {}", cos);
    }

    #[test]
    fn cooccurrence_pairs_counted() {
        let mut layer = SemanticLayer::new();
        layer.ingest_text("the cat sat on the mat the dog ran");
        assert!(layer.cooccur_pairs() > 0);
    }
}
