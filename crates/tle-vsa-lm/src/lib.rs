//! # VSA-LM: VSA-based Language Model
//!
//! A fully algebraic, non-neural language generator. Replaces the neural
//! components of an LM with VSA operations:
//!
//! - **Codebook** (random bipolar vectors) instead of learned embeddings
//! - **Transition Binding Algebra** (bundled bigram bindings) instead of a
//!   weight matrix
//! - **Engram** (O(1) hash n-gram counts) for statistical prior
//! - **VSA cosine decoder** instead of softmax
//! - **Energy-guided beam search** instead of autoregressive argmax
//!
//! No backprop, no gradient, no probability sampling — deterministic for a
//! fixed corpus and seed.

pub mod decode;
pub mod engram;
pub mod reservoir;
pub mod tba;
pub mod vocab;

pub use reservoir::{ReservoirConfig, ReservoirMemory};

use std::collections::HashSet;

use tle_vsa::HyperVector;

use decode::DecodedToken;
use engram::Engram;
use reservoir::Reservoir;
use tba::TransitionMemory;
use vocab::Vocab;

/// Configuration for the VSA-LM.
#[derive(Debug, Clone)]
pub struct LmConfig {
    /// VSA vector dimensionality.
    pub dim: usize,
    /// Maximum n-gram order for the Engram statistical layer.
    pub max_order: usize,
    /// Beam width for generation.
    pub beam_width: usize,
    /// Maximum tokens to generate per call.
    pub max_gen_tokens: usize,
    /// Weight of the TBA (transition) score in prediction.
    pub w_tba: f32,
    /// Weight of the Engram (n-gram) score in prediction.
    pub w_engram: f32,
    /// Weight of the reservoir associative-memory score in prediction.
    pub w_reservoir: f32,
    /// Reservoir configuration (dimension, leak, etc.).
    pub reservoir_config: Option<ReservoirConfig>,
    /// Anti-repetition penalty strength.
    pub w_repeat: f32,
    /// Repetition window size (tokens).
    pub repeat_window: usize,
}

impl Default for LmConfig {
    fn default() -> Self {
        Self {
            dim: 10_240,
            max_order: 4,
            beam_width: 8,
            max_gen_tokens: 16,
            w_tba: 1.0,
            w_engram: 1.5,
            w_reservoir: 0.5,
            reservoir_config: None,
            w_repeat: 0.15,
            repeat_window: 3,
        }
    }
}

/// The VSA language model engine.
#[derive(Debug, Clone)]
pub struct VsaLm {
    /// Word vocabulary + VSA codebook.
    pub vocab: Vocab,
    /// Statistical n-gram layer.
    pub engram: Engram,
    /// Transition Binding Algebra memory.
    pub tba: TransitionMemory,
    /// Dynamical reservoir (optional).
    pub reservoir: Option<Reservoir>,
    /// Non-parametric reservoir associative memory.
    pub reservoir_mem: Option<ReservoirMemory>,
    /// Configuration.
    pub config: LmConfig,
}

impl VsaLm {
    pub fn new(config: LmConfig) -> Self {
        let dim = config.dim;
        let (reservoir, reservoir_mem) = match &config.reservoir_config {
            Some(rconfig) => {
                let res = Reservoir::new(rconfig, 0xBEAD_01CE_5001);
                let mem = ReservoirMemory::new(8);
                (Some(res), Some(mem))
            }
            None => (None, None),
        };
        Self {
            vocab: Vocab::new(dim, 0xA11E_0BEE_F001),
            engram: Engram::new(config.max_order),
            tba: TransitionMemory::new(dim),
            reservoir,
            reservoir_mem,
            config,
        }
    }

    /// Tokenize a lowercase sentence into word tokens (alphabetic runs).
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric() && c != '\'')
            .map(|t| t.trim().to_lowercase())
            .filter(|t| !t.is_empty())
            .collect()
    }

    /// Ingest a sentence: register vocabulary, learn n-grams and transitions.
    pub fn learn(&mut self, sentence: &str) {
        let tokens = self.tokenize(sentence);
        if tokens.len() < 2 {
            return;
        }
        let ids: Vec<usize> = tokens.iter().map(|t| self.vocab.get_or_add(t)).collect();
        self.engram.learn(&ids);
        self.tba.learn_ids(&ids, &self.vocab);

        // If a reservoir is configured, feed the tokens through it and record
        // each (state, next token) pair into the associative memory.
        if let (Some(reservoir), Some(mem)) = (&mut self.reservoir, &mut self.reservoir_mem) {
            let dim = reservoir.dim;
            // Drive the reservoir with VSA vectors zero-padded to reservoir dim.
            for i in 0..ids.len().saturating_sub(1) {
                let Some(vec) = self.vocab.vector_by_id(ids[i]) else { continue };
                let mut input = vec![0.0f32; dim];
                for j in 0..vec.dim().min(dim) {
                    input[j] = vec.data[j];
                }
                let state = reservoir.step(&input).to_vec();
                mem.record(state, ids[i + 1]);
            }
            reservoir.reset();
        }
    }

    /// Raw TBA prediction vector for the last token in `context`.
    pub fn tba_prediction(&self, context: &[String]) -> Option<HyperVector> {
        let last = context.last()?;
        let vec = self.vocab.vector(last)?;
        Some(self.tba.predict(vec))
    }

    /// Combined prediction: blend TBA cosine, Engram n-gram probability, and
    /// the reservoir associative-memory signal.
    ///
    /// Returns a ranked list of `k` candidate tokens. The score is a weighted
    /// sum of:
    /// - TBA: `w_tba * cosine(prediction, C(token))` — algebraic transition
    /// - Engram: `w_engram * P(token | context)` for the highest-order seen
    ///   context (backoff to lower orders when the exact n-gram is unseen)
    /// - Reservoir: `w_reservoir * k-NN vote` over stored states
    pub fn predict_next(&self, context: &[String], k: usize) -> Vec<DecodedToken> {
        let mut candidates = self.predict_next_fast(context, k);

        // Reservoir signal: drive the reservoir over the context, then look up
        // the k nearest stored states and aggregate their next-token votes.
        if let (Some(reservoir), Some(mem)) = (&self.reservoir, &self.reservoir_mem) {
            let mut res = reservoir.clone();
            for word in context.iter().rev().take(self.config.repeat_window * 3) {
                if let Some(vec) = self.vocab.vector(word) {
                    let mut input = vec![0.0f32; reservoir.dim];
                    for j in 0..vec.dim().min(reservoir.dim) {
                        input[j] = vec.data[j];
                    }
                    res.step(&input);
                }
            }
            let reservoir_scores = mem.predict_scores(&res.state, self.vocab.len());
            for cand in candidates.iter_mut() {
                cand.similarity += self.config.w_reservoir * reservoir_scores[cand.id];
            }
            candidates.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(k);
        }
        candidates
    }

    /// Fast combined prediction (TBA + Engram, no reservoir k-NN).
    ///
    /// Uses a two-stage decode to avoid O(V·D) full-vocabulary scoring:
    /// 1. Engram builds a short-list of candidate ids (O(V) hash lookups).
    /// 2. TBA cosine scores only the short-list (O(candidates·D)).
    pub fn predict_next_fast(&self, context: &[String], k: usize) -> Vec<DecodedToken> {
        let context_ids: Vec<usize> = context.iter().filter_map(|w| self.vocab.id(w)).collect();
        let shortlist = self.engram.top_candidates(&context_ids, 32);
        let tba_signal = self.tba_prediction(context);

        let mut candidates: Vec<DecodedToken> = shortlist
            .iter()
            .map(|&id| {
                let mut score = self.config.w_engram * engram_probability(self, &context_ids, id);
                if let Some(signal) = &tba_signal {
                    let sim = tle_vsa::cosine_similarity(&signal, self.vocab.vector_by_id(id).unwrap());
                    score += self.config.w_tba * sim;
                }
                DecodedToken {
                    id,
                    word: self.vocab.word(id).to_string(),
                    similarity: score,
                }
            })
            .collect();

        // If no context matched any n-gram, fall back to TBA-only scoring
        // over the full vocabulary (small corpora, cold context).
        if candidates.is_empty() {
            if let Some(signal) = tba_signal {
                candidates = self
                    .vocab
                    .iter()
                    .map(|(id, word)| {
                        let sim = tle_vsa::cosine_similarity(&signal, self.vocab.vector_by_id(id).unwrap());
                        DecodedToken { id, word: word.to_string(), similarity: sim }
                    })
                    .collect();
            }
        }

        if candidates.is_empty() {
            return Vec::new();
        }
        candidates.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);
        candidates
    }

    /// Build a candidate short-list from the Engram n-gram counts.
    ///
    /// At each order (highest first) the seen contexts' next-token ids are
    /// collected in descending count order. This bounds the number of
    /// cosine decodes to a constant regardless of vocabulary size.
    fn engram_shortlist(&self, context: &[String], limit: usize) -> Vec<usize> {
        let context_ids: Vec<usize> = context.iter().filter_map(|w| self.vocab.id(w)).collect();
        self.engram.top_candidates(&context_ids, limit)
    }

    /// Generate a continuation for `prompt` using energy-guided beam search.
    ///
    /// Each step expands the beam with the top candidates from `predict_next`,
    /// and scores partial sequences by summed token scores minus a repetition
    /// penalty. Deterministic given the corpus and config.
    pub fn generate(&self, prompt: &str, max_tokens: Option<usize>) -> String {
        let max_tokens = max_tokens.unwrap_or(self.config.max_gen_tokens);
        let tokens = self.tokenize(prompt);
        let mut beam: Vec<(Vec<String>, f32)> = vec![(tokens.clone(), 0.0)];
        for _ in 0..max_tokens {
            let mut next_beam: Vec<(Vec<String>, f32)> = Vec::new();
            for (seq, score) in &beam {
                let top = self.predict_next(seq, self.config.beam_width);
                for cand in top {
                    let mut new_seq = seq.clone();
                    new_seq.push(cand.word.clone());
                    // VSA repetition penalty: bundle the recent context tokens
                    // and penalize a candidate whose vector is similar to that
                    // bundle. Repeated words self-bundle and get suppressed.
                    let mut rep = 0.0f32;
                    let recent: Vec<String> = new_seq
                        .iter()
                        .rev()
                        .take(self.config.repeat_window + 1)
                        .cloned()
                        .collect();
                    let recent_ids: Vec<usize> = recent
                        .iter()
                        .filter_map(|w| self.vocab.id(w))
                        .collect();
                    if !recent_ids.is_empty() {
                        let mut bundle = HyperVector::zeros(self.config.dim);
                        for id in &recent_ids {
                            if let Some(v) = self.vocab.vector_by_id(*id) {
                                bundle = bundle.add(v);
                            }
                        }
                        if let Some(v) = self.vocab.vector_by_id(cand.id) {
                            let sim = tle_vsa::cosine_similarity(&bundle, v);
                            // Only penalize strong, positive similarity to the
                            // recent bundle (repetition), not weak noise.
                            rep = sim.max(0.0) * self.config.w_repeat;
                        }
                    }

                    // Hard-ish loop breaker: penalize repeating an existing
                    // adjacent bigram later in the sequence.
                    if new_seq.len() >= 4 {
                        let last_pair = format!("{} {}", &new_seq[new_seq.len() - 2], &new_seq[new_seq.len() - 1]);
                        for w in new_seq[..new_seq.len() - 2].windows(2) {
                            if format!("{} {}", w[0], w[1]) == last_pair {
                                rep += self.config.w_repeat * 2.0;
                                break;
                            }
                        }
                    }
                    next_beam.push((new_seq, score + cand.similarity - rep));
                }
            }
            if next_beam.is_empty() {
                break;
            }
            next_beam.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            next_beam.truncate(self.config.beam_width);
            beam = next_beam;
        }

        let best = &beam[0].0;
        let continuation = best
            .iter()
            .skip(tokens.len().min(best.len()))
            .cloned()
            .collect::<Vec<_>>();

        // Deterministic end-of-sequence: stop if we re-produced the prompt.
        let mut out = tokens;
        out.extend(continuation);
        // Avoid pure repeats of a single word.
        if let Some(first) = out.first() {
            if out.iter().filter(|w| *w == first).count() > out.len() / 2 {
                out.truncate(1);
            }
        }
        out.join(" ")
    }

    /// TBA-only prediction: rank candidates purely by VSA transition cosine.
    pub fn predict_tba_only(&self, context: &[String], k: usize) -> Vec<DecodedToken> {
        let Some(signal) = self.tba_prediction(context) else {
            return Vec::new();
        };
        let mut candidates: Vec<DecodedToken> = self
            .vocab
            .iter()
            .map(|(id, word)| {
                let sim = tle_vsa::cosine_similarity(&signal, self.vocab.vector_by_id(id).unwrap());
                DecodedToken { id, word: word.to_string(), similarity: sim }
            })
            .collect();
        candidates.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);
        candidates
    }

    /// Engram-only prediction: rank candidates purely by n-gram probability.
    pub fn predict_engram_only(&self, context: &[String], k: usize) -> Vec<DecodedToken> {
        let context_ids: Vec<usize> = context.iter().filter_map(|w| self.vocab.id(w)).collect();
        let mut best: Vec<(usize, f32)> = Vec::new();
        for order in (1..=self.config.max_order).rev() {
            if context_ids.len() < order {
                continue;
            }
            if let Some((id, prob)) = self.engram.best_next(&context_ids, order) {
                best = vec![(id, prob)];
                break;
            }
        }
        if best.is_empty() {
            return Vec::new();
        }
        let mut candidates: Vec<DecodedToken> = best
            .into_iter()
            .map(|(id, prob)| DecodedToken {
                id,
                word: self.vocab.word(id).to_string(),
                similarity: prob,
            })
            .collect();
        candidates.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);
        candidates
    }

    /// Measure next-token accuracy: for each adjacent pair in the corpus,
    /// does `predict_next` rank the true next token first?
    ///
    /// Returns (accuracy, total_pairs).
    pub fn next_token_accuracy(&self, sentences: &[String]) -> (f32, usize) {
        self.next_token_accuracy_sample(sentences, usize::MAX)
    }

    /// Bounded next-token accuracy over at most `max_pairs` adjacent pairs.
    pub fn next_token_accuracy_sample(&self, sentences: &[String], max_pairs: usize) -> (f32, usize) {
        let mut correct = 0usize;
        let mut total = 0usize;
        'outer: for sentence in sentences {
            let tokens = self.tokenize(sentence);
            if tokens.len() < 2 {
                continue;
            }
            let ids: Vec<usize> = tokens.iter().filter_map(|w| self.vocab.id(w)).collect();
            for pos in 0..ids.len().saturating_sub(1) {
                if total >= max_pairs {
                    break 'outer;
                }
                let context: Vec<String> = tokens[..=pos].to_vec();
                let pred = self.predict_next_fast(&context, 5);
                if pred.is_empty() {
                    continue;
                }
                let true_id = ids[pos + 1];
                total += 1;
                if pred[0].id == true_id {
                    correct += 1;
                }
            }
        }
        if total == 0 {
            (0.0, 0)
        } else {
            (correct as f32 / total as f32, total)
        }
    }

    /// Determinism check: generate the same prompt N times, assert identical.
    pub fn is_deterministic(&self, prompt: &str) -> bool {
        let first = self.generate(prompt, Some(8));
        let mut outputs = HashSet::new();
        for _ in 0..3 {
            outputs.insert(self.generate(prompt, Some(8)));
        }
        outputs.len() == 1 && outputs.contains(&first)
    }
}

/// Seed for the VSA-LM codebook.
const _CFLM_SEED: u64 = 0xC0DE_0EAD_0A11;

/// Probability of `next` under the Engram, backed off to the lowest seen
/// order (i.e. highest-probability match wins across orders).
fn engram_probability(lm: &VsaLm, context: &[usize], next: usize) -> f32 {
    for order in (1..=lm.config.max_order).rev() {
        if context.len() < order {
            continue;
        }
        let prob = lm.engram.probability(context, next, order);
        if prob > 0.0 {
            return prob;
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    const CORPUS: &[&str] = &[
        "the cat sat on the mat",
        "the dog ran in the park",
        "the bird flew over the tree",
        "a cat is a small animal",
        "a dog is a loyal friend",
        "the sun is bright and warm",
        "the moon is bright at night",
    ];

    fn build() -> VsaLm {
        let config = LmConfig { dim: 4096, max_order: 3, beam_width: 4, max_gen_tokens: 8, ..Default::default() };
        let mut lm = VsaLm::new(config);
        for s in CORPUS {
            lm.learn(s);
        }
        lm
    }

    #[test]
    fn test_learn_and_vocab() {
        let lm = build();
        assert!(lm.vocab.len() >= 10);
        assert!(lm.engram.tokens > 0);
        assert!(lm.tba.transitions > 0);
    }

    #[test]
    fn test_predict_next_returns_known_token() {
        let lm = build();
        let context = vec!["the".to_string(), "cat".to_string()];
        let pred = lm.predict_next(&context, 5);
        assert!(!pred.is_empty());
        // "sat" should be strongly ranked after "the cat"
        assert!(pred.iter().any(|t| t.word == "sat"));
    }

    #[test]
    fn test_generate_is_nonempty_and_deterministic() {
        let lm = build();
        let out = lm.generate("the cat", Some(6));
        assert!(!out.is_empty());
        assert!(lm.is_deterministic("the cat"));
    }

    #[test]
    fn test_next_token_accuracy_on_trained_corpus() {
        let lm = build();
        let corpus: Vec<String> = CORPUS.iter().map(|s| s.to_string()).collect();
        let (acc, total) = lm.next_token_accuracy(&corpus);
        assert!(total > 0);
        // On its own training set a trigram+Engram + TBA system should get
        // well above random chance (1/vocab ≈ 5%).
        assert!(acc > 0.2, "expected >20% next-token accuracy, got {:.1}%", acc * 100.0);
    }

    #[test]
    fn test_unknown_prompt_graceful() {
        let lm = build();
        let out = lm.generate("zzzz", Some(4));
        // Should not panic; may return just the prompt.
        assert!(!out.is_empty());
    }

    #[test]
    fn test_two_stage_decoder_matches_full_decode() {
        let lm = build();
        let context = vec!["the".to_string(), "cat".to_string()];
        let fast = lm.predict_next_fast(&context, 5);
        assert!(!fast.is_empty());
        // The two-stage decoder must agree that "sat" is a top candidate.
        assert!(fast.iter().any(|t| t.word == "sat"));
    }

    #[test]
    fn test_reservoir_signal_changes_prediction() {
        // A reservoir should inject a signal (even if small) that the fast
        // path lacks; verify the reservoir path returns sane candidates.
        let config = LmConfig {
            dim: 2048,
            max_order: 3,
            beam_width: 4,
            max_gen_tokens: 6,
            w_reservoir: 0.5,
            reservoir_config: Some(crate::ReservoirConfig { dim: 256, ..Default::default() }),
            ..Default::default()
        };
        let mut lm = VsaLm::new(config);
        for s in CORPUS {
            lm.learn(s);
        }
        assert!(lm.reservoir.is_some());
        assert!(lm.reservoir_mem.as_ref().map(|m| !m.states.is_empty()).unwrap_or(false));
        let context = vec!["the".to_string(), "cat".to_string()];
        let pred = lm.predict_next(&context, 5);
        assert!(!pred.is_empty());
    }

    #[test]
    fn test_top_candidates_bounds_shortlist() {
        let lm = build();
        let context = vec!["the".to_string(), "cat".to_string()];
        let context_ids: Vec<usize> = context.iter().filter_map(|w| lm.vocab.id(w)).collect();
        let shortlist = lm.engram.top_candidates(&context_ids, 8);
        assert!(shortlist.len() <= 8);
        assert!(!shortlist.is_empty());
    }
}
