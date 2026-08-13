//! Kneser-Ney 5-gram candidate shortlist (G0/H3 finding).
//!
//! On the wiki corpus the KN-5 top-32 shortlist recall is 48% vs the Engram's
//! 29.3% — replacing the candidate pool with KN-5 raises the
//! candidate-restricted ceiling from ~15% to ~24%+. This module is a compact,
//! deterministic KN-5 built over `usize` token ids (the VsaLm vocab), exposing
//! `top_candidates` for the decode shortlist and `predict_distribution` for
//! scoring.

use std::collections::HashMap;

pub const MAX_ORDER: usize = 5;
const DISCOUNT: f32 = 0.75;

fn hash_context(ctx: &[usize]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &t in ctx {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Compact KN-5 n-gram model.
#[derive(Clone)]
pub struct Kn5Model {
    tables: Vec<HashMap<u64, HashMap<usize, u32>>>,
    continuation: Vec<u32>,
    total_continuation: u32,
    vocab_size: usize,
    trained: bool,
}

impl Kn5Model {
    pub fn new() -> Self {
        Self {
            tables: (0..=MAX_ORDER).map(|_| HashMap::new()).collect(),
            continuation: Vec::new(),
            total_continuation: 0,
            vocab_size: 0,
            trained: false,
        }
    }
    /// Train on a token stream (one call per sentence).
    pub fn train(&mut self, ids: &[usize]) {
        let max_id = ids.iter().max().map(|&m| m + 1).unwrap_or(0);
        if max_id > self.vocab_size {
            self.vocab_size = max_id;
            self.continuation.resize(max_id, 0);
        }
        self.trained = true;
        for n in 1..=MAX_ORDER {
            for i in n..ids.len() {
                let ctx = &ids[i - n..i];
                let word = ids[i];
                let hash = hash_context(ctx);
                *self.tables[n].entry(hash).or_default().entry(word).or_insert(0) += 1;
            }
        }
        // continuation[w] = number of distinct contexts w continues
        let mut seen: HashMap<(usize, u64), ()> = HashMap::new();
        for n in 1..=MAX_ORDER {
            for (hash, entries) in self.tables[n].iter() {
                for w in entries.keys() {
                    seen.insert((*w, *hash), ());
                }
            }
        }
        let mut per_word: HashMap<usize, u32> = HashMap::new();
        for (w, _) in seen.keys() {
            *per_word.entry(*w).or_insert(0) += 1;
        }
        for (w, c) in per_word {
            if w < self.continuation.len() {
                self.continuation[w] = c;
            }
        }
        self.total_continuation = self.continuation.iter().sum();
    }

    /// Top-K candidate token ids by KN-5 probability given the context.
    pub fn top_candidates(&self, context: &[usize], k: usize) -> Vec<usize> {
        if !self.trained || self.vocab_size == 0 {
            return Vec::new();
        }
        let mut prob: Vec<f32> = Vec::with_capacity(self.vocab_size);
        self.predict_distribution(context, &mut prob);
        let mut order: Vec<usize> = (0..self.vocab_size).collect();
        order.sort_by(|&a, &b| prob[b].partial_cmp(&prob[a]).unwrap_or(std::cmp::Ordering::Equal));
        order.truncate(k);
        order
    }

    /// Full KN-5 probability distribution over the vocab.
    pub fn predict_distribution(&self, context: &[usize], out: &mut Vec<f32>) {
        let v = self.vocab_size;
        if out.len() < v {
            out.resize(v, 0.0);
        }
        out[..v].fill(0.0);
        let mut best_order = 0usize;
        let mut best_hash = 0u64;
        for n in (1..=MAX_ORDER).rev() {
            if context.len() < n {
                continue;
            }
            let ctx = &context[context.len() - n..];
            let hash = hash_context(ctx);
            if self.tables[n].contains_key(&hash) {
                best_order = n;
                best_hash = hash;
                break;
            }
        }
        if best_order > 0 {
            let entries = &self.tables[best_order][&best_hash];
            let total: u32 = entries.values().sum();
            let n_unique = entries.len() as f32;
            let lambda = DISCOUNT * n_unique / total as f32;
            let t_cont = self.total_continuation as f32 + self.vocab_size as f32 * 0.5;
            for w in 0..v {
                out[w] = lambda * ((self.continuation[w] as f32 + 0.5) / t_cont);
            }
            for (&word, &count) in entries.iter() {
                let p_high = (count as f32 - DISCOUNT).max(0.0) / total as f32;
                out[word] += p_high;
            }
        } else {
            let t_cont = self.total_continuation as f32 + self.vocab_size as f32 * 0.5;
            for w in 0..v {
                out[w] = (self.continuation[w] as f32 + 0.5) / t_cont;
            }
        }
        let sum: f32 = out[..v].iter().sum();
        if sum > 0.0 {
            for w in 0..v {
                out[w] /= sum;
            }
        }
    }

    pub fn is_trained(&self) -> bool {
        self.trained
    }
}
