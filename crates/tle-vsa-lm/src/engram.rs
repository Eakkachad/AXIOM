//! Engram: O(1) hash-addressed n-gram statistical memory.
//!
//! This is the statistical layer of the VSA-LM. It counts how often each
//! next-token follows a given n-gram context, using a rolling hash for O(1)
//! lookup — no neural training, pure counting (single pass over the corpus).

use std::collections::HashMap;

/// Multi-order n-gram counter keyed by a rolling FNV hash of the context.
#[derive(Debug, Clone)]
pub struct Engram {
    max_order: usize,
    /// counts[order][context_hash][next_token_id] = count
    counts: Vec<HashMap<u64, HashMap<usize, u64>>>,
    /// Number of (context, next) pairs observed per order (for smoothing).
    totals: Vec<u64>,
    /// Total tokens seen.
    pub tokens: u64,
}

impl Engram {
    pub fn new(max_order: usize) -> Self {
        let counts = (0..max_order).map(|_| HashMap::new()).collect();
        let totals = vec![0u64; max_order];
        Self { max_order, counts, totals, tokens: 0 }
    }

    /// Learn from a sequence of token ids, updating every order 1..=max_order.
    pub fn learn(&mut self, ids: &[usize]) {
        for pos in 1..ids.len() {
            let next = ids[pos];
            for order in 1..=self.max_order {
                if pos >= order {
                    let context = &ids[pos - order..pos];
                    let hash = fx_hash(context);
                    *self.counts[order - 1]
                        .entry(hash)
                        .or_default()
                        .entry(next)
                        .or_insert(0) += 1;
                    self.totals[order - 1] += 1;
                }
            }
        }
        self.tokens += ids.len() as u64;
    }

    /// Raw count of `next` following `context` at the given order.
    pub fn count(&self, context: &[usize], next: usize, order: usize) -> u64 {
        if order == 0 || order > self.max_order || context.len() < order {
            return 0;
        }
        let window = &context[context.len() - order..];
        let hash = fx_hash(window);
        self.counts[order - 1]
            .get(&hash)
            .and_then(|m| m.get(&next))
            .copied()
            .unwrap_or(0)
    }

    /// Additive-smoothed probability of `next` given the trailing context
    /// window of `order` tokens. Returns 0 if the context never occurred.
    pub fn probability(&self, context: &[usize], next: usize, order: usize) -> f32 {
        if order == 0 || order > self.max_order || context.len() < order {
            return 0.0;
        }
        let window = &context[context.len() - order..];
        let hash = fx_hash(window);
        match self.counts[order - 1].get(&hash) {
            None => 0.0,
            Some(m) => {
                let total: u64 = m.values().sum();
                if total == 0 {
                    return 0.0;
                }
                // Additive smoothing (alpha = 1) against unseen vocabulary.
                let alpha = 1.0f32;
                let n_vocab = m.len() as f32;
                let count = m.get(&next).copied().unwrap_or(0) as f32;
                (count + alpha) / (total as f32 + alpha * n_vocab)
            }
        }
    }

    /// Highest-probability next token for a context at an order.
    /// Returns `None` if the context is unseen.
    pub fn best_next(&self, context: &[usize], order: usize) -> Option<(usize, f32)> {
        if order == 0 || order > self.max_order || context.len() < order {
            return None;
        }
        let window = &context[context.len() - order..];
        let hash = fx_hash(window);
        let m = self.counts[order - 1].get(&hash)?;
        let total: u64 = m.values().sum();
        let (best, count) = m.iter().max_by_key(|(_, &c)| c)?;
        let alpha = 1.0f32;
        let n_vocab = m.len() as f32;
        let prob = (*count as f32 + alpha) / (total as f32 + alpha * n_vocab);
        Some((*best, prob))
    }

    /// Top `limit` next-token ids for a context window (count order),
    /// aggregated across all orders from highest to lowest.
    ///
    /// This is the candidate short-list generator: it bounds decoding cost to
    /// a constant rather than the full vocabulary.
    pub fn top_candidates(&self, context: &[usize], limit: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, u64, usize)> = Vec::new(); // (id, count, order)
        for order in (1..=self.max_order).rev() {
            if context.len() < order {
                continue;
            }
            let window = &context[context.len() - order..];
            let hash = fx_hash(window);
            if let Some(m) = self.counts[order - 1].get(&hash) {
                for (&next, &count) in m {
                    scored.push((next, count, order));
                }
            }
        }
        // Higher orders dominate: sort by (count desc, order desc).
        scored.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)));
        scored.dedup_by_key(|&mut (id, _, _)| id);
        scored.into_iter().take(limit).map(|(id, _, _)| id).collect()
    }
}

/// FNV-1a rolling hash over a window of token ids.
pub fn fx_hash(ids: &[usize]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &id in ids {
        hash ^= id as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engram_bigram() {
        let mut engram = Engram::new(3);
        // "the cat sat"
        let ids = vec![0, 1, 2];
        engram.learn(&ids);

        // "the" -> "cat" should have prob 1.0 at order 1
        let p = engram.probability(&[0], 1, 1);
        assert!((p - 1.0).abs() < 1e-6);
        let (best, _) = engram.best_next(&[0], 1).unwrap();
        assert_eq!(best, 1);
    }

    #[test]
    fn test_engram_trigram() {
        let mut engram = Engram::new(3);
        // "a b c" repeated, then "a b d"
        let ids = vec![0, 1, 2, 0, 1, 3];
        engram.learn(&ids);

        // context [0,1] -> next 2 or 3, each 1/2 at order 2
        let p2 = engram.probability(&[0, 1], 2, 2);
        let p3 = engram.probability(&[0, 1], 3, 2);
        assert!((p2 - 0.5).abs() < 1e-6, "p2={}", p2);
        assert!((p3 - 0.5).abs() < 1e-6, "p3={}", p3);
    }

    #[test]
    fn test_engram_unseen_returns_zero() {
        let mut engram = Engram::new(2);
        engram.learn(&[0, 1]);
        assert_eq!(engram.probability(&[5], 9, 1), 0.0);
        assert!(engram.best_next(&[5], 1).is_none());
    }

    #[test]
    fn test_fx_hash_is_deterministic() {
        let h1 = fx_hash(&[1, 2, 3]);
        let h2 = fx_hash(&[1, 2, 3]);
        assert_eq!(h1, h2);
        let h3 = fx_hash(&[1, 2, 4]);
        assert_ne!(h1, h3);
    }
}
