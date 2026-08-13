//! `kn5-acc` — G0 decision gate: measure a Kneser-Ney 5-gram's next-token
//! accuracy + top-32 shortlist recall on the SAME wiki split as vsalm-scale.
//! This determines whether a statistical filler (vs the VSA-LM's 11%) is a
//! viable local-coherence backbone for "chat like an LLM".
//!
//! Usage: kn5-acc <corpus.txt> [sentences_limit] [train_ratio]

use std::collections::HashMap;
use std::time::Instant;

const MAX_ORDER: usize = 5;
const DISCOUNT: f32 = 0.75;

fn main() {
    let path = std::env::args().nth(1).expect("usage: kn5-acc <corpus.txt> [limit] [ratio]");
    let limit: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(3000);
    let train_ratio: f32 = std::env::args().nth(3).and_then(|v| v.parse().ok()).unwrap_or(0.8);

    let raw = std::fs::read_to_string(&path).expect("read corpus");
    let mut sentences: Vec<Vec<String>> = raw
        .split(|c: char| matches!(c, '.' | '!' | '?' | '\n'))
        .map(|s| s.split_whitespace().map(|w| w.to_lowercase()).collect())
        .filter(|s: &Vec<String>| s.len() >= 4 && s.len() <= 60)
        .take(limit)
        .collect();
    let split = (sentences.len() as f32 * train_ratio) as usize;
    let test: Vec<Vec<String>> = sentences.split_off(split);
    let train = sentences;

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  G0: Kneser-Ney 5-gram ceiling probe                     ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("corpus: {} ({} train / {} test)", path, train.len(), test.len());

    // vocab from train
    let mut freq: HashMap<String, usize> = HashMap::new();
    for s in &train {
        for w in s {
            *freq.entry(w.clone()).or_insert(0) += 1;
        }
    }
    let mut sorted: Vec<(String, usize)> = freq.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let w2i: HashMap<String, u16> = sorted.iter().enumerate().map(|(i, (w, _))| (w.clone(), i as u16)).collect();
    let i2w: Vec<String> = sorted.iter().map(|(w, _)| w.clone()).collect();
    let vocab_size = w2i.len().max(1) as u16;
    println!("  vocab: {} words", vocab_size);

    // train
    let t0 = Instant::now();
    let mut ng = NgramCounts::new(vocab_size);
    for s in &train {
        let ids: Vec<u16> = s.iter().filter_map(|w| w2i.get(w).copied()).collect();
        ng.train(&ids);
    }
    println!("  trained in {:.2?}", t0.elapsed());

    // evaluate
    let t0 = Instant::now();
    let (acc, short_recall, short128, total) = evaluate(&ng, &w2i, &test, vocab_size);
    println!("\n  TEST next-token accuracy (full-vocab argmax): {:.1}% ({})", acc * 100.0, total);
    println!("  TEST top-32 shortlist recall: {:.1}%", short_recall * 100.0);
    println!("  TEST top-128 shortlist recall: {:.1}%", short128 * 100.0);
    println!("  ceiling estimate (recall × argmax-on-hit): see shortlist recall");
    println!("  elapsed: {:.2?}", t0.elapsed());
}

struct NgramCounts {
    tables: Vec<HashMap<u64, HashMap<u16, u32>>>,
    continuation: Vec<u32>,
    total_continuation: u32,
    unigram: Vec<u32>,
    vocab_size: u16,
}

fn hash_context(ctx: &[u16]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &t in ctx {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl NgramCounts {
    fn new(vocab_size: u16) -> Self {
        Self {
            tables: (0..=MAX_ORDER).map(|_| HashMap::new()).collect(),
            continuation: vec![0; vocab_size as usize],
            total_continuation: 0,
            unigram: vec![0; vocab_size as usize],
            vocab_size,
        }
    }

    fn train(&mut self, token_ids: &[u16]) {
        for &t in token_ids {
            self.unigram[t as usize] += 1;
        }
        for n in 1..=MAX_ORDER {
            for i in n..token_ids.len() {
                let ctx = &token_ids[i - n..i];
                let word = token_ids[i];
                let hash = hash_context(ctx);
                *self.tables[n].entry(hash).or_default().entry(word).or_insert(0) += 1;
            }
        }
        // continuation[w] = number of distinct contexts w appears in
        // (efficient: one flat set of (word, context_hash) pairs)
        let mut ctx_seen: std::collections::HashSet<(u16, u64)> =
            std::collections::HashSet::new();
        for n in 1..=MAX_ORDER {
            for (hash, entries) in self.tables[n].iter() {
                for &w in entries.keys() {
                    ctx_seen.insert((w, *hash));
                }
            }
        }
        let mut per_word: HashMap<u16, u32> = HashMap::new();
        for (w, _) in ctx_seen {
            *per_word.entry(w).or_insert(0) += 1;
        }
        for (w, c) in per_word {
            self.continuation[w as usize] = c;
        }
        self.total_continuation = self.continuation.iter().sum();
    }

    fn predict_distribution(&self, context: &[u16], out: &mut [f32]) {
        let v = self.vocab_size as usize;
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
                out[word as usize] += p_high;
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
}

fn evaluate(ng: &NgramCounts, w2i: &HashMap<String, u16>, test: &[Vec<String>], vocab: u16) -> (f32, f32, f32, usize) {
    let mut correct = 0usize;
    let mut short_hit = 0usize;
    let mut short128 = 0usize;
    let mut total = 0usize;
    let v = vocab as usize;
    let mut buf = vec![0.0f32; v];
    'outer: for s in test {
        let ids: Vec<u16> = s.iter().filter_map(|w| w2i.get(w).copied()).collect();
        for pos in 0..ids.len().saturating_sub(1) {
            if total >= 300 {
                break 'outer;
            }
            ng.predict_distribution(&ids[..=pos], &mut buf);
            let want = ids[pos + 1] as usize;
            total += 1;
            // argmax
            let mut best = 0usize;
            let mut best_p = -1.0f32;
            for (i, &p) in buf.iter().enumerate() {
                if p > best_p {
                    best_p = p;
                    best = i;
                }
            }
            if best == want {
                correct += 1;
            }
            // top-32 / top-128 recall
            let mut order: Vec<usize> = (0..v).collect();
            order.sort_by(|&a, &b| buf[b].partial_cmp(&buf[a]).unwrap_or(std::cmp::Ordering::Equal));
            if order[..32.min(v)].contains(&want) {
                short_hit += 1;
            }
            if order[..128.min(v)].contains(&want) {
                short128 += 1;
            }
        }
    }
    let t = total as f32;
    (correct as f32 / t, short_hit as f32 / t, short128 as f32 / t, total)
}
