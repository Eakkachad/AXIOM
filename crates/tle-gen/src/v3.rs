//! # TLE-Gen v3: Optimized KN-5 + Interactive Generation (Pure Rust)
//!
//! Improvements over v1:
//! - Sparse prediction: only score words that appear in matched context (not full V)
//! - Proper interpolated KN smoothing with backoff
//! - Interactive generation CLI
//! - GloVe semantic smoothing on sparse predictions
//! - Target: V=2000, >5000 tok/s, ppl comparable to v1

use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Write};
use std::fs::File;
use std::time::Instant;

const MAX_VOCAB: usize = 2000;
const MAX_TOKENS: usize = 80000;
const GLOVE_DIM: usize = 50;
const MAX_ORDER: usize = 5;
const DISCOUNT: f32 = 0.75;
const ALPHA_SEM: f32 = 0.08;
const GLOVE_TEMP: f32 = 3.0;

const GLOVE_PATH: &str = "/home/eggchad/eakject/research/Deep_Man/topological-latent-engine/data/glove.6B.50d.txt";
const WIKI_PATH: &str = "/home/eggchad/eakject/research/Deep_Man/topological-latent-engine/data/wiki_train.txt";

// ═══════════════════ OPTIMIZED N-GRAM ═══════════════════

struct FastNgram {
    /// For each order n: context_hash → [(word_id, count)]
    tables: Vec<HashMap<u64, Vec<(u16, u32)>>>,
    /// Total counts per context (cached for speed)
    totals: Vec<HashMap<u64, u32>>,
    /// Unique continuations per context (for KN lambda)
    n_unique: Vec<HashMap<u64, u16>>,
    /// Continuation counts for KN lower-order
    continuation: Vec<u32>,
    total_cont: u32,
    /// Unigram
    unigram: Vec<u32>,
    total_tokens: u32,
    vocab_size: u16,
}

impl FastNgram {
    fn new(v: u16) -> Self {
        Self {
            tables: (0..=MAX_ORDER).map(|_| HashMap::new()).collect(),
            totals: (0..=MAX_ORDER).map(|_| HashMap::new()).collect(),
            n_unique: (0..=MAX_ORDER).map(|_| HashMap::new()).collect(),
            continuation: vec![0; v as usize],
            total_cont: 0,
            unigram: vec![0; v as usize],
            total_tokens: 0,
            vocab_size: v,
        }
    }

    fn train(&mut self, ids: &[u16]) {
        self.total_tokens = ids.len() as u32;
        for &t in ids { self.unigram[t as usize] += 1; }

        for n in 1..=MAX_ORDER {
            for i in n..ids.len() {
                let hash = hash_ctx(&ids[i - n..i]);
                let word = ids[i];
                let entry = self.tables[n].entry(hash).or_default();
                if let Some(item) = entry.iter_mut().find(|x| x.0 == word) {
                    item.1 += 1;
                } else {
                    entry.push((word, 1));
                }
            }
        }

        // Cache totals and n_unique
        for n in 1..=MAX_ORDER {
            for (hash, entries) in &self.tables[n] {
                let total: u32 = entries.iter().map(|&(_, c)| c).sum();
                self.totals[n].insert(*hash, total);
                self.n_unique[n].insert(*hash, entries.len() as u16);
            }
        }

        // Continuation counts
        for entries in self.tables[1].values() {
            for &(w, _) in entries { self.continuation[w as usize] += 1; }
        }
        self.total_cont = self.continuation.iter().sum();
    }

    /// FAST sparse prediction: return scored candidates only
    /// Returns (word_id, probability) for words with non-trivial probability
    fn predict_sparse(&self, context: &[u16]) -> Vec<(u16, f32)> {
        // Find deepest matching context
        for n in (1..=MAX_ORDER).rev() {
            if context.len() < n { continue; }
            let ctx = &context[context.len() - n..];
            let hash = hash_ctx(ctx);

            if let Some(entries) = self.tables[n].get(&hash) {
                let total = self.totals[n][&hash] as f32;
                let n_uniq = self.n_unique[n][&hash] as f32;
                let lambda = DISCOUNT * n_uniq / total;

                let mut results: Vec<(u16, f32)> = entries.iter().map(|&(w, c)| {
                    let p_high = (c as f32 - DISCOUNT).max(0.0) / total;
                    let p_low = self.kn_unigram(w);
                    (w, p_high + lambda * p_low)
                }).collect();

                // Add unigram floor for unseen words (top-20 by frequency)
                let seen: Vec<u16> = results.iter().map(|&(w, _)| w).collect();
                let mut unseen_budget = 20usize;
                for (w, &count) in self.unigram.iter().enumerate() {
                    if count == 0 || seen.contains(&(w as u16)) { continue; }
                    results.push((w as u16, lambda * self.kn_unigram(w as u16)));
                    unseen_budget -= 1;
                    if unseen_budget == 0 { break; }
                }

                // Normalize
                let sum: f32 = results.iter().map(|&(_, p)| p).sum();
                if sum > 0.0 {
                    for r in results.iter_mut() { r.1 /= sum; }
                }
                return results;
            }
        }

        // No context match: return top unigram
        let mut results: Vec<(u16, f32)> = self.unigram.iter().enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(w, &c)| (w as u16, c as f32 / self.total_tokens as f32))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(50);
        results
    }

    #[inline]
    fn kn_unigram(&self, w: u16) -> f32 {
        (self.continuation[w as usize] as f32 + 0.5)
            / (self.total_cont as f32 + self.vocab_size as f32 * 0.5)
    }

    /// Full distribution (for perplexity calculation)
    fn prob(&self, word: u16, context: &[u16]) -> f32 {
        for n in (1..=MAX_ORDER).rev() {
            if context.len() < n { continue; }
            let ctx = &context[context.len() - n..];
            let hash = hash_ctx(ctx);

            if let Some(entries) = self.tables[n].get(&hash) {
                let total = self.totals[n][&hash] as f32;
                let n_uniq = self.n_unique[n][&hash] as f32;
                let lambda = DISCOUNT * n_uniq / total;

                let count_w = entries.iter()
                    .find(|&&(w, _)| w == word).map(|&(_, c)| c).unwrap_or(0);
                let p_high = (count_w as f32 - DISCOUNT).max(0.0) / total;
                let p_low = self.kn_unigram(word);
                return p_high + lambda * p_low;
            }
        }
        self.kn_unigram(word)
    }
}

#[inline]
fn hash_ctx(ctx: &[u16]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &w in ctx {
        h ^= w as u64;
        h = h.wrapping_mul(0x100000001b3);
        h ^= (w as u64) << 16;
        h = h.wrapping_mul(0x517cc1b727220a95);
    }
    h
}

// ═══════════════════ GLOVE (same as before) ═══════════════════

struct GloVe {
    data: Vec<f32>, // [V × 50] normalized
    vocab_size: usize,
}

impl GloVe {
    fn load(path: &str, w2i: &HashMap<String, u16>, v: usize) -> Self {
        let mut data = vec![0.0f32; v * GLOVE_DIM];
        if let Ok(file) = File::open(path) {
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line.unwrap();
                let mut parts = line.split_whitespace();
                if let Some(word) = parts.next() {
                    if let Some(&id) = w2i.get(word) {
                        let off = id as usize * GLOVE_DIM;
                        for (i, val) in parts.enumerate().take(GLOVE_DIM) {
                            if let Ok(v) = val.parse::<f32>() { data[off + i] = v; }
                        }
                    }
                }
            }
        }
        for i in 0..v {
            let off = i * GLOVE_DIM;
            let norm: f32 = data[off..off + GLOVE_DIM].iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 { for j in 0..GLOVE_DIM { data[off + j] /= norm; } }
        }
        Self { data, vocab_size: v }
    }

    /// Semantic boost for a word given context (cosine similarity)
    fn semantic_score(&self, word: u16, context: &[u16]) -> f32 {
        if context.is_empty() { return 0.0; }
        let ctx_start = context.len().saturating_sub(5);
        let mut ctx_vec = [0.0f32; GLOVE_DIM];
        let mut count = 0.0f32;
        for &w in &context[ctx_start..] {
            let off = w as usize * GLOVE_DIM;
            for j in 0..GLOVE_DIM { ctx_vec[j] += self.data[off + j]; }
            count += 1.0;
        }
        if count == 0.0 { return 0.0; }
        for j in 0..GLOVE_DIM { ctx_vec[j] /= count; }
        let norm: f32 = ctx_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-8 { return 0.0; }
        for j in 0..GLOVE_DIM { ctx_vec[j] /= norm; }

        let off = word as usize * GLOVE_DIM;
        let mut dot = 0.0f32;
        for j in 0..GLOVE_DIM { dot += self.data[off + j] * ctx_vec[j]; }
        dot
    }
}

// ═══════════════════ GENERATION ENGINE ═══════════════════

struct GenEngine {
    ngram: FastNgram,
    glove: GloVe,
    w2i: HashMap<String, u16>,
    i2w: Vec<String>,
    vocab_size: usize,
}

impl GenEngine {
    /// Generate next token (deterministic)
    fn next_token(&self, context: &[u16]) -> u16 {
        let candidates = self.ngram.predict_sparse(context);
        if candidates.is_empty() { return 0; }

        // Apply GloVe semantic boost + anti-repetition
        let recent: Vec<u16> = context.iter().rev().take(5).copied().collect();
        let mut best_id = candidates[0].0;
        let mut best_score = f32::NEG_INFINITY;

        for &(word, kn_prob) in &candidates {
            let sem = self.glove.semantic_score(word, context);
            let mut score = (1.0 - ALPHA_SEM) * kn_prob + ALPHA_SEM * (sem * GLOVE_TEMP).exp();

            // Anti-repetition
            if recent.contains(&word) { score *= 0.05; }

            if score > best_score {
                best_score = score;
                best_id = word;
            }
        }
        best_id
    }

    /// Generate a sequence
    fn generate(&self, prompt: &str, max_tokens: usize) -> String {
        let mut ids: Vec<u16> = prompt.split_whitespace()
            .filter_map(|w| self.w2i.get(w).copied())
            .collect();
        if ids.is_empty() { return prompt.to_string(); }

        for _ in 0..max_tokens {
            let ctx = &ids[ids.len().saturating_sub(MAX_ORDER)..];
            let next = self.next_token(ctx);
            ids.push(next);
        }

        ids.iter().map(|&id| self.i2w[id as usize].as_str()).collect::<Vec<_>>().join(" ")
    }

    /// Evaluate perplexity
    fn evaluate(&self, test_ids: &[u16]) -> (f32, f32) {
        let mut log_p = 0.0f64;
        let mut correct = 0u32;
        let mut total = 0u32;

        for i in MAX_ORDER..test_ids.len() {
            let ctx = &test_ids[i.saturating_sub(MAX_ORDER)..i];
            let target = test_ids[i];
            let p = self.ngram.prob(target, ctx).max(1e-10);
            log_p += (p as f64).ln();

            let candidates = self.ngram.predict_sparse(ctx);
            if candidates.first().map(|&(w, _)| w) == Some(target) { correct += 1; }
            total += 1;
        }

        let ppl = (-log_p / total as f64).exp() as f32;
        let acc = correct as f32 / total as f32 * 100.0;
        (ppl, acc)
    }
}

// ═══════════════════ MAIN ═══════════════════

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  TLE-Gen v3: Optimized KN-5 + GloVe + Interactive CLI       ║");
    println!("║  Sparse prediction • Fast inference • 100% Rust             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load
    print!("Loading WikiText-2... ");
    io::stdout().flush().unwrap();
    let t0 = Instant::now();

    let file = BufReader::new(File::open(WIKI_PATH).unwrap());
    let mut all_tokens: Vec<String> = Vec::new();
    for line in file.lines() {
        let line = line.unwrap();
        let t = line.trim();
        if t.is_empty() || t.starts_with('=') { continue; }
        for w in t.split_whitespace() {
            let lower = w.to_lowercase();
            if lower.chars().all(|c| c.is_alphabetic()) && lower != "unk" && lower.len() > 1 {
                all_tokens.push(lower);
            }
        }
        if all_tokens.len() >= MAX_TOKENS { break; }
    }
    println!("{} tokens [{:.2?}]", all_tokens.len(), t0.elapsed());

    // Build vocab
    let mut freq: HashMap<&str, u32> = HashMap::new();
    for t in &all_tokens { *freq.entry(t.as_str()).or_default() += 1; }
    let mut sorted: Vec<(&str, u32)> = freq.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    sorted.truncate(MAX_VOCAB);

    let mut w2i: HashMap<String, u16> = HashMap::new();
    let mut i2w: Vec<String> = Vec::new();
    for (i, (word, _)) in sorted.iter().enumerate() {
        w2i.insert(word.to_string(), i as u16);
        i2w.push(word.to_string());
    }
    let v = i2w.len();

    let all_ids: Vec<u16> = all_tokens.iter().filter_map(|w| w2i.get(w).copied()).collect();
    let split = all_ids.len() * 80 / 100;
    let train_ids = &all_ids[..split];
    let test_ids = &all_ids[split..];
    println!("Vocab: {}, Train: {}, Test: {}", v, train_ids.len(), test_ids.len());

    // Train KN-5
    print!("Training KN-5... ");
    io::stdout().flush().unwrap();
    let t0 = Instant::now();
    let mut ngram = FastNgram::new(v as u16);
    ngram.train(train_ids);
    println!("{:.2?}", t0.elapsed());

    // Load GloVe
    print!("Loading GloVe... ");
    io::stdout().flush().unwrap();
    let t0 = Instant::now();
    let glove = GloVe::load(GLOVE_PATH, &w2i, v);
    println!("{:.2?}", t0.elapsed());

    let engine = GenEngine { ngram, glove, w2i: w2i.clone(), i2w: i2w.clone(), vocab_size: v };

    // Evaluate
    print!("\nEvaluating... ");
    io::stdout().flush().unwrap();
    let t0 = Instant::now();
    let (ppl, acc) = engine.evaluate(test_ids);
    let eval_time = t0.elapsed();
    let speed = test_ids.len() as f64 / eval_time.as_secs_f64();
    println!("{:.2?}", eval_time);

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Perplexity:   {:.1}", ppl);
    println!("  Accuracy:     {:.1}%", acc);
    println!("  Speed:        {:.0} tokens/sec", speed);
    println!("  Vocab:        {}", v);
    println!("  Deterministic: YES");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Generation demo
    println!("━━━ GENERATION ━━━\n");
    let prompts = ["the president of", "in the first", "it was a",
                   "the city of", "he was the", "they were not",
                   "she said that", "the game was"];
    for prompt in &prompts {
        let t0 = Instant::now();
        let text = engine.generate(prompt, 12);
        let gen_time = t0.elapsed();
        println!("  \"{}\" → \"{}\" [{:.2?}]", prompt, text, gen_time);
    }

    // Interactive mode
    println!("\n━━━ INTERACTIVE MODE (type a prompt, 'quit' to exit) ━━━\n");
    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush().unwrap();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 { break; }
        let input = line.trim();
        if input.is_empty() { continue; }
        if input == "quit" || input == "exit" { break; }

        let t0 = Instant::now();
        let output = engine.generate(input, 15);
        let elapsed = t0.elapsed();
        println!("  {} [{:.2?}]\n", output, elapsed);
    }
}
