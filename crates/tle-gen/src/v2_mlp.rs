//! # TLE-Gen v2: KN-5 + GloVe + Trained MLP Output (Pure Rust)
//!
//! Architecture: KN-5 gives base distribution → 1-layer MLP refines it
//! - KN-5 + GloVe smoothing = ppl 383 (proven baseline)
//! - Add 1 hidden layer (512 units, ReLU) trained with SGD
//! - Input: [KN-5 top-100 probs ∥ GloVe context embedding] = 150d
//! - Output: correction to KN-5 distribution
//! - Training: ~2 min SGD on CPU
//! - Target: beat pure KN-5 (ppl < 383)

use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Write};
use std::fs::File;
use std::time::Instant;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const MAX_VOCAB: usize = 200;
const MAX_TOKENS: usize = 80000;
const GLOVE_DIM: usize = 50;
const HIDDEN: usize = 64;
const INPUT_DIM: usize = 100; // top-50 KN probs + 50d GloVe context
const LR: f32 = 0.02;
const EPOCHS: usize = 3;
const BATCH: usize = 512;
const MAX_ORDER: usize = 5;
const DISCOUNT: f32 = 0.75;

const GLOVE_PATH: &str = "/home/eggchad/eakject/research/Deep_Man/topological-latent-engine/data/glove.6B.50d.txt";
const WIKI_PATH: &str = "/home/eggchad/eakject/research/Deep_Man/topological-latent-engine/data/wiki_train.txt";

// ═══════════════════ N-GRAM (same as tle-gen) ═══════════════════

struct NgramLM {
    tables: Vec<HashMap<u64, Vec<(u16, u32)>>>,
    continuation: Vec<u32>,
    total_cont: u32,
    vocab_size: u16,
}

impl NgramLM {
    fn new(v: u16) -> Self {
        Self { tables: (0..=MAX_ORDER).map(|_| HashMap::new()).collect(),
               continuation: vec![0; v as usize], total_cont: 0, vocab_size: v }
    }
    fn train(&mut self, ids: &[u16]) {
        for n in 1..=MAX_ORDER {
            for i in n..ids.len() {
                let hash = Self::hash(&ids[i-n..i]);
                let e = self.tables[n].entry(hash).or_default();
                if let Some(item) = e.iter_mut().find(|x| x.0 == ids[i]) { item.1 += 1; }
                else { e.push((ids[i], 1)); }
            }
        }
        for entries in self.tables[1].values() {
            for &(w, _) in entries { self.continuation[w as usize] += 1; }
        }
        self.total_cont = self.continuation.iter().sum();
    }
    fn hash(ctx: &[u16]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &w in ctx { h ^= w as u64; h = h.wrapping_mul(0x100000001b3); h ^= (w as u64) << 16; h = h.wrapping_mul(0x517cc1b727220a95); }
        h
    }
    /// Get top-K predictions with scores (fast: only look at stored entries)
    fn top_predictions(&self, context: &[u16], k: usize) -> Vec<(u16, f32)> {
        for n in (1..=MAX_ORDER).rev() {
            if context.len() < n { continue; }
            let ctx = &context[context.len()-n..];
            let hash = Self::hash(ctx);
            if let Some(entries) = self.tables[n].get(&hash) {
                let total: u32 = entries.iter().map(|&(_, c)| c).sum();
                let mut preds: Vec<(u16, f32)> = entries.iter()
                    .map(|&(w, c)| (w, c as f32 / total as f32)).collect();
                preds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                preds.truncate(k);
                return preds;
            }
        }
        // Fallback: unigram (continuation counts)
        let mut preds: Vec<(u16, f32)> = self.continuation.iter().enumerate()
            .map(|(i, &c)| (i as u16, (c as f32 + 0.5) / (self.total_cont as f32 + self.vocab_size as f32 * 0.5)))
            .collect();
        preds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        preds.truncate(k);
        preds
    }
}

// ═══════════════════ MLP (1 hidden layer, SGD) ═══════════════════

struct MLP {
    w1: Vec<f32>,  // HIDDEN × INPUT_DIM
    b1: Vec<f32>,  // HIDDEN
    w2: Vec<f32>,  // VOCAB × HIDDEN
    b2: Vec<f32>,  // VOCAB
    vocab_size: usize,
}

impl MLP {
    fn new(vocab_size: usize, seed: u64) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let s1 = (2.0 / INPUT_DIM as f32).sqrt();
        let s2 = (2.0 / HIDDEN as f32).sqrt();
        Self {
            w1: (0..HIDDEN*INPUT_DIM).map(|_| rng.gen_range(-s1..s1)).collect(),
            b1: vec![0.0; HIDDEN],
            w2: (0..vocab_size*HIDDEN).map(|_| rng.gen_range(-s2..s2)).collect(),
            b2: vec![0.0; vocab_size],
            vocab_size,
        }
    }

    fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        // Hidden = ReLU(W1 × input + b1)
        let mut hidden = vec![0.0f32; HIDDEN];
        for h in 0..HIDDEN {
            let mut s = self.b1[h];
            for i in 0..INPUT_DIM { s += self.w1[h * INPUT_DIM + i] * input[i]; }
            hidden[h] = s.max(0.0);
        }
        // Output = W2 × hidden + b2 → softmax
        let mut logits = vec![0.0f32; self.vocab_size];
        for v in 0..self.vocab_size {
            let mut s = self.b2[v];
            for h in 0..HIDDEN { s += self.w2[v * HIDDEN + h] * hidden[h]; }
            logits[v] = s;
        }
        (hidden, logits)
    }

    fn train_batch(&mut self, inputs: &[Vec<f32>], targets: &[u16], lr: f32) -> f32 {
        let n = inputs.len();
        let v = self.vocab_size;
        let mut total_loss = 0.0f32;

        // Accumulate gradients
        let mut dw1 = vec![0.0f32; HIDDEN * INPUT_DIM];
        let mut db1 = vec![0.0f32; HIDDEN];
        let mut dw2 = vec![0.0f32; v * HIDDEN];
        let mut db2 = vec![0.0f32; v];

        for idx in 0..n {
            let (hidden, logits) = self.forward(&inputs[idx]);

            // Softmax
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
            let sum: f32 = probs.iter().sum();
            for p in probs.iter_mut() { *p /= sum; }

            total_loss += -(probs[targets[idx] as usize].max(1e-10)).ln();

            // dL/d_logits = probs - onehot
            let mut d_logits = probs;
            d_logits[targets[idx] as usize] -= 1.0;

            // Gradients for W2, b2
            for vi in 0..v {
                db2[vi] += d_logits[vi];
                for h in 0..HIDDEN { dw2[vi * HIDDEN + h] += d_logits[vi] * hidden[h]; }
            }

            // Backprop through ReLU
            let mut d_hidden = vec![0.0f32; HIDDEN];
            for h in 0..HIDDEN {
                if hidden[h] > 0.0 {
                    for vi in 0..v { d_hidden[h] += d_logits[vi] * self.w2[vi * HIDDEN + h]; }
                }
            }

            // Gradients for W1, b1
            for h in 0..HIDDEN {
                db1[h] += d_hidden[h];
                for i in 0..INPUT_DIM { dw1[h * INPUT_DIM + i] += d_hidden[h] * inputs[idx][i]; }
            }
        }

        // Update
        let scale = lr / n as f32;
        for i in 0..self.w1.len() { self.w1[i] -= scale * dw1[i]; }
        for i in 0..self.b1.len() { self.b1[i] -= scale * db1[i]; }
        for i in 0..self.w2.len() { self.w2[i] -= scale * dw2[i]; }
        for i in 0..self.b2.len() { self.b2[i] -= scale * db2[i]; }

        total_loss / n as f32
    }

    fn predict_probs(&self, input: &[f32]) -> Vec<f32> {
        let (_, logits) = self.forward(input);
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() { *p /= sum; }
        probs
    }
}

// ═══════════════════ FEATURE EXTRACTION ═══════════════════

fn build_features(ngram: &NgramLM, glove: &[f32], context: &[u16], v: usize) -> Vec<f32> {
    let mut feat = vec![0.0f32; INPUT_DIM];

    // First 50 features: top-50 KN predictions (word_id normalized to 0..1)
    let top_preds = ngram.top_predictions(context, 50);
    for (i, &(word, score)) in top_preds.iter().enumerate() {
        feat[i] = score; // probability score
    }

    // Next 50 features: GloVe context average (last 5 words)
    let ctx_start = context.len().saturating_sub(5);
    let mut count = 0.0f32;
    for &w in &context[ctx_start..] {
        let offset = w as usize * GLOVE_DIM;
        for j in 0..GLOVE_DIM { feat[50 + j] += glove[offset + j]; }
        count += 1.0;
    }
    if count > 0.0 {
        for j in 50..100 { feat[j] /= count; }
    }

    feat
}

// ═══════════════════ MAIN ═══════════════════

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  TLE-Gen v2: KN-5 + GloVe + Trained MLP (Pure Rust)        ║");
    println!("║  Target: Beat pure KN-5 (ppl 383) • CPU only • Fast train  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load
    print!("Loading... ");
    io::stdout().flush().unwrap();
    let t_start = Instant::now();

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

    let mut freq: HashMap<&str, u32> = HashMap::new();
    for t in &all_tokens { *freq.entry(t.as_str()).or_default() += 1; }
    let mut sorted: Vec<(&str, u32)> = freq.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    sorted.truncate(MAX_VOCAB);
    let w2i: HashMap<String, u16> = sorted.iter().enumerate().map(|(i, (w, _))| (w.to_string(), i as u16)).collect();
    let i2w: Vec<String> = sorted.iter().map(|(w, _)| w.to_string()).collect();
    let v = i2w.len();

    let all_ids: Vec<u16> = all_tokens.iter().filter_map(|w| w2i.get(w).copied()).collect();
    let split = all_ids.len() * 80 / 100;
    let train_ids = all_ids[..split].to_vec();
    let test_ids = all_ids[split..].to_vec();

    // Load GloVe
    let mut glove = vec![0.0f32; v * GLOVE_DIM];
    let file = BufReader::new(File::open(GLOVE_PATH).unwrap());
    for line in file.lines() {
        let line = line.unwrap();
        let mut parts = line.split_whitespace();
        if let Some(word) = parts.next() {
            if let Some(&id) = w2i.get(word) {
                let off = id as usize * GLOVE_DIM;
                for (i, val) in parts.enumerate().take(GLOVE_DIM) {
                    if let Ok(v) = val.parse::<f32>() { glove[off + i] = v; }
                }
            }
        }
    }
    // Normalize
    for i in 0..v {
        let off = i * GLOVE_DIM;
        let norm: f32 = glove[off..off+GLOVE_DIM].iter().map(|x| x*x).sum::<f32>().sqrt();
        if norm > 1e-8 { for j in 0..GLOVE_DIM { glove[off+j] /= norm; } }
    }

    println!("done [{:.2?}]", t_start.elapsed());
    println!("  V={}, Train={}, Test={}\n", v, train_ids.len(), test_ids.len());

    // Train KN-5
    print!("Training KN-5... ");
    io::stdout().flush().unwrap();
    let t0 = Instant::now();
    let mut ngram = NgramLM::new(v as u16);
    ngram.train(&train_ids);
    println!("{:.2?}", t0.elapsed());

    // Collect training features for MLP
    print!("Collecting features... ");
    io::stdout().flush().unwrap();
    let t0 = Instant::now();
    let n_train = train_ids.len().min(5000) - MAX_ORDER;
    let mut train_feats: Vec<Vec<f32>> = Vec::with_capacity(n_train);
    let mut train_targets: Vec<u16> = Vec::with_capacity(n_train);

    for i in MAX_ORDER..MAX_ORDER + n_train {
        let ctx = &train_ids[i.saturating_sub(MAX_ORDER)..i];
        let feat = build_features(&ngram, &glove, ctx, v);
        train_feats.push(feat);
        train_targets.push(train_ids[i]);
    }
    println!("{} samples [{:.2?}]", train_feats.len(), t0.elapsed());

    // Train MLP
    println!("\nTraining MLP ({} epochs, hidden={}, lr={})...", EPOCHS, HIDDEN, LR);
    let t0 = Instant::now();
    let mut mlp = MLP::new(v, 42);
    let mut rng = ChaCha20Rng::seed_from_u64(99);

    for epoch in 0..EPOCHS {
        let mut indices: Vec<usize> = (0..train_feats.len()).collect();
        // Shuffle
        for i in (1..indices.len()).rev() { indices.swap(i, rng.gen_range(0..=i)); }

        let mut epoch_loss = 0.0f32;
        let mut n_batches = 0;
        let lr_now = LR * (0.7f32).powi(epoch as i32 / 5);

        for start in (0..indices.len()).step_by(BATCH) {
            let end = (start + BATCH).min(indices.len());
            let batch_feats: Vec<Vec<f32>> = indices[start..end].iter().map(|&i| train_feats[i].clone()).collect();
            let batch_targets: Vec<u16> = indices[start..end].iter().map(|&i| train_targets[i]).collect();
            let loss = mlp.train_batch(&batch_feats, &batch_targets, lr_now);
            epoch_loss += loss;
            n_batches += 1;
        }

        if (epoch + 1) % 3 == 0 || epoch == 0 {
            println!("  Epoch {:>2}/{}: loss={:.4}, lr={:.4} [{:.1?}]",
                     epoch + 1, EPOCHS, epoch_loss / n_batches as f32, lr_now, t0.elapsed());
        }
    }
    println!("  Total training: {:.2?}\n", t0.elapsed());

    // Evaluate
    println!("Evaluating...");
    let t0 = Instant::now();
    let mut log_p_mlp = 0.0f64;
    let mut log_p_kn = 0.0f64;
    let mut correct_mlp = 0u32;
    let mut correct_kn = 0u32;
    let mut total = 0u32;

    for i in MAX_ORDER..test_ids.len() - 1 {
        let ctx = &test_ids[i.saturating_sub(MAX_ORDER)..i];
        let target = test_ids[i] as usize;

        // MLP prediction
        let feat = build_features(&ngram, &glove, ctx, v);
        let probs_mlp = mlp.predict_probs(&feat);
        log_p_mlp += (probs_mlp[target].max(1e-10) as f64).ln();
        if probs_mlp.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 == target { correct_mlp += 1; }

        // Pure KN baseline
        let kn_preds = ngram.top_predictions(ctx, v);
        let kn_prob = kn_preds.iter().find(|&&(w, _)| w as usize == target).map(|&(_, p)| p).unwrap_or(1.0 / v as f32);
        log_p_kn += (kn_prob.max(1e-10) as f64).ln();
        if kn_preds.first().map(|&(w, _)| w as usize) == Some(target) { correct_kn += 1; }

        total += 1;
    }

    let ppl_mlp = (-log_p_mlp / total as f64).exp() as f32;
    let ppl_kn = (-log_p_kn / total as f64).exp() as f32;
    let acc_mlp = correct_mlp as f32 / total as f32 * 100.0;
    let acc_kn = correct_kn as f32 / total as f32 * 100.0;
    let speed = total as f64 / t0.elapsed().as_secs_f64();

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  FINAL RESULTS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  KN-5 (no training):    ppl={:>7.1}  acc={:.1}%", ppl_kn, acc_kn);
    println!("  KN-5 + MLP (trained):  ppl={:>7.1}  acc={:.1}%", ppl_mlp, acc_mlp);
    println!("  Improvement:           {:.1}%", (1.0 - ppl_mlp / ppl_kn) * 100.0);
    println!("  Speed:                 {:.0} tok/s", speed);
    println!("  Training time:         {:?}", t0.elapsed());
    println!("  100% Rust, CPU only");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    if ppl_mlp < ppl_kn {
        println!("\n  🎉 MLP BEATS pure KN-5! ({:.1} < {:.1})", ppl_mlp, ppl_kn);
    }
}
