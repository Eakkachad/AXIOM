//! # Word2Vec Skip-gram Trainer + MPFA Generation (Pure Rust)
//!
//! Train predictive embeddings from scratch in Rust, then use with MPFA.
//! - Skip-gram with negative sampling (Mikolov et al. 2013)
//! - Simple SGD, ~5 min on CPU
//! - Embeddings encode "what comes NEAR this word" (sequential info!)
//! - Then: MPFA attention + trained embeddings → predict next word

use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Write};
use std::fs::File;
use std::time::Instant;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

// ═══════════════════ CONFIG ═══════════════════
const EMBED_DIM: usize = 64;
const MAX_VOCAB: usize = 2000;
const WINDOW: usize = 5;
const NEG_SAMPLES: usize = 5;
const LEARNING_RATE: f32 = 0.025;
const MIN_LR: f32 = 0.001;
const EPOCHS: usize = 5;
const MAX_TOKENS: usize = 80000;

// MPFA config
const DECAY_ALPHA: f32 = 0.15;
const RIDGE_LAMBDA: f32 = 0.1;

const WIKI_PATH: &str = "/home/eggchad/eakject/research/Deep_Man/topological-latent-engine/data/wiki_train.txt";

// ═══════════════════ VOCAB ═══════════════════

struct Vocab {
    w2i: HashMap<String, u16>,
    i2w: Vec<String>,
    freq: Vec<u32>,
    size: usize,
    // Negative sampling table (unigram^0.75 distribution)
    neg_table: Vec<u16>,
}

impl Vocab {
    fn build(tokens: &[String], max_size: usize) -> Self {
        let mut freq_map: HashMap<&str, u32> = HashMap::new();
        for t in tokens { *freq_map.entry(t.as_str()).or_insert(0) += 1; }
        let mut sorted: Vec<(&str, u32)> = freq_map.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(max_size);

        let mut w2i = HashMap::new();
        let mut i2w = Vec::new();
        let mut freq = Vec::new();
        for (i, (word, count)) in sorted.iter().enumerate() {
            w2i.insert(word.to_string(), i as u16);
            i2w.push(word.to_string());
            freq.push(*count);
        }
        let size = i2w.len();

        // Build negative sampling table (unigram^0.75)
        let mut neg_table = Vec::with_capacity(100000);
        let total_pow: f64 = freq.iter().map(|&f| (f as f64).powf(0.75)).sum();
        for (i, &f) in freq.iter().enumerate() {
            let proportion = (f as f64).powf(0.75) / total_pow;
            let count = (proportion * 100000.0) as usize;
            for _ in 0..count.max(1) {
                neg_table.push(i as u16);
            }
        }

        Self { w2i, i2w, freq, size, neg_table }
    }

    fn neg_sample(&self, rng: &mut ChaCha20Rng) -> u16 {
        self.neg_table[rng.gen_range(0..self.neg_table.len())]
    }
}

// ═══════════════════ WORD2VEC TRAINER ═══════════════════

struct Word2Vec {
    /// Target word embeddings [V × D]
    w_target: Vec<f32>,
    /// Context word embeddings [V × D]
    w_context: Vec<f32>,
    dim: usize,
    vocab_size: usize,
}

impl Word2Vec {
    fn new(vocab_size: usize, dim: usize, seed: u64) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let scale = 0.5 / dim as f32;
        let n = vocab_size * dim;
        let w_target: Vec<f32> = (0..n).map(|_| rng.gen_range(-scale..scale)).collect();
        let w_context: Vec<f32> = vec![0.0; n]; // Context starts at zero

        Self { w_target, w_context, dim, vocab_size }
    }

    /// Train skip-gram with negative sampling on token IDs
    fn train(&mut self, token_ids: &[u16], vocab: &Vocab, epochs: usize) {
        let n_tokens = token_ids.len();
        let total_pairs = n_tokens * WINDOW * 2 * epochs;
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut processed = 0u64;

        println!("    Training Word2Vec Skip-gram...");
        println!("    {} tokens, {} epochs, window={}, neg={}", n_tokens, epochs, WINDOW, NEG_SAMPLES);

        let t0 = Instant::now();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0f64;
            let mut pairs = 0u64;

            for i in 0..n_tokens {
                let target = token_ids[i] as usize;

                // Dynamic window
                let window = rng.gen_range(1..=WINDOW);

                for j in (i.saturating_sub(window))..=(i + window).min(n_tokens - 1) {
                    if j == i { continue; }
                    let context = token_ids[j] as usize;

                    // Learning rate decay
                    let progress = processed as f32 / total_pairs as f32;
                    let lr = LEARNING_RATE * (1.0 - progress).max(0.0) + MIN_LR;

                    // Positive sample: sigmoid(target · context) → 1
                    let dot = self.dot_tc(target, context);
                    let sig = sigmoid(dot);
                    let grad = (1.0 - sig) * lr;
                    self.update_pair(target, context, grad);
                    epoch_loss += -(sig.max(1e-10) as f64).ln();

                    // Negative samples: sigmoid(target · neg) → 0
                    for _ in 0..NEG_SAMPLES {
                        let neg = vocab.neg_sample(&mut rng) as usize;
                        if neg == context { continue; }
                        let dot_neg = self.dot_tc(target, neg);
                        let sig_neg = sigmoid(dot_neg);
                        let grad_neg = -sig_neg * lr;
                        self.update_pair(target, neg, grad_neg);
                        epoch_loss += -(1.0 - sig_neg).max(1e-10 as f32) as f64;
                    }

                    pairs += 1;
                    processed += 1;
                }
            }

            let elapsed = t0.elapsed();
            let avg_loss = epoch_loss / pairs.max(1) as f64;
            println!("    Epoch {}/{}: loss={:.4}, pairs={}, [{:.1?}]", epoch + 1, epochs, avg_loss, pairs, elapsed);
        }

        // Normalize target embeddings
        for i in 0..self.vocab_size {
            let offset = i * self.dim;
            let norm: f32 = self.w_target[offset..offset + self.dim].iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for j in 0..self.dim { self.w_target[offset + j] /= norm; }
            }
        }
    }

    #[inline]
    fn dot_tc(&self, target: usize, context: usize) -> f32 {
        let t_off = target * self.dim;
        let c_off = context * self.dim;
        let mut dot = 0.0f32;
        for j in 0..self.dim {
            dot += self.w_target[t_off + j] * self.w_context[c_off + j];
        }
        dot
    }

    #[inline]
    fn update_pair(&mut self, target: usize, context: usize, grad: f32) {
        let t_off = target * self.dim;
        let c_off = context * self.dim;
        for j in 0..self.dim {
            let t_val = self.w_target[t_off + j];
            let c_val = self.w_context[c_off + j];
            self.w_target[t_off + j] += grad * c_val;
            self.w_context[c_off + j] += grad * t_val;
        }
    }

    fn get_embed(&self, id: u16) -> &[f32] {
        let offset = id as usize * self.dim;
        &self.w_target[offset..offset + self.dim]
    }

    fn cosine(&self, a: u16, b: u16) -> f32 {
        let ea = self.get_embed(a);
        let eb = self.get_embed(b);
        ea.iter().zip(eb).map(|(x, y)| x * y).sum()
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-15.0, 15.0)).exp())
}

// ═══════════════════ MPFA + RIDGE ═══════════════════

struct MpfaEngine {
    w2v: Word2Vec,
    w_out: Vec<f32>, // [V × D]
    vocab_size: usize,
    dim: usize,
}

impl MpfaEngine {
    fn new(w2v: Word2Vec) -> Self {
        let vocab_size = w2v.vocab_size;
        let dim = w2v.dim;
        Self {
            w2v,
            w_out: vec![0.0; vocab_size * dim],
            vocab_size,
            dim,
        }
    }

    /// MPFA attention: context-aware representation using trained embeddings
    fn attend(&self, context: &[u16], pos: usize) -> Vec<f32> {
        let d = self.dim;
        let mut output = vec![0.0f32; d];
        let mut total_w = 0.0f32;

        let start = pos.saturating_sub(10);
        for j in start..=pos {
            let dist = (pos - j) as f32;

            // Head 1: Positional decay
            let a1 = (-DECAY_ALPHA * dist).exp();

            // Head 2: Semantic similarity (now with TRAINED embeddings!)
            let a2 = if j < pos {
                self.w2v.cosine(context[pos], context[j]).max(0.0)
            } else { 0.0 };

            // Head 3: Induction
            let a3 = if pos > 0 && j > 0 && j < pos {
                let sim = self.w2v.cosine(context[j - 1], context[pos - 1]);
                (sim * sim).max(0.0)
            } else { 0.0 };

            let w = 0.4 * a1 + 0.35 * a2 + 0.25 * a3;
            if w > 1e-6 {
                let ej = self.w2v.get_embed(context[j]);
                for k in 0..d { output[k] += w * ej[k]; }
                total_w += w;
            }
        }
        if total_w > 1e-6 {
            for k in 0..d { output[k] /= total_w; }
        }
        output
    }

    /// Train output layer with ridge regression
    fn train_readout(&mut self, train_ids: &[u16]) {
        let d = self.dim;
        let v = self.vocab_size;
        let n = train_ids.len().min(40000) - 1;

        println!("  Training MPFA readout (ridge, {} samples)...", n);
        let t0 = Instant::now();

        // Accumulate Gram and target sums
        let mut gram = vec![0.0f32; d * d];
        let mut target_sums = vec![0.0f32; v * d];

        for i in 1..n {
            let h = self.attend(train_ids, i);
            let target = train_ids[i + 1] as usize;

            // Gram += h × h^T
            for r in 0..d {
                for c in r..d {
                    let val = h[r] * h[c];
                    gram[r * d + c] += val;
                    if r != c { gram[c * d + r] += val; }
                }
            }

            // Target sums
            let offset = target * d;
            for k in 0..d { target_sums[offset + k] += h[k]; }
        }

        // Regularize
        for i in 0..d { gram[i * d + i] += RIDGE_LAMBDA; }

        // Cholesky solve for each vocab word
        // First factorize Gram
        let mut l = vec![0.0f32; d * d];
        for i in 0..d {
            for j in 0..=i {
                let mut s = 0.0f32;
                for k in 0..j { s += l[i * d + k] * l[j * d + k]; }
                if i == j {
                    l[i * d + j] = (gram[i * d + i] - s).max(1e-10).sqrt();
                } else {
                    l[i * d + j] = (gram[i * d + j] - s) / l[j * d + j];
                }
            }
        }

        // Solve for each word: w_out[word] = L^{-T} L^{-1} target_sums[word]
        for word in 0..v {
            let ts = &target_sums[word * d..(word + 1) * d];
            if ts.iter().all(|&x| x.abs() < 1e-10) { continue; }

            // Forward: Ly = ts
            let mut y = vec![0.0f32; d];
            for i in 0..d {
                let mut s = 0.0f32;
                for j in 0..i { s += l[i * d + j] * y[j]; }
                y[i] = (ts[i] - s) / l[i * d + i];
            }
            // Backward: L^T x = y
            for i in (0..d).rev() {
                let mut s = 0.0f32;
                for j in (i + 1)..d { s += l[j * d + i] * self.w_out[word * d + j]; }
                self.w_out[word * d + i] = (y[i] - s) / l[i * d + i];
            }
        }

        println!("  Done in {:.2?}", t0.elapsed());
    }

    /// Predict next token
    fn predict(&self, context: &[u16], pos: usize) -> Vec<f32> {
        let h = self.attend(context, pos);
        let d = self.dim;
        let v = self.vocab_size;

        let mut scores = vec![0.0f32; v];
        for w in 0..v {
            let mut s = 0.0f32;
            for j in 0..d { s += self.w_out[w * d + j] * h[j]; }
            scores[w] = s;
        }

        // Softmax
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() { *s = (*s - max).exp(); sum += *s; }
        for s in scores.iter_mut() { *s /= sum; }
        scores
    }

    fn generate(&self, prompt: &[u16], max_tokens: usize) -> Vec<u16> {
        let mut context = prompt.to_vec();
        for _ in 0..max_tokens {
            let pos = context.len() - 1;
            let probs = self.predict(&context, pos);
            let recent: Vec<u16> = context.iter().rev().take(4).copied().collect();
            let mut best_id = 0u16;
            let mut best_s = f32::NEG_INFINITY;
            for w in 0..self.vocab_size {
                let mut s = probs[w];
                if recent.contains(&(w as u16)) { s *= 0.05; }
                if s > best_s { best_s = s; best_id = w as u16; }
            }
            context.push(best_id);
        }
        context
    }
}

// ═══════════════════ MAIN ═══════════════════

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  MPFA + Word2Vec: Train Embeddings → Attention → Generate   ║");
    println!("║  Pure Rust • Predictive Embeddings • CPU Training            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load corpus
    print!("Loading WikiText-2... ");
    io::stdout().flush().unwrap();
    let t0 = Instant::now();
    let file = BufReader::new(File::open(WIKI_PATH).unwrap());
    let mut all_tokens: Vec<String> = Vec::new();
    for line in file.lines() {
        let line = line.unwrap();
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('=') { continue; }
        for word in trimmed.split_whitespace() {
            let lower = word.to_lowercase();
            if lower.chars().all(|c| c.is_alphabetic()) && lower != "unk" && lower.len() > 1 {
                all_tokens.push(lower);
            }
        }
        if all_tokens.len() >= MAX_TOKENS { break; }
    }
    println!("{} tokens [{:.2?}]", all_tokens.len(), t0.elapsed());

    // Build vocab
    let vocab = Vocab::build(&all_tokens, MAX_VOCAB);
    println!("Vocab: {} words\n", vocab.size);

    // Encode
    let all_ids: Vec<u16> = all_tokens.iter().filter_map(|w| vocab.w2i.get(w).copied()).collect();
    let split = all_ids.len() * 80 / 100;
    let train_ids = &all_ids[..split];
    let test_ids = &all_ids[split..];
    println!("Train: {} tokens, Test: {} tokens\n", train_ids.len(), test_ids.len());

    // ═══ Train Word2Vec ═══
    println!("Step 1: Training Word2Vec (Skip-gram, {} epochs)...", EPOCHS);
    let t0 = Instant::now();
    let mut w2v = Word2Vec::new(vocab.size, EMBED_DIM, 42);
    w2v.train(train_ids, &vocab, EPOCHS);
    println!("  Total training: {:.2?}\n", t0.elapsed());

    // ═══ Build MPFA with trained embeddings ═══
    println!("Step 2: Building MPFA with trained embeddings...");
    let mut engine = MpfaEngine::new(w2v);
    engine.train_readout(train_ids);
    println!();

    // ═══ Evaluate ═══
    println!("Step 3: Evaluating on test set...");
    let t0 = Instant::now();
    let mut log_prob = 0.0f64;
    let mut correct = 0u32;
    let mut total = 0u32;
    let eval_n = test_ids.len().min(5000) - 1;

    for i in 1..eval_n {
        let probs = engine.predict(test_ids, i);
        let target = test_ids[i + 1] as usize;
        let p = probs[target].max(1e-10);
        log_prob += (p as f64).ln();
        if probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 == target {
            correct += 1;
        }
        total += 1;
    }

    let ppl = (-log_prob / total as f64).exp() as f32;
    let acc = correct as f32 / total as f32 * 100.0;
    let eval_time = t0.elapsed();
    let speed = total as f64 / eval_time.as_secs_f64();

    println!("  Done in {:.2?}\n", eval_time);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  RESULTS: MPFA + Word2Vec");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Perplexity:       {:.1}", ppl);
    println!("  Accuracy:         {:.1}%", acc);
    println!("  Speed:            {:.0} tokens/sec", speed);
    println!("  Embedding training: {} epochs SGD (CPU)", EPOCHS);
    println!("  Attention params: 0 (fixed algebraic MPFA)");
    println!("  Readout:          Closed-form ridge");
    println!("  100% Rust, deterministic after training");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Generation
    println!("━━━ GENERATION ━━━\n");
    let prompts = ["the president of", "in the first", "it was a", "he was the", "they were not"];
    for prompt in &prompts {
        let ids: Vec<u16> = prompt.split_whitespace().filter_map(|w| vocab.w2i.get(w).copied()).collect();
        if ids.is_empty() { continue; }
        let gen = engine.generate(&ids, 12);
        let text: String = gen.iter().map(|&id| vocab.i2w[id as usize].as_str()).collect::<Vec<_>>().join(" ");
        println!("  \"{}\" → \"{}\"", prompt, text);
    }
}
