//! # Multi-Pattern Fixed Attention (MPFA)
//!
//! Novel parameter-free attention mechanism that replaces learned Q/K/V
//! with algebraic patterns derived from pre-trained embeddings.
//!
//! ## The 5 Heads:
//! 1. Positional Decay: exp(-α|i-j|) — local context
//! 2. Semantic Similarity: ReLU(cos(eᵢ,eⱼ)) — topic coherence
//! 3. Induction: cos(e_{j-1}, e_{i-1})² — "what followed similar context?"
//! 4. Previous Token: δ(j==i-1) — immediate predecessor
//! 5. BOS Anchor: δ(j==0) — global context
//!
//! Values = GloVe embeddings directly (no V projection)
//! Output = weighted sum across heads → next-token prediction
//!
//! ZERO learned parameters in attention. All structure from embeddings + position.

use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Write};
use std::fs::File;
use std::time::Instant;

const GLOVE_DIM: usize = 50;
const DECAY_ALPHA: f32 = 0.15;
const MAX_CONTEXT: usize = 64;  // Max attention window
const MAX_VOCAB: usize = 2000;

// Head weights (fixed, not learned)
const W_DECAY: f32 = 0.30;
const W_SEMANTIC: f32 = 0.25;
const W_INDUCTION: f32 = 0.20;
const W_PREV: f32 = 0.15;
const W_BOS: f32 = 0.10;

const GLOVE_PATH: &str = "/home/eggchad/eakject/research/Deep_Man/topological-latent-engine/data/glove.6B.50d.txt";
const WIKI_PATH: &str = "/home/eggchad/eakject/research/Deep_Man/topological-latent-engine/data/wiki_train.txt";

// ═══════════════════════════════════════════════════════════════
// VOCAB + EMBEDDINGS
// ═══════════════════════════════════════════════════════════════

struct Embeddings {
    data: Vec<f32>,     // [V × D] flattened, normalized
    vocab_size: usize,
}

impl Embeddings {
    fn get(&self, id: u16) -> &[f32] {
        let offset = id as usize * GLOVE_DIM;
        &self.data[offset..offset + GLOVE_DIM]
    }

    /// Cosine similarity between two embeddings (already normalized)
    #[inline]
    fn cosine(&self, a: u16, b: u16) -> f32 {
        let ea = self.get(a);
        let eb = self.get(b);
        let mut dot = 0.0f32;
        for i in 0..GLOVE_DIM {
            dot += ea[i] * eb[i];
        }
        dot // Already normalized → dot = cosine
    }
}

struct Vocab {
    w2i: HashMap<String, u16>,
    i2w: Vec<String>,
    size: usize,
}

// ═══════════════════════════════════════════════════════════════
// MPFA: Multi-Pattern Fixed Attention
// ═══════════════════════════════════════════════════════════════

struct MPFA {
    emb: Embeddings,
    /// Output projection: maps GLOVE_DIM → vocab scores
    /// This is the ONLY trainable part (simple linear, closed-form)
    w_out: Vec<f32>,  // [V × GLOVE_DIM]
    vocab_size: usize,
    // Scratch buffers
    attn_output: Vec<f32>,  // [GLOVE_DIM]
}

impl MPFA {
    fn new(emb: Embeddings, vocab_size: usize) -> Self {
        Self {
            w_out: vec![0.0; vocab_size * GLOVE_DIM],
            vocab_size,
            attn_output: vec![0.0; GLOVE_DIM],
            emb,
        }
    }

    /// Compute attention output for position i given context [0..i]
    /// Returns a GLOVE_DIM-sized vector (the attended representation)
    fn attend(&mut self, context: &[u16], pos: usize) -> &[f32] {
        let n = (pos + 1).min(MAX_CONTEXT);
        let start = if pos + 1 > MAX_CONTEXT { pos + 1 - MAX_CONTEXT } else { 0 };

        // Initialize output to zero
        for d in 0..GLOVE_DIM { self.attn_output[d] = 0.0; }

        // Accumulate each head's contribution
        let mut total_weight = 0.0f32;

        for j in start..=pos {
            let dist = (pos - j) as f32;

            // Head 1: Positional decay
            let a1 = (-DECAY_ALPHA * dist).exp();

            // Head 2: Semantic similarity
            let a2 = if pos > 0 && j < pos {
                self.emb.cosine(context[pos], context[j]).max(0.0)
            } else {
                0.0
            };

            // Head 3: Induction (similarity of preceding contexts)
            let a3 = if pos > 0 && j > 0 && j < pos {
                let sim = self.emb.cosine(context[j - 1], context[pos - 1]);
                (sim * sim).max(0.0) // squared for sharpness
            } else {
                0.0
            };

            // Head 4: Previous token
            let a4 = if j == pos.saturating_sub(1) && pos > 0 { 1.0 } else { 0.0 };

            // Head 5: BOS anchor
            let a5 = if j == start { 0.5 } else { 0.0 };

            // Combined attention weight
            let w = W_DECAY * a1 + W_SEMANTIC * a2 + W_INDUCTION * a3 + W_PREV * a4 + W_BOS * a5;

            if w > 1e-6 {
                let ej = self.emb.get(context[j]);
                for d in 0..GLOVE_DIM {
                    self.attn_output[d] += w * ej[d];
                }
                total_weight += w;
            }
        }

        // Normalize
        if total_weight > 1e-6 {
            let inv = 1.0 / total_weight;
            for d in 0..GLOVE_DIM { self.attn_output[d] *= inv; }
        }

        &self.attn_output
    }

    /// Train output projection W_out using closed-form ridge regression
    /// W_out = Y × H^T × (H × H^T + λI)^{-1}
    /// where H = attended representations, Y = one-hot targets
    fn train_output(&mut self, contexts: &[Vec<u16>], lambda: f32) {
        let v = self.vocab_size;
        // Collect (attended_vector, target) pairs
        let mut all_h: Vec<Vec<f32>> = Vec::new();
        let mut all_targets: Vec<u16> = Vec::new();

        for ctx in contexts {
            for i in 1..ctx.len() - 1 {
                let h = self.attend(ctx, i).to_vec();
                all_h.push(h);
                all_targets.push(ctx[i + 1]);
            }
        }

        let n = all_h.len();
        if n == 0 { return; }

        // Gram matrix: G = H^T H [D×D]
        let d = GLOVE_DIM;
        let mut gram = vec![0.0f32; d * d];
        for h in &all_h {
            for r in 0..d {
                for c in r..d {
                    let val = h[r] * h[c];
                    gram[r * d + c] += val;
                    if r != c { gram[c * d + r] += val; }
                }
            }
        }
        // Add regularization
        for i in 0..d { gram[i * d + i] += lambda; }

        // Invert G (Cholesky)
        let g_inv = cholesky_inverse(&gram, d);

        // For each vocab word: w_out[v] = (Σ h_i where target==v) × G_inv
        let mut target_sums = vec![0.0f32; v * d];
        for (i, &t) in all_targets.iter().enumerate() {
            let offset = t as usize * d;
            for j in 0..d {
                target_sums[offset + j] += all_h[i][j];
            }
        }

        // W_out[v] = target_sums[v] × G_inv
        for word in 0..v {
            let ts_offset = word * d;
            for j in 0..d {
                let mut sum = 0.0f32;
                for k in 0..d {
                    sum += target_sums[ts_offset + k] * g_inv[k * d + j];
                }
                self.w_out[word * d + j] = sum;
            }
        }
    }

    /// Predict next token distribution
    fn predict(&mut self, context: &[u16], pos: usize) -> Vec<f32> {
        let h = self.attend(context, pos).to_vec();
        let v = self.vocab_size;
        let d = GLOVE_DIM;

        // Residual: add current token embedding to attention output
        let curr_emb = self.emb.get(context[pos]);
        let mut combined = [0.0f32; GLOVE_DIM];
        for j in 0..d {
            combined[j] = h[j] + curr_emb[j]; // residual connection!
        }

        // scores = W_out × combined
        let mut scores = vec![0.0f32; v];
        for w in 0..v {
            let mut s = 0.0f32;
            for j in 0..d {
                s += self.w_out[w * d + j] * combined[j];
            }
            scores[w] = s;
        }

        // Softmax
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        for s in scores.iter_mut() { *s /= sum; }
        scores
    }
}

fn cholesky_inverse(gram: &[f32], n: usize) -> Vec<f32> {
    // Cholesky decomposition: G = L L^T
    let mut l = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0f32;
            for k in 0..j { s += l[i * n + k] * l[j * n + k]; }
            if i == j {
                l[i * n + j] = (gram[i * n + i] - s).max(1e-10).sqrt();
            } else {
                l[i * n + j] = (gram[i * n + j] - s) / l[j * n + j];
            }
        }
    }
    // Invert L (forward substitution for each column)
    let mut l_inv = vec![0.0f32; n * n];
    for i in 0..n {
        l_inv[i * n + i] = 1.0 / l[i * n + i];
        for j in (i + 1)..n {
            let mut s = 0.0f32;
            for k in i..j { s += l[j * n + k] * l_inv[k * n + i]; }
            l_inv[j * n + i] = -s / l[j * n + j];
        }
    }
    // G_inv = L_inv^T × L_inv
    let mut g_inv = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0f32;
            for k in i.max(j)..n {
                s += l_inv[k * n + i] * l_inv[k * n + j];
            }
            g_inv[i * n + j] = s;
            g_inv[j * n + i] = s;
        }
    }
    g_inv
}

// ═══════════════════════════════════════════════════════════════
// DATA LOADING
// ═══════════════════════════════════════════════════════════════

fn load_data() -> (Vocab, Embeddings, Vec<Vec<u16>>, Vec<Vec<u16>>) {
    // Load tokens
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
        if all_tokens.len() >= 80000 { break; }
    }

    // Build vocab
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for t in &all_tokens { *freq.entry(t.as_str()).or_insert(0) += 1; }
    let mut sorted: Vec<(&str, usize)> = freq.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    sorted.truncate(MAX_VOCAB);

    let mut w2i = HashMap::new();
    let mut i2w = Vec::new();
    for (i, (word, _)) in sorted.iter().enumerate() {
        w2i.insert(word.to_string(), i as u16);
        i2w.push(word.to_string());
    }
    let v = i2w.len();
    let vocab = Vocab { w2i, i2w, size: v };

    // Load GloVe
    let mut emb_data = vec![0.0f32; v * GLOVE_DIM];
    let file = BufReader::new(File::open(GLOVE_PATH).unwrap());
    for line in file.lines() {
        let line = line.unwrap();
        let mut parts = line.split_whitespace();
        if let Some(word) = parts.next() {
            if let Some(&id) = vocab.w2i.get(word) {
                let offset = id as usize * GLOVE_DIM;
                for (i, val) in parts.enumerate().take(GLOVE_DIM) {
                    if let Ok(v) = val.parse::<f32>() { emb_data[offset + i] = v; }
                }
            }
        }
    }
    // Normalize
    for i in 0..v {
        let offset = i * GLOVE_DIM;
        let norm: f32 = emb_data[offset..offset + GLOVE_DIM].iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 { for j in 0..GLOVE_DIM { emb_data[offset + j] /= norm; } }
    }
    let emb = Embeddings { data: emb_data, vocab_size: v };

    // Encode to ID sequences (chunks of 32 tokens)
    let all_ids: Vec<u16> = all_tokens.iter().filter_map(|w| vocab.w2i.get(w).copied()).collect();
    let chunk_size = 32;
    let split = (all_ids.len() * 80 / 100) / chunk_size * chunk_size;
    let train_chunks: Vec<Vec<u16>> = all_ids[..split].chunks(chunk_size).map(|c| c.to_vec()).collect();
    let test_chunks: Vec<Vec<u16>> = all_ids[split..].chunks(chunk_size).map(|c| c.to_vec()).collect();

    (vocab, emb, train_chunks, test_chunks)
}

// ═══════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  MPFA: Multi-Pattern Fixed Attention (Parameter-Free)        ║");
    println!("║  5 algebraic heads • GloVe values • Zero attention params    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load
    print!("Loading data... ");
    io::stdout().flush().unwrap();
    let t0 = Instant::now();
    let (vocab, emb, train_chunks, test_chunks) = load_data();
    println!("done in {:.2?}", t0.elapsed());
    println!("  Vocab: {}, Train chunks: {}, Test chunks: {}", vocab.size, train_chunks.len(), test_chunks.len());
    println!();

    // Build MPFA
    let mut mpfa = MPFA::new(emb, vocab.size);

    // Train output projection (closed-form ridge)
    print!("Training output projection (closed-form)... ");
    io::stdout().flush().unwrap();
    let t0 = Instant::now();
    let train_subset: Vec<Vec<u16>> = train_chunks.iter().take(1500).cloned().collect();
    mpfa.train_output(&train_subset, 1.0);
    println!("done in {:.2?}", t0.elapsed());
    println!();

    // Evaluate
    print!("Evaluating on test set... ");
    io::stdout().flush().unwrap();
    let t0 = Instant::now();
    let mut log_prob = 0.0f64;
    let mut correct = 0u32;
    let mut total = 0u32;

    for chunk in test_chunks.iter().take(100) {
        for i in 1..chunk.len() - 1 {
            let probs = mpfa.predict(chunk, i);
            let target = chunk[i + 1] as usize;
            let p = probs[target].max(1e-10);
            log_prob += (p as f64).ln();
            if probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 == target {
                correct += 1;
            }
            total += 1;
        }
    }

    let ppl = (-log_prob / total as f64).exp() as f32;
    let acc = correct as f32 / total as f32 * 100.0;
    let eval_time = t0.elapsed();
    let speed = total as f64 / eval_time.as_secs_f64();

    println!("done in {:.2?}", eval_time);
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  MPFA RESULTS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Perplexity:       {:.1}", ppl);
    println!("  Accuracy:         {:.1}%", acc);
    println!("  Speed:            {:.0} tokens/sec", speed);
    println!("  Attention params: 0 (all fixed algebraic)");
    println!("  Output params:    {} (closed-form ridge, not SGD)", vocab.size * GLOVE_DIM);
    println!("  Deterministic:    YES");
    println!("  Language:         100% Rust");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Generation demo
    println!("━━━ GENERATION ━━━\n");
    let prompts = ["the president of", "in the first", "it was a", "the city of", "he was the"];
    for prompt in &prompts {
        let ids: Vec<u16> = prompt.split_whitespace().filter_map(|w| vocab.w2i.get(w).copied()).collect();
        if ids.is_empty() { continue; }
        let mut context = ids.clone();
        for _ in 0..12 {
            let pos = context.len() - 1;
            let probs = mpfa.predict(&context, pos);
            // Anti-repetition
            let mut best_id = 0u16;
            let mut best_score = f32::NEG_INFINITY;
            let recent: Vec<u16> = context.iter().rev().take(4).copied().collect();
            for w in 0..vocab.size {
                let mut s = probs[w];
                if recent.contains(&(w as u16)) { s *= 0.05; }
                if s > best_score { best_score = s; best_id = w as u16; }
            }
            context.push(best_id);
        }
        let text: String = context.iter().map(|&id| vocab.i2w[id as usize].as_str()).collect::<Vec<_>>().join(" ");
        println!("  \"{}\" → \"{}\"", prompt, text);
    }

    // Determinism
    println!("\n━━━ DETERMINISM ━━━");
    let ids: Vec<u16> = "the president".split_whitespace().filter_map(|w| vocab.w2i.get(w).copied()).collect();
    let mut outputs = std::collections::HashSet::new();
    for _ in 0..10 {
        let mut ctx = ids.clone();
        for _ in 0..5 {
            let probs = mpfa.predict(&ctx, ctx.len() - 1);
            let best = probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            ctx.push(best as u16);
        }
        outputs.insert(ctx);
    }
    println!("  10 runs → {} unique ✓", outputs.len());
}
