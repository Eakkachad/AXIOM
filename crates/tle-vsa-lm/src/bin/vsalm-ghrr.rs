//! `vsalm-ghrr` — GHRR block-unitary transition-memory LM (A3 prototype).
//!
//! Replaces the random-bipolar TBA bigram/trigram encoding with GHRR
//! block-unitary (O(4)) binding. GHRR unbinding is EXACT for orthogonal
//! blocks (`unbind(bind(A,B), A) == B`), so a single learned transition
//! recovers the next-token vector perfectly; bundled transitions recover a
//! superposition, scored by blockwise cosine vs the vocabulary.
//!
//! Usage: vsalm-ghrr <corpus.txt> [sentences_limit] [train_ratio]

use std::collections::HashMap;
use std::time::Instant;

use tle_ghrr::{GhrrCodebook, GhrrVector};

const DIM: usize = tle_ghrr::DIM; // 2048 = 128 × 16

fn main() {
    let path = std::env::args().nth(1).expect("usage: vsalm-ghrr <corpus.txt> [limit] [ratio]");
    let limit: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(3000);
    let train_ratio: f32 = std::env::args().nth(3).and_then(|v| v.parse().ok()).unwrap_or(0.8);
    let order: usize = std::env::args().nth(4).and_then(|v| v.parse().ok()).unwrap_or(2);

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
    println!("║  VSA-LM GHRR transition memory (A3 prototype)          ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("corpus: {} ({} train / {} test) · order={} · dim={}", path, train.len(), test.len(), order, DIM);

    let mut lm = GhrrLm::new();
    let t0 = Instant::now();
    for s in &train {
        lm.learn(s, order);
    }
    println!("  vocab {} · transitions {} in {:.2?}", lm.vocab.len(), lm.tm.len(), t0.elapsed());

    let t0 = Instant::now();
    let (tr_acc, tr_n) = lm.accuracy(&train, 500);
    let (te_acc, te_n) = lm.accuracy(&test, 300);
    println!("\n  TRAIN next-token: {:.1}% ({})", tr_acc * 100.0, tr_n);
    println!("  TEST  next-token: {:.1}% ({})", te_acc * 100.0, te_n);
    println!("  elapsed: {:.2?}", t0.elapsed());

    let prompts = ["the game", "the player", "the series"];
    for p in prompts {
        println!("  \"{}\" → \"{}\"", p, lm.generate(p, 10));
    }
}

struct GhrrLm {
    vocab: Vec<String>,
    word_id: HashMap<String, usize>,
    codebook: GhrrCodebook,
    /// key → bundled transition memory (bigram: key=src; trigram: key=2-tuple
    /// hashed). Value = Σ_next Bind(ctx, C(next)).
    tm: HashMap<u64, GhrrVector>,
}

impl GhrrLm {
    fn new() -> Self {
        Self {
            vocab: Vec::new(),
            word_id: HashMap::new(),
            codebook: GhrrCodebook::new(0xA5E1_2016_1D12_5EED),
            tm: HashMap::new(),
        }
    }

    fn id(&mut self, w: &str) -> usize {
        if let Some(&i) = self.word_id.get(w) {
            return i;
        }
        let i = self.vocab.len();
        self.vocab.push(w.to_string());
        self.word_id.insert(w.to_string(), i);
        i
    }

    fn vec(&mut self, w: &str) -> GhrrVector {
        self.codebook.get_or_insert(w)
    }

    /// Learn transitions: for order=2 (bigram) key=w_i, value += Bind(w_i,w_{i+1}).
    /// For order=3 (trigram) key=(w_i,w_{i+1}), value += Bind(Bind(w_i,w_{i+1}), w_{i+2}).
    fn learn(&mut self, sentence: &[String], order: usize) {
        let ctxs: Vec<usize> = sentence.iter().map(|w| self.id(w)).collect();
        for (i, &src) in ctxs.iter().enumerate() {
            if i + order >= ctxs.len() {
                break;
            }
            let key = if order == 2 {
                src as u64
            } else {
                // pair key: src * V + next
                (src as u64) * (self.vocab.len() as u64).max(1) + ctxs[i + 1] as u64
            };
            // context vector: for bigram, C(src); for trigram, Bind(C(src), C(next))
            let ctx_vec = if order == 2 {
                self.vec(&sentence[i])
            } else {
                let a = self.vec(&sentence[i]);
                let b = self.vec(&sentence[i + 1]);
                GhrrVector::bind_path(&[&a, &b])
            };
            let next = self.vec(&sentence[i + order]);
            let binding = GhrrVector::bind_path(&[&ctx_vec, &next]);
            self.tm.entry(key)
                .and_modify(|e| *e = bundle(e, &binding))
                .or_insert(binding);
        }
    }

    /// Predict next token after `context` (uses last `order-1` words as key).
    fn predict(&mut self, context: &[String], order: usize, k: usize) -> Vec<(usize, f32)> {
        if context.len() < order {
            return Vec::new();
        }
        let n = context.len();
        let key = if order == 2 {
            self.word_id.get(&context[n - 1]).copied().unwrap_or(usize::MAX) as u64
        } else {
            let a = self.word_id.get(&context[n - 2]).copied().unwrap_or(usize::MAX) as u64;
            let b = self.word_id.get(&context[n - 1]).copied().unwrap_or(usize::MAX) as u64;
            a * (self.vocab.len() as u64).max(1) + b
        };
        let Some(mem) = self.tm.get(&key).cloned() else {
            return Vec::new();
        };
        let ctx_vec = if order == 2 {
            self.vec(&context[n - 1])
        } else {
            let a = self.vec(&context[n - 2]);
            let b = self.vec(&context[n - 1]);
            GhrrVector::bind_path(&[&a, &b])
        };
        let recovered = mem.unbind(&ctx_vec);
        // score vs vocab
        let mut scored: Vec<(usize, f32)> = (0..self.vocab.len())
            .map(|i| {
                let word = self.vocab[i].clone();
                (i, recovered.blockwise_cosine(&self.codebook.get_or_insert(&word)))
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    fn accuracy(&mut self, sentences: &[Vec<String>], max_pairs: usize) -> (f32, usize) {
        let mut correct = 0usize;
        let mut total = 0usize;
        let debug = std::env::var("GHRR_DEBUG").is_ok();
        'outer: for s in sentences {
            for i in 0..s.len().saturating_sub(1) {
                if total >= max_pairs {
                    break 'outer;
                }
                let pred = self.predict(&s[..=i], 2, 5);
                let want = self.word_id.get(&s[i + 1]).copied();
                total += 1;
                if let Some(&(id, _)) = pred.first() {
                    if Some(id) == want {
                        correct += 1;
                    }
                }
                if debug && total <= 10 {
                    let top1 = pred.first().map(|(id, s)| (self.vocab[*id].clone(), *s));
                    let want_word = want.map(|id| self.vocab[id].clone());
                    let want_sim = want.and_then(|wid| {
                        pred.iter().find(|(id, _)| *id == wid).map(|(_, s)| *s)
                    });
                    println!("  [{}] ctx='...{}' want='{:?}'({:.3}) top1='{:?}'({:.3})",
                        total, s[i], want_word, want_sim.unwrap_or(-9.0), top1, top1.as_ref().map(|t| t.1).unwrap_or(-9.0));
                }
            }
        }
        if total == 0 {
            (0.0, 0)
        } else {
            (correct as f32 / total as f32, total)
        }
    }

    fn generate(&mut self, prompt: &str, max_tokens: usize) -> String {
        let mut ctx: Vec<String> = prompt.split_whitespace().map(|w| w.to_lowercase()).collect();
        let mut out = ctx.clone();
        for _ in 0..max_tokens {
            let pred = self.predict(&ctx, 2, 1);
            let Some(&(id, _)) = pred.first() else { break };
            let w = self.vocab[id].clone();
            out.push(w.clone());
            ctx.push(w);
            if ctx.len() > 8 {
                ctx.remove(0);
            }
        }
        out.join(" ")
    }
}

/// Bundle (superpose) two GHRR vectors: blockwise sum, then renormalize.
fn bundle(a: &GhrrVector, b: &GhrrVector) -> GhrrVector {
    use tle_ghrr::block::{M, normalize};
    let mut blocks = Vec::with_capacity(tle_ghrr::D_BLOCKS);
    for j in 0..tle_ghrr::D_BLOCKS {
        let mut c = [[0.0f32; M]; M];
        for i in 0..M {
            for k in 0..M {
                c[i][k] = a.blocks[j][i][k] + b.blocks[j][i][k];
            }
        }
        blocks.push(normalize(&c));
    }
    GhrrVector::from_blocks(blocks)
}
