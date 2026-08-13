//! VSA-LM scale benchmark: ingest a real text corpus and measure next-token
//! accuracy on a held-out test set.  This is the LM-breakthrough metric:
//! can VSA algebra (no softmax, no backprop) generalize beyond n-gram
//! memorisation on real text?
//!
//! Usage: vsalm-scale <corpus.txt> [sentences_limit] [train_ratio]

use std::fs;
use std::time::Instant;

use tle_vsa_lm::{LmConfig, VsaLm};

fn env_f(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Conditional rerank accuracy: among pairs where the correct token is in the
/// engram top-K shortlist, how often does each signal rank it #1?
fn conditional_rerank(lm: &VsaLm, sentences: &[String], k: usize, max_pairs: usize)
    -> (f32, f32, f32, f32, usize) {
    let mut combined = 0usize;
    let mut tba = 0usize;
    let mut tri = 0usize;
    let mut eng = 0usize;
    let mut total = 0usize;
    'outer: for sentence in sentences {
        let tokens = lm.tokenize(sentence);
        for pos in 0..tokens.len().saturating_sub(1) {
            if total >= max_pairs {
                break 'outer;
            }
            let ctx: Vec<String> = tokens[..=pos].to_vec();
            let ctx_ids: Vec<usize> = ctx.iter().filter_map(|w| lm.vocab.id(w)).collect();
            let cands = lm.engram.top_candidates(&ctx_ids, k);
            let want = lm.vocab.id(&tokens[pos + 1]);
            let Some(want) = want else { continue };
            if !cands.contains(&want) {
                continue; // only condition on shortlist hit
            }
            total += 1;
            let combined_top = lm.predict_next_fast(&ctx, 1).first().map(|c| c.id);
            let tba_top = lm.predict_tba_only(&ctx, 1).first().map(|c| c.id);
            let tri_top = predict_trigram_only(lm, &ctx, 1).first().map(|c| c.id);
            let eng_top = lm.predict_engram_only(&ctx, 1).first().map(|c| c.id);
            if combined_top == Some(want) { combined += 1; }
            if tba_top == Some(want) { tba += 1; }
            if tri_top == Some(want) { tri += 1; }
            if eng_top == Some(want) { eng += 1; }
        }
    }
    if total == 0 {
        (0.0, 0.0, 0.0, 0.0, 0)
    } else {
        let t = total as f32;
        (combined as f32 / t, tba as f32 / t, tri as f32 / t, eng as f32 / t, total)
    }
}

/// Fraction of next-token pairs where the correct token is in the engram
/// top-K shortlist (the ceiling of the shortlist+rerank architecture).
fn shortlist_recall(lm: &VsaLm, sentences: &[String], k: usize, max_pairs: usize) -> (f32, usize) {
    let mut hit = 0usize;
    let mut total = 0usize;
    'outer: for sentence in sentences {
        let tokens = lm.tokenize(sentence);
        for pos in 0..tokens.len().saturating_sub(1) {
            if total >= max_pairs {
                break 'outer;
            }
            let ctx: Vec<String> = tokens[..=pos].to_vec();
            let ctx_ids: Vec<usize> = ctx.iter().filter_map(|w| lm.vocab.id(w)).collect();
            let cands = lm.engram.top_candidates(&ctx_ids, k);
            let want = lm.vocab.id(&tokens[pos + 1]);
            total += 1;
            if let Some(want) = want {
                if cands.contains(&want) {
                    hit += 1;
                }
            }
        }
    }
    if total == 0 {
        (0.0, 0)
    } else {
        (hit as f32 / total as f32, total)
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: vsalm-scale <corpus.txt> [limit] [ratio]");
    let limit: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(10_000);
    let train_ratio: f32 = std::env::args().nth(3).and_then(|v| v.parse().ok()).unwrap_or(0.8);

    let raw = fs::read_to_string(&path).expect("read corpus");
    // Split into sentence-like units; filter noise tokens.
    let mut sentences: Vec<String> = raw
        .split(|c: char| matches!(c, '.' | '!' | '?' | '\n'))
        .map(|s| s.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|s| s.split_whitespace().count() >= 4)
        .filter(|s| s.split_whitespace().count() <= 60)
        .filter(|s| !s.contains("<unk>") && !s.contains("@-@") && !s.contains('='))
        .take(limit)
        .map(|s| s.to_string())
        .collect();

    let split = (sentences.len() as f32 * train_ratio) as usize;
    let test: Vec<String> = sentences.split_off(split);
    let train = sentences;

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  VSA-LM scale — real corpus, next-token generalization  ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("Corpus: {} ({} train / {} test)", path, train.len(), test.len());

    let config = LmConfig {
        dim: env_f("AXIOM_LM_DIM", 4096.0) as usize,
        max_order: 4,
        beam_width: 8,
        max_gen_tokens: 12,
        w_trigram: env_f("AXIOM_LM_W_TRI", 0.6),
        w_tba: env_f("AXIOM_LM_W_TBA", 1.0),
        w_engram: env_f("AXIOM_LM_W_ENG", 1.5),
        w_reservoir: env_f("AXIOM_LM_W_RES", 0.5),
        ..Default::default()
    };
    println!("  weights: dim={} tba={} tri={} eng={} res={}", config.dim, config.w_tba, config.w_trigram, config.w_engram, config.w_reservoir);
    let mut lm = VsaLm::new(config);

    println!("\nStep 1: Ingesting {} train sentences...", train.len());
    let t0 = Instant::now();
    for s in &train {
        lm.learn(s);
    }
    println!("  {} words in {:.2?}", lm.vocab.len(), t0.elapsed());
    print!("  Building TBA TopK cache... ");
    let t0 = Instant::now();
    lm.build_tba_cache();
    println!("done in {:.2?}", t0.elapsed());

    println!("\nStep 2: Next-token accuracy (no softmax, no backprop)");
    let t0 = Instant::now();
    let (train_acc, train_n) = lm.next_token_accuracy_sample(&train, 500);
    let (test_acc, test_n) = lm.next_token_accuracy_sample(&test, 300);
    println!("  TRAIN: {:.1}% ({}/{})", train_acc * 100.0, train_n, train_n);
    println!("  TEST:  {:.1}% ({}/{})", test_acc * 100.0, test_n, test_n);
    println!("  elapsed (accuracy): {:.2?}", t0.elapsed());

    // A3: shortlist recall — is the correct next token IN the engram top-K?
    // This is the ceiling of the shortlist+rerank architecture.
    for k in [16usize, 32, 64, 128] {
        let (recall, _) = shortlist_recall(&lm, &test, k, 300);
        println!("  TEST engram top-{} shortlist recall: {:.1}%", k, recall * 100.0);
    }
    // A3: conditional accuracy — when the correct token IS in the top-32,
    // how often does each signal (and the combined) rank it #1? This isolates
    // the rerank quality (shortlist recall 28.7% vs combined accuracy 11% ⇒
    // the rerank loses ~17.7pt).
    {
        let (cond_combined, cond_tba, cond_tri, cond_eng, n) = conditional_rerank(&lm, &test, 32, 300);
        println!("  of {n} TEST pairs with correct in top-32 shortlist:");
        println!("    combined picks it: {:.1}%", cond_combined * 100.0);
        println!("    TBA-only picks it:  {:.1}%", cond_tba * 100.0);
        println!("    trigram-only:       {:.1}%", cond_tri * 100.0);
        println!("    engram-only:        {:.1}%", cond_eng * 100.0);
    }
    // A3: UNION shortlist recall (engram top-32 ∪ TBA-cache top-32 ∪ trigram
    // top-32) — the candidate pool of a multi-source architecture.
    {
        let (union_recall, _) = union_shortlist_recall(&lm, &test, 32, 300);
        println!("  TEST UNION shortlist recall (eng∪tba∪tri): {:.1}%", union_recall * 100.0);
    }

    println!("\nStep 3: Signal decomposition (small sample)");
    let t0 = Instant::now();
    let (tba_acc, _) = tba_only_accuracy_sample(&lm, &train, 100);
    let (tri_acc, _) = trigram_only_accuracy_sample(&lm, &train, 100);
    let (eng_acc, _) = engram_only_accuracy_sample(&lm, &train, 100);
    println!("  TRAIN TBA (bigram VSA): {:.1}%", tba_acc * 100.0);
    println!("  TRAIN Trigram (trigram VSA): {:.1}%", tri_acc * 100.0);
    println!("  TRAIN Engram (n-gram): {:.1}%", eng_acc * 100.0);

    let (tba_t, _) = tba_only_accuracy_sample(&lm, &test, 300);
    let (tri_t, _) = trigram_only_accuracy_sample(&lm, &test, 300);
    let (eng_t, _) = engram_only_accuracy_sample(&lm, &test, 300);
    println!("  TEST  TBA (bigram VSA): {:.1}%", tba_t * 100.0);
    println!("  TEST  Trigram (trigram VSA): {:.1}%", tri_t * 100.0);
    println!("  TEST  Engram (n-gram): {:.1}%", eng_t * 100.0);
    println!("  elapsed: {:.2?}", t0.elapsed());

    println!("\nStep 4: Generation (deterministic)");
    let prompts = ["the game", "the player", "the series", "a character", "the story"];
    for prompt in prompts {
        let out = lm.generate(prompt, Some(12));
        println!("  \"{}\" → \"{}\"", prompt, out);
    }

    println!("\nStep 5: Determinism check");
    let mut unique = std::collections::HashSet::new();
    for _ in 0..5 {
        unique.insert(lm.generate("the game", Some(6)));
    }
    println!("  Unique from 5 runs: {} (expected 1) → {}", unique.len(), if unique.len() == 1 { "✓" } else { "✗" });

    println!("\n━━━ VSA-LM SCALE RESULTS ━━━");
    println!("  Vocab: {} words", lm.vocab.len());
    println!("  TRAIN next-token acc: {:.1}%", train_acc * 100.0);
    println!("  TEST  next-token acc: {:.1}%", test_acc * 100.0);
    println!("  No backprop / no softmax / deterministic: ✓");
}

fn tba_only_accuracy_sample(lm: &VsaLm, sentences: &[String], max_pairs: usize) -> (f32, usize) {
    let mut correct = 0usize;
    let mut total = 0usize;
    'outer: for sentence in sentences {
        let tokens = lm.tokenize(sentence);
        for pos in 0..tokens.len().saturating_sub(1) {
            if total >= max_pairs { break 'outer; }
            let ctx: Vec<String> = tokens[..=pos].to_vec();
            let pred = lm.predict_tba_only(&ctx, 5);
            if pred.is_empty() { continue; }
            let tid = lm.vocab.id(&tokens[pos + 1]);
            total += 1;
            if Some(pred[0].id) == tid { correct += 1; }
        }
    }
    if total == 0 { (0.0, 0) } else { (correct as f32 / total as f32, total) }
}

fn trigram_only_accuracy_sample(lm: &VsaLm, sentences: &[String], max_pairs: usize) -> (f32, usize) {
    let mut correct = 0usize;
    let mut total = 0usize;
    'outer: for sentence in sentences {
        let tokens = lm.tokenize(sentence);
        for pos in 0..tokens.len().saturating_sub(1) {
            if total >= max_pairs { break 'outer; }
            let ctx: Vec<String> = tokens[..=pos].to_vec();
            let pred = predict_trigram_only(lm, &ctx, 5);
            if pred.is_empty() { continue; }
            let tid = lm.vocab.id(&tokens[pos + 1]);
            total += 1;
            if Some(pred[0].id) == tid { correct += 1; }
        }
    }
    if total == 0 { (0.0, 0) } else { (correct as f32 / total as f32, total) }
}

fn predict_trigram_only(lm: &VsaLm, context: &[String], k: usize) -> Vec<tle_vsa_lm::decode::DecodedToken> {
    let signal = lm.trigram_prediction(context);
    let Some(signal) = signal else { return Vec::new(); };
    let mut cands: Vec<tle_vsa_lm::decode::DecodedToken> = lm.vocab.iter()
        .map(|(id, word)| {
            let sim = tle_vsa::cosine_similarity(&signal, lm.vocab.vector_by_id(id).unwrap());
            tle_vsa_lm::decode::DecodedToken { id, word: word.to_string(), similarity: sim }
        })
        .collect();
    cands.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
    cands.truncate(k);
    cands
}

fn engram_only_accuracy_sample(lm: &VsaLm, sentences: &[String], max_pairs: usize) -> (f32, usize) {
    let mut correct = 0usize;
    let mut total = 0usize;
    'outer: for sentence in sentences {
        let tokens = lm.tokenize(sentence);
        for pos in 0..tokens.len().saturating_sub(1) {
            if total >= max_pairs { break 'outer; }
            let ctx: Vec<String> = tokens[..=pos].to_vec();
            let pred = lm.predict_engram_only(&ctx, 5);
            if pred.is_empty() { continue; }
            let tid = lm.vocab.id(&tokens[pos + 1]);
            total += 1;
            if Some(pred[0].id) == tid { correct += 1; }
        }
    }
    if total == 0 { (0.0, 0) } else { (correct as f32 / total as f32, total) }
}

/// Union shortlist recall: correct token in (engram top-K ∪ TBA-cache top-K
/// ∪ trigram-VSA top-K). The candidate pool of a multi-source architecture.
fn union_shortlist_recall(lm: &VsaLm, sentences: &[String], k: usize, max_pairs: usize) -> (f32, usize) {
    use std::collections::HashSet;
    let mut hit = 0usize;
    let mut total = 0usize;
    'outer: for sentence in sentences {
        let tokens = lm.tokenize(sentence);
        for pos in 0..tokens.len().saturating_sub(1) {
            if total >= max_pairs {
                break 'outer;
            }
            let ctx: Vec<String> = tokens[..=pos].to_vec();
            let ctx_ids: Vec<usize> = ctx.iter().filter_map(|w| lm.vocab.id(w)).collect();
            let mut pool: HashSet<usize> = lm.engram.top_candidates(&ctx_ids, k).into_iter().collect();
            // TBA cache top-k (per-source bigram transitions)
            if let Some(last) = ctx_ids.last() {
                for (id, _) in lm.tba_cache_top_k(*last, k) {
                    pool.insert(id);
                }
            }
            // trigram VSA top-k
            if let Some(sig) = lm.trigram_prediction(&ctx) {
                let mut scored: Vec<(usize, f32)> = lm.vocab.iter()
                    .map(|(id, word)| (id, tle_vsa::cosine_similarity(&sig, lm.vocab.vector_by_id(id).unwrap())))
                    .collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (id, _) in scored.into_iter().take(k) {
                    pool.insert(id);
                }
            }
            let want = lm.vocab.id(&tokens[pos + 1]);
            total += 1;
            if let Some(want) = want {
                if pool.contains(&want) {
                    hit += 1;
                }
            }
        }
    }
    if total == 0 {
        (0.0, 0)
    } else {
        (hit as f32 / total as f32, total)
    }
}
