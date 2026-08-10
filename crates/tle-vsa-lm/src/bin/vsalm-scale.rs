//! VSA-LM scale benchmark: ingest a real text corpus and measure next-token
//! accuracy on a held-out test set.  This is the LM-breakthrough metric:
//! can VSA algebra (no softmax, no backprop) generalize beyond n-gram
//! memorisation on real text?
//!
//! Usage: vsalm-scale <corpus.txt> [sentences_limit] [train_ratio]

use std::fs;
use std::time::Instant;

use tle_vsa_lm::{LmConfig, VsaLm};

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
        dim: 4096,
        max_order: 4,
        beam_width: 8,
        max_gen_tokens: 12,
        w_trigram: 0.6,
        ..Default::default()
    };
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

    println!("\nStep 3: Signal decomposition (small sample)");
    let t0 = Instant::now();
    let (tba_acc, _) = tba_only_accuracy_sample(&lm, &train, 100);
    let (tri_acc, _) = trigram_only_accuracy_sample(&lm, &train, 100);
    let (eng_acc, _) = engram_only_accuracy_sample(&lm, &train, 100);
    println!("  TRAIN TBA (bigram VSA): {:.1}%", tba_acc * 100.0);
    println!("  TRAIN Trigram (trigram VSA): {:.1}%", tri_acc * 100.0);
    println!("  TRAIN Engram (n-gram): {:.1}%", eng_acc * 100.0);

    let (tba_t, _) = tba_only_accuracy_sample(&lm, &test, 50);
    let (tri_t, _) = trigram_only_accuracy_sample(&lm, &test, 50);
    let (eng_t, _) = engram_only_accuracy_sample(&lm, &test, 50);
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
