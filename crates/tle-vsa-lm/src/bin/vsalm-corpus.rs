//! VSA-LM corpus benchmark: learn from a real text file and measure
//! next-token accuracy + generation at scale.
//!
//! Usage: vsalm-corpus <corpus.txt> [train_ratio]

use std::fs;
use std::time::Instant;

use tle_vsa_lm::{LmConfig, VsaLm};

fn main() {
    let path = std::env::args().nth(1).expect("usage: vsalm-corpus <corpus.txt> [train_ratio]");
    let train_ratio: f32 = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(0.8);

    let raw = fs::read_to_string(&path).expect("read corpus");
    // Split into sentence-like units on punctuation, filter noise.
    let mut sentences: Vec<String> = raw
        .split(|c: char| matches!(c, '.' | '!' | '?' | '\n'))
        .map(|s| s.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|s| s.split_whitespace().count() >= 4)
        .filter(|s| s.split_whitespace().count() <= 60)
        .filter(|s| !s.contains("<unk>") && !s.contains("@-@") && !s.contains('='))
        .take(300)
        .map(|s| s.to_string())
        .collect();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  VSA-LM corpus benchmark — real text, no neural net          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("Corpus file: {}", path);
    println!("Sentences: {}", sentences.len());

    // Deterministic split (train_ratio of first sentences, then rest).
    let split = (sentences.len() as f32 * train_ratio) as usize;
    let test: Vec<String> = sentences.split_off(split);
    let train = sentences;

    let config = LmConfig {
        dim: 4096,
        max_order: 4,
        beam_width: 8,
        max_gen_tokens: 12,
        reservoir_config: None,
        ..Default::default()
    };
    let mut lm = VsaLm::new(config);

    println!("\nStep 1: Ingesting {} train sentences...", train.len());
    let t0 = Instant::now();
    for s in &train {
        lm.learn(s);
    }
    println!("  {} words in {:.2?}", lm.vocab.len(), t0.elapsed());

    println!("\nStep 2: Next-token accuracy (sampled)");
    let t0 = Instant::now();
    let (train_acc, train_n) = lm.next_token_accuracy_sample(&train, 2000);
    let (test_acc, test_n) = lm.next_token_accuracy_sample(&test, 400);
    println!("  TRAIN: {:.1}% ({}/{})", train_acc * 100.0, train_n, train_n);
    println!("  TEST:  {:.1}% ({}/{})", test_acc * 100.0, test_n, test_n);
    println!("  elapsed: {:.2?}", t0.elapsed());

    println!("\nStep 3: Signal decomposition (sampled)");
    let t0 = Instant::now();
    let (tba_acc, tba_n) = tba_only_accuracy_sample(&lm, &train, 2000);
    let (eng_acc, eng_n) = engram_only_accuracy_sample(&lm, &train, 2000);
    let (tba_test, tba_test_n) = tba_only_accuracy_sample(&lm, &test, 400);
    let (eng_test, eng_test_n) = engram_only_accuracy_sample(&lm, &test, 400);
    println!("  TRAIN TBA-only: {:.1}% ({})   TEST TBA-only: {:.1}% ({})", tba_acc * 100.0, tba_n, tba_test * 100.0, tba_test_n);
    println!("  TRAIN Engram-only: {:.1}% ({})   TEST Engram-only: {:.1}% ({})", eng_acc * 100.0, eng_n, eng_test * 100.0, eng_test_n);
    println!("  elapsed: {:.2?}", t0.elapsed());

    println!("\nStep 4: Generation (deterministic)");
    let prompts = ["the game", "the player", "the series", "a character", "the story"];
    for prompt in prompts {
        let out = lm.generate(prompt, Some(12));
        println!("  \"{}\" → \"{}\"", prompt, out);
    }

    println!("\nStep 5: Determinism");
    let mut unique = std::collections::HashSet::new();
    for _ in 0..5 {
        unique.insert(lm.generate("the game", Some(6)));
    }
    println!("  Unique from 5 runs: {} (expected 1) → {}", unique.len(), if unique.len() == 1 { "✓" } else { "✗" });

    println!("\n━━━ VSA-LM CORPUS RESULTS ━━━");
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
        if tokens.len() < 2 {
            continue;
        }
        for pos in 0..tokens.len() - 1 {
            if total >= max_pairs {
                break 'outer;
            }
            let context: Vec<String> = tokens[..=pos].to_vec();
            let pred = lm.predict_tba_only(&context, 5);
            if pred.is_empty() {
                continue;
            }
            let true_id = lm.vocab.id(&tokens[pos + 1]);
            total += 1;
            if Some(pred[0].id) == true_id {
                correct += 1;
            }
        }
    }
    if total == 0 {
        (0.0, 0)
    } else {
        (correct as f32 / total as f32, total)
    }
}

fn engram_only_accuracy_sample(lm: &VsaLm, sentences: &[String], max_pairs: usize) -> (f32, usize) {
    let mut correct = 0usize;
    let mut total = 0usize;
    'outer: for sentence in sentences {
        let tokens = lm.tokenize(sentence);
        if tokens.len() < 2 {
            continue;
        }
        for pos in 0..tokens.len() - 1 {
            if total >= max_pairs {
                break 'outer;
            }
            let context: Vec<String> = tokens[..=pos].to_vec();
            let pred = lm.predict_engram_only(&context, 5);
            if pred.is_empty() {
                continue;
            }
            let true_id = lm.vocab.id(&tokens[pos + 1]);
            total += 1;
            if Some(pred[0].id) == true_id {
                correct += 1;
            }
        }
    }
    if total == 0 {
        (0.0, 0)
    } else {
        (correct as f32 / total as f32, total)
    }
}
