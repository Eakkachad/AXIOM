//! Engram Demo — Build from WikiText corpus and demonstrate generation.
//!
//! Usage: cargo run --release -p tle-engram

use std::fs;
use std::time::Instant;

use tle_engram::{EngramBuilder, SigmoidFusion};
use tle_engram::builder::BuilderConfig;

fn main() {
    println!("═══════════════════════════════════════════════════");
    println!("  ENGRAM — Multi-Head N-gram Hash Table Demo");
    println!("  Deep Man Architecture: Layer 1 (Fast Facts)");
    println!("═══════════════════════════════════════════════════\n");

    // Load WikiText corpus
    let data_path = "data/wiki_train.txt";
    let data = match fs::read_to_string(data_path) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("Could not find {}. Using built-in demo corpus.", data_path);
            include_str!("demo_corpus.txt").to_string()
        }
    };

    let config = BuilderConfig {
        max_order: 5,
        min_count: 2,       // ignore hapax legomena
        max_vocab: 5000,    // top 5K words
        max_candidates_per_entry: 30,
    };

    // Build Engram
    println!("[1] Building Engram from corpus...");
    let start = Instant::now();
    let mut builder = EngramBuilder::with_config(config);

    let mut line_count = 0;
    for line in data.lines() {
        let trimmed = line.trim().to_lowercase();
        if !trimmed.is_empty() && trimmed.len() > 10 {
            builder.ingest_line(&trimmed);
            line_count += 1;
        }
    }

    let build_time = start.elapsed();
    println!("    Lines ingested: {}", line_count);
    println!("    Tokens processed: {}", builder.total_tokens);
    println!("    Vocabulary size: {}", builder.vocab.len());
    println!("    Build time: {:?}", build_time);
    println!(
        "    Throughput: {:.0} tokens/sec\n",
        builder.total_tokens as f64 / build_time.as_secs_f64()
    );

    // Freeze tables
    let start = Instant::now();
    let engram = builder.build();
    let freeze_time = start.elapsed();

    println!("[2] Frozen tables:");
    for table in &engram.tables {
        let stats = table.stats();
        println!("    {}", stats);
    }
    println!("    Freeze time: {:?}\n", freeze_time);

    // Demonstrate generation
    println!("[3] Generation demo (greedy argmax from Engram):\n");

    let prompts = vec![
        "the president of",
        "the city of",
        "he was the",
        "in the united",
        "the first",
    ];

    let fusion = SigmoidFusion::new(engram.vocab.len());

    for prompt in prompts {
        let tokens: Vec<&str> = prompt.split_whitespace().collect();
        let mut context: Vec<u16> = tokens
            .iter()
            .filter_map(|t| engram.vocab.get_id(t))
            .collect();

        if context.is_empty() {
            println!("  \"{}\" → [unknown tokens]", prompt);
            continue;
        }

        let mut generated = Vec::new();
        let start = Instant::now();

        // Generate up to 10 tokens
        for _ in 0..10 {
            let results = engram.query(&context);
            if results.is_empty() {
                break;
            }

            let fused = fusion.fuse(
                &results
                    .iter()
                    .map(|(order, conf, entry)| (*order, *conf, *entry))
                    .collect::<Vec<_>>(),
            );

            match SigmoidFusion::select_best(&fused) {
                Some(best_id) => {
                    generated.push(best_id);
                    context.push(best_id);
                }
                None => break,
            }
        }

        let gen_time = start.elapsed();
        let gen_str: String = generated
            .iter()
            .filter_map(|&id| engram.vocab.get_token(id))
            .collect::<Vec<_>>()
            .join(" ");

        println!("  \"{}\" → {} [{:?}]", prompt, gen_str, gen_time);
    }

    // Performance summary
    println!("\n[4] Performance summary:");
    println!("    Vocab: {} tokens", engram.vocab.len());
    println!("    Total N-gram contexts stored:");
    let total_contexts: usize = engram.tables.iter().map(|t| t.len()).sum();
    println!("      {} contexts across {} heads", total_contexts, engram.tables.len());
    let total_memory: f64 = engram.tables.iter().map(|t| t.stats().memory_kb).sum();
    println!("    Estimated memory: {:.1} KB", total_memory);
    println!("    Deterministic: YES (same input → same output)");
    println!("    Training: NONE (single-pass counting only)");
}
