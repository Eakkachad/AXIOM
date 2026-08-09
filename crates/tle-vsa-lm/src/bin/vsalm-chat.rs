//! vsalm-chat: Corpus-trained VSA-LM + KnowledgePrior = fluent, fact-grounded QA.
//!
//! Pipeline: corpus → VSA-LM (fluency) + facts → KnowledgePrior (truth)
//!          → question → KG-steered generation → fluent answer
//!
//! Usage: vsalm-chat <corpus.txt> [corpus_limit]

use std::fs;
use std::time::Instant;

use tle_vsa_lm::{LmConfig, VsaLm};

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| "data/wiki_train.txt".to_string());
    let corpus_limit: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(3000);

    // ---- 1. Load corpus for fluency ----
    let raw = fs::read_to_string(&corpus_path).expect("read corpus");
    let sentences: Vec<String> = raw
        .split(|c: char| matches!(c, '.' | '!' | '?' | '\n'))
        .map(|s| s.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|s| s.split_whitespace().count() >= 4)
        .filter(|s| s.split_whitespace().count() <= 60)
        .filter(|s| !s.contains("<unk>") && !s.contains("@-@") && !s.contains('='))
        .take(corpus_limit)
        .map(|s| s.to_string())
        .collect();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  AXIOM Chat — corpus fluency + knowledge-grounded answers    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    // ---- 2. Build VSA-LM ----
    let config = LmConfig {
        dim: 4096,
        max_order: 4,
        beam_width: 8,
        max_gen_tokens: 8,  // short answers — fact chain exhausted quickly
        w_trigram: 0.0,
        w_knowledge: 8.0,
        knowledge_only: false, // keep false: corpus fluency enriches answers
        ..Default::default()
    };
    let mut lm = VsaLm::new(config);

    println!("\nLearning corpus ({} sentences)...", sentences.len());
    let t0 = Instant::now();
    for s in &sentences {
        lm.learn(s);
    }
    println!("  {} words in {:.2?}", lm.vocab.len(), t0.elapsed());

    // ---- 3. Teach facts into KnowledgePrior ----
    println!("\nTeaching facts...");
    let facts: &[(&str, &str, &str)] = &[
        ("sky", "is", "blue"),
        ("blue", "has", "short_wavelength"),
        ("short_wavelength", "scatters", "in_atmosphere"),
        ("scattering", "causes", "blue_sky"),
        ("cat", "is", "animal"),
        ("animal", "has", "heart"),
        ("dog", "is", "animal"),
        ("water", "is", "liquid"),
        ("ice", "is", "frozen_water"),
        ("sun", "is", "bright"),
        ("sun", "rises", "in_east"),
        ("sun", "sets", "in_west"),
        ("moon", "is", "bright_at_night"),
        ("bird", "has", "wings"),
        ("bird", "can", "fly"),
        ("fish", "has", "gills"),
        ("fish", "lives", "in_water"),
        ("earth", "orbits", "sun"),
        ("earth", "is", "third_planet"),
        ("martina_hingis", "is", "tennis_player"),
        ("martina_hingis", "won", "australian_open"),
        ("einstein", "developed", "relativity"),
        ("mars", "is", "red_planet"),
        ("mars", "is", "fourth_planet"),
    ];
    for (s, r, o) in facts {
        for word in [s, r, o] {
            for w in word.split('_') {
                lm.vocab.get_or_add(w);
            }
        }
        lm.knowledge.add_fact(s, r, o);
    }
    println!("  {} facts learned", lm.knowledge.facts);

    // ---- 4. QA loop ----
    let questions = [
        "why is the sky blue",
        "what is a cat",
        "does a cat have a heart",
        "what is water",
        "what is ice",
        "where does the sun rise",
        "what does a bird have",
        "can a bird fly",
        "what does a fish have",
        "where does a fish live",
        "what orbits the sun",
        "what did Einstein develop",
        "what is Mars",
    ];

    println!("\n━━━ Knowledge-Grounded Answers ━━━");
    for q in &questions {
        // Seed the generator with knowledge-fact words so the n-gram corpus
        // patterns start from a fact-grounded position. Without this seed,
        // the corpus n-grams dominate and the answer drifts into random text.
        let seed = knowledge_seed(q);
        let out = if seed.is_empty() {
            lm.generate(q, Some(8))
        } else {
            lm.generate(&seed, Some(8))
        };
        println!("  Q: {}", q);
        println!("  A: {}", out);
        println!();
    }

    // ---- 5. Determinism ----
    let mut unique = std::collections::HashSet::new();
    for _ in 0..5 {
        unique.insert(lm.generate("why is the sky blue", Some(12)));
    }
    println!("Determinism: {} unique from 5 runs → {}",
        unique.len(), if unique.len() == 1 { "✓" } else { "✗" });
}

/// Build a knowledge-seeded prompt: find facts whose subjects appear in the
/// question and chain them.  E.g. "does a cat have a heart" → "cat is animal
/// and animal has heart" — multi-hop chaining from the question.
fn knowledge_seed(question: &str) -> String {
    let facts: &[(&str, &str, &str)] = &[
        ("sky", "is", "blue"),
        ("blue", "has", "short_wavelength"),
        ("short_wavelength", "scatters", "in_atmosphere"),
        ("cat", "is", "animal"),
        ("animal", "has", "heart"),
        ("water", "is", "liquid"),
        ("ice", "is", "frozen_water"),
        ("sun", "rises", "in_east"),
        ("sun", "sets", "in_west"),
        ("bird", "has", "wings"),
        ("bird", "can", "fly"),
        ("fish", "has", "gills"),
        ("fish", "lives", "in_water"),
        ("earth", "orbits", "sun"),
        ("earth", "is", "third_planet"),
        ("einstein", "developed", "relativity"),
        ("mars", "is", "red_planet"),
    ];
    let q_lower = question.to_lowercase();

    // Find matching facts and chain them.
    let mut seeds: Vec<String> = Vec::new();
    for (s, r, o) in facts {
        let subj_words: Vec<&str> = s.split('_').collect();
        if subj_words.iter().any(|w| q_lower.contains(w)) {
            let subj = s.replace('_', " ");
            let obj = o.replace('_', " ");
            seeds.push(format!("{} {} {}", subj, r, obj));
        }
    }
    // Multi-hop: if a fact's object is the subject of another fact AND that
    // object's relation is hinted at in the question, chain it.
    for (s, r, o) in facts {
        let obj_words: Vec<&str> = o.split('_').collect();
        if obj_words.iter().any(|w| q_lower.contains(w)) {
            let subj = s.replace('_', " ");
            let obj = o.replace('_', " ");
            let cand = format!("{} {} {}", subj, r, obj);
            if !seeds.contains(&cand) {
                seeds.push(cand);
            }
        }
    }
    seeds.dedup();
    seeds.join(" ")
}
