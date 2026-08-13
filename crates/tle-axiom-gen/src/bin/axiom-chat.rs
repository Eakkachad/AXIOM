//! `axiom-chat` — Hybrid Conversational AXIOM (Phase B).
//!
//! A deterministic, zero-training, CPU-only chat experience that combines:
//!   1. **Graph reasoning** (AxiomGen): `/teach sky is blue` → knowledge graph →
//!      multi-hop compositional answers ("why is the sky blue?" → "because the
//!      blue has short wavelength").
//!   2. **VSA-LM fluency** (tle-vsa-lm): `/teach` also trains the non-neural
//!      language model; `gen <prompt>` generates fluent free-form continuations.
//!   3. **Conversational handling**: greetings / thanks / what-can-you-do /
//!      honest "I don't know" (never hallucinates — it just says it doesn't know).
//!
//! Deterministic: same input → same output. No GPU, no gradients, no sampling.

use std::io::{self, Write};
use std::time::Instant;

use tle_axiom_gen::AxiomGen;
use tle_vsa_lm::{LmConfig, VsaLm};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   AXIOM-CHAT — deterministic conversational reasoning         ║");
    println!("║   zero-training · CPU-only · 100% reproducible                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    let mut graph = AxiomGen::new(2048);
    graph.search_config.max_hops = 3;
    graph.search_config.beam_width = 16;

    let mut lm = VsaLm::new(LmConfig {
        dim: 4096,
        max_order: 4,
        beam_width: 8,
        max_gen_tokens: 24,
        w_knowledge: 3.0,
        ..Default::default()
    });

    // optional corpus for VSA-LM fluency (gen mode): `axiom-chat --corpus <file>`
    if std::env::args().any(|a| a == "--corpus") {
        if let Some(path) = std::env::args().nth(2) {
            let text = std::fs::read_to_string(&path)?;
            let mut n = 0;
            for line in text.lines().filter(|l| l.len() > 8).take(2000) {
                lm.learn(line);
                n += 1;
            }
            println!("  VSA-LM corpus: learned {} sentences from {}", n, path);
        }
    }

    // optional starter facts so a first chat isn't empty
    for fact in [("sky", "is", "blue"), ("blue", "has", "short_wavelength")] {
        teach(&mut graph, &mut lm, &format!("{} {} {}", fact.0, fact.1, fact.2));
    }

    println!();
    println!("  Commands:");
    println!("    /teach <fact>     learn a fact  (e.g. /teach cats are animals)");
    println!("    /ask <subject> <rel>   query the graph directly");
    println!("    gen <prompt>      VSA-LM free-form generation");
    println!("    /stats /quit");
    println!("  Try:  \"why is the sky blue?\"   or   \"/teach water is liquid\"");
    println!();

    let stdin = io::stdin();
    let mut last_subject: Option<String> = None;

    loop {
        eprint!("you> ");
        io::stderr().flush().ok();
        let mut input = String::new();
        match stdin.read_line(&mut input) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }
        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        let start = Instant::now();
        let lower = trimmed.to_lowercase();

        if trimmed == "/quit" || trimmed == "/exit" || trimmed == "/q" {
            println!("  Goodbye!");
            break;
        }
        if let Some(fact) = trimmed.strip_prefix("/teach ") {
            teach(&mut graph, &mut lm, fact);
            println!("  Got it — taught \"{}\" [deterministic ✓]", fact);
            continue;
        }
        if let Some(q) = trimmed.strip_prefix("/ask ") {
            let parts: Vec<&str> = q.split_whitespace().collect();
            if parts.len() >= 2 {
                let subj = parts[0];
                let rel = parts[1];
                let mut found = false;
                if let Some(sid) = graph.graph.entity_id(subj) {
                    for t in &graph.graph.triples {
                        if t.subject_id == sid {
                            let r = graph.graph.relation_name(t.relation_id);
                            if r == rel {
                                println!("  {} {} {} [{:?}]", capitalize(subj), rel,
                                    graph.graph.entity_name(t.object_id), start.elapsed());
                                found = true;
                            }
                        }
                    }
                }
                if !found {
                    println!("  I don't know \"{} {}\" yet. Teach me with /teach.", subj, rel);
                }
            } else {
                println!("  Usage: /ask <subject> <relation>");
            }
            continue;
        }
        if let Some(prompt) = trimmed.strip_prefix("gen ") {
            let out = lm.generate(prompt, Some(24));
            println!("  {} [{:?}]", out, start.elapsed());
            continue;
        }
        if trimmed == "/stats" {
            println!("  graph triples: {}", graph.graph.triples.len());
            println!("  vocab: {} words", lm.vocab.len());
            continue;
        }

        // ── conversational handling ────────────────────────────────────────
        if is_greeting(&lower) {
            println!("  Hello! I'm AXIOM — a deterministic reasoning engine.");
            println!("  I don't guess: teach me facts with /teach, then ask me questions.");
            println!("  I can chain facts, e.g. teach \"cats are animals\" + \"animals have hearts\"");
            println!("  then ask \"do cats have hearts?\"");
            continue;
        }
        if is_thanks(&lower) {
            println!("  You're welcome! Teach me more whenever you like.");
            continue;
        }
        if lower.contains("what can you do") || lower.contains("help") || lower.contains("who are you") {
            println!("  I reason over facts you teach me. Examples:");
            println!("    /teach earth orbits the sun");
            println!("    /teach the sun is a star");
            println!("    \"what orbits the sun?\"   →   earth");
            println!("    \"what is the sun?\"       →   a star");
            continue;
        }
        if lower.contains("how are you") {
            println!("  Operating at 100% determinism — every run identical. That's my kind of stable.");
            continue;
        }

        // ── factual question ───────────────────────────────────────────────
        let gen = graph.generate(&trimmed);
        if gen.path_length >= 1 && !gen.sentence.is_empty() {
            println!("  {}", gen.sentence);
            if !gen.reasoning.is_empty() {
                println!("  [trace: {}]", gen.reasoning.join(" → "));
            }
            if !gen.answer.is_empty() {
                last_subject = Some(gen.answer.clone());
            }
        } else if let Some(prev) = last_subject.take() {
            // follow-up on the last topic
            let gen = graph.generate(&format!("{} {}", prev, trimmed));
            if gen.path_length >= 1 && !gen.sentence.is_empty() {
                println!("  {}", gen.sentence);
            } else {
                println!("  I don't know about \"{}\" yet. Teach me with /teach.", trimmed);
            }
        } else {
            println!("  I don't know \"{}\" yet. Teach me with /teach, or ask me to generate with \"gen ...\".", trimmed);
        }
        println!("  [{:?}]", start.elapsed());
    }
    Ok(())
}

/// Teach a fact to both the graph and the VSA-LM.
fn teach(graph: &mut AxiomGen, lm: &mut VsaLm, fact: &str) {
    // extract triples the same way the pipeline does
    for d in tle_axiom_gen::decompose::decompose_sentence(fact, "") {
        graph.add_fact(&d.subject, &d.relation, &d.object);
    }
    lm.learn(fact);
    graph.sync_into_vsa_lm(lm);
}

fn is_greeting(lower: &str) -> bool {
    ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "yo", "sup"]
        .iter()
        .any(|g| lower == *g || lower.starts_with(&format!("{},", g)) || lower.contains(g))
}

fn is_thanks(lower: &str) -> bool {
    ["thanks", "thank you", "cheers", "thx", "appreciate it"]
        .iter()
        .any(|t| lower.contains(t))
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}
