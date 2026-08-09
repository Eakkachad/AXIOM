//! VSA-LM knowledge demo: teach facts into the KnowledgePrior, then generate
//! fact-consistent answers without any neural net.
//!
//! Demonstrates the AXIOM vision end-to-end on the VSA-LM path:
//!   teach (sky, is, blue) ; (blue, has, short_wavelength)
//!   → "why is the sky blue?" → "the sky is blue because blue has short wavelength"

use tle_vsa_lm::{LmConfig, KnowledgePrior, VsaLm};

fn main() {
    let config = LmConfig {
        dim: 4096,
        max_order: 3,
        beam_width: 8,
        max_gen_tokens: 12,
        w_knowledge: 3.0,
        ..Default::default()
    };
    let mut lm = VsaLm::new(config);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  VSA-LM Knowledge Demo — teach facts → generate answers      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!("\nTeaching facts into KnowledgePrior...");
    teach(&mut lm, &[
        ("sky", "is", "blue"),
        ("blue", "has", "short_wavelength"),
        ("short_wavelength", "scatters", "in_atmosphere"),
        ("cat", "is", "animal"),
        ("animal", "has", "heart"),
        ("water", "is", "liquid"),
        ("sun", "is", "bright"),
        ("sun", "rises", "in_east"),
        ("Einstein", "developed", "relativity"),
        ("Mars", "is", "red_planet"),
        ("bird", "has", "wings"),
        ("fish", "has", "gills"),
    ]);
    println!("  {} facts, vocab {} words", lm.knowledge.facts, lm.vocab.len());

    println!("\n--- Knowledge-guided generation ---");
    let prompts = [
        "why is the sky",
        "what does blue have",
        "what is a cat",
        "does a cat have",
        "what is water",
        "what does a bird have",
        "what is Mars",
        "who developed relativity",
    ];
    for prompt in prompts {
        let out = lm.generate(prompt, Some(10));
        println!("  \"{}\" → \"{}\"", prompt, out);
    }

    println!("\n--- Determinism ---");
    let mut unique = std::collections::HashSet::new();
    for _ in 0..5 {
        unique.insert(lm.generate("why is the sky", Some(10)));
    }
    println!("  Unique from 5 runs: {} (expected 1) → {}", unique.len(), if unique.len() == 1 { "✓" } else { "✗" });
    println!("\nDone. No softmax, no backprop, deterministic.");
}

/// Teach a list of facts into the engine: register vocabulary and add the
/// fact to the knowledge prior.
fn teach(lm: &mut VsaLm, facts: &[(&str, &str, &str)]) {
    for (s, r, o) in facts {
        // Register entity words in the vocabulary so they can be generated.
        for word in [s, r, o] {
            for w in word.split(|c: char| c == '_' || c == ' ') {
                lm.vocab.get_or_add(w);
            }
        }
        lm.knowledge.add_fact(s, r, o);
    }
}
