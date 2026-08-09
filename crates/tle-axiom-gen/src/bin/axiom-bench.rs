//! Deterministic AXIOM-Gen quality benchmark.

use std::time::Instant;
use tle_axiom_gen::AxiomGen;

fn main() {
    let mut engine = AxiomGen::new(2048);
    for index in 0..10 {
        engine.add_fact(&format!("node_{}", index), "leads_to", &format!("node_{}", index + 1));
    }
    engine.search_config.max_hops = 10;
    engine.search_config.beam_width = 32;
    engine.energy_config.target_length = 10;
    engine.energy_config.lambda_length = 0.5;

    let start = Instant::now();
    let first = engine.generate("why does node_0 reach node_10?");
    let first_elapsed = start.elapsed();

    let start = Instant::now();
    let second = engine.generate("why does node_0 reach node_10?");
    let second_elapsed = start.elapsed();

    println!("AXIOM-Gen benchmark");
    println!("  path_length: {}", first.path_length);
    println!("  deterministic: {}", first.sentence == second.sentence && first.reasoning == second.reasoning);
    println!("  first_latency: {:?}", first_elapsed);
    println!("  second_latency: {:?}", second_elapsed);
    println!("  sentence: {}", first.sentence);
    println!("  reasoning_steps: {}", first.reasoning.len());

    assert!(first.path_length >= 10, "expected a 10-hop path");
    assert_eq!(first.sentence, second.sentence, "generation must be deterministic");
    assert_eq!(first.reasoning, second.reasoning, "reasoning trace must be deterministic");
}
