//! # Topological Latent Engine - Benchmark Suite
//!
//! Runs the three verification benchmarks:
//! 1. Absolute Determinism Test (100 runs bit-identical)
//! 2. Crosstalk Noise Stress Test (SNR under deep superposition)
//! 3. Latency & Throughput Benchmark

use std::time::Instant;
use tle_vsa::{bind, bundle, cosine_similarity, HyperVector, DEFAULT_DIM, ops};
use tle_resonator::{ResonatorNetwork, ResonatorConfig, CleanupRule};
use tle_memory::MemoryBank;
use tle_pipeline::{LatentEngine, engine::EngineConfig};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Topological Latent Execution Engine - Benchmark Suite      ║");
    println!("║  Model-less Non-Parametric Multi-Node Latent MoE            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    bench_determinism();
    println!();
    bench_crosstalk_snr();
    println!();
    bench_latency();
    println!();
    bench_memory_capacity();
}

/// Benchmark 1: Absolute Determinism Test
/// Validates 100% bit-identical outputs across 100 consecutive runs.
fn bench_determinism() {
    println!("━━━ Benchmark 1: Absolute Determinism (100 runs) ━━━");

    let config = EngineConfig::default();
    let input = "the cat sat on the mat";
    let num_runs = 100;

    let mut outputs = Vec::with_capacity(num_runs);
    let start = Instant::now();

    for _ in 0..num_runs {
        let mut engine = LatentEngine::new(config.clone());
        let result = engine.process(input);
        outputs.push(result.output);
    }

    let elapsed = start.elapsed();
    let first = &outputs[0];
    let all_identical = outputs.iter().all(|o| o == first);

    println!("  Input: \"{}\"", input);
    println!("  Output: \"{}\"", first);
    println!("  Runs: {}", num_runs);
    println!("  All identical: {} ✓", all_identical);
    println!("  Total time: {:.2?}", elapsed);
    println!("  Avg per run: {:.2?}", elapsed / num_runs as u32);

    if !all_identical {
        let mismatches: Vec<usize> = outputs
            .iter()
            .enumerate()
            .filter(|(_, o)| *o != first)
            .map(|(i, _)| i)
            .collect();
        println!("  ✗ FAILED: {} mismatches at indices {:?}", mismatches.len(), &mismatches[..5.min(mismatches.len())]);
    } else {
        println!("  ✓ PASSED: Zero-sampling determinism verified");
    }
}

/// Benchmark 2: Crosstalk Noise Stress Test
/// Measures SNR recovery accuracy under deep multi-fact superposition.
fn bench_crosstalk_snr() {
    println!("━━━ Benchmark 2: Crosstalk Noise Stress Test ━━━");

    let dim = DEFAULT_DIM;
    let test_sizes = [5, 10, 25, 50, 100, 200, 500];

    println!("  D = {}", dim);
    println!("  {:>6} | {:>12} | {:>12} | {:>12} | {:>8}", "k", "Theoretical", "Raw Unbind", "After Cleanup", "Recovery");
    println!("  {}", "-".repeat(70));

    for &k in &test_sizes {
        // Generate random bindings
        let roles: Vec<HyperVector> = (0..k)
            .map(|i| HyperVector::random_bipolar(dim, i as u64 * 1000))
            .collect();
        let fillers: Vec<HyperVector> = (0..k)
            .map(|i| HyperVector::random_bipolar(dim, i as u64 * 1000 + 500))
            .collect();

        // Create superposition
        let bindings: Vec<HyperVector> = roles
            .iter()
            .zip(fillers.iter())
            .map(|(r, f)| bind(r, f))
            .collect();
        let binding_refs: Vec<&HyperVector> = bindings.iter().collect();
        let composite = bundle(&binding_refs);

        // Measure raw unbinding quality
        let raw_unbound = tle_vsa::unbind(&roles[0], &composite);
        let raw_sim = cosine_similarity(&raw_unbound, &fillers[0]);

        // Measure after sign cleanup
        let cleaned = raw_unbound.sign();
        let cleaned_sim = cosine_similarity(&cleaned, &fillers[0]);

        // Theoretical SNR
        let theoretical = ops::theoretical_snr(dim, k);

        // Recovery success (> 0.9 similarity)
        let recovery = if cleaned_sim > 0.9 { "PASS" } else if cleaned_sim > 0.5 { "WEAK" } else { "FAIL" };

        println!(
            "  {:>6} | {:>12.2} | {:>12.4} | {:>12.4} | {:>8}",
            k, theoretical, raw_sim, cleaned_sim, recovery
        );
    }

    println!();
    println!("  Legend: Theoretical = √(D/(k-1)), Raw = cos(unbind, target),");
    println!("         Cleanup = cos(sign(unbind), target)");
}

/// Benchmark 3: Latency & Throughput
fn bench_latency() {
    println!("━━━ Benchmark 3: Latency & Throughput ━━━");

    let config = EngineConfig::default();
    let sentences = [
        "the cat sat on the mat",
        "I love programming in Rust",
        "hello world",
        "the big red dog ran fast",
        "we think that is good",
    ];

    // Warmup
    let mut engine = LatentEngine::new(config.clone());
    engine.process("warmup");

    // Measure individual sentence latency
    println!("  Per-sentence latency:");
    let mut total_tokens = 0;
    let mut total_time = std::time::Duration::ZERO;

    for sentence in &sentences {
        let mut engine = LatentEngine::new(config.clone());
        let start = Instant::now();
        let result = engine.process(sentence);
        let elapsed = start.elapsed();

        total_tokens += result.tokens_processed;
        total_time += elapsed;

        println!("    \"{}\" → {:.2?} ({} tokens)", sentence, elapsed, result.tokens_processed);
    }

    println!();
    println!("  Aggregate:");
    println!("    Total sentences: {}", sentences.len());
    println!("    Total tokens: {}", total_tokens);
    println!("    Total time: {:.2?}", total_time);
    println!("    Avg latency/sentence: {:.2?}", total_time / sentences.len() as u32);
    println!("    Throughput: {:.1} tokens/sec", total_tokens as f64 / total_time.as_secs_f64());

    // Measure VSA operation throughput
    println!();
    println!("  VSA operation microbenchmarks (D={}):", DEFAULT_DIM);

    let a = HyperVector::random_bipolar(DEFAULT_DIM, 1);
    let b = HyperVector::random_bipolar(DEFAULT_DIM, 2);
    let iters = 10_000;

    let start = Instant::now();
    for _ in 0..iters {
        let _ = bind(&a, &b);
    }
    let bind_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..iters {
        let _ = cosine_similarity(&a, &b);
    }
    let sim_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..iters {
        let _ = a.dot(&b);
    }
    let dot_time = start.elapsed();

    println!("    Bind (Hadamard):     {:.1?}/op ({} ops)", bind_time / iters as u32, iters);
    println!("    Cosine Similarity:   {:.1?}/op ({} ops)", sim_time / iters as u32, iters);
    println!("    Dot Product:         {:.1?}/op ({} ops)", dot_time / iters as u32, iters);
}

/// Benchmark 4: Memory Bank Capacity
fn bench_memory_capacity() {
    println!("━━━ Benchmark 4: Memory Capacity & Retrieval ━━━");

    let dim = DEFAULT_DIM;
    let test_counts = [10, 50, 100, 200, 500];

    println!("  {:>6} | {:>12} | {:>12} | {:>12}", "Facts", "Avg Sim", "Min Sim", "Est. SNR");
    println!("  {}", "-".repeat(55));

    for &k in &test_counts {
        let mut bank = MemoryBank::new(dim);

        let roles: Vec<HyperVector> = (0..k)
            .map(|i| HyperVector::random_bipolar(dim, i as u64 * 100))
            .collect();
        let fillers: Vec<HyperVector> = (0..k)
            .map(|i| HyperVector::random_bipolar(dim, i as u64 * 100 + 50))
            .collect();

        for i in 0..k {
            bank.store(&roles[i], &fillers[i], 1.0);
        }

        // Measure retrieval quality
        let sample_size = k.min(20); // Sample first 20 for speed
        let mut sims = Vec::new();
        for i in 0..sample_size {
            let (retrieved, _) = bank.retrieve(&roles[i]);
            let sim = cosine_similarity(&retrieved, &fillers[i]);
            sims.push(sim);
        }

        let avg_sim: f32 = sims.iter().sum::<f32>() / sims.len() as f32;
        let min_sim: f32 = sims.iter().cloned().fold(f32::INFINITY, f32::min);
        let stats = bank.stats();

        println!("  {:>6} | {:>12.4} | {:>12.4} | {:>12.2}", k, avg_sim, min_sim, stats.estimated_snr);
    }
}
