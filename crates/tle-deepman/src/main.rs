//! # AXIOM — Algebraic neXt-token Inference On Memory
//!
//! Solve for X. No training required.
//!
//! Combines:
//! - **Layer 1 (Engram)**: O(1) multi-head N-gram hash lookup — fast path
//! - **Layer 2 (TBA)**: VSA Transition Binding Algebra — fallback for unseen contexts
//! - **AFC**: Algebraic Flow Composition — composable energy scoring
//!
//! ## Architecture
//!
//! ```text
//! Input: "the president of" (tokenized)
//!   │
//!   ├─→ [LAYER 1: Engram]  O(1) hash → candidates + confidence
//!   │     confidence > θ?
//!   │       YES → use Engram scores (fast path, ~1μs)
//!   │       NO  ↓
//!   ├─→ [LAYER 2: TBA]     VSA unbind → transition candidates (~50μs)
//!   │
//!   └─→ [AFC Energy Scoring]
//!         E(token) = α·engram + β·transition + γ·context - δ·repetition - ε·diversity
//!         argmax → next token (deterministic)
//! ```
//!
//! ## Key Properties
//! - 100% deterministic (same input → same output, always)
//! - Zero training (single-pass corpus ingestion)
//! - CPU-only, single-threaded
//! - Incremental: add data → immediately smarter

use std::collections::HashMap;
use std::fs;
use std::time::Instant;

use tle_vsa::{cosine_similarity, HyperVector, Codebook};
use tle_engram::builder::{BuilderConfig, EngramBuilder, Vocab};
use tle_engram::fusion::SigmoidFusion;
use tle_engram::hash::NgramHash;

// ═══════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════

/// Deep Man engine configuration.
#[derive(Clone, Debug)]
struct DeepManConfig {
    /// Engram configuration.
    max_ngram_order: usize,
    min_ngram_count: u32,
    max_candidates: usize,

    /// TBA (VSA) configuration.
    vsa_dim: usize,
    vsa_seed: u64,

    /// AFC energy weights.
    alpha_engram: f32,      // Engram score weight
    beta_transition: f32,   // TBA transition weight
    gamma_context: f32,     // Context coherence weight
    delta_repetition: f32,  // Repetition penalty
    epsilon_diversity: f32, // Diversity penalty

    /// Generation parameters.
    max_gen_tokens: usize,
    engram_confidence_threshold: f32,
    context_decay: f32,
    repetition_window: usize,
}

impl Default for DeepManConfig {
    fn default() -> Self {
        Self {
            max_ngram_order: 5,
            min_ngram_count: 2,
            max_candidates: 30,

            vsa_dim: 2048, // Fast for demo (still reliable for <200 transitions per bundle)
            vsa_seed: 0xDEAD_FACE_CAFE_0001,

            alpha_engram: 1.0,
            beta_transition: 0.3,
            gamma_context: 0.2,
            delta_repetition: 3.0,
            epsilon_diversity: 0.2,

            max_gen_tokens: 30,
            engram_confidence_threshold: 0.4,
            context_decay: 0.7,
            repetition_window: 8,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// DEEP MAN ENGINE
// ═══════════════════════════════════════════════════════════════════════

/// Fast sigmoid for gating.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x > 15.0 { return 1.0; }
    if x < -15.0 { return 0.0; }
    1.0 / (1.0 + (-x).exp())
}

/// The unified Deep Man generation engine.
struct DeepManEngine {
    // --- Layer 1: Engram (fast hash lookup) ---
    engram_tables: Vec<tle_engram::table::EngramTable>,
    engram_hasher: NgramHash,
    engram_fusion: SigmoidFusion,

    // --- Layer 2: TBA (VSA transition memory) ---
    transition_memory: HyperVector,
    codebook: Codebook,

    // --- Shared ---
    vocab: Vocab,
    config: DeepManConfig,

    // --- Statistics ---
    stats: EngineStats,
}

#[derive(Default, Clone, Debug)]
struct EngineStats {
    engram_hits: usize,
    engram_misses: usize,
    tba_activations: usize,
    total_tokens_generated: usize,
}

impl DeepManEngine {
    /// Build the engine from a text corpus.
    fn build_from_corpus(text: &str, config: DeepManConfig) -> Self {
        let start = Instant::now();

        // === Phase 1: Build Engram ===
        let engram_config = BuilderConfig {
            max_order: config.max_ngram_order,
            min_count: config.min_ngram_count,
            max_vocab: 0,
            max_candidates_per_entry: config.max_candidates,
        };

        let mut builder = EngramBuilder::with_config(engram_config);
        let mut line_count = 0;

        for line in text.lines() {
            let trimmed = line.trim().to_lowercase();
            if !trimmed.is_empty() && trimmed.len() > 5 {
                builder.ingest_line(&trimmed);
                line_count += 1;
            }
        }

        let _vocab_snapshot = builder.vocab.clone();
        let total_tokens = builder.total_tokens;
        let engram = builder.build();
        let engram_time = start.elapsed();

        // === Phase 2: Build TBA Transition Memory ===
        let tba_start = Instant::now();
        let mut codebook = Codebook::new(config.vsa_dim, config.vsa_seed);

        // Encode all vocab words into hypervectors
        for i in 0..engram.vocab.len() {
            if let Some(token) = engram.vocab.get_token(i as u16) {
                codebook.get_or_insert(token);
            }
        }

        // Build transition memory: TM = Σ π(w_i) ⊗ w_{i+1}
        let mut tm = HyperVector::zeros(config.vsa_dim);
        let mut transition_count = 0usize;
        let max_tba_lines = 5000; // Limit TBA for speed (Engram handles the rest)
        let mut tba_line_count = 0;

        for line in text.lines() {
            if tba_line_count >= max_tba_lines {
                break;
            }
            let trimmed = line.trim().to_lowercase();
            if trimmed.is_empty() || trimmed.len() <= 5 {
                continue;
            }
            tba_line_count += 1;

            let tokens: Vec<&str> = trimmed.split_whitespace().collect();
            for window in tokens.windows(2) {
                if let (Some(from_vec), Some(to_vec)) = (
                    codebook.get(window[0]),
                    codebook.get(window[1]),
                ) {
                    // T(A→B) = π(A) ⊗ B
                    let shifted = from_vec.permute(1);
                    let transition = shifted.hadamard(to_vec);
                    tm = tm.add(&transition);
                    transition_count += 1;
                }
            }
        }

        let tba_time = tba_start.elapsed();

        // Print build stats
        println!("  [Build] Engram: {} lines, {} tokens, {} vocab in {:?}",
            line_count, total_tokens, engram.vocab.len(), engram_time);
        println!("  [Build] TBA: {} transitions encoded in D={} in {:?}",
            transition_count, config.vsa_dim, tba_time);
        println!("  [Build] Engram contexts: {}",
            engram.tables.iter().map(|t| t.len()).sum::<usize>());

        let vocab_size = engram.vocab.len();
        let fusion = SigmoidFusion::new(vocab_size);

        Self {
            engram_tables: engram.tables,
            engram_hasher: engram.hasher,
            engram_fusion: fusion,
            transition_memory: tm,
            codebook,
            vocab: engram.vocab,
            config,
            stats: EngineStats::default(),
        }
    }

    /// Generate a sequence of tokens from a prompt.
    ///
    /// Returns: (generated_token_ids, generation_time)
    fn generate(&mut self, prompt: &str) -> (Vec<u16>, std::time::Duration) {
        let start = Instant::now();

        // Tokenize prompt
        let prompt_tokens: Vec<&str> = prompt.split_whitespace().collect();
        let mut context: Vec<u16> = prompt_tokens
            .iter()
            .filter_map(|t| self.vocab.get_id(&t.to_lowercase()))
            .collect();

        if context.is_empty() {
            return (Vec::new(), start.elapsed());
        }

        let mut generated = Vec::new();
        let mut context_vec = HyperVector::zeros(self.config.vsa_dim);

        // Initialize context vector from prompt
        for &id in &context {
            if let Some(token) = self.vocab.get_token(id) {
                if let Some(vec) = self.codebook.get(token) {
                    let decayed = context_vec.scale(self.config.context_decay);
                    let fresh = vec.scale(1.0 - self.config.context_decay);
                    context_vec = decayed.add(&fresh);
                }
            }
        }

        // Generation loop
        for _step in 0..self.config.max_gen_tokens {
            let vocab_size = self.vocab.len();
            let mut scores = vec![0.0f32; vocab_size];

            // === Layer 1: Engram query ===
            let engram_fused = {
                // Query engram - collect results to owned data to release borrow
                let mut head_hits: Vec<(usize, f32, Vec<u16>, Vec<f32>)> = Vec::new();
                for (head_idx, key) in self.engram_hasher.hash_all_heads(&context) {
                    let order = head_idx + 1;
                    if let Some(entry) = self.engram_tables[head_idx].lookup(key) {
                        let confidence = entry.confidence(3.0);
                        if confidence > self.config.engram_confidence_threshold || order >= 3 {
                            head_hits.push((
                                order,
                                confidence,
                                entry.candidates.clone(),
                                entry.scores.clone(),
                            ));
                        }
                    }
                }

                if !head_hits.is_empty() {
                    // Manually fuse scores (avoid borrowing entry references)
                    let mut fused = vec![0.0f32; vocab_size];
                    for (order, confidence, candidates, entry_scores) in &head_hits {
                        let gate = sigmoid(
                            *confidence + 0.5 * (*order as f32 - 1.0),
                        );
                        for (i, &token_id) in candidates.iter().enumerate() {
                            if (token_id as usize) < vocab_size {
                                let log_prob = entry_scores.get(i).copied().unwrap_or(-10.0);
                                fused[token_id as usize] += gate * log_prob.exp();
                            }
                        }
                    }
                    Some(fused)
                } else {
                    None
                }
            };
            let engram_confident = engram_fused.is_some();

            if engram_confident {
                self.stats.engram_hits += 1;
            } else {
                self.stats.engram_misses += 1;
            }

            if let Some(ref fused) = engram_fused {
                for (i, &s) in fused.iter().enumerate() {
                    if i < vocab_size {
                        scores[i] += self.config.alpha_engram * s;
                    }
                }
            }

            // === Layer 2: TBA fallback (only when Engram misses or low confidence) ===
            if !engram_confident {
                let tba_weight = self.config.beta_transition * 2.0;

                if let Some(current_token) = context.last().and_then(|&id| self.vocab.get_token(id)) {
                    if let Some(current_vec) = self.codebook.get(current_token).cloned() {
                        self.stats.tba_activations += 1;
                        let shifted = current_vec.permute(1);
                        let predicted = shifted.hadamard(&self.transition_memory);

                        // Score only top candidates (not full vocab) for speed
                        for i in 0..vocab_size.min(scores.len()).min(1000) {
                            if let Some(token) = self.vocab.get_token(i as u16) {
                                if let Some(candidate_vec) = self.codebook.get(token) {
                                    let sim = cosine_similarity(&predicted, candidate_vec);
                                    scores[i] += tba_weight * sim;
                                }
                            }
                        }
                    }
                }
            }

            // === Context coherence scoring (only when TBA is active — helps disambiguate) ===
            if !engram_confident && context_vec.norm() > 0.01 {
                for i in 0..vocab_size.min(scores.len()).min(1000) {
                    if let Some(token) = self.vocab.get_token(i as u16) {
                        if let Some(candidate_vec) = self.codebook.get(token) {
                            let sim = cosine_similarity(&context_vec, candidate_vec);
                            scores[i] += self.config.gamma_context * sim;
                        }
                    }
                }
            }

            // === Repetition penalty (strengthened) ===
            let window_start = context.len().saturating_sub(self.config.repetition_window);
            let recent = &context[window_start..];
            for &recent_id in recent {
                if (recent_id as usize) < scores.len() {
                    scores[recent_id as usize] -= self.config.delta_repetition;
                }
            }

            // Bigram repetition: if last 2 tokens appeared as a bigram before, penalize
            if context.len() >= 2 {
                // Check if any candidate would form a repeated bigram
                for i in 0..vocab_size.min(scores.len()) {
                    let candidate_id = i as u16;
                    // Would selecting this create a bigram we've seen in last 10 tokens?
                    let recent_start = context.len().saturating_sub(10);
                    for w in context[recent_start..].windows(2) {
                        if w[0] == context[context.len() - 1] && w[1] == candidate_id {
                            scores[i] -= self.config.delta_repetition * 1.5;
                            break;
                        }
                    }
                }
            }

            // === Diversity penalty ===
            let mut freq_map: HashMap<u16, usize> = HashMap::new();
            for &id in generated.iter() {
                *freq_map.entry(id).or_insert(0) += 1;
            }
            for (&id, &count) in &freq_map {
                if (id as usize) < scores.len() {
                    scores[id as usize] -= self.config.epsilon_diversity * (1.0 + count as f32).ln();
                }
            }

            // === Argmax selection (deterministic) ===
            // Block <unk> and special tokens
            for blocked in &["<unk>", "=", "@-@", "@.@"] {
                if let Some(id) = self.vocab.get_id(blocked) {
                    scores[id as usize] = f32::NEG_INFINITY;
                }
            }

            let best_idx = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            let best_id = best_idx as u16;

            // End-of-sentence detection: stop after generating a sentence-ending token
            if let Some(token) = self.vocab.get_token(best_id) {
                if token == "." || token == "!" || token == "?" {
                    generated.push(best_id);
                    self.stats.total_tokens_generated += 1;
                    break;
                }
            }

            // Update state
            context.push(best_id);
            generated.push(best_id);
            self.stats.total_tokens_generated += 1;

            // Update context vector
            if let Some(token) = self.vocab.get_token(best_id) {
                if let Some(vec) = self.codebook.get(token) {
                    let decayed = context_vec.scale(self.config.context_decay);
                    let fresh = vec.scale(1.0 - self.config.context_decay);
                    context_vec = decayed.add(&fresh);
                }
            }
        }

        (generated, start.elapsed())
    }

    /// Decode token IDs to text.
    fn decode(&self, ids: &[u16]) -> String {
        ids.iter()
            .filter_map(|&id| self.vocab.get_token(id))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Print engine statistics.
    fn print_stats(&self) {
        let total = self.stats.engram_hits + self.stats.engram_misses;
        let hit_rate = if total > 0 {
            self.stats.engram_hits as f64 / total as f64 * 100.0
        } else {
            0.0
        };

        println!("\n── Engine Statistics ──");
        println!("  Engram hit rate: {:.1}% ({}/{})",
            hit_rate, self.stats.engram_hits, total);
        println!("  TBA activations: {}", self.stats.tba_activations);
        println!("  Total tokens generated: {}", self.stats.total_tokens_generated);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MAIN — INTERACTIVE REPL
// ═══════════════════════════════════════════════════════════════════════

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   AXIOM — Algebraic neXt-token Inference On Memory       ║");
    println!("║                                                          ║");
    println!("║   Solve for X. No training required.                     ║");
    println!("║                                                          ║");
    println!("║   Layer 1: Engram (O(1) hash)                           ║");
    println!("║   Layer 2: TBA (VSA transitions)                        ║");
    println!("║   Layer 3: KG (algebraic fact store)                    ║");
    println!("║   Training: ZERO | Deterministic: 100%                  ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Load corpus
    let data_path = "data/wiki_train.txt";
    let data = match fs::read_to_string(data_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("  Note: {} not found ({}), using built-in corpus.", data_path, e);
            FALLBACK_CORPUS.to_string()
        }
    };

    // Build engine
    println!("[*] Building AXIOM engine...");
    let build_start = Instant::now();
    let config = DeepManConfig::default();
    let mut engine = DeepManEngine::build_from_corpus(&data, config);
    let build_time = build_start.elapsed();
    println!("  Ready in {:?} | Vocab: {} | Engram contexts: {}",
        build_time, engine.vocab.len(),
        engine.engram_tables.iter().map(|t| t.len()).sum::<usize>());

    // Initialize IncrementalStore for live learning
    let mut store = tle_afc::IncrementalStore::new();

    println!("\n── Commands ──");
    println!("  /teach <fact>       Learn a fact (e.g., /teach Bangkok is the capital of Thailand)");
    println!("  /ask <S> <R>        Query knowledge (e.g., /ask bangkok capital_of)");
    println!("  /stats              Show engine statistics");
    println!("  /quit               Exit");
    println!("  <anything else>     Generate continuation from your input");
    println!("────────────────────────────────────────────────────────────\n");

    // REPL loop
    let stdin = std::io::stdin();
    loop {
        // Prompt
        eprint!("AXIOM> ");
        use std::io::Write;
        std::io::stderr().flush().ok();

        let mut input = String::new();
        match stdin.read_line(&mut input) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(_) => break,
        }

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Parse commands
        if trimmed == "/quit" || trimmed == "/exit" || trimmed == "/q" {
            println!("  Goodbye!");
            break;
        }

        if trimmed == "/stats" {
            engine.print_stats();
            let st = store.stats();
            println!("  Incremental: {} facts, {} tokens learned, {} transitions",
                st.facts_added, st.tokens_ingested, st.transitions_added);
            continue;
        }

        if let Some(fact_text) = trimmed.strip_prefix("/teach ") {
            handle_teach(&mut store, &mut engine, fact_text);
            continue;
        }

        if let Some(query_text) = trimmed.strip_prefix("/ask ") {
            handle_ask(&store, query_text);
            continue;
        }

        // Default: generate continuation
        handle_generate(&mut engine, &store, trimmed);
    }
}

/// Handle /teach command — learn facts and text.
fn handle_teach(
    store: &mut tle_afc::IncrementalStore,
    engine: &mut DeepManEngine,
    fact_text: &str,
) {
    let trimmed = fact_text.trim();
    if trimmed.is_empty() {
        println!("  Usage: /teach <sentence or fact>");
        return;
    }

    // Try to parse as triple: "X is Y" / "X is_a Y" / "X relation Y"
    let parts: Vec<&str> = trimmed.splitn(3, ' ').collect();

    if parts.len() >= 3 {
        // Check for common patterns
        let (subject, relation, object) = if parts[1] == "is" || parts[1] == "are" {
            // "Bangkok is the capital of Thailand" → subject=Bangkok, rel=is, obj=rest
            (parts[0], parts[1], parts[2..].join(" "))
        } else {
            (parts[0], parts[1], parts[2..].join(" "))
        };

        // Learn as structured fact
        store.learn_fact(subject, relation, &object);
        println!("  ✓ Fact: {} {} {}", subject, relation, object);
    }

    // Always also learn as text (for N-gram + TBA)
    store.learn_text(trimmed);

    // Update engine's Engram inline (add to N-gram counts)
    // For now, the IncrementalStore handles prediction independently
    println!("  ✓ Learned: \"{}\"", trimmed);
}

/// Handle /ask command — query the knowledge graph.
fn handle_ask(store: &tle_afc::IncrementalStore, query_text: &str) {
    let parts: Vec<&str> = query_text.trim().split_whitespace().collect();

    if parts.len() < 2 {
        println!("  Usage: /ask <subject> <relation>");
        println!("  Example: /ask bangkok capital_of");
        return;
    }

    let subject = parts[0];
    let relation = parts[1];

    match store.query_fact(subject, relation) {
        Some((answer, confidence)) => {
            println!("  → {} {} {} (confidence: {:.3})", subject, relation, answer, confidence);
        }
        None => {
            // Try N-gram prediction as fallback
            let context: Vec<&str> = parts.iter().copied().collect();
            let predictions = store.predict_next(&context);
            if !predictions.is_empty() {
                let top: Vec<String> = predictions.iter().take(5)
                    .map(|(t, s)| format!("{} ({:.2})", t, s))
                    .collect();
                println!("  → N-gram predictions: {}", top.join(", "));
            } else {
                println!("  → I don't know about '{}' '{}' yet. Teach me with /teach!", subject, relation);
            }
        }
    }
}

/// Handle chat input — generate continuation using AXIOM engine + IncrementalStore.
fn handle_generate(
    engine: &mut DeepManEngine,
    store: &tle_afc::IncrementalStore,
    input: &str,
) {
    let start = Instant::now();

    // First check if IncrementalStore has a direct answer
    let input_tokens: Vec<&str> = input.split_whitespace().collect();
    let incr_predictions = store.predict_next(&input_tokens);

    // Generate from main engine
    let (generated, gen_time) = engine.generate(input);
    let output = engine.decode(&generated);

    if !output.is_empty() {
        println!("  {} [{:?}]", output, gen_time);
    } else if !incr_predictions.is_empty() {
        // Fallback to incremental predictions
        let predicted: Vec<&str> = incr_predictions.iter().take(8).map(|(t, _)| t.as_str()).collect();
        println!("  {} [{:?}]", predicted.join(" "), start.elapsed());
    } else {
        println!("  (no prediction available — teach me more with /teach!)");
    }
}

const FALLBACK_CORPUS: &str = "\
the cat sat on the mat in the morning sun
the dog ran in the park chasing the ball
the president of the united states spoke today
in the united states of america people vote
he was the first to arrive at the meeting
the city of new york is very large and busy
she was born in london in nineteen eighty
the team won the championship last year
it was released in two thousand and five
the album was recorded in los angeles
according to the report the numbers are rising
at the end of the day we went home
the bird flew over the tree and into the sky
the fish swam in the deep blue sea
the sun set behind the mountains slowly
";
