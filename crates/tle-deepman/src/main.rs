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
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::Instant;

use tle_vsa::{cosine_similarity, HyperVector, Codebook};
use tle_engram::builder::{BuilderConfig, EngramBuilder, Vocab};
use tle_engram::fusion::SigmoidFusion;
use tle_engram::hash::NgramHash;

mod web_learning;
mod daemon;

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

            // === Layer 1: Engram query (sparse — only get candidates) ===
            let mut sparse_candidates: Vec<u16> = Vec::new();
            let mut sparse_scores: Vec<f32> = Vec::new();

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

            let engram_confident = !head_hits.is_empty();

            if engram_confident {
                self.stats.engram_hits += 1;

                // Collect unique candidate IDs from all heads
                let mut candidate_set: HashMap<u16, f32> = HashMap::new();
                for (order, confidence, candidates, entry_scores) in &head_hits {
                    let gate = sigmoid(*confidence + 0.5 * (*order as f32 - 1.0));
                    for (i, &token_id) in candidates.iter().enumerate() {
                        let log_prob = entry_scores.get(i).copied().unwrap_or(-10.0);
                        *candidate_set.entry(token_id).or_insert(0.0) +=
                            self.config.alpha_engram * gate * log_prob.exp();
                    }
                }

                // Apply penalties ONLY to these sparse candidates
                let window_start = context.len().saturating_sub(self.config.repetition_window);
                let recent = &context[window_start..];

                let mut freq_map: HashMap<u16, usize> = HashMap::new();
                for &id in generated.iter() {
                    *freq_map.entry(id).or_insert(0) += 1;
                }

                for (&token_id, score) in candidate_set.iter_mut() {
                    // Repetition penalty
                    if recent.contains(&token_id) {
                        *score -= self.config.delta_repetition;
                    }

                    // Bigram repetition
                    if context.len() >= 2 {
                        let last = context[context.len() - 1];
                        let recent_start = context.len().saturating_sub(10);
                        for w in context[recent_start..].windows(2) {
                            if w[0] == last && w[1] == token_id {
                                *score -= self.config.delta_repetition * 1.5;
                                break;
                            }
                        }
                    }

                    // Diversity penalty
                    if let Some(&count) = freq_map.get(&token_id) {
                        *score -= self.config.epsilon_diversity * (1.0 + count as f32).ln();
                    }

                    // Block special tokens
                    if let Some(token) = self.vocab.get_token(token_id) {
                        if token == "<unk>" || token == "=" || token == "@-@" || token == "@.@" {
                            *score = f32::NEG_INFINITY;
                        }
                    }
                }

                // Convert to sparse arrays
                for (&id, &score) in &candidate_set {
                    sparse_candidates.push(id);
                    sparse_scores.push(score);
                }
            } else {
                // === Engram miss → TBA fallback (scan limited vocab) ===
                self.stats.engram_misses += 1;
                let tba_weight = self.config.beta_transition * 2.0;
                let scan_limit = 500; // scan top-500 vocab only

                if let Some(current_token) = context.last().and_then(|&id| self.vocab.get_token(id)) {
                    if let Some(current_vec) = self.codebook.get(current_token).cloned() {
                        self.stats.tba_activations += 1;
                        let shifted = current_vec.permute(1);
                        let predicted = shifted.hadamard(&self.transition_memory);

                        for i in 0..vocab_size.min(scan_limit) {
                            if let Some(token) = self.vocab.get_token(i as u16) {
                                if token == "<unk>" || token == "=" {
                                    continue;
                                }
                                if let Some(candidate_vec) = self.codebook.get(token) {
                                    let mut score = tba_weight * cosine_similarity(&predicted, candidate_vec);

                                    // Context coherence
                                    if context_vec.norm() > 0.01 {
                                        score += self.config.gamma_context * cosine_similarity(&context_vec, candidate_vec);
                                    }

                                    sparse_candidates.push(i as u16);
                                    sparse_scores.push(score);
                                }
                            }
                        }
                    }
                }
            }

            // === Argmax over sparse candidates (deterministic) ===
            if sparse_candidates.is_empty() {
                break;
            }

            let best_pos = sparse_scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            let best_id = sparse_candidates[best_pos];

            // End-of-sentence detection
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

    // Initialize AXIOM-Gen for compositional generation
    let mut axiom_gen = tle_axiom_gen::AxiomGen::new(2048);
    axiom_gen.template_bank = tle_axiom_gen::templates::TemplateBank::extract_from_corpus(&data);

    // Initialize Delta Memory for conversation context
    let mut delta_mem = tle_afc::DeltaMem::new(2048);

    // Initialize VSA Intent Detector (replaces rule-based)
    let mut intent_codebook = tle_vsa::Codebook::new(2048, 0x1A7E_CAFE_0001);
    let vsa_intent_detector = tle_afc::VsaIntentDetector::build(&mut intent_codebook);

    // Initialize Paragraph Generator
    let paragraph_gen = tle_afc::paragraph::ParagraphGenerator::new();
    let mut response_style = tle_afc::paragraph::ResponseStyle::Casual;

    // Initialize Morphological Tokenizer (VSA subword composition)
    let morph_tokenizer = tle_afc::MorphTokenizer::new();

    // Initialize Attractor Reasoner (iterative refinement)
    let mut attractor = tle_afc::attractor::AttractorReasoner::new();

    println!("\n── Commands ──");
    println!("  /teach <fact>       Learn a fact (e.g., /teach Bangkok is the capital of Thailand)");
    println!("  /ask <S> <R>        Query knowledge (e.g., /ask bangkok capital_of)");
    println!("  /load <file.txt>    Load and learn from a text file");
    println!("  /learn-url <url>    Fetch a web page and learn its facts");
    println!("  /learn-url-bg <url> Fetch a web page in the background");
    println!("  /queue-url <url>   Persist a URL for daemon processing");
    println!("  /save <file.json>   Save learned knowledge to file");
    println!("  /restore <file.json> Restore previously saved knowledge");
    println!("  /stats              Show engine statistics");
    println!("  /style <mode>       Set response style: casual, formal, brief");
    println!("  /quit               Exit");
    println!("  <anything else>     Chat — ask questions or generate text");
    println!("────────────────────────────────────────────────────────────\n");

    // REPL loop
    let stdin = std::io::stdin();
    let mut web_queue = daemon::WebLearnQueue::load("data/axiom_web_queue.tsv");
    let mut background_jobs: Vec<Receiver<(String, bool, Result<web_learning::ExtractedPage, String>)>> = Vec::new();
    loop {
        poll_background_learns(&mut background_jobs, &mut web_queue, &mut store, &mut axiom_gen);
        start_next_queued_job(&mut background_jobs, &mut web_queue);

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
            println!("  Compaction: {} runs, {} facts pruned, {} values merged",
                st.compactions, st.facts_pruned, st.facts_merged);
            println!("  AXIOM-Gen contradictions: {}",
                axiom_gen.graph.contradictions().len());
            println!("  Web queue: {} pending jobs", web_queue.len());
            continue;
        }

        if let Some(style_name) = trimmed.strip_prefix("/style ") {
            response_style = match style_name.trim().to_lowercase().as_str() {
                "casual" => tle_afc::paragraph::ResponseStyle::Casual,
                "formal" => tle_afc::paragraph::ResponseStyle::Formal,
                "brief" => tle_afc::paragraph::ResponseStyle::Brief,
                _ => {
                    println!("  Usage: /style casual|formal|brief");
                    continue;
                }
            };
            println!("  ✓ Response style: {:?}", response_style);
            continue;
        }

        if let Some(fact_text) = trimmed.strip_prefix("/teach ") {
            handle_teach(&mut store, &mut engine, &mut axiom_gen, fact_text);
            delta_mem.update_topic(fact_text);
            // Add subject as attractor basin for reasoning (using morphological encoding)
            let lower = fact_text.to_lowercase();
            if let Some(pos) = lower.find(' ') {
                let subject = &lower[..pos];
                if subject.len() > 2 {
                    let subj_vec = morph_tokenizer.encode(subject, &mut intent_codebook);
                    attractor.add_attractor(subject, subj_vec);
                }
            }
            continue;
        }

        if let Some(file_path) = trimmed.strip_prefix("/load ") {
            handle_load(&mut store, file_path.trim());
            continue;
        }

        if let Some(url) = trimmed.strip_prefix("/learn-url ") {
            handle_learn_url(&mut store, &mut axiom_gen, url.trim());
            continue;
        }

        if let Some(url) = trimmed.strip_prefix("/learn-url-bg ") {
            handle_learn_url_background(&mut background_jobs, url.trim());
            continue;
        }

        if let Some(url) = trimmed.strip_prefix("/queue-url ") {
            match web_queue.enqueue(url.trim()) {
                Ok(()) => println!("  ✓ Queued '{}' ({} pending)", url.trim(), web_queue.len()),
                Err(error) => println!("  ✗ Could not queue URL: {}", error),
            }
            continue;
        }

        if let Some(file_path) = trimmed.strip_prefix("/save ") {
            handle_save(&store, file_path.trim());
            continue;
        }

        if let Some(file_path) = trimmed.strip_prefix("/restore ") {
            handle_restore(&mut store, file_path.trim());
            continue;
        }

        if let Some(query_text) = trimmed.strip_prefix("/ask ") {
            handle_ask(&store, query_text);
            continue;
        }

        // Default: generate/conversation
        // First: resolve pronouns using conversation memory
        let resolved = delta_mem.resolve_pronoun(trimmed);
        let effective_input = if resolved != trimmed.to_lowercase() {
            resolved.clone()
        } else {
            trimmed.to_lowercase()
        };

        // Track topic for future pronoun resolution
        delta_mem.update_topic(trimmed);

        // Process through intent detection (VSA-based with rule-based fallback)
        let start = Instant::now();

        // Compose morpheme vectors before intent matching so OOV forms share
        // algebraic signal with known roots and affixes.
        let morph_query = morph_tokenizer.encode_sentence(&effective_input, &mut intent_codebook);
        let attractor_result = attractor.reason(&morph_query);
        let linked_query = attractor
            .identify_attractor(&attractor_result.state)
            .filter(|(_, confidence)| *confidence > 0.2)
            .map(|(label, _)| format!("{} {}", effective_input, label))
            .unwrap_or_else(|| effective_input.clone());

        // Try VSA intent detection first
        let (vsa_intent, vsa_confidence) = vsa_intent_detector.detect(&effective_input, &mut intent_codebook);
        let intent = if vsa_confidence > 0.1 {
            // Map VsaIntent → local Intent enum
            match vsa_intent {
                tle_afc::VsaIntent::Why | tle_afc::VsaIntent::How => {
                    // Try AXIOM-Gen for compositional answer
                    let gen_result = axiom_gen.generate(&linked_query);
                    if gen_result.path_length >= 2 {
                        println!("  {} [{:?}]", gen_result.sentence, start.elapsed());
                        if !gen_result.reasoning.is_empty() {
                            println!("  [Reasoning: {}]", gen_result.reasoning.join(" → "));
                        }
                        continue;
                    }
                    detect_intent(&effective_input)
                }
                tle_afc::VsaIntent::What => detect_intent(&effective_input),
                tle_afc::VsaIntent::Where => detect_intent(&effective_input),
                tle_afc::VsaIntent::Who => detect_intent(&effective_input),
                tle_afc::VsaIntent::YesNo => Intent::YesNo(effective_input.clone()),
                tle_afc::VsaIntent::Greeting => Intent::Greeting,
                tle_afc::VsaIntent::Thanks => Intent::Thanks,
                _ => detect_intent(&effective_input),
            }
        } else {
            detect_intent(&effective_input)
        };

        match intent {
            Intent::Greeting => {
                println!("  Hello! I'm AXIOM. Ask me anything, or teach me with /teach.");
            }
            Intent::Thanks => {
                println!("  You're welcome! Ask me more or teach me something new.");
            }
            Intent::WhatIs(subject) => {
                respond_what_is(&mut engine, &store, &mut axiom_gen, &subject, start, response_style);
            }
            Intent::WhoIs(subject) => {
                respond_what_is(&mut engine, &store, &mut axiom_gen, &subject, start, response_style);
            }
            Intent::WhereIs(subject) => {
                respond_where_is(&mut engine, &store, &subject, start);
            }
            Intent::YesNo(question) => {
                respond_yes_no(&store, &question, start);
            }
            Intent::Question(topic) => {
                // Try AXIOM-Gen compositional generation first
                let gen_result = axiom_gen.generate(&linked_query);
                if gen_result.path_length >= 2 {
                    println!("  {} [{:?}]", gen_result.sentence, start.elapsed());
                    if !gen_result.reasoning.is_empty() {
                        println!("  [Reasoning: {}]", gen_result.reasoning.join(" → "));
                    }
                } else {
                    respond_question(&mut engine, &store, &topic, &effective_input, start);
                }
            }
            Intent::Generate => {
                respond_generate(&mut engine, &store, &effective_input, start);
            }
        }
    }
}

/// Handle /teach command — learn facts and text.
fn handle_teach(
    store: &mut tle_afc::IncrementalStore,
    _engine: &mut DeepManEngine,
    axiom_gen: &mut tle_axiom_gen::AxiomGen,
    fact_text: &str,
) {
    let trimmed = fact_text.trim();
    if trimmed.is_empty() {
        println!("  Usage: /teach <sentence or fact>");
        return;
    }

    // Extract structured facts from common patterns
    let lower = trimmed.to_lowercase();
    let mut extracted = false;

    // Thai copula/property patterns do not require whitespace tokenization.
    for (pattern, relation) in [("อยู่ใน", "located_in"), ("อยู่ที่", "located_in"),
        ("เกิดใน", "born_in"), ("เป็น", "is"), ("คือ", "is"), ("มี", "has")] {
        if let Some(pos) = lower.find(pattern) {
            let subject = trimmed[..pos].trim();
            let object = trimmed[pos + pattern.len()..].trim();
            if !subject.is_empty() && !object.is_empty() {
                store.learn_fact(subject, relation, object);
                println!("  ✓ Fact: {} → {} → {}", subject, relation, object);
                extracted = true;
                break;
            }
        }
    }

    // Pattern: "X is Y" / "X is a Y" / "X is the Y"
    if !extracted { if let Some(pos) = lower.find(" is ") {
        let subject = &trimmed[..pos];
        let object = &trimmed[pos + 4..];
        store.learn_fact(subject, "is", object);
        println!("  ✓ Fact: {} → is → {}", subject, object);
        extracted = true;
    }}
    // Pattern: "X are Y"
    else if !extracted { if let Some(pos) = lower.find(" are ") {
        let subject = &trimmed[..pos];
        let object = &trimmed[pos + 5..];
        store.learn_fact(subject, "are", object);
        println!("  ✓ Fact: {} → are → {}", subject, object);
        extracted = true;
    }}
    // Pattern: "X has Y" / "X have Y"
    else if let Some(pos) = lower.find(" has ") {
        let subject = &trimmed[..pos];
        let object = &trimmed[pos + 5..];
        store.learn_fact(subject, "has", object);
        println!("  ✓ Fact: {} → has → {}", subject, object);
        extracted = true;
    }
    else if let Some(pos) = lower.find(" have ") {
        let subject = &trimmed[..pos];
        let object = &trimmed[pos + 6..];
        store.learn_fact(subject, "have", object);
        println!("  ✓ Fact: {} → have → {}", subject, object);
        extracted = true;
    }
    // Pattern: "X can Y"
    else if let Some(pos) = lower.find(" can ") {
        let subject = &trimmed[..pos];
        let object = &trimmed[pos + 5..];
        store.learn_fact(subject, "can", object);
        println!("  ✓ Fact: {} → can → {}", subject, object);
        extracted = true;
    }
    // Pattern: "X lives in Y" / "X located in Y"
    else if let Some(pos) = lower.find(" lives in ") {
        let subject = &trimmed[..pos];
        let object = &trimmed[pos + 10..];
        store.learn_fact(subject, "lives_in", object);
        println!("  ✓ Fact: {} → lives_in → {}", subject, object);
        extracted = true;
    }
    else if let Some(pos) = lower.find(" located in ") {
        let subject = &trimmed[..pos];
        let object = &trimmed[pos + 12..];
        store.learn_fact(subject, "located_in", object);
        println!("  ✓ Fact: {} → located_in → {}", subject, object);
        extracted = true;
    }
    // Pattern: "X was born in Y"
    else if let Some(pos) = lower.find(" was born in ") {
        let subject = &trimmed[..pos];
        let object = &trimmed[pos + 13..];
        store.learn_fact(subject, "born_in", object);
        println!("  ✓ Fact: {} → born_in → {}", subject, object);
        extracted = true;
    }

    // Always learn as text (for N-gram + TBA + sentence memory)
    store.learn_text(trimmed);

    // Also add to AXIOM-Gen knowledge graph (for compositional generation)
    // Re-extract subject/relation/object for axiom_gen
    let lower = trimmed.to_lowercase();
    for pattern in &[" is ", " are ", " has ", " have ", " can ", " causes ", " makes ", " scatters ",
                     " produces ", " leads to ", " comes from ", " results in ",
                     " contains ", " requires ", " enables ", " creates "] {
        if let Some(pos) = lower.find(pattern) {
            let subj = &lower[..pos];
            let rel = pattern.trim();
            let obj = &lower[pos + pattern.len()..];
            if subj.len() > 1 && obj.len() > 1 {
                axiom_gen.add_fact(subj, rel, obj);
            }
            break;
        }
    }

    if !extracted {
        println!("  ✓ Learned: \"{}\"", trimmed);
    }
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

/// Handle chat input — detect intent and respond appropriately.
fn handle_generate(
    engine: &mut DeepManEngine,
    store: &tle_afc::IncrementalStore,
    input: &str,
) {
    let lower = input.trim().to_lowercase();
    let start = Instant::now();

    // === Intent Detection ===
    let intent = detect_intent(&lower);

    match intent {
        Intent::Greeting => {
            println!("  Hello! I'm AXIOM. Ask me anything, or teach me with /teach.");
        }
        Intent::Thanks => {
            println!("  You're welcome! Ask me more or teach me something new.");
        }
        Intent::WhatIs(subject) => {
            respond_what_is(engine, store, &mut tle_axiom_gen::AxiomGen::new(2048), &subject, start, tle_afc::paragraph::ResponseStyle::Casual);
        }
        Intent::WhoIs(subject) => {
            respond_what_is(engine, store, &mut tle_axiom_gen::AxiomGen::new(2048), &subject, start, tle_afc::paragraph::ResponseStyle::Casual);
        }
        Intent::WhereIs(subject) => {
            respond_where_is(engine, store, &subject, start);
        }
        Intent::YesNo(question) => {
            respond_yes_no(store, &question, start);
        }
        Intent::Question(topic) => {
            respond_question(engine, store, &topic, &lower, start);
        }
        Intent::Generate => {
            respond_generate(engine, store, &lower, start);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CONVERSATION ENGINE
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug)]
enum Intent {
    Greeting,
    Thanks,
    WhatIs(String),
    WhoIs(String),
    WhereIs(String),
    YesNo(String),
    Question(String),
    Generate,
}

/// Detect user intent from input.
fn detect_intent(input: &str) -> Intent {
    let words: Vec<&str> = input.split_whitespace().collect();

    // Greetings
    if matches!(words.first(), Some(&"hello") | Some(&"hi") | Some(&"hey") | Some(&"howdy")) {
        return Intent::Greeting;
    }

    // Thanks
    if input.contains("thank") || input.contains("thanks") || input == "ok" {
        return Intent::Thanks;
    }

    // "what is X?" / "what are X?"
    if input.starts_with("what is ") || input.starts_with("what's ") {
        let subject = input
            .trim_start_matches("what is ")
            .trim_start_matches("what's ")
            .trim_end_matches('?')
            .trim();
        return Intent::WhatIs(subject.to_string());
    }

    if input.starts_with("what are ") {
        let subject = input.trim_start_matches("what are ").trim_end_matches('?').trim();
        return Intent::WhatIs(subject.to_string());
    }

    // "who is X?"
    if input.starts_with("who is ") || input.starts_with("who's ") {
        let subject = input
            .trim_start_matches("who is ")
            .trim_start_matches("who's ")
            .trim_end_matches('?')
            .trim();
        return Intent::WhoIs(subject.to_string());
    }

    // "where is X?"
    if input.starts_with("where is ") || input.starts_with("where's ") {
        let subject = input
            .trim_start_matches("where is ")
            .trim_start_matches("where's ")
            .trim_end_matches('?')
            .trim();
        return Intent::WhereIs(subject.to_string());
    }

    // Yes/No: "is X a Y?" / "does X have Y?" / "can X Y?" / "do X Y?"
    if input.starts_with("is ") || input.starts_with("does ") || input.starts_with("can ")
        || input.starts_with("do ") {
        let topic = input.trim_end_matches('?').trim().to_string();
        return Intent::YesNo(topic);
    }

    // General questions (contains ?)
    if input.contains('?') {
        let topic = input.trim_end_matches('?').trim().to_string();
        return Intent::Question(topic);
    }

    // "tell me about X"
    if input.starts_with("tell me about ") {
        let subject = input.trim_start_matches("tell me about ").trim();
        return Intent::WhatIs(subject.to_string());
    }

    // Default: generate continuation
    Intent::Generate
}

/// Respond to "what is X?" questions.
fn respond_what_is(
    engine: &mut DeepManEngine,
    store: &tle_afc::IncrementalStore,
    axiom_gen: &mut tle_axiom_gen::AxiomGen,
    subject: &str,
    start: Instant,
    style: tle_afc::paragraph::ResponseStyle,
) {
    // Priority 1: Exact fact store (always correct if taught)
    let subject_clean = subject
        .trim_start_matches("the ")
        .trim_start_matches("a ")
        .trim_start_matches("an ");
    let facts = store.get_facts(subject_clean);
    let facts = if facts.is_empty() { store.get_facts(subject) } else { facts };
    if !facts.is_empty() {
        if facts.len() >= 2 {
            // Multiple facts → use ParagraphGenerator
            let para_gen = tle_afc::paragraph::ParagraphGenerator::new();
            let owned_facts: Vec<(String, String)> = facts.iter()
                .map(|(r, o)| (r.to_string(), o.to_string()))
                .collect();
            let paragraph = para_gen.generate_with_style(subject_clean, &owned_facts, style);
            println!("  {} [{:?}]", paragraph, start.elapsed());
            return;
        }
        // Single fact
        for (rel, obj) in &facts {
            if *rel == "is" || *rel == "are" {
                println!("  {} {} {}. [{:?}]", capitalize(subject), rel, obj, start.elapsed());
                return;
            }
        }
        // Show first available fact
        let (rel, obj) = &facts[0];
        println!("  {} {} {}. [{:?}]", capitalize(subject), rel, obj, start.elapsed());
        return;
    }

    // Priority 2: Sentence memory (exact recall of taught sentences)
    let sentences = store.get_sentences(subject_clean);
    let sentences = if sentences.is_empty() { store.get_sentences(subject) } else { sentences };
    if !sentences.is_empty() {
        // Find sentence with "is" or "are"
        for sent in &sentences {
            if sent.contains(" is ") || sent.contains(" are ") {
                // Limit to first sentence (up to first period)
                let first_sent = sent.split('.').next().unwrap_or(sent);
                if first_sent.len() < 200 {
                    println!("  {}. [{:?}]", capitalize(first_sent.trim()), start.elapsed());
                    return;
                }
            }
        }
        // Show first sentence of first entry
        let first = sentences[0].split('.').next().unwrap_or(sentences[0]);
        if first.len() < 200 {
            println!("  {}. [{:?}]", capitalize(first.trim()), start.elapsed());
            return;
        }
    }

    // Priority 3: AXIOM-Gen graph composition
    let generated = axiom_gen.generate(&format!("what is {}?", subject));
    if generated.path_length >= 1 && !generated.sentence.is_empty() {
        println!("  {} [{:?}]", generated.sentence, start.elapsed());
        return;
    }

    // Priority 4: Engram-based generation
    let prompt = format!("{} is", subject);
    let (generated, gen_time) = engine.generate(&prompt);
    let output = engine.decode(&generated);

    if !output.is_empty() {
        println!("  {} is {} [{:?}]", capitalize(subject), output, gen_time);
    } else {
        println!("  I don't know about '{}' yet. Teach me with /teach!", subject);
    }
}

/// Respond to "where is X?" questions.
fn respond_where_is(
    engine: &mut DeepManEngine,
    store: &tle_afc::IncrementalStore,
    subject: &str,
    start: Instant,
) {
    // Check fact store for location-related facts
    let facts = store.get_facts(subject);
    for (rel, obj) in &facts {
        if rel.contains("in") || rel.contains("located") || rel.contains("capital") {
            println!("  {} is in {}. [{:?}]", capitalize(subject), obj, start.elapsed());
            return;
        }
    }

    // Check sentence memory
    let sentences = store.get_sentences(subject);
    for sent in &sentences {
        if sent.contains(" in ") || sent.contains(" at ") || sent.contains(" located ") {
            println!("  {} [{:?}]", capitalize(sent), start.elapsed());
            return;
        }
    }

    // Generate from Engram
    let prompt = format!("{} is located in", subject);
    let (generated, gen_time) = engine.generate(&prompt);
    let output = engine.decode(&generated);

    if !output.is_empty() {
        println!("  {} is located in {} [{:?}]", capitalize(subject), output, gen_time);
    } else {
        println!("  I don't know where '{}' is. Teach me with /teach!", subject);
    }
}

/// Respond to yes/no questions using direct facts + analogical inference.
///
/// "is a dog an animal?" → check facts, then try analogy
/// "can elephants swim?" → check facts, then try analogy
fn respond_yes_no(
    store: &tle_afc::IncrementalStore,
    question: &str,
    start: Instant,
) {
    let words: Vec<&str> = question.split_whitespace().collect();

    // Parse: "is [subject] [relation] [object]" or "can [subject] [verb]" etc.
    let (subject, relation, object) = parse_yes_no_question(&words);

    if subject.is_empty() {
        println!("  I'm not sure how to answer that. Try rephrasing?");
        return;
    }

    // 1. Check direct facts (try singular, plural, and original)
    let subject_variants = vec![
        subject.clone(),
        format!("{}s", subject),                           // add s
        subject.trim_end_matches('s').to_string(),         // remove s
        format!("{}es", subject),                          // add es
        subject.trim_end_matches("es").to_string(),        // remove es
    ];

    for subj_v in &subject_variants {
        let facts = store.get_facts(subj_v);
        for (rel, obj) in &facts {
            if relation_matches_loose(rel, &relation) {
                if object.is_empty() || obj.contains(&object) || object.contains(&**obj) {
                    println!("  Yes! {} {} {}. [{:?}]", capitalize(subj_v), rel, obj, start.elapsed());
                    return;
                }
            }
        }
    }

    // 2. Try analogical inference
    let analogy_engine = tle_afc::AnalogicalEngine::new(store.codebook(), store.fact_store());
    let results = analogy_engine.infer(&subject, &relation);

    if !results.is_empty() {
        let best = &results[0];
        if object.is_empty() || best.answer.contains(&object) || object.contains(&best.answer) {
            println!("  Probably yes — by analogy: {} [{:?}]",
                best.reasoning, start.elapsed());
        } else {
            println!("  By analogy with {}: {} {} {} (confidence: {:.2}) [{:?}]",
                best.source_subject, subject, relation, best.answer, best.confidence, start.elapsed());
        }
        return;
    }

    // 3. Multi-hop transitive reasoning
    //    "does cat have a heart?" → cat is_a animal, animal has heart → YES
    let hop_result = try_multi_hop(store, &subject, &relation, &object);
    if let Some((answer, chain)) = hop_result {
        println!("  {} (Reasoning: {}) [{:?}]", answer, chain, start.elapsed());
        return;
    }

    // 4. Check sentence memory
    let sentences = store.get_sentences(&subject);
    for sent in &sentences {
        if !relation.is_empty() && sent.contains(&relation) {
            println!("  Based on what I know: {} [{:?}]", capitalize(sent), start.elapsed());
            return;
        }
    }

    println!("  I don't have enough information to answer that. Teach me more about {}!", subject);
}

/// Try multi-hop transitive reasoning.
///
/// Pattern: subject → is_a → category → has/can → object
/// Example: "cat" → is "an animal" → check if "animal" has "heart"
fn try_multi_hop(
    store: &tle_afc::IncrementalStore,
    subject: &str,
    relation: &str,
    object: &str,
) -> Option<(String, String)> {
    let subject_lower = subject.to_lowercase();

    // Try subject variants (singular/plural)
    let variants = vec![
        subject_lower.clone(),
        format!("{}s", subject_lower),
        subject_lower.trim_end_matches('s').to_string(),
    ];

    let mut subject_facts: Vec<(&str, &str)> = Vec::new();
    let mut matched_subject = subject_lower.clone();
    for v in &variants {
        let facts = store.get_facts(v);
        if !facts.is_empty() {
            subject_facts = facts;
            matched_subject = v.clone();
            break;
        }
    }

    if subject_facts.is_empty() {
        return None;
    }

    // For each "is" or "are" fact of the subject, check if that category has the property
    for (rel, category) in &subject_facts {
        if *rel != "is" && *rel != "are" {
            continue;
        }

        // Extract category words, strip articles
        let category_clean = category
            .trim_start_matches("a ")
            .trim_start_matches("an ")
            .trim_start_matches("the ");

        // Try: the exact category, singular, plural
        let candidates = vec![
            category_clean.to_string(),
            format!("{}s", category_clean),    // plural
            category_clean.trim_end_matches('s').to_string(), // singular
        ];

        for cat_key in &candidates {
            let cat_facts = store.get_facts(cat_key);
            for (cat_rel, cat_obj) in &cat_facts {
                if relation_matches_loose(cat_rel, relation) {
                    // Normalize object for comparison (strip articles, try plural/singular)
                    let obj_clean = object.to_lowercase()
                        .replace("a ", "").replace("an ", "").replace("the ", "");
                    let obj_variants = vec![
                        obj_clean.clone(),
                        format!("{}s", obj_clean),
                        obj_clean.trim_end_matches('s').to_string(),
                    ];

                    let matches = object.is_empty() || obj_variants.iter().any(|v| {
                        cat_obj.contains(v.as_str()) || v.contains(&**cat_obj)
                    });

                    if matches {
                        let chain = format!(
                            "{} {} {}, and {} {} {}",
                            subject, rel, category, cat_key, cat_rel, cat_obj
                        );
                        let answer = format!(
                            "Yes! Because {} {} {}, and {} {} {}.",
                            subject, rel, category, cat_key, cat_rel, cat_obj
                        );
                        return Some((answer, chain));
                    }
                }
            }
        }
    }

    None
}
fn parse_yes_no_question(words: &[&str]) -> (String, String, String) {
    if words.len() < 3 {
        return (String::new(), String::new(), String::new());
    }

    // "is [subject] a/an [object]" → (subject, "is", object)
    if words[0] == "is" || words[0] == "are" {
        let subject = words[1].to_lowercase();
        // Find "a", "an", "the" or just take the rest
        let rest_start = if words.len() > 2 && (words[2] == "a" || words[2] == "an" || words[2] == "the") {
            3
        } else {
            2
        };
        let object = words[rest_start..].join(" ").to_lowercase();
        return (subject, "is".to_string(), object);
    }

    // "does [subject] have [object]" → (subject, "has", object)
    if words[0] == "does" || words[0] == "do" {
        let subject = words[1].to_lowercase();
        if words.len() > 2 {
            let verb = words[2].to_lowercase();
            let relation = match verb.as_str() {
                "have" => "has".to_string(),
                _ => verb,
            };
            let object = if words.len() > 3 {
                words[3..].join(" ").to_lowercase()
            } else {
                String::new()
            };
            return (subject, relation, object);
        }
    }

    // "can [subject] [verb] [rest]" → (subject, "can", verb+rest)
    if words[0] == "can" {
        let subject = words[1].to_lowercase();
        let object = if words.len() > 2 {
            words[2..].join(" ").to_lowercase()
        } else {
            String::new()
        };
        return (subject, "can".to_string(), object);
    }

    (String::new(), String::new(), String::new())
}

/// Loose relation matching for yes/no questions.
fn relation_matches_loose(stored: &str, query: &str) -> bool {
    if stored == query { return true; }
    match (stored, query) {
        ("is", "are") | ("are", "is") => true,
        ("has", "have") | ("have", "has") => true,
        _ => stored.contains(query) || query.contains(stored),
    }
}

/// Respond to general questions.
fn respond_question(
    engine: &mut DeepManEngine,
    store: &tle_afc::IncrementalStore,
    topic: &str,
    _full_input: &str,
    start: Instant,
) {
    let words: Vec<&str> = topic.split_whitespace().collect();

    // Try to find relevant sentences using multiple keywords
    let mut best_sentence: Option<&str> = None;
    let mut best_score = 0;

    for word in &words {
        if word.len() <= 2 { continue; }
        let sentences = store.get_sentences(word);
        for sent in &sentences {
            // Score by how many query words appear in the sentence
            let score: usize = words.iter()
                .filter(|w| w.len() > 2 && sent.contains(&w.to_lowercase()))
                .count();
            if score > best_score {
                best_score = score;
                best_sentence = Some(sent);
            }
        }
    }

    if let Some(sent) = best_sentence {
        if best_score >= 2 {
            // Limit to first sentence
            let first = sent.split('.').next().unwrap_or(sent);
            if first.len() < 200 {
                println!("  {}. [{:?}]", capitalize(first.trim()), start.elapsed());
            } else {
                println!("  {}... [{:?}]", capitalize(&first[..150]), start.elapsed());
            }
            return;
        }
    }

    // Try fact store with first meaningful word
    for word in &words {
        if word.len() <= 2 { continue; }
        let clean = word.trim_start_matches("the ").trim_start_matches("a ");
        let facts = store.get_facts(clean);
        if !facts.is_empty() {
            let mut response_parts: Vec<String> = Vec::new();
            for (rel, obj) in facts.iter().take(3) {
                response_parts.push(format!("{} {} {}", clean, rel, obj));
            }
            println!("  {} [{:?}]", capitalize(&response_parts.join(". ")), start.elapsed());
            return;
        }
    }

    // Fallback: generate from Engram
    let (generated, gen_time) = engine.generate(topic);
    let output = engine.decode(&generated);

    if !output.is_empty() {
        println!("  {} [{:?}]", output, gen_time);
    } else {
        println!("  I'm not sure about that. Teach me with /teach!");
    }
}

/// Generate text continuation (default mode).
fn respond_generate(
    engine: &mut DeepManEngine,
    store: &tle_afc::IncrementalStore,
    input: &str,
    start: Instant,
) {
    // Check incremental store first
    let input_tokens: Vec<&str> = input.split_whitespace().collect();
    let incr_predictions = store.predict_next(&input_tokens);

    // Generate from main engine
    let (generated, gen_time) = engine.generate(input);
    let output = engine.decode(&generated);

    if !output.is_empty() {
        println!("  {} [{:?}]", output, gen_time);
    } else if !incr_predictions.is_empty() {
        let predicted: Vec<&str> = incr_predictions.iter().take(8).map(|(t, _)| t.as_str()).collect();
        println!("  {} [{:?}]", predicted.join(" "), start.elapsed());
    } else {
        println!("  (I need more context. Try a longer phrase or teach me with /teach!)");
    }
}

/// Handle /load command — ingest a text file.
fn handle_load(store: &mut tle_afc::IncrementalStore, path: &str) {
    let start = Instant::now();
    match fs::read_to_string(path) {
        Ok(content) => {
            let mut line_count = 0;
            let mut fact_count = 0;

            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.len() < 5 {
                    continue;
                }

                // Learn as text
                store.learn_text(trimmed);
                line_count += 1;

                // Try to extract facts from simple patterns:
                // "X is Y", "X are Y", "X has Y", "X can Y"
                let lower = trimmed.to_lowercase();
                for pattern in &[" is ", " are ", " has ", " can ", " was ", " scatters "] {
                    if let Some(pos) = lower.find(pattern) {
                        let subject = &trimmed[..pos];
                        let relation = pattern.trim();
                        let object = &trimmed[pos + pattern.len()..];
                        if subject.len() > 1 && object.len() > 1 {
                            store.learn_fact(subject, relation, object);
                            fact_count += 1;
                        }
                        break;
                    }
                }
            }

            println!("  ✓ Loaded '{}': {} lines, {} facts extracted [{:?}]",
                path, line_count, fact_count, start.elapsed());
        }
        Err(e) => {
            println!("  ✗ Error loading '{}': {}", path, e);
        }
    }
}

/// Fetch a web page, extract bounded text/facts, and teach both production stores.
fn handle_learn_url(store: &mut tle_afc::IncrementalStore, axiom_gen: &mut tle_axiom_gen::AxiomGen, url: &str) {
    if url.is_empty() {
        println!("  Usage: /learn-url <http(s)://...>");
        return;
    }

    let start = Instant::now();
    match web_learning::fetch_html(url) {
        Ok(html) => {
            let page = web_learning::extract_html(&html);
            let (title, sentences, facts) = learn_extracted_page(store, axiom_gen, &page);
            println!("  ✓ Learned '{}': {} sentences, {} facts [{:?}]", title, sentences, facts, start.elapsed());
        }
        Err(error) => println!("  ✗ Could not learn '{}': {}", url, error),
    }
}

fn handle_learn_url_background(
    jobs: &mut Vec<Receiver<(String, bool, Result<web_learning::ExtractedPage, String>)>>,
    url: &str,
) {
    if url.is_empty() {
        println!("  Usage: /learn-url-bg <http(s)://...>");
        return;
    }
    let url = url.to_string();
    let worker_url = url.clone();
    let result_url = url.clone();
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let result = web_learning::fetch_html(&worker_url).map(|html| web_learning::extract_html(&html));
        let _ = sender.send((result_url, false, result));
    });
    jobs.push(receiver);
    println!("  ✓ Background learning started for '{}'", url);
}

fn poll_background_learns(
    jobs: &mut Vec<Receiver<(String, bool, Result<web_learning::ExtractedPage, String>)>>,
    queue: &mut daemon::WebLearnQueue,
    store: &mut tle_afc::IncrementalStore,
    axiom_gen: &mut tle_axiom_gen::AxiomGen,
) {
    let mut remaining = Vec::with_capacity(jobs.len());
    for receiver in jobs.drain(..) {
        match receiver.try_recv() {
            Ok((url, queued, Ok(page))) => {
                let (title, sentences, facts) = learn_extracted_page(store, axiom_gen, &page);
                println!("  ✓ Background learned '{}': {} sentences, {} facts", title, sentences, facts);
                if queued {
                    let _ = queue.complete(&url);
                }
            }
            Ok((url, queued, Err(error))) => {
                println!("  ✗ Background learning failed: {}", error);
                if queued {
                    let _ = queue.fail(&url);
                }
            }
            Err(TryRecvError::Empty) => remaining.push(receiver),
            Err(TryRecvError::Disconnected) => println!("  ✗ Background learning worker disconnected"),
        }
    }
    *jobs = remaining;
}

fn start_next_queued_job(
    jobs: &mut Vec<Receiver<(String, bool, Result<web_learning::ExtractedPage, String>)>>,
    queue: &mut daemon::WebLearnQueue,
) {
    if !jobs.is_empty() {
        return;
    }
    let Some(job) = queue.next() else { return; };
    let url = job.url;
    let worker_url = url.clone();
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let result = web_learning::fetch_html(&worker_url).map(|html| web_learning::extract_html(&html));
        let _ = sender.send((url, true, result));
    });
    jobs.push(receiver);
}

fn learn_extracted_page(
    store: &mut tle_afc::IncrementalStore,
    axiom_gen: &mut tle_axiom_gen::AxiomGen,
    page: &web_learning::ExtractedPage,
) -> (String, usize, usize) {
    for sentence in &page.sentences {
        store.learn_text(sentence);
    }
    for fact in &page.facts {
        store.learn_fact(&fact.subject, &fact.relation, &fact.object);
        axiom_gen.add_fact(&fact.subject.to_lowercase(), &fact.relation, &fact.object.to_lowercase());
    }
    (page.title.clone().unwrap_or_else(|| "untitled page".to_string()), page.sentences.len(), page.facts.len())
}

/// Handle /save command — save learned knowledge to JSON.
fn handle_save(store: &tle_afc::IncrementalStore, path: &str) {
    let start = Instant::now();

    // Serialize the IncrementalStore's vocabulary and stats
    let stats = store.stats();
    let save_data = format!(
        "{{\"axiom_version\": \"0.1.0\", \"facts\": {}, \"tokens\": {}, \"transitions\": {}, \"vocab_size\": {}}}",
        stats.facts_added, stats.tokens_ingested, stats.transitions_added, stats.vocab_size
    );

    // Save transition memory as binary
    let tm_path = format!("{}.tm", path);
    let tm_data: Vec<u8> = store.transition_memory.data
        .iter()
        .flat_map(|&f| f.to_le_bytes())
        .collect();

    // Save KG memory
    let kg_path = format!("{}.kg", path);
    let kg_data: Vec<u8> = store.kg_memory.data
        .iter()
        .flat_map(|&f| f.to_le_bytes())
        .collect();

    match fs::write(path, &save_data) {
        Ok(_) => {
            let _ = fs::write(&tm_path, &tm_data);
            let _ = fs::write(&kg_path, &kg_data);
            println!("  ✓ Saved to '{}' (+.tm +.kg) [{:?}]", path, start.elapsed());
            println!("    {} facts, {} tokens, {} transitions",
                stats.facts_added, stats.tokens_ingested, stats.transitions_added);
        }
        Err(e) => {
            println!("  ✗ Error saving to '{}': {}", path, e);
        }
    }
}

/// Handle /restore command — restore knowledge from saved files.
fn handle_restore(store: &mut tle_afc::IncrementalStore, path: &str) {
    let start = Instant::now();

    // Restore transition memory
    let tm_path = format!("{}.tm", path);
    match fs::read(&tm_path) {
        Ok(data) => {
            let dim = store.transition_memory.data.len();
            if data.len() == dim * 4 {
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                store.transition_memory = HyperVector::new(floats);
                println!("  ✓ Restored transition memory from '{}' [{:?}]", tm_path, start.elapsed());
            } else {
                println!("  ⚠ TM dimension mismatch (expected {}, got {})", dim * 4, data.len());
            }
        }
        Err(e) => {
            println!("  ✗ Cannot restore '{}': {}", tm_path, e);
        }
    }

    // Restore KG memory
    let kg_path = format!("{}.kg", path);
    match fs::read(&kg_path) {
        Ok(data) => {
            let dim = store.kg_memory.data.len();
            if data.len() == dim * 4 {
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                store.kg_memory = HyperVector::new(floats);
                println!("  ✓ Restored KG memory from '{}'", kg_path);
            }
        }
        Err(_) => {} // KG is optional
    }

    // Note: N-gram counts are NOT restored (they're in the HashMap, would need full serialization)
    // For now, TM + KG give the VSA-based retrieval back
    println!("  Note: N-gram counts need re-learning from text (VSA memories restored)");
}

/// Capitalize first letter of a string.
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
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
