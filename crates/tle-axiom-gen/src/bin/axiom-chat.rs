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

/// H4: deterministic TF-IDF sentence retrieval over a corpus (RAG). Query →
/// top-K best-matching corpus sentences, scored by Σ tf×idf (cosine-ish).
struct SentenceIndex {
    sents: Vec<String>,
    df: std::collections::HashMap<String, usize>,
    postings: std::collections::HashMap<String, Vec<(usize, u32)>>,
    n: usize,
}

impl SentenceIndex {
    fn new() -> Self {
        Self { sents: Vec::new(), df: std::collections::HashMap::new(), postings: std::collections::HashMap::new(), n: 0 }
    }

    fn tokens(s: &str) -> Vec<String> {
        s.split(|c: char| !c.is_alphanumeric())
            .map(|t| t.trim().to_lowercase())
            .filter(|t| t.len() >= 3)
            .collect()
    }

    fn add(&mut self, sentence: &str) {
        let idx = self.n;
        self.sents.push(sentence.to_string());
        let toks = Self::tokens(sentence);
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut tf: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        for t in toks {
            *tf.entry(t.clone()).or_insert(0) += 1;
            seen.insert(t);
        }
        for t in seen {
            let tfv = tf.get(&t).copied().unwrap_or(0);
            *self.df.entry(t.clone()).or_insert(0) += 1;
            self.postings.entry(t).or_default().push((idx, tfv));
        }
        self.n += 1;
    }

    fn idf(&self, term: &str) -> f32 {
        let d = self.df.get(term).copied().unwrap_or(0) as f32;
        if d == 0.0 {
            0.0
        } else {
            (1.0 + self.n as f32 / d).ln()
        }
    }

    /// Top-K sentences by TF-IDF cosine similarity to the query.
    fn recall(&self, query: &str, k: usize) -> Vec<(f32, String)> {
        let qt: Vec<String> = Self::tokens(query);
        if qt.is_empty() {
            return Vec::new();
        }
        let mut scores = vec![0.0f32; self.n];
        let mut qnorms = vec![0.0f32; self.n];
        let mut qidfs: Vec<(String, f32)> = qt.iter().map(|t| (t.clone(), self.idf(t))).collect();
        let mut q_norm = 0.0f32;
        for (_, idf) in &qidfs {
            q_norm += idf * idf;
        }
        q_norm = q_norm.sqrt().max(1e-6);
        for (t, idf) in &qidfs {
            if let Some(postings) = self.postings.get(t) {
                for (idx, tf) in postings {
                    scores[*idx] += idf * idf * *tf as f32;
                }
            }
        }
        // length-normalize: divide by |sentence|
        let mut ranked: Vec<(f32, usize)> = (0..self.n)
            .map(|i| {
                let len = self.sents[i].split_whitespace().count().max(1) as f32;
                (scores[i] / len / q_norm, i)
            })
            .collect();
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        ranked.into_iter().take(k).map(|(s, i)| (s, self.sents[i].clone())).collect()
    }

    fn is_empty(&self) -> bool {
        self.n == 0
    }
}

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

    // optional corpus for VSA-LM fluency + RAG recall: `axiom-chat --corpus <file>`
    let mut corpus: SentenceIndex = SentenceIndex::new();
    if std::env::args().any(|a| a == "--corpus") {
        if let Some(path) = std::env::args().nth(2) {
            let text = std::fs::read_to_string(&path)?;
            let mut n = 0;
            for line in text.lines().filter(|l| l.len() > 8).take(2000) {
                lm.learn(line);
                corpus.add(line);
                n += 1;
            }
            println!("  corpus: learned {} sentences (VSA-LM + RAG index)", n);
        }
    }
    // H5: load a sample of the Wikipedia evidence corpus into the RAG index.
    // `--evidence <dir> [limit]` (default limit 3000 files)
    if std::env::args().any(|a| a == "--evidence") {
        let dir = std::env::args().nth(2).unwrap_or_default();
        let limit: usize = std::env::args().nth(3).and_then(|v| v.parse().ok()).unwrap_or(3000);
        let t0 = Instant::now();
        let mut n_sent = 0usize;
        let mut n_file = 0usize;
        for entry in std::fs::read_dir(&dir).into_iter().flatten() {
            if n_file >= limit {
                break;
            }
            let Ok(entry) = entry else { continue };
            let path = entry.path();
            if path.extension().map(|e| e != "txt").unwrap_or(true) {
                continue;
            }
            if let Ok(text) = std::fs::read_to_string(&path) {
                for s in text.split(|c: char| matches!(c, '.' | '\n')) {
                    let s = s.trim();
                    let clean: &str = s.trim_start_matches(|c: char| matches!(c, '*' | '#' | '|' | '-' | '=' | ':' | ' '));
                    if clean.len() >= 30
                        && clean.len() < 300
                        && !clean.contains("http")
                        && !clean.contains("://")
                        && !clean.contains('|')
                        && !clean.contains('=')
                        && !clean.contains('<')
                        && !clean.contains('>')
                        && !clean.contains('{')
                        && clean.split_whitespace().count() >= 6
                    {
                        corpus.add(clean);
                        n_sent += 1;
                    }
                }
                n_file += 1;
            }
        }
        println!("  evidence: {} sentences from {} files in {:.2?}", n_sent, n_file, t0.elapsed());
    }

    // optional starter facts so a first chat isn't empty
    for fact in [("sky", "is", "blue"), ("blue", "has", "short_wavelength")] {
        teach(&mut graph, &mut lm, &format!("{} {} {}", fact.0, fact.1, fact.2));
    }
    lm.kn5.finalize();

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
    // H2 turn memory: resolve pronouns ("it/they/that") → last discussed topic.
    let mut mem = tle_afc::DeltaMem::new(2048);

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
        // H2: resolve pronouns against the last discussed topic BEFORE intent
        // handling, so "what is it?" continues the previous turn.
        let mut resolved = mem.resolve_pronoun(&lower);
        // word-boundary pronoun swap (handles "it?" / "that." / "they,")
        if let Some(ls) = last_subject.clone() {
            let words: Vec<String> = resolved.split_whitespace().map(|w| {
                let bare = w.trim_matches(|c: char| !c.is_alphabetic());
                if matches!(bare, "it" | "it's" | "that" | "they" | "them") {
                    if let Some(s) = ls.split_whitespace().next() {
                        w.replace(bare, s)
                    } else {
                        w.to_string()
                    }
                } else {
                    w.to_string()
                }
            }).collect();
            resolved = words.join(" ");
        }
        let question = if resolved != lower { &resolved } else { &lower };

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
            // H7: KN-5 greedy argmax generation (16.7% TEST vs the fused
            // VSA-LM 11%; deterministic, better local coherence).
            let out = lm.generate_kn5(prompt, 24);
            println!("  {} [{:?}]", out, start.elapsed());
            continue;
        }
        if trimmed == "/stats" {
            println!("  graph triples: {}", graph.graph.triples.len());
            println!("  vocab: {} words", lm.vocab.len());
            println!("  corpus sentences: {}", corpus.n);
            if !corpus.is_empty() {
                let (r1, r10) = self_recall(&corpus, 200);
                println!("  self-retrieval R@1: {:.1}%  R@10: {:.1}% (H4 diagnostic)", r1 * 100.0, r10 * 100.0);
            }
            continue;
        }
        if let Some(q) = trimmed.strip_prefix("/recall ") {
            // H4 RAG: retrieve the best-matching corpus sentences
            if corpus.is_empty() {
                println!("  No corpus loaded. Run with --corpus <file> first.");
            } else {
                let results = corpus.recall(q, 3);
                for (score, sent) in results {
                    println!("  [{:>4.2}] {}", score, sent);
                }
            }
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
        // fast path: "what/who is X" → answer the direct is/are fact cleanly
        // (avoids over-chaining "A sun is a star, that is An earth orbits a sun")
        let mut answered = false;
        for prefix in ["what is ", "what's ", "who is ", "what was ", "what are ", "who was "] {
            if let Some(subj) = question.strip_prefix(prefix) {
                let subj = subj.trim().trim_matches('?').trim();
                let subj = subj
                    .strip_prefix("the ")
                    .or_else(|| subj.strip_prefix("a "))
                    .unwrap_or(subj);
                if let Some(sid) = graph.graph.entity_id(subj) {
                    for t in &graph.graph.triples {
                        if t.subject_id == sid {
                            let r = graph.graph.relation_name(t.relation_id);
                            if r == "is" || r == "is_a" || r == "are" || r == "has" || r == "has_a" {
                                println!("  {} is {}.", capitalize(subj), graph.graph.entity_name(t.object_id));
                                println!("  [{:?}]", start.elapsed());
                                // H2: register the topic so "what is it?" resolves
                                last_subject = Some(subj.to_string());
                                mem.update_topic(subj);
                                answered = true;
                                break;
                            }
                        }
                    }
                }
                if answered {
                    break;
                }
            }
        }
        if answered {
            continue;
        }
        // 1-hop fast path: "what does X <verb>?" → direct object (no over-chain)
        for subj_prefix in ["what does ", "what do ", "what did ", "who does ", "which "] {
            if let Some(rest) = question.strip_prefix(subj_prefix) {
                let words: Vec<&str> = rest.split_whitespace().collect();
                let subj = words
                    .first()
                    .map(|w| w.trim_matches(|c: char| c == ',' || c == '?'))
                    .unwrap_or("");
                if let Some(sid) = graph.graph.entity_id(subj) {
                    for t in &graph.graph.triples {
                        if t.subject_id == sid {
                            let r = graph.graph.relation_name(t.relation_id);
                            let o = graph.graph.entity_name(t.object_id);
                            println!("  {} {} {}.", capitalize(subj), r, o);
                            println!("  [{:?}]", start.elapsed());
                            answered = true;
                            break;
                        }
                    }
                }
                if answered {
                    break;
                }
            }
        }
        if answered {
            continue;
        }

        let gen = graph.generate(question);
        if gen.path_length >= 1 && !gen.sentence.is_empty() {
            println!("  {}", gen.sentence);
            if !gen.reasoning.is_empty() {
                println!("  [trace: {}]", gen.reasoning.join(" → "));
            }
            if !gen.answer.is_empty() {
                last_subject = Some(gen.answer.clone());
                mem.update_topic(&gen.answer);
            }
        } else if is_followup(&lower) {
            // short continuation on the last topic ("and what about it?")
            if let Some(prev) = last_subject.take() {
                let gen = graph.generate(&format!("what is {}?", prev));
                if gen.path_length >= 1 && !gen.sentence.is_empty() {
                    println!("  {}", gen.sentence);
                } else {
                    println!("  I don't know more about \"{}\" yet.", prev);
                }
            } else {
                println!("  I don't know \"{}\" yet. Teach me with /teach.", trimmed);
            }
        } else if !corpus.is_empty() {
            // H4 RAG fallback: the graph doesn't know — retrieve the best
            // corpus sentence (deterministic, attributable, no hallucination).
            let results = corpus.recall(&trimmed, 1);
            if let Some((score, sent)) = results.first().filter(|(s, _)| *s > 0.05) {
                println!("  (from my reading, relevance {:.2}) {}", score, sent);
            } else {
                println!("  I don't know \"{}\" yet. Teach me with /teach.", trimmed);
            }
        } else {
            println!("  I don't know \"{}\" yet. Teach me with /teach, or ask me to generate with \"gen ...\".", trimmed);
        }
        println!("  [{:?}]", start.elapsed());
    }
    Ok(())
}

/// H4 diagnostic: self-retrieval R@1/R@10 — query each sentence with its own
/// first 6 words, does the index retrieve it in the top-K? (a standard RAG
/// sanity metric)
fn self_recall(index: &SentenceIndex, max: usize) -> (f32, f32) {
    let mut r1 = 0usize;
    let mut r10 = 0usize;
    let mut total = 0usize;
    for (i, s) in index.sents.iter().take(max).enumerate() {
        let words: Vec<&str> = s.split_whitespace().collect();
        if words.len() < 8 {
            continue;
        }
        let query = words[..6].join(" ");
        let hits = index.recall(&query, 10);
        total += 1;
        let found = hits.iter().position(|(_, sent)| *sent == *s);
        if let Some(pos) = found {
            if pos == 0 {
                r1 += 1;
            }
            if pos < 10 {
                r10 += 1;
            }
        }
        let _ = i;
    }
    if total == 0 {
        (0.0, 0.0)
    } else {
        (r1 as f32 / total as f32, r10 as f32 / total as f32)
    }
}

/// A short vague continuation that refers to the last topic.
fn is_followup(lower: &str) -> bool {
    let words: Vec<&str> = lower.split_whitespace().collect();
    words.len() <= 4
        && (lower.contains("about it") || lower.contains("more") || lower.contains("tell me")
            || lower.contains("what else") || lower.contains("and"))
}

/// Teach a fact to both the graph and the VSA-LM.
fn teach(graph: &mut AxiomGen, lm: &mut VsaLm, fact: &str) {
    // extract triples the same way the pipeline does
    let mut added = false;
    for d in tle_axiom_gen::decompose::decompose_sentence(fact, "") {
        graph.add_fact(&d.subject, &d.relation, &d.object);
        added = true;
    }
    // fallback parse for verbs the decomposer doesn't know:
    // "earth orbits the sun" → (earth, orbits, sun)
    if !added {
        let words: Vec<&str> = fact.split_whitespace().collect();
        if words.len() >= 3 {
            let subj = words[0].trim_matches(|c: char| c == ',');
            let mut body: Vec<&str> = words[1..].to_vec();
            // object = longest tail after the LAST preposition ("boils at 100
            // degrees" → object "100 degrees"), else the last word
            let mut obj_idx = words.len() - 1;
            for i in (1..words.len() - 1).rev() {
                if matches!(words[i], "at" | "in" | "on" | "by" | "from" | "to" | "for" | "with") {
                    obj_idx = i + 1;
                    break;
                }
            }
            let obj = words[obj_idx..].join(" ").trim_matches('.').to_string();
            body.truncate(obj_idx - 1); // relation = words[1..obj_idx]
            let rel: String = body
                .iter()
                .filter(|w| !matches!(**w, "the" | "a" | "an"))
                .map(|w| w.to_string())
                .collect::<Vec<_>>()
                .join("_");
            if !subj.is_empty() && !obj.is_empty() && !rel.is_empty() {
                graph.add_fact(subj, &rel, &obj);
                println!("  (parsed: {} {} {})", subj, rel, obj);
                added = true;
            }
        }
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
