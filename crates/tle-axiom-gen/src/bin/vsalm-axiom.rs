//! vsalm-axiom: AXIOM-Gen → VSA-LM integration benchmark.
//!
//! The full conversational-AXIOM path, all non-neural:
//!   1. Evidence → decompose → AxiomGen knowledge graph (fact reasoning)
//!   2. Graph triples → `sync_into_vsa_lm` → KnowledgePrior
//!   3. Question → VSA-LM energy beam search → fluent fact-grounded answer
//!
//! Usage:
//!   vsalm-axiom <qa.json> <evidence_dir> [record_limit]

use std::env;
use std::fs;
use std::io::Read;
use std::time::{Duration, Instant};

use serde_json::Value;
use tle_axiom_gen::AxiomGen;
use tle_axiom_gen::decompose::decompose_sentence;
use tle_vsa_lm::{LmConfig, VsaLm};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let qa_path = env::args().nth(1).unwrap_or_else(|| "data/triviaqa/qa/verified-wikipedia-dev.json".to_string());
    let evidence_dir = env::args().nth(2).unwrap_or_else(|| "data/triviaqa/evidence/wikipedia".to_string());
    let limit = env::args().nth(3).and_then(|v| v.parse().ok());

    let records = load_qa(&qa_path)?;
    let limit = limit.unwrap_or(records.len());

    let mut total = 0usize;
    let mut substring = 0usize;
    let mut candidate_hits = 0usize;
    let mut total_latency = Duration::ZERO;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  AXIOM-Gen → VSA-LM integration (conversational AXIOM)      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("QA: {}", qa_path);
    println!("Evidence dir: {}", evidence_dir);
    println!("Records to run: {}", limit);

    for record in records.iter().take(limit) {
        let mut engine = AxiomGen::new(4096);
        let facts = extract_document_facts(&evidence_dir, &record.evidence_files, &record.question);

        for fact in &facts {
            engine.add_fact(&fact[0], &fact[1], &fact[2]);
        }

        // Build the VSA-LM with the AXIOM-Gen graph as its knowledge prior.
        let config = LmConfig {
            dim: 4096,
            max_order: 4,
            beam_width: 8,
            max_gen_tokens: 8,
            w_knowledge: 3.0,
            knowledge_only: true,
            ..Default::default()
        };
        let mut lm = VsaLm::new(config);
        engine.sync_into_vsa_lm(&mut lm);

        let start = Instant::now();
        let out = lm.generate(&record.question, Some(8));
        total_latency += start.elapsed();

        let output = out.to_lowercase();
        let answers: Vec<String> = record.answers.iter().map(|a| a.to_lowercase()).collect();
        if answers.iter().any(|answer| output.contains(answer)) {
            substring += 1;
        }

        // Candidate answer: the first fact-grounded word after the prompt.
        let engine_answer = first_novel_word(&out, &record.question).unwrap_or_default();
        if !engine_answer.is_empty()
            && answers.iter().any(|answer| engine_answer.contains(answer) || answer.contains(&engine_answer))
        {
            candidate_hits += 1;
        }

        if total < 12 {
            println!("\n  Q: {}", record.question);
            println!("     gold: {:?}", record.answers);
            println!("     out:  {}", out);
            println!("     answer: {:?}", engine_answer);
        }

        total += 1;
    }

    println!("\n━━━ AXIOM → VSA-LM RESULTS ━━━");
    println!("  records: {}", total);
    println!("  substring_accuracy: {:.2}%", percentage(substring, total));
    println!("  candidate_answer_accuracy: {:.2}%", percentage(candidate_hits, total));
    println!("  avg_latency: {:?}", if total == 0 { Duration::ZERO } else { total_latency / total as u32 });
    Ok(())
}

/// The first content word the model emitted that is not part of the prompt
/// and not a stopword.
fn first_novel_word(output: &str, prompt: &str) -> Option<String> {
    let prompt_words: Vec<String> = prompt
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect();
    output
        .to_lowercase()
        .split_whitespace()
        .find(|w| !prompt_words.contains(&w.to_string()) && !is_stopword(w))
        .map(str::to_string)
}

fn is_stopword(word: &str) -> bool {
    matches!(
        word,
        "the" | "a" | "an" | "and" | "or" | "but" | "of" | "in" | "on" | "at" | "to"
            | "for" | "with" | "by" | "as" | "is" | "are" | "was" | "were" | "be"
            | "been" | "being" | "has" | "have" | "had" | "it" | "its" | "this"
            | "that" | "these" | "those" | "which" | "who" | "whom" | "whose"
            | "what" | "when" | "where" | "why" | "how" | "no" | "not" | "nor"
            | "from" | "up" | "down" | "out" | "about" | "into" | "over" | "under"
            | "again" | "then" | "once" | "here" | "there" | "all" | "any" | "both"
            | "each" | "few" | "more" | "most" | "other" | "some" | "such" | "than"
            | "too" | "very" | "can" | "will" | "just" | "should" | "would" | "could"
            | "may" | "might" | "must" | "shall" | "am" | "do" | "did" | "does"
    )
}

#[derive(Debug)]
struct Record {
    question: String,
    answers: Vec<String>,
    evidence_files: Vec<String>,
}

fn load_qa(path: &str) -> Result<Vec<Record>, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let document: Value = serde_json::from_str(&content)?;
    let items = document.get("Data").and_then(Value::as_array).unwrap();
    let mut records = Vec::new();
    for item in items {
        let question = item.get("Question").and_then(Value::as_str).unwrap_or_default().to_string();
        let mut answers = Vec::new();
        if let Some(answer) = item.get("Answer") {
            if let Some(value) = answer.get("Value").and_then(Value::as_str) {
                answers.push(value.to_string());
            }
            if let Some(aliases) = answer.get("Aliases").and_then(Value::as_array) {
                answers.extend(aliases.iter().filter_map(Value::as_str).map(str::to_string));
            }
        }
        let evidence_files = item.get("EntityPages").and_then(Value::as_array)
            .map(|pages| pages.iter().filter_map(|p| p.get("Filename").and_then(Value::as_str).map(str::to_string)).collect())
            .unwrap_or_default();
        records.push(Record { question, answers, evidence_files });
    }
    Ok(records)
}

/// Decompose the top question-overlap sentences of each evidence file into
/// structured triples (mirrors triviaqa-bench's extraction).
fn extract_document_facts(directory: &str, files: &[String], question: &str) -> Vec<[String; 3]> {
    let mut facts = Vec::new();
    for filename in files {
        let path = std::path::Path::new(directory).join(filename);
        let Ok(file) = fs::File::open(path) else { continue; };
        let mut text = String::new();
        if file.take(256 * 1024).read_to_string(&mut text).is_err() {
            continue;
        }
        let subject = filename.trim_end_matches(".txt").replace('_', " ");
        let clean = clean_wikipedia_text(&text);

        let mut sentences: Vec<(usize, String)> = clean
            .split(|c| matches!(c, '.' | '!' | '?'))
            .take(300)
            .map(|s| s.split_whitespace().collect::<Vec<_>>().join(" "))
            .filter(|s| is_likely_sentence(s))
            .map(|s| (question_overlap(question, &s), s))
            .collect();
        sentences.sort_by(|a, b| b.0.cmp(&a.0));

        let mut seen = std::collections::HashSet::new();
        for (_, sentence) in sentences.into_iter().take(12) {
            for fact in decompose_sentence(&sentence, &subject) {
                let key = (fact.subject.clone(), fact.relation.clone(), fact.object.clone());
                if seen.insert(key.clone()) {
                    facts.push([fact.subject, fact.relation, fact.object]);
                }
            }
        }
    }
    facts
}

fn clean_wikipedia_text(text: &str) -> String {
    let mut cleaned = String::with_capacity(text.len());
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let lower = trimmed.to_lowercase();
        if lower.starts_with("references") || lower.starts_with("external links")
            || lower.starts_with("see also") || lower.starts_with("notes")
            || lower.starts_with("bibliography") || lower.starts_with("further reading")
            || lower.starts_with("== ") || trimmed.starts_with('|') || trimmed.starts_with('!')
        {
            continue;
        }
        cleaned.push_str(trimmed);
        cleaned.push(' ');
    }

    let mut out = String::with_capacity(cleaned.len());
    let mut depth = 0usize;
    for ch in cleaned.chars() {
        match ch {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }

    let mut refs_removed = String::with_capacity(out.len());
    let mut chars = out.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '[' {
            while let Some(next) = chars.next() {
                if next == ']' {
                    break;
                }
            }
        } else {
            refs_removed.push(ch);
        }
    }
    refs_removed
}

fn is_likely_sentence(sentence: &str) -> bool {
    let words: Vec<&str> = sentence.split_whitespace().collect();
    if words.len() < 4 || words.len() > 60 {
        return false;
    }
    let first = words[0].to_lowercase();
    !matches!(first.as_str(), "and" | "or" | "but" | "by" | "which" | "that" | "for" | "in" | "at" | "an" | "of")
        && !sentence.contains("==")
        && !sentence.starts_with('|')
        && !sentence.starts_with('*')
        && !sentence.starts_with('#')
        && !sentence.starts_with(';')
}

fn question_overlap(question: &str, sentence: &str) -> usize {
    let question_words: Vec<String> = question.to_lowercase().split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| w.len() >= 4 && !matches!(w.as_str(), "what" | "which" | "where" | "when" | "does" | "have"))
        .collect();
    let lower_sentence = sentence.to_lowercase();
    question_words.iter().filter(|w| lower_sentence.contains(w.as_str())).count()
}

fn percentage(correct: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        correct as f64 * 100.0 / total as f64
    }
}
