//! TriviaQA-compatible JSONL benchmark for taught structured facts.

use std::env;
use std::fs::File;
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::Value;
use tle_axiom_gen::decompose::decompose_sentence;
use tle_axiom_gen::AxiomGen;

#[derive(Debug, Deserialize)]
struct Record {
    #[serde(default)]
    id: String,
    question: String,
    answers: Vec<String>,
    #[serde(default)]
    facts: Vec<[String; 3]>,
    #[serde(default)]
    evidence_files: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EvidenceRecord {
    id: String,
    #[serde(default)]
    facts: Vec<[String; 3]>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).unwrap_or_else(|| "data/axiom_triviaqa.jsonl".to_string());
    let evidence_path = env::args().nth(2);
    let evidence_dir = env::args().nth(3);
    let records = load_records(&path)?;
    let evidence = match evidence_path.as_deref() {
        None | Some("-") => HashMap::new(),
        Some(path) => load_evidence(path)?,
    };
    let mut total = 0usize;
    let mut exact = 0usize;
    let mut substring = 0usize;
    let mut candidate_hits = 0usize;
    let mut answer_entity_recall = 0usize;
    let mut evidence_hits = 0usize;
    let mut total_latency = Duration::ZERO;

    let limit = env::var("AXIOM_TRIVIA_LIMIT").ok().and_then(|value| value.parse().ok());
    let debug = env::var("AXIOM_TRIVIA_DEBUG").ok().map(|_| ());
    for record in records.into_iter().take(limit.unwrap_or(usize::MAX)) {
        let mut engine = new_qa_engine();
        let evidence_facts = evidence.get(&record.id).cloned().unwrap_or_default();
        let document_facts = evidence_dir.as_deref()
            .map(|directory| extract_document_facts(directory, &record.evidence_files, &record.question))
            .unwrap_or_default();
        let document_text = evidence_dir.as_deref()
            .map(|directory| read_evidence_text(directory, &record.evidence_files))
            .unwrap_or_default();
        if record.answers.iter().any(|answer| document_text.to_lowercase().contains(&answer.to_lowercase())) {
            evidence_hits += 1;
        }
        for fact in record.facts.iter().chain(evidence_facts.iter()).chain(document_facts.iter()) {
            engine.add_fact(&fact[0], &fact[1], &fact[2]);
        }
        let start = Instant::now();
        let result = engine.generate(&record.question);
        total_latency += start.elapsed();
        let output = result.sentence.to_lowercase();
        let answers: Vec<String> = record.answers.iter().map(|answer| answer.to_lowercase()).collect();
        if answers.iter().any(|answer| output.trim() == *answer) { exact += 1; }
        if answers.iter().any(|answer| output.contains(answer)) { substring += 1; }

        // Candidate answer accuracy: the answer entity selected by AXIOM-Gen
        // itself from the best path, scored structurally (novelty + role bias
        // + VSA relevance). No answer oracle is consulted at runtime.
        let engine_answer = result.answer.to_lowercase();
        if !engine_answer.is_empty()
            && answers.iter().any(|answer| engine_answer.contains(answer) || answer.contains(&engine_answer)) {
            candidate_hits += 1;
        }

        // Diagnostic: is any gold answer even present as a graph node? This
        // separates "answer not extracted into graph" from "ranking/generation
        // failed", so we never mistake one failure mode for another.
        let lower_entities: Vec<String> = engine.graph.entities.iter().map(|e| e.to_lowercase()).collect();
        if answers.iter().any(|answer| lower_entities.iter().any(|entity| entity.contains(answer) || answer.contains(entity))) {
            answer_entity_recall += 1;
        }
        if debug.is_some() && total < 8 {
            println!("---");
            println!("Q: {}", record.question);
            println!("  gold: {:?}", record.answers);
            println!("  answer(entity): {:?}", engine_answer);
            println!("  entities({}): {:?}", engine.graph.entities.len(),
                engine.graph.entities.iter().take(20).collect::<Vec<_>>());
            println!("  sentence: {}", output);
        }
        if total > 0 && total % 50 == 0 {
            println!("  [{} records done, avg so far: {:?}]", total, total_latency / total as u32);
        }
        if total < 10 {
            println!("  [{}/{} {:?}] Q: {}", total + 1, limit.unwrap_or(318), start.elapsed(), record.question);
        }
        total += 1;
    }

    println!("TriviaQA-compatible AXIOM benchmark");
    println!("  records: {}", total);
    println!("  exact_accuracy: {:.2}%", percentage(exact, total));
    println!("  substring_accuracy: {:.2}%", percentage(substring, total));
    println!("  candidate_answer_accuracy: {:.2}%", percentage(candidate_hits, total));
    println!("  answer_entity_recall: {:.2}%", percentage(answer_entity_recall, total));
    println!("  evidence_answer_recall: {:.2}%", percentage(evidence_hits, total));
    println!("  avg_latency: {:?}", if total == 0 { Duration::ZERO } else { total_latency / total as u32 });
    Ok(())
}

/// Build an AXIOM-Gen engine tuned for QA: answers are usually 1-3 hops, so a
/// bounded hop depth and narrower beam avoid exponential beam-search blowup on
/// noisy decomposition graphs (which can have hundreds of triples per page).
fn new_qa_engine() -> AxiomGen {
    let mut engine = AxiomGen::new(2048);
    engine.search_config.max_hops = 3;
    engine.search_config.beam_width = 16;
    engine.search_config.early_exit_on_stall = true;
    engine
}

fn load_records(path: &str) -> Result<Vec<Record>, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    if let Ok(document) = serde_json::from_str::<Value>(&content) {
        if let Some(items) = document.get("Data").and_then(Value::as_array) {
            let mut records = Vec::new();
            for item in items {
                let id = item.get("QuestionId").and_then(Value::as_str).unwrap_or_default().to_string();
                let question = item.get("Question").and_then(Value::as_str).unwrap_or_default().to_string();
                let mut answers = Vec::new();
                if let Some(answer) = item.get("Answer") {
                    if let Some(value) = answer.get("Value").and_then(Value::as_str) { answers.push(value.to_string()); }
                    if let Some(aliases) = answer.get("Aliases").and_then(Value::as_array) {
                        answers.extend(aliases.iter().filter_map(Value::as_str).map(str::to_string));
                    }
                }
                let evidence_files = item.get("EntityPages").and_then(Value::as_array)
                    .map(|pages| pages.iter().filter_map(|page| page.get("Filename").and_then(Value::as_str).map(str::to_string)).collect())
                    .unwrap_or_default();
                records.push(Record { id, question, answers, facts: Vec::new(), evidence_files });
            }
            return Ok(records);
        }
    }

    let reader = BufReader::new(File::open(path)?);
    reader.lines()
        .filter(|line| line.as_ref().map(|line| !line.trim().is_empty()).unwrap_or(true))
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect()
}

fn extract_document_facts(directory: &str, files: &[String], question: &str) -> Vec<[String; 3]> {
    let mut facts = Vec::new();
    for filename in files {
        let path = std::path::Path::new(directory).join(filename);
        let Ok(file) = File::open(path) else { continue; };
        let mut text = String::new();
        if file.take(256 * 1024).read_to_string(&mut text).is_err() { continue; }
        let subject = filename.trim_end_matches(".txt").replace('_', " ");
        let clean = clean_wikipedia_text(&text);

        // Rank cleaned sentences by question-word overlap and decompose the
        // top ones into structured triples so entities inside the sentence
        // become graph nodes the beam search can traverse.
        let mut sentences: Vec<(usize, String)> = clean
            .split(|character| matches!(character, '.' | '!' | '?'))
            .take(300)
            .map(|sentence| sentence.split_whitespace().collect::<Vec<_>>().join(" "))
            .filter(|sentence| is_likely_sentence(sentence))
            .map(|sentence| (question_overlap(question, &sentence), sentence))
            .collect();
        sentences.sort_by(|left, right| right.0.cmp(&left.0));

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

/// Strip the noise that dominates raw Wikipedia dumps: reference markers,
/// parenthetical disambiguation, table/infobox cells, and trailing sections.
fn clean_wikipedia_text(text: &str) -> String {
    let mut cleaned = String::with_capacity(text.len());
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        let lower = trimmed.to_lowercase();
        if lower.starts_with("references") || lower.starts_with("external links")
            || lower.starts_with("see also") || lower.starts_with("notes")
            || lower.starts_with("bibliography") || lower.starts_with("further reading")
            || lower.starts_with("== ") || trimmed.starts_with('|') || trimmed.starts_with('!') {
            continue;
        }
        cleaned.push_str(trimmed);
        cleaned.push(' ');
    }

    // Remove content inside parentheses (often pronunciation/disambiguation).
    let mut out = String::with_capacity(cleaned.len());
    let mut depth = 0usize;
    for ch in cleaned.chars() {
        match ch {
            '(' => depth += 1,
            ')' => { depth = depth.saturating_sub(1); }
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }

    // Remove bracketed reference markers like [1] or [citation needed].
    let mut refs_removed = String::with_capacity(out.len());
    let mut chars = out.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '[' {
            while let Some(next) = chars.next() {
                if next == ']' { break; }
            }
        } else {
            refs_removed.push(ch);
        }
    }
    refs_removed
}

fn is_likely_sentence(sentence: &str) -> bool {
    let words: Vec<&str> = sentence.split_whitespace().collect();
    if words.len() < 4 || words.len() > 60 { return false; }
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
        .map(|word| word.trim_matches(|character: char| !character.is_alphanumeric()).to_string())
        .filter(|word| word.len() >= 4 && !matches!(word.as_str(), "what" | "which" | "where" | "when" | "does" | "have"))
        .collect();
    let lower_sentence = sentence.to_lowercase();
    question_words.iter().filter(|word| lower_sentence.contains(word.as_str())).count()
}

fn read_evidence_text(directory: &str, files: &[String]) -> String {
    let mut text = String::new();
    for filename in files {
        let path = std::path::Path::new(directory).join(filename);
        if let Ok(file) = File::open(path) {
            let mut limited = file.take(256 * 1024);
            let _ = limited.read_to_string(&mut text);
        }
    }
    text
}

fn load_evidence(path: &str) -> Result<HashMap<String, Vec<[String; 3]>>, Box<dyn std::error::Error>> {
    let reader = BufReader::new(File::open(path)?);
    let mut evidence = HashMap::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let record: EvidenceRecord = serde_json::from_str(&line)?;
        evidence.insert(record.id, record.facts);
    }
    Ok(evidence)
}

fn percentage(correct: usize, total: usize) -> f64 {
    if total == 0 { 0.0 } else { correct as f64 * 100.0 / total as f64 }
}
