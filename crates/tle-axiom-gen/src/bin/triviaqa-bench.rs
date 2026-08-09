//! TriviaQA-compatible JSONL benchmark for taught structured facts.

use std::env;
use std::fs::File;
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::Value;
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
    let mut evidence_hits = 0usize;
    let mut total_latency = Duration::ZERO;

    let limit = env::var("AXIOM_TRIVIA_LIMIT").ok().and_then(|value| value.parse().ok());
    for record in records.into_iter().take(limit.unwrap_or(usize::MAX)) {
        let mut engine = AxiomGen::new(2048);
        let evidence_facts = evidence.get(&record.id).cloned().unwrap_or_default();
        let document_facts = evidence_dir.as_deref()
            .map(|directory| extract_document_facts(directory, &record.evidence_files))
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
        total += 1;
    }

    println!("TriviaQA-compatible AXIOM benchmark");
    println!("  records: {}", total);
    println!("  exact_accuracy: {:.2}%", percentage(exact, total));
    println!("  substring_accuracy: {:.2}%", percentage(substring, total));
    println!("  evidence_answer_recall: {:.2}%", percentage(evidence_hits, total));
    println!("  avg_latency: {:?}", if total == 0 { Duration::ZERO } else { total_latency / total as u32 });
    Ok(())
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

fn extract_document_facts(directory: &str, files: &[String]) -> Vec<[String; 3]> {
    let mut facts = Vec::new();
    for filename in files {
        let path = std::path::Path::new(directory).join(filename);
        let Ok(file) = File::open(path) else { continue; };
        let mut text = String::new();
        if file.take(256 * 1024).read_to_string(&mut text).is_err() { continue; }
        let subject = filename.trim_end_matches(".txt").replace('_', " ");
        for sentence in text.split(|character| matches!(character, '.' | '!' | '?')).take(20) {
            if let Some((relation, object)) = extract_relation(sentence) {
                if object.split_whitespace().count() <= 30 && object.len() >= 2 {
                    facts.push([subject.clone(), relation.to_string(), object]);
                }
            }
        }
    }
    facts
}

fn extract_relation(sentence: &str) -> Option<(&'static str, String)> {
    let sentence = sentence.split_whitespace().collect::<Vec<_>>().join(" ");
    let lower = sentence.to_lowercase();
    for (pattern, relation) in [(" is located in ", "located_in"), (" was born in ", "born_in"),
        (" was founded in ", "founded_in"), (" is known for ", "known_for"),
        (" consists of ", "consists_of"), (" contains ", "contains"),
        (" includes ", "includes"), (" causes ", "causes"), (" produces ", "produces"),
        (" supports ", "supports"), (" uses ", "uses"), (" used by ", "used_by"),
        (" developed by ", "developed_by"), (" created by ", "created_by"),
        (" became ", "became"), (" served as ", "served_as"),
        (" has ", "has"), (" was ", "was"), (" are ", "are"), (" is ", "is")] {
        if let Some(position) = lower.find(pattern) {
            let object = sentence[position + pattern.len()..].trim().to_lowercase();
            if !object.is_empty() { return Some((relation, object)); }
        }
    }
    None
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
