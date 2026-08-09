//! TriviaQA-compatible JSONL benchmark for taught structured facts.

use std::env;
use std::fs::File;
use std::fs;
use std::io::{BufRead, BufReader};
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
    let records = load_records(&path)?;
    let evidence = evidence_path.as_deref().map(load_evidence).transpose()?.unwrap_or_default();
    let mut engine = AxiomGen::new(2048);
    let mut total = 0usize;
    let mut exact = 0usize;
    let mut substring = 0usize;
    let mut total_latency = Duration::ZERO;

    for record in records {
        let evidence_facts = evidence.get(&record.id).cloned().unwrap_or_default();
        for fact in record.facts.iter().chain(evidence_facts.iter()) {
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
                records.push(Record { id, question, answers, facts: Vec::new() });
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
