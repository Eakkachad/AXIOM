//! vsalm-wiki: Wikipedia → AXIOM knowledge ingestion + conversational QA
//!
//! Pipeline:
//!   1. Fetch Wikipedia page HTML via ureq
//!   2. Clean HTML → plain text
//!   3. Decompose sentences → facts via AxiomGen decompose engine
//!   4. Store facts in VSA-LM KnowledgePrior
//!   5. Ask questions → fact-grounded VSA-LM generation
//!
//! Usage:
//!   vsalm-wiki <wikipedia_url>            — learn one page, then interactive chat
//!   vsalm-wiki <url> <url> ...            — learn multiple pages

use std::io::{self, Write, BufRead, BufReader, Read};
use std::time::Instant;

use tle_axiom_gen::AxiomGen;
use tle_axiom_gen::decompose::decompose_sentence;
use tle_vsa_lm::{LmConfig, VsaLm};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let urls: Vec<String> = std::env::args().skip(1).collect();
    if urls.is_empty() {
        eprintln!("usage: vsalm-wiki <wikipedia_url> [url...]");
        std::process::exit(1);
    }

    let config = LmConfig { dim: 4096, max_order: 4, beam_width: 8, max_gen_tokens: 12, w_knowledge: 3.0, w_engram: 0.0, w_tba: 0.0, knowledge_only: true, ..Default::default() };
    let mut lm = VsaLm::new(config);
    // Answer-first engine: AxiomGen extract_answer finds the entity, VSA-LM
    // is the fluency fallback.  Both share the same ingested facts.
    let mut engine = AxiomGen::new(2048);
    engine.search_config.max_hops = 3;
    engine.search_config.beam_width = 16;
    let mut total_facts = 0usize;

    for url in &urls {
        print!("Fetching {}... ", url);
        io::stdout().flush()?;
        let t0 = Instant::now();
        let html = fetch_html(url)?;
        let (title, text) = extract_wikipedia_text(&html);
        let subject = title.trim_end_matches(" - Wikipedia").to_string().replace(' ', "_");
        println!("{} ({:.1} KB, {:.2?})", title, text.len() / 1024, t0.elapsed());

        // Decompose into facts using the advanced clause-based engine.
        let sentences: Vec<String> = text
            .split(|c: char| matches!(c, '.' | '!' | '?'))
            .map(|s| s.split_whitespace().collect::<Vec<_>>().join(" "))
            .filter(|s| s.split_whitespace().count() >= 4 && s.split_whitespace().count() <= 60)
            .take(200)
            .collect();

        let mut seen = std::collections::HashSet::new();
        let mut facts = 0usize;
        for sentence in &sentences {
            for fact in decompose_sentence(sentence, &subject) {
                let key = (fact.subject.clone(), fact.relation.clone(), fact.object.clone());
                if seen.insert(key.clone()) {
                    lm.knowledge.add_fact(&fact.subject, &fact.relation, &fact.object);
                    engine.add_fact(&fact.subject, &fact.relation, &fact.object);
                    for w in fact.subject.split(|c: char| c == '_' || c == ' ') { if !w.is_empty() { lm.vocab.get_or_add(w); } }
                    for w in fact.relation.split(|c: char| c == '_' || c == ' ') { if !w.is_empty() { lm.vocab.get_or_add(w); } }
                    for w in fact.object.split(|c: char| c == '_' || c == ' ') { if !w.is_empty() { lm.vocab.get_or_add(w); } }
                    facts += 1;
                }
            }
        }
        total_facts += facts;
        println!("  {} facts from {} sentences", facts, sentences.len());
    }

    println!("\nVocabulary: {} words, {} facts in KnowledgePrior", lm.vocab.len(), total_facts);
    print!("Building TBA cache... "); io::stdout().flush()?;
    let t0 = Instant::now(); lm.build_tba_cache();
    println!("done ({:.2?})", t0.elapsed());
    println!("Ready. Ask a question (Ctrl-D to quit):\n");

    let stdin = BufReader::new(io::stdin());
    for line in stdin.lines() {
        let question = line?.trim().to_string();
        if question.is_empty() { continue; }
        let t0 = Instant::now();
        // Answer-first: extract_answer finds the entity directly from the
        // knowledge graph.  Fall back to VSA-LM fluency if no entity found.
        let result = engine.generate(&question);
        let answer = if !result.answer.is_empty() {
            let entity = result.answer;
            if entity.split_whitespace().count() <= 5 && entity.to_lowercase() != question.to_lowercase() {
                entity
            } else {
                lm.generate(&question, Some(8))
            }
        } else {
            lm.generate(&question, Some(8))
        };
        println!("  → {}  [{:.2?}]", answer, t0.elapsed());
    }
    Ok(())
}

fn fetch_html(url: &str) -> Result<String, Box<dyn std::error::Error>> {
    let response = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .get(url)
        .set("User-Agent", "AXIOM-Wiki/1.0 (research)")
        .set("Accept", "text/html")
        .call()?;
    let mut body = String::new();
    let limit = 2 * 1024 * 1024;
    response.into_reader().take(limit as u64).read_to_string(&mut body)?;
    Ok(body)
}

fn extract_wikipedia_text(html: &str) -> (String, String) {
    let title = html.find("<title>").and_then(|i| {
        let rest = &html[i+7..];
        rest.find("</title>").map(|j| rest[..j].to_string())
    }).unwrap_or_default();

    let mut text = String::with_capacity(html.len() / 2);
    let mut skip_level = 0u32;
    let mut in_tag = false;
    let mut tag_buf = String::new();

    for c in html.chars() {
        if c == '<' { in_tag = true; tag_buf.clear(); continue; }
        if in_tag {
            if c == '>' {
                in_tag = false;
                let lower = tag_buf.trim().to_lowercase();
                if lower == "p" || lower == "/p" || lower == "br" || lower == "br/"
                    || lower == "/div" || lower == "/li" || lower == "/h1" || lower == "/h2" || lower == "/h3"
                    || lower.starts_with("br ") { text.push('\n'); }
                if lower.starts_with("script") || lower.starts_with("style") || lower.starts_with("svg")
                    || lower.starts_with("noscript") { skip_level += 1; }
                if lower.starts_with("/script") || lower.starts_with("/style") || lower.starts_with("/svg")
                    || lower.starts_with("/noscript") { skip_level = skip_level.saturating_sub(1); }
                if lower.starts_with("nav") || lower.starts_with("footer") || lower.starts_with("aside")
                    || lower.starts_with("head") || lower.starts_with("form")
                    { skip_level += 1; }
                if lower.starts_with("/nav") || lower.starts_with("/footer") || lower.starts_with("/aside")
                    || lower.starts_with("/head") || lower.starts_with("/form")
                    { skip_level = skip_level.saturating_sub(1); }
                continue;
            }
            tag_buf.push(c);
            continue;
        }
        if skip_level > 0 { continue; }
        text.push(c);
    }

    // Decode HTML entities and normalize.
    let text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        .replace("&quot;", "\"").replace("&#39;", "'").replace("&nbsp;", " ");
    let text: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    (title, text)
}
