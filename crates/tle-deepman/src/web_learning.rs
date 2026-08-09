//! Bounded web learning: fetch a page, extract readable text, and identify facts.

use std::time::Duration;

const MAX_HTML_BYTES: usize = 4 * 1024 * 1024;
const MAX_SENTENCES: usize = 2_000;
const MAX_FACTS: usize = 1_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactCandidate {
    pub subject: String,
    pub relation: String,
    pub object: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExtractedPage {
    pub title: Option<String>,
    pub sentences: Vec<String>,
    pub facts: Vec<FactCandidate>,
}

/// Fetch an HTML page with a bounded response body and timeout.
pub fn fetch_html(url: &str) -> Result<String, String> {
    if !matches!(url.split_once("://").map(|(scheme, _)| scheme), Some("http" | "https")) {
        return Err("URL must use http:// or https://".to_string());
    }

    let response = ureq::get(url)
        .timeout(Duration::from_secs(15))
        .call()
        .map_err(|error| format!("request failed: {}", error))?;
    let reader = response.into_reader();
    let mut body = String::new();
    reader
        .take((MAX_HTML_BYTES + 1) as u64)
        .read_to_string(&mut body)
        .map_err(|error| format!("could not read response: {}", error))?;
    if body.len() > MAX_HTML_BYTES {
        return Err(format!("response exceeds {} MB limit", MAX_HTML_BYTES / (1024 * 1024)));
    }
    Ok(body)
}

/// Extract bounded readable content and simple subject-relation-object facts.
pub fn extract_html(html: &str) -> ExtractedPage {
    let title = extract_title(html);
    let text = strip_markup(html);
    let sentences: Vec<String> = text
        .split(|c| matches!(c, '.' | '!' | '?' | '\n'))
        .map(normalize_text)
        .filter(|sentence| sentence.len() >= 12 && (sentence.split_whitespace().count() >= 3
            || sentence.chars().any(|character| !character.is_ascii())))
        .take(MAX_SENTENCES)
        .collect();

    let mut facts = Vec::new();
    for sentence in &sentences {
        for fact in extract_facts(sentence) {
            if !facts.contains(&fact) {
                facts.push(fact);
            }
            if facts.len() >= MAX_FACTS {
                break;
            }
        }
        if facts.len() >= MAX_FACTS {
            break;
        }
    }

    // Wikipedia-style infoboxes expose useful facts as table labels and values.
    if let Some(page_subject) = title.as_deref().and_then(page_subject) {
        for line in text.lines() {
            if let Some(fact) = extract_labeled_fact(line, &page_subject) {
                if !facts.contains(&fact) {
                    facts.push(fact);
                }
                if facts.len() >= MAX_FACTS {
                    break;
                }
            }
        }
    }

    ExtractedPage { title, sentences, facts }
}

fn extract_title(html: &str) -> Option<String> {
    let lower = html.to_lowercase();
    let start = lower.find("<title")?;
    let content_start = html[start..].find('>')? + start + 1;
    let end = lower[content_start..].find("</title>")? + content_start;
    let title = normalize_text(&strip_markup(&html[content_start..end]));
    (!title.is_empty()).then_some(title)
}

fn strip_markup(html: &str) -> String {
    let mut output = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut tag = String::new();
    let mut skip_depth = 0usize;
    let mut chars = html.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '<' {
            in_tag = true;
            tag.clear();
            continue;
        }
        if in_tag {
            if ch == '>' {
                in_tag = false;
                let name = tag.trim().trim_start_matches('/').split_whitespace().next().unwrap_or("");
                let closing = tag.trim_start().starts_with('/');
                if matches!(name, "script" | "style" | "noscript" | "svg" | "head" | "title" | "nav" | "footer" | "aside" | "form") {
                    if closing { skip_depth = skip_depth.saturating_sub(1); } else { skip_depth += 1; }
                }
                if skip_depth == 0 && closing && name == "th" {
                    output.push_str(": ");
                } else if skip_depth == 0 && closing && matches!(name, "th" | "dt") {
                    output.push_str(": ");
                } else if skip_depth == 0 && (name == "br" || (closing && matches!(name, "p" | "div" | "li" | "h1" | "h2" | "h3" | "section" | "article" | "tr" | "td" | "dd"))) {
                    output.push('\n');
                }
            } else {
                tag.push(ch.to_ascii_lowercase());
            }
            continue;
        }
        if skip_depth == 0 {
            output.push(ch);
        }
    }
    decode_entities(&output)
}

fn decode_entities(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

fn normalize_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ").trim_matches(|c: char| c == ',' || c == ';' || c == ':').to_string()
}

fn extract_fact(sentence: &str) -> Option<FactCandidate> {
    let lower = sentence.to_lowercase();
    let patterns = [
        ("อยู่ใน", "located_in"), ("อยู่ที่", "located_in"),
        ("เกิดใน", "born_in"), ("สามารถ", "can"), ("เป็น", "is"),
        ("คือ", "is"), ("มี", "has"),
        (" was born in ", "born_in"), (" located in ", "located_in"),
        (" lives in ", "lives_in"), (" leads to ", "leads_to"),
        (" causes ", "causes"), (" produces ", "produces"),
        (" contains ", "contains"), (" enables ", "enables"),
        (" creates ", "creates"), (" is ", "is"), (" are ", "are"),
        (" has ", "has"), (" have ", "have"), (" can ", "can"),
        (" was ", "was"), (" were ", "were"), (" became ", "became"),
        (" includes ", "includes"), (" included ", "included"),
        (" uses ", "uses"), (" used ", "used"), (" supports ", "supports"),
        (" consists of ", "consists_of"), (" based on ", "based_on"),
        (" developed by ", "developed_by"), (" created by ", "created_by"),
        (" designed by ", "designed_by"), (" written in ", "written_in"),
        (" released in ", "released_in"), (" introduced in ", "introduced_in"),
        (" known as ", "known_as"), (" called ", "called"),
        (" refers to ", "refers_to"), (" provides ", "provides"),
        (" allows ", "allows"), (" requires ", "requires"),
        (" features ", "features"), (" follows ", "follows"),
        (" originated in ", "originated_in"), (" originated from ", "originated_from"),
        (" part of ", "part_of"), (" occurs in ", "occurs_in"),
        (" found in ", "found_in"), (" built on ", "built_on"),
        (" compatible with ", "compatible_with"), (" available on ", "available_on"),
        (" runs on ", "runs_on"), (" written by ", "written_by"),
        (" maintained by ", "maintained_by"), (" governed by ", "governed_by"),
        (" influenced by ", "influenced_by"), (" derived from ", "derived_from"),
        (" adopted by ", "adopted_by"), (" known for ", "known_for"),
        (" serves as ", "serves_as"), (" acts as ", "acts_as"),
        (" functions as ", "functions_as"), (" associated with ", "associated_with"),
        (" linked to ", "linked_to"), (" related to ", "related_to"),
        (" documented in ", "documented_in"), (" published in ", "published_in"),
        (" introduced by ", "introduced_by"), (" named ", "named"),
        (" evolved from ", "evolved_from"), (" superseded by ", "superseded_by"),
        (" replaced by ", "replaced_by"), (" used for ", "used_for"),
        (" developed in ", "developed_in"), (" implemented in ", "implemented_in"),
        (" implemented by ", "implemented_by"), (" distributed under ", "distributed_under"),
        (" licensed under ", "licensed_under"), (" written using ", "written_using"),
        (" used by ", "used_by"), (" supported by ", "supported_by"),
    ];
    let (_, pattern, relation) = patterns.iter().filter_map(|(pattern, relation)| {
        lower.find(pattern).map(|position| (position, *pattern, *relation))
    }).min_by_key(|(position, _, _)| *position)?;
    let position = lower.find(pattern)?;
    let mut subject = normalize_subject(&sentence[..position]);
    if subject.split_whitespace().count() > 8 {
        if let Some((_, tail)) = subject.rsplit_once(',') {
            subject = normalize_subject(tail);
        }
    }
    let object = normalize_text(&sentence[position + pattern.len()..]);
    if subject.split_whitespace().count() > 8 || object.split_whitespace().count() > 40 || subject.len() < 2 || object.len() < 2 {
        return None;
    }
    Some(FactCandidate { subject, relation: relation.to_string(), object })
}

fn extract_facts(sentence: &str) -> Vec<FactCandidate> {
    let mut facts = Vec::new();
    let clauses: Vec<&str> = sentence.split(" and ").collect();
    // Prefer the complete sentence so introductory clauses containing "and"
    // do not detach the real subject before relation extraction.
    let first = extract_fact(sentence);
    if let Some(fact) = first.clone() {
        facts.push(fact.clone());
        for clause in clauses.iter().skip(1) {
            let candidate = extract_fact(clause).or_else(|| {
                let joined = format!("{} {}", fact.subject, clause);
                extract_fact(&joined)
            });
            if let Some(candidate) = candidate {
                if !facts.contains(&candidate) {
                    facts.push(candidate);
                }
            }
        }
    } else {
        for clause in clauses.iter().skip(1) {
            if let Some(fact) = extract_fact(clause) {
                facts.push(fact);
            }
        }
    }
    facts
}

fn normalize_subject(text: &str) -> String {
    let subject = normalize_text(text);
    subject
        .strip_prefix("the ")
        .or_else(|| subject.strip_prefix("The "))
        .or_else(|| subject.strip_prefix("a "))
        .or_else(|| subject.strip_prefix("A "))
        .or_else(|| subject.strip_prefix("an "))
        .or_else(|| subject.strip_prefix("An "))
        .unwrap_or(&subject)
        .to_string()
}

fn page_subject(title: &str) -> Option<String> {
    let subject = title
        .strip_suffix(" - Wikipedia")
        .or_else(|| title.strip_suffix(" – Wikipedia"))
        .unwrap_or(title);
    let subject = normalize_subject(subject);
    (subject.len() >= 2).then_some(subject)
}

fn extract_labeled_fact(line: &str, subject: &str) -> Option<FactCandidate> {
    let (label, object) = line.split_once(':')?;
    let label = normalize_text(label);
    let object = normalize_text(object);
    if label.is_empty() || object.len() < 2 || label.len() > 40 || object.len() > 120 {
        return None;
    }
    if label.split_whitespace().count() > 5 || label.chars().any(|c| !c.is_alphanumeric() && c != ' ' && c != '-') {
        return None;
    }
    Some(FactCandidate {
        subject: subject.to_string(),
        relation: label.to_lowercase().replace([' ', '-'], "_"),
        object,
    })
}

use std::io::Read;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_readable_text_and_facts() {
        let page = extract_html(r#"
            <html><head><title>Sky</title><script>ignore me</script></head>
            <body><nav>Menu</nav><article><h1>The Sky</h1>
            <p>The sky is blue.</p><p>Blue light has a short wavelength.</p>
            <table><tr><th>Color</th><td>Blue</td></tr></table>
            </article></body></html>
        "#);
        assert_eq!(page.title.as_deref(), Some("Sky"));
        assert!(page.sentences.iter().any(|s| s == "The sky is blue"));
        assert!(page.facts.iter().any(|f| f.subject == "sky" && f.relation == "is" && f.object == "blue"));
        assert!(page.facts.iter().any(|f| f.relation == "color" && f.object == "Blue"));
        assert!(!page.sentences.iter().any(|s| s.contains("ignore me")));
    }

    #[test]
    fn rejects_invalid_urls() {
        assert!(fetch_html("ftp://example.com").is_err());
        assert!(fetch_html("example.com").is_err());
    }

    #[test]
    fn fetches_from_a_local_http_server() {
        use std::io::Write;
        use std::net::TcpListener;
        use std::thread;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0u8; 512];
            let _ = std::io::Read::read(&mut stream, &mut request);
            let response = "HTTP/1.1 200 OK\r\nContent-Length: 57\r\nConnection: close\r\n\r\n<title>Local</title><p>Rust is fast systems software.</p>";
            stream.write_all(response.as_bytes()).unwrap();
        });

        let html = fetch_html(&format!("http://{}", address)).unwrap();
        server.join().unwrap();
        let page = extract_html(&html);
        assert_eq!(page.title.as_deref(), Some("Local"));
        assert_eq!(page.facts[0].subject, "Rust");
        assert_eq!(page.facts[0].object, "fast systems software");
    }

    #[test]
    fn extracted_facts_are_usable_by_incremental_store() {
        let page = extract_html("<p>Rust is fast systems software.</p>");
        let mut store = tle_afc::IncrementalStore::new();
        for fact in page.facts {
            store.learn_fact(&fact.subject, &fact.relation, &fact.object);
        }
        assert_eq!(store.get_facts("rust"), vec![("is", "fast systems software")]);
    }

    #[test]
    fn extracts_400_facts_under_five_seconds() {
        let mut html = String::from("<html><body>");
        for index in 0..400 {
            html.push_str(&format!("<p>Entity {} is property {}.</p>", index, index));
        }
        html.push_str("</body></html>");

        let start = std::time::Instant::now();
        let page = extract_html(&html);
        let elapsed = start.elapsed();

        assert_eq!(page.sentences.len(), 400);
        assert_eq!(page.facts.len(), 400);
        assert!(elapsed.as_secs() < 5, "extraction took {:?}", elapsed);
    }

    #[test]
    fn trims_long_introductory_subject_clauses() {
        let page = extract_html("<p>In the long and documented history of modern systems programming, Rust is memory safe.</p>");
        assert!(page.facts.iter().any(|fact| {
            fact.subject == "Rust" && fact.relation == "is" && fact.object == "memory safe"
        }), "facts: {:?}", page.facts);
    }

    #[test]
    fn extracts_thai_facts_without_whitespace_tokenization() {
        let page = extract_html("<p>ท้องฟ้าเป็นสีฟ้า</p><p>แมวมีสี่ขา</p>");
        assert!(page.facts.iter().any(|fact| fact.relation == "is" && fact.subject == "ท้องฟ้า"));
        assert!(page.facts.iter().any(|fact| fact.relation == "has" && fact.subject == "แมว"));
    }
}
