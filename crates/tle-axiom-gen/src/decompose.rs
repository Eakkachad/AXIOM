//! Clause-based fact decomposition — extract structured triples from natural
//! language evidence without a trained parser.
//!
//! The insight: an evidence sentence like
//! "Martina Hingis is a Swiss player who won the Australian Open in 1997"
//! is compositionally richer than a single `(page, describes, sentence)` fact.
//! We split it into clauses on grammatical boundaries (commas, relative
//! pronouns, coordinators) and anchor each clause on a relational predicate
//! from a closed class. Clauses that omit an explicit subject inherit the
//! subject of the preceding clause, producing a connected chain of triples.

/// A single extracted fact triple.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecomposedFact {
    pub subject: String,
    pub relation: String,
    pub object: String,
}

/// Relational predicates, longest phrase first so `was born in` is matched
/// before the bare `was`.
const RELATIONAL_PHRASES: &[(&str, &str)] = &[
    ("was born in", "born_in"),
    ("was born on", "born_on"),
    ("is located in", "located_in"),
    ("was located in", "located_in"),
    ("is located at", "located_at"),
    ("was founded in", "founded_in"),
    ("was established in", "founded_in"),
    ("was created in", "created_in"),
    ("was created by", "created_by"),
    ("was developed by", "developed_by"),
    ("was developed in", "developed_in"),
    ("was written by", "written_by"),
    ("was designed by", "designed_by"),
    ("was released in", "released_in"),
    ("was released on", "released_on"),
    ("was published in", "published_in"),
    ("is known for", "known_for"),
    ("was known for", "known_for"),
    ("is known as", "known_as"),
    ("was known as", "known_as"),
    ("is considered", "considered"),
    ("was considered", "considered"),
    ("is famous for", "known_for"),
    ("was famous for", "known_for"),
    ("served as", "served_as"),
    ("served in", "served_in"),
    ("worked at", "worked_at"),
    ("worked as", "worked_as"),
    ("lived in", "lived_in"),
    ("died in", "died_in"),
    ("took place in", "took_place_in"),
    ("occurred in", "occurred_in"),
    ("is part of", "part_of"),
    ("is home to", "home_to"),
    ("is the capital of", "capital_of"),
    ("was the capital of", "capital_of"),
    ("is the mother of", "has_mother"),
    ("was the mother of", "has_mother"),
    ("is the father of", "has_father"),
    ("was the father of", "has_father"),
    ("is the parent of", "has_parent"),
    ("was the parent of", "has_parent"),
    ("is the daughter of", "daughter_of"),
    ("was the daughter of", "daughter_of"),
    ("is the son of", "son_of"),
    ("was the son of", "son_of"),
    ("is the wife of", "wife_of"),
    ("was the wife of", "wife_of"),
    ("is the husband of", "husband_of"),
    ("was the husband of", "husband_of"),
    ("is the sister of", "sister_of"),
    ("was the sister of", "sister_of"),
    ("is the brother of", "brother_of"),
    ("was the brother of", "brother_of"),
    ("is the founder of", "founder_of"),
    ("was the founder of", "founder_of"),
    ("is the leader of", "leader_of"),
    ("was the leader of", "leader_of"),
    ("is the president of", "president_of"),
    ("was the president of", "president_of"),
    ("is the author of", "author_of"),
    ("was the author of", "author_of"),
    ("is the director of", "director_of"),
    ("was the director of", "director_of"),
    ("is the largest", "is_largest"),
    ("is the smallest", "is_smallest"),
    ("is one of", "is_one_of"),
    ("is the only", "is_only"),
    ("is referred to as", "referred_to_as"),
    ("is also known as", "known_as"),
    ("also known as", "known_as"),
    ("refers to", "refers_to"),
    ("helps control", "helps_control"),
    ("helps regulate", "helps_regulate"),
    ("helps to", "helps_to"),
    ("controls the", "controls"),
    ("regulates the", "regulates"),
    ("regulates", "regulates"),
    ("controls", "controls"),
    ("triggers the", "triggers"),
    ("triggers", "triggers"),
    ("stimulates the", "stimulates"),
    ("stimulates", "stimulates"),
    ("governs the", "governs"),
    ("governs", "governs"),
    ("secretes", "secretes"),
    ("produces the", "produces"),
    ("produces", "produces"),
    ("releases", "releases"),
    ("uses the", "uses"),
    ("uses", "uses"),
    ("made of", "made_of"),
    ("made from", "made_from"),
    ("comprised of", "consists_of"),
    ("comprises", "consists_of"),
    ("features", "features"),
    ("starred", "starred"),
    ("directed", "directed"),
    ("painted", "painted"),
    ("discovered", "discovered"),
    ("invented", "invented"),
    ("founded", "founded"),
    ("established", "founded"),
    ("designed", "designed"),
    ("built", "built"),
    ("wrote", "wrote"),
    ("composed", "composed"),
    ("released", "released"),
    ("published", "published"),
    ("won", "won"),
    ("played for", "played_for"),
    ("played", "played"),
    ("became", "became"),
    ("contains", "contains"),
    ("includes", "includes"),
    ("produced", "produced"),
    ("created", "created"),
    ("developed", "developed"),
    ("worked", "worked"),
    ("studied", "studied"),
    ("taught", "taught"),
    ("led", "led"),
    ("ruled", "ruled"),
    ("owned", "owned"),
    ("has", "has"),
    ("have", "have"),
    ("had", "had"),
    ("is", "is"),
    ("are", "are"),
    ("was", "was"),
    ("were", "were"),
];

/// Common words that should not become fact subjects when a clause has no
/// explicit subject of its own.
fn is_discardable(word: &str) -> bool {
    matches!(
        word,
        "the" | "a" | "an" | "this" | "that" | "these" | "those" | "it" | "its"
            | "which" | "who" | "whose" | "where" | "and" | "but" | "or" | "then"
            | "however" | "also" | "later" | "he" | "she" | "they" | "his" | "her"
            | "their" | "him" | "herself" | "himself" | "we" | "you" | "them" | "us"
            | "there" | "here" | "one" | "two" | "some" | "many" | "more" | "most"
            | "after" | "before" | "since" | "until" | "while" | "although" | "because"
            | "according" | "together" | "addition" | "though" | "accordingly" | "such"
            | "eventually" | "finally" | "meanwhile" | "besides" | "elsewhere" | "including"
            | "through"
    )
}

/// A subject that begins with a date/number token (e.g. "2013, Hingis") is
/// not a valid entity — the year is a temporal marker, not part of the subject.
fn subject_is_date_prefix(subject: &str) -> bool {
    subject
        .split_whitespace()
        .next()
        .map(|w| w.chars().all(|c| c.is_ascii_digit()))
        .unwrap_or(false)
}

fn normalize(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches(|c: char| c == ',' || c == '.' || c == ';' || c == ':' || c == '"' || c == '\'' || c == '(' || c == ')')
        .to_string()
}

/// Strip leading articles for a cleaner subject key.
fn canonical_subject(text: &str) -> String {
    let mut words = text.split_whitespace().peekable();
    let mut out = Vec::new();
    for word in words.by_ref() {
        if is_discardable(&word.to_lowercase()) && out.is_empty() {
            continue;
        }
        out.push(word);
    }
    out.join(" ")
}

/// Find a whole-word occurrence of `phrase` in `lower`, returning the byte
/// index such that the phrase is bounded by non-alphanumeric characters on
/// both sides. This prevents predicates like "is" or "has" from matching
/// inside words such as "Hingis" or "Swiss".
fn find_word_boundary(lower: &str, phrase: &str) -> Option<usize> {
    if phrase.is_empty() || phrase.len() > lower.len() { return None; }
    let mut search_from = 0;
    let mut iterations = 0usize;
    let max_iter = lower.len() + 1;
    while let Some(position) = lower[search_from..].find(phrase) {
        iterations += 1;
        if iterations > max_iter { return None; }
        let absolute = search_from + position;
        let before = absolute.checked_sub(1).and_then(|i| lower.as_bytes().get(i)).copied();
        let after = lower.as_bytes().get(absolute + phrase.len()).copied();
        let before_ok = before.map(|b| !b.is_ascii_alphanumeric() && b != b'-').unwrap_or(true);
        let after_ok = after.map(|b| !b.is_ascii_alphanumeric() && b != b'-').unwrap_or(true);
        if before_ok && after_ok {
            return Some(absolute);
        }
        search_from = absolute + phrase.len();
        if search_from > lower.len() { return None; }
    }
    None
}

/// Find the relational predicate in a clause.
///
/// Returns `(relation, object)` anchored at the first matching phrase. The
/// subject is everything before the phrase.
fn find_predicate(clause: &str) -> Option<(usize, &'static str, String)> {
    let lower = clause.to_lowercase();
    for (phrase, relation) in RELATIONAL_PHRASES {
        if let Some(position) = find_word_boundary(&lower, phrase) {
            let object = normalize(&clause[position + phrase.len()..]);
            if !object.is_empty() {
                return Some((position, *relation, object));
            }
        }
    }
    None
}

/// Split a sentence into clauses on grammatical boundaries.
///
/// Commas and relative pronouns are always boundaries. Coordinators (`and`,
/// `but`, `;`) are boundaries only when the right-hand segment contains a
/// relational predicate, to avoid splitting enumerated lists.
fn split_clauses(sentence: &str) -> Vec<String> {
    let mut clauses = Vec::new();
    let mut current = String::new();
    let mut parts: Vec<String> = Vec::new();

    for token in sentence.split_whitespace() {
        let lower = token.to_lowercase();
        let is_comma = token.ends_with(',');
        let is_boundary = is_comma
            || lower == "and" || lower == "but" || lower == ";" || token.ends_with(';');
        // A comma-boundary token belongs to the CURRENT (left) clause so the
        // word before it stays attached: "the capital of France," keeps
        // "France" with the left clause. So commas do NOT trigger the generic
        // pre-push; they are appended to current and then the left clause is
        // committed.
        if is_boundary && !is_comma && !current.is_empty() {
            parts.push(current.trim().to_string());
            current.clear();
        }
        current.push(' ');
        current.push_str(token);
        if is_comma {
            parts.push(current.trim().to_string());
            current.clear();
        }
        if token.ends_with('.') {
            parts.push(current.trim().to_string());
            current.clear();
        }
    }
    if !current.trim().is_empty() {
        parts.push(current.trim().to_string());
    }

    for part in parts {
        let mut piece = part;
        for relative in ["which", "who", "that", "whose", "where"] {
            let marker = format!(" {} ", relative);
            if let Some(position) = piece.to_lowercase().find(&marker) {
                let (left, right) = piece.split_at(position);
                clauses.push(left.trim().to_string());
                piece = right.trim_start().to_string();
            }
        }
        // If the remaining piece has a predicate, keep it; else drop it.
        if find_predicate(&piece).is_some() {
            clauses.push(piece);
        }
    }
    clauses.retain(|clause| clause.len() >= 3);
    clauses
}

/// A ranked answer candidate extracted from a generated path.
#[derive(Debug, Clone, PartialEq)]
pub struct AnswerCandidate {
    pub text: String,
    pub score: f32,
}

fn is_question_stop_word(word: &str) -> bool {
    matches!(word, "what" | "which" | "where" | "when" | "how" | "does" | "have"
        | "who" | "with" | "from" | "that" | "this" | "whose" | "why" | "the" | "was" | "did")
}

/// Rank answer candidates for a query from a set of candidate strings.
///
/// The score combines exact question-word overlap (tokens length >= 4) with a
/// lexical bonus for the last content word of the question, which is often the
/// answer anchor (e.g. "Who won X" → X). No answer oracle is used: this only
/// reads the question and the candidate surface forms.
pub fn rank_answer_candidates(query: &str, candidates: &[String]) -> Vec<AnswerCandidate> {
    let lower_query = query.to_lowercase();
    let query_words: Vec<String> = lower_query
        .split(|c: char| !c.is_alphanumeric() && c != ' ' && c != '-')
        .filter(|w| w.len() >= 4)
        .filter(|w| !is_question_stop_word(w))
        .map(|w| w.to_string())
        .collect();

    let mut ranked: Vec<AnswerCandidate> = candidates
        .iter()
        .map(|candidate| {
            let lower = candidate.to_lowercase();
            let overlap = query_words.iter().filter(|word| lower.contains(word.as_str())).count();
            let anchor = query_words.last().map(String::as_str).unwrap_or("");
            let anchor_bonus = if !anchor.is_empty() && lower.contains(anchor) { 2.0 } else { 0.0 };
            AnswerCandidate {
                text: candidate.clone(),
                score: overlap as f32 + anchor_bonus,
            }
        })
        .collect();

    ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

/// Truncate a long object at clause boundaries so we keep the short, entity-like
/// head of the object rather than the whole trailing clause.
///
/// "the Eiffel Tower, which was built in 1889" → "the Eiffel Tower"
/// "a Swiss professional tennis player who won ..." → "a Swiss professional tennis player"
///
/// We ONLY cut at strong clause boundaries (commas, relative pronouns,
/// coordinators) — NOT at prepositions like " in ", " of ", " with " which are
/// part of real entity names ("United States of America", "Mount McKinley").
fn truncate_object(object: &str) -> String {
    for marker in [", ", " who ", " which ", " that ", " where ", " and ", " but ", " such ", " including "] {
        if let Some(position) = object.find(marker) {
            let head = &object[..position];
            let trimmed = head.trim();
            if !trimmed.is_empty() {
                return normalize(trimmed);
            }
        }
    }
    normalize(object)
}

/// Is the subject a plausible entity (not a long lowercase descriptive phrase)?
///
/// A valid subject is either capitalized (proper noun) or short (≤3 words).
/// Lowercase phrases longer than 3 words (e.g. "a change of heart two months
/// later") are not entities and are dropped.
fn is_entity_like(subject: &str) -> bool {
    let words: Vec<&str> = subject.split_whitespace().collect();
    if words.is_empty() {
        return false;
    }
    // Leading article is fine ("the sky", "a cat").
    let content: Vec<&str> = words
        .iter()
        .filter(|w| !matches!(w.to_lowercase().as_str(), "the" | "a" | "an"))
        .copied()
        .collect();
    if content.is_empty() {
        return false;
    }
    // Short subject → keep regardless of capitalization (handles lowercase facts).
    if content.len() <= 3 {
        return true;
    }
    // Long subject → must contain a capitalized token (proper noun).
    content.iter().any(|w| w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
}

/// Decompose a sentence into connected fact triples.
///
/// `fallback_subject` is used only when no clause in the sentence yields an
/// explicit subject (e.g. a bare continuation).
pub fn decompose_sentence(sentence: &str, fallback_subject: &str) -> Vec<DecomposedFact> {
    let mut facts = Vec::new();
    let clauses = split_clauses(sentence);
    let mut inherited_subject: Option<String> = None;

    for clause in clauses {
        let Some((position, relation, object)) = find_predicate(&clause) else {
            continue;
        };
        let raw_object = object.clone();
        let mut subject = normalize(&clause[..position]);
        let subject_canonical = canonical_subject(&subject);
        // Whether the subject came from the clause (derived) or was inherited /
        // the fallback page title (trusted — always a valid entity).
        let mut trusted_subject = false;

        if subject_canonical.is_empty() {
            // No explicit subject — inherit from the previous clause.
            match inherited_subject.as_ref() {
                Some(previous) => {
                    subject = previous.clone();
                    trusted_subject = true;
                }
                None => {
                    subject = fallback_subject.to_string();
                    trusted_subject = true;
                }
            }
        } else {
            inherited_subject = Some(subject_canonical.clone());
            subject = subject_canonical;
        }

        // Reject garbage derived subjects: date-prefixed ("2013, Hingis"), or
        // long lowercase descriptive phrases that are not real entities.
        // Trusted (inherited / page-title) subjects always pass.
        if subject.is_empty()
            || subject.len() > 60
            || (!trusted_subject && (subject_is_date_prefix(&subject) || !is_entity_like(&subject)))
        {
            continue;
        }

        // Truncate the object to its entity-like head span.
        let object = truncate_object(&object);
        if object.is_empty() {
            continue;
        }

        // Extract a year reference from the FULL pre-truncation object and
        // link it as a temporal anchor: (subject, happened_in, year).
        for word in raw_object.split_whitespace() {
            let cleaned = word.trim_matches(|c: char| !c.is_ascii_digit());
            if cleaned.len() == 4 && cleaned.starts_with('1') || cleaned.len() == 4 && cleaned.starts_with('2') {
                facts.push(DecomposedFact {
                    subject: subject.clone(),
                    relation: "happened_in".to_string(),
                    object: cleaned.to_string(),
                });
                break;
            }
        }

        facts.push(DecomposedFact {
            subject: subject.clone(),
            relation: relation.to_string(),
            object: object.clone(),
        });

        // Surface embedded proper-noun phrases from the object as standalone
        // entities. E.g. "the inventor of the lightweight baby buggy" yields
        // the additional fact (subject, is_related_to, Baby Buggy). TriviaQA
        // answers are often capitalized phrases buried inside long objects.
        // NOTE: extract_proper_nouns is expensive on long objects (O(n·m) with
        // many clauses); applied only when the object is short enough.
        if object.split_whitespace().count() <= 20 {
            for phrase in extract_proper_nouns(&object) {
                if !facts.iter().any(|f| f.subject == subject && f.object == phrase) {
                    facts.push(DecomposedFact {
                        subject: subject.clone(),
                        relation: "is_related_to".to_string(),
                        object: phrase,
                    });
                }
            }
        }
    }

    facts
}

/// Extract capitalized multi-word proper-noun phrases from a string.
///
/// Looks for runs of tokens where at least the first is capitalized and not at
/// sentence start (heuristically: the token before is lowercase or a
/// preposition). Returns the longest capitalized phrase per window.
fn extract_proper_nouns(text: &str) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let w = words[i];
        let starts_cap = w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
        // A proper-noun phrase: capitalized word possibly followed by more
        // capitalized/lowercase words until a clause marker or preposition.
        if starts_cap {
            let mut j = i;
            // Stop the phrase at relative pronouns, prepositions, and
            // coordinators so "Baby Buggy with a collapsible ..." keeps only
            // "Baby Buggy".
            while j < words.len()
                && !matches!(
                    words[j].to_lowercase().as_str(),
                    "who" | "which" | "that" | "where" | "with" | "in" | "on" | "at"
                        | "of" | "to" | "from" | "and" | "but" | "for" | "the" | "a"
                        | "an" | "his" | "her" | "its" | "was" | "is" | "are" | "were"
                )
            {
                j += 1;
            }
            let phrase_words = &words[i..j];
            if phrase_words.len() >= 2 && phrase_words.len() <= 5 {
                let phrase = phrase_words.join(" ");
                let first = phrase_words[0].to_lowercase();
                if !matches!(first.as_str(), "a" | "an" | "the" | "his" | "her" | "its") {
                    out.push(phrase);
                }
            }
            // Advance past the phrase.  If j == i (the very first word was a
            // stop word), consume it and move on — otherwise the loop would
            // hang forever because i never advances.
            i = if j == i { i + 1 } else { j };
        } else {
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decomposes_relative_clause_into_chain() {
        let facts = decompose_sentence(
            "Martina Hingis is a Swiss tennis player who won the Australian Open in 1997.",
            "Martina Hingis",
        );
        assert!(facts.iter().any(|f| f.subject == "Martina Hingis" && f.relation == "is"));
        assert!(facts.iter().any(|f| f.subject == "Martina Hingis" && f.relation == "won"));
        assert!(facts.iter().any(|f| f.subject == "Martina Hingis" && f.relation == "happened_in" && f.object == "1997"));
    }

    #[test]
    fn inherits_subject_across_clauses() {
        let facts = decompose_sentence(
            "Paris is the capital of France and is known for the Eiffel Tower.",
            "Paris",
        );
        // The more specific "is the capital of" phrase is preferred over bare "is".
        assert!(facts.iter().any(|f| f.subject == "Paris" && f.relation == "capital_of"));
        assert!(facts.iter().any(|f| f.subject == "Paris" && f.relation == "known_for"));
    }

    #[test]
    fn born_in_phrase_matches_longest_first() {
        let facts = decompose_sentence(
            "Albert Einstein was born in Ulm in 1879.",
            "Albert Einstein",
        );
        assert!(facts.iter().any(|f| f.subject == "Albert Einstein" && f.relation == "born_in"));
        assert!(facts.iter().any(|f| f.subject == "Albert Einstein" && f.relation == "happened_in" && f.object == "1879"));
    }

    #[test]
    fn ignores_discardable_pseudo_subjects() {
        let facts = decompose_sentence("It was founded in 1901 in Chicago.", "Company X");
        assert!(facts.iter().any(|f| f.subject == "Company X" && f.relation == "founded_in"));
    }

    #[test]
    fn rejects_long_lowercase_descriptive_subjects() {
        // "a change of heart two months later" is not an entity — the clause
        // should be dropped (no inherited subject survives the filter).
        let facts = decompose_sentence(
            "Hingis failed in 1997; a change of heart two months later just before the French Open.",
            "Hingis",
        );
        assert!(!facts.iter().any(|f| f.subject.contains("change of heart")));
    }

    #[test]
    fn rejects_date_prefixed_subjects() {
        let facts = decompose_sentence(
            "2013, Hingis was elected into the International Tennis Hall of Fame.",
            "Hingis",
        );
        // The subject must be Hingis (inherited), not "2013, Hingis".
        assert!(!facts.iter().any(|f| f.subject.starts_with('2')));
    }

    #[test]
    fn surfaces_embedded_proper_nouns() {
        let facts = decompose_sentence(
            "He was the inventor of the lightweight Baby Buggy with a collapsible support assembly.",
            "Maclaren",
        );
        assert!(
            facts.iter().any(|f| f.object == "Baby Buggy"),
            "expected embedded proper noun 'Baby Buggy' surfaced, got {:?}",
            facts.iter().map(|f| f.object.clone()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn truncates_object_at_relative_clause() {
        let facts = decompose_sentence(
            "Paris is the capital of France, which is in Europe.",
            "Paris",
        );
        // The capital_of object must be the clean head "France", not the whole
        // trailing clause.
        assert!(facts.iter().any(|f| f.relation == "capital_of" && f.object == "France"));
        // The relative clause "which is in Europe" becomes its own fact, but
        // the capital_of object must NOT contain "Europe".
        assert!(!facts.iter().any(|f| f.relation == "capital_of" && f.object.contains("Europe")));
    }
}
