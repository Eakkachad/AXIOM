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

/// Quality gate for decomposed facts: reject triples whose entities are
/// visibly junk (long lowercase phrases, bare copulas with weak subjects).
/// This filter runs AT INGESTION — junk never enters the graph, so
/// reasoning methods get a clean signal.
pub fn is_fact_worthy(fact: &DecomposedFact) -> bool {
    let subj_words = fact.subject.split_whitespace().count();
    let obj_words = fact.object.split_whitespace().count();
    if subj_words > 8 || obj_words > 8 { return false; }
    if subj_words == 0 || obj_words == 0 { return false; }
    // Reject subjects that are only bare verbs used as subjects.
    let subj_lower = fact.subject.to_lowercase();
    if subj_words <= 2
        && (subj_lower == "has" || subj_lower == "had"
            || subj_lower == "been" || subj_lower == "being")
    {
        return false;
    }
    let rel = fact.relation.as_str();
    // Copula (SVC) clauses: "X is/was Y".  If Y starts with an article
    // and has no mid-sentence capitalised word, it is a descriptive
    // complement ("a tennis player"), not an answer entity — reject it.
    // Shorter complements may still hold genuine entities ("the sky",
    // "the capital of France") that are useful in the graph.
    if matches!(rel, "is" | "was" | "are" | "were") {
        let obj_first = fact.object.split_whitespace().next().unwrap_or("");
        let starts_article = matches!(obj_first.to_lowercase().as_str(), "a" | "an" | "the");
        if starts_article && !has_noninitial_capital(&fact.object) && obj_words >= 3 {
            return false;
        }
    }
    // "mentions" / "is_related_to" link page subjects to proper nouns.
    // Only admit when the object is a genuine proper noun: 2+ words,
    // every non-article word capitalised.  Filters "Swiss tennis" (lowercase
    // "tennis") while keeping "Baby Buggy", "Martina Hingis".
    if matches!(rel, "mentions" | "is_related_to") {
        let has_cap = fact.object.split_whitespace()
            .filter(|w| !matches!(w.to_lowercase().as_str(), "a" | "an" | "the"))
            .any(|w| w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false));
        if has_cap { return true; }
        // Lowercase fallback: 3+ word all-lowercase noun phrases from the
        // tail pass (specific answers like "collapsible support assembly").
        // 1-2 word lowercase phrases are too noisy to admit.
        if obj_words >= 3 {
            let all_lower = fact.object.split_whitespace()
                .all(|w| w.chars().all(|c| !c.is_uppercase()));
            if all_lower { return true; }
        }
        return false;
    }
    true
}

fn has_noninitial_capital(text: &str) -> bool {
    text.split_whitespace()
        .enumerate()
        .any(|(i, w)| i > 1 && w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
}

/// Relational predicates, longest phrase first so `was born in` is matched
/// before the bare `was`.
const RELATIONAL_PHRASES: &[(&str, &str)] = &[
    ("was born in", "born_in"),
    ("was born on", "born_on"),
    ("born on", "born_on"),
    ("was born as", "born_as"),
    ("born as", "born_as"),
    ("was born to", "child_of"),
    ("born to", "child_of"),
    ("was married to", "married_to"),
    ("married to", "married_to"),
    ("was named after", "named_after"),
    ("named after", "named_after"),
    ("was named for", "named_after"),
    ("named for", "named_after"),
    ("was renamed to", "renamed_to"),
    ("renamed to", "renamed_to"),
    ("was voiced by", "played_by"),
    ("is voiced by", "played_by"),
    ("voiced by", "played_by"),
    ("was derived from", "derived_from"),
    ("is derived from", "derived_from"),
    ("derived from", "derived_from"),
    ("graduated from", "graduated_from"),
    ("studied at", "studied_at"),
    ("is located in", "located_in"),
    ("was located in", "located_in"),
    ("is located at", "located_at"),
    ("is a village in", "located_in"),
    ("is a town in", "located_in"),
    ("is a city in", "located_in"),
    ("is a municipality in", "located_in"),
    ("is a commune in", "located_in"),
    ("is a district in", "located_in"),
    ("is a province in", "located_in"),
    ("is a county in", "located_in"),
    ("is a country in", "located_in"),
    ("is a state in", "located_in"),
    ("is an island in", "located_in"),
    ("is a region in", "located_in"),
    ("is a village near", "located_near"),
    ("is a town near", "located_near"),
    ("is a city near", "located_near"),
    ("is from", "from"),
    ("was from", "from"),
    ("was founded in", "founded_in"),
    ("was established in", "founded_in"),
    ("was created in", "created_in"),
    ("was created by", "created_by"),
    ("was developed by", "developed_by"),
    ("was developed in", "developed_in"),
    ("was written by", "written_by"),
    ("was designed by", "designed_by"),
    // T1.10d: passive creator/author/director patterns. Without these, "Swan
    // Lake ... composed by Pyotr Ilyich Tchaikovsky" decomposes to a bare
    // "composed" with the WRONG subject ("is a ballet"), and the composer only
    // enters the graph as weak is_related_to (0.8) instead of strong created_by
    // (2.0) → gold sinks to deep rank. Same for wrote/directed/played/portrayed.
    ("was composed by", "created_by"),
    ("is composed by", "created_by"),
    ("was written by", "written_by"),
    ("was directed by", "directed_by"),
    ("is directed by", "directed_by"),
    ("was played by", "played_by"),
    ("is played by", "played_by"),
    ("was portrayed by", "portrayed_by"),
    ("is portrayed by", "portrayed_by"),
    ("was created by", "created_by"),
    ("was founded by", "founded_by"),
    ("was invented by", "invented_by"),
    ("was discovered by", "discovered_by"),
    ("was painted by", "painted_by"),
    ("was built by", "built_by"),
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
    ("written by", "written_by"),
    ("written in", "written_in"),
    ("directed by", "directed_by"),
    ("played for", "played_for"),
    ("played by", "played_by"),
    ("portrayed by", "portrayed_by"),
    ("painted by", "painted_by"),
    ("invented by", "invented_by"),
    ("discovered by", "discovered_by"),
    ("built by", "built_by"),
    ("produced by", "created_by"),
    ("recorded by", "created_by"),
    ("performed by", "played_by"),
    ("composed by", "created_by"),
    ("made of", "made_of"),
    ("made from", "made_from"),
    ("starred in", "starred_in"),
    ("stars in", "starred_in"),
    ("premiered in", "released_in"),
    ("premiered on", "released_on"),
    ("headquartered in", "located_in"),
    ("situated in", "located_in"),
    ("based in", "located_in"),
    ("buried in", "died_in"),
    ("buried at", "died_in"),
    ("died on", "died_on"),
    ("died in", "died_in"),
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
    // T1.19b high-precision multi-word relations common in TriviaQA media /
    // film / location questions — absent before, golds entered via weak
    // is_related_to instead of these typed relations.
    ("happened in", "happened_in"),
    ("originated in", "originated_in"),
    ("led by", "led_by"),
    ("first appeared in", "first_appeared_in"),
    ("featured in", "featured_in"),
    ("hosted", "hosted"),
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
        let is_abbrev = (token.len() <= 2 && token.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
            || matches!(lower.trim_end_matches('.'), "st" | "dr" | "mr" | "mrs" | "ms" | "prof" | "jr" | "sr" | "u.s" | "e.g" | "i.e");
        if token.ends_with('.') && !is_abbrev {
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
        } else if tail_enabled()
            && piece.split_whitespace().count() <= 3
            && piece.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
        {
            // T1.19a: keep a short bare proper-noun continuation ("Scotland",
            // "Paris") so a following location clause can extend the chain.
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

/// Is this a question stopword / function word that should NOT contribute
/// to semantic vectors or content matching?  Content words are what carry
/// the answer signal; filtering these prevents VSA noise.
pub fn is_question_stop_word(word: &str) -> bool {
    matches!(word, "what" | "which" | "where" | "when" | "how" | "does" | "have"
        | "who" | "with" | "from" | "that" | "this" | "whose" | "why" | "the" | "was" | "did"
        | "a" | "an" | "is" | "are" | "were" | "been" | "being" | "of" | "in" | "on" | "at"
        | "to" | "for" | "by" | "as" | "and" | "or" | "but" | "not" | "no" | "it" | "its"
        | "there" | "here" | "then" | "than" | "more" | "most" | "some" | "such" | "can"
        | "will" | "would" | "could" | "should" | "may" | "might" | "must" | "shall" | "you")
}

/// Extract the relation(s) a question implies, by scanning the question for
/// the RELATIONAL_PHRASES vocabulary (the same map the decomposer uses, so the
/// returned relations align with the graph's relation vocabulary). Used as the
/// PathHD query relation INTENT.
pub fn query_relations(query: &str) -> Vec<String> {
    let lower = query.to_lowercase();
    let mut found: Vec<(usize, String)> = Vec::new();
    for (phrase, relation) in RELATIONAL_PHRASES {
        // skip degenerate copula/have phrases ("is", "was", "has", "had", ...)
        // — they carry no query-intent signal
        if matches!(*relation, "is" | "are" | "was" | "were" | "has" | "have" | "had") {
            continue;
        }
        if let Some(pos) = find_word_boundary(&lower, phrase) {
            found.push((pos, relation.to_string()));
        }
    }
    found.sort_by_key(|(pos, _)| *pos);
    let mut out: Vec<String> = Vec::new();
    for (_, r) in found {
        if !out.contains(&r) {
            out.push(r);
        }
    }
    out
}

/// Does the question contain an EXCLUSION/CONTRAST cue (NegEx-style)? If so,
/// the entities named in the question are REFERENCES (the answer is the OTHER
/// one), never the answer themselves. Examples: "Buddy Holly ... Who was the
/// other one?", "besides X", "apart from X", "two of the three musicians".
/// High-precision cue lexicon — only fires on strong signals.
pub fn has_exclusion_cue(query: &str) -> bool {
    let lower = query.to_lowercase();
    const CUES: &[&str] = &[
        "the other", "other than", "other one", "who was the other",
        "besides", "apart from", "except", "except for", "excluding",
        "one of the", "two of the", "three of the", "four of the",
        "alongside", "as well as",
    ];
    for cue in CUES {
        if find_word_boundary(&lower, cue).is_some() {
            return true;
        }
    }
    // "not X but Y" contrast pattern
    let has_not = find_word_boundary(&lower, "not").is_some();
    let has_but = find_word_boundary(&lower, "but").is_some();
    if has_not && has_but {
        return true;
    }
    false
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
/// "the Eiffel Tower, which was built in 1889" → ("the Eiffel Tower", "which was built in 1889")
/// "a Swiss professional tennis player who won ..." → ("a Swiss professional tennis player", "who won ...")
///
/// Returns (head, tail) — the tail often contains additional entities.
fn truncate_object(object: &str) -> (String, Option<String>) {
    for marker in [", ", " who ", " which ", " that ", " where ", " and ", " but ", " such ", " including "] {
        if let Some(position) = object.find(marker) {
            let head = object[..position].trim();
            let tail = object[position..].trim();
            if !head.is_empty() {
                return (normalize(head), Some(tail.to_string()));
            }
        }
    }
    // For very long objects (5+ words), also cut at prepositions to keep
    // the entity head clean.  Short objects ("United States of America")
    // are left intact to preserve real entity names.
    let words: Vec<&str> = object.split_whitespace().collect();
    if words.len() >= 5 {
        for marker in [" in ", " on ", " at ", " for ", " from ", " by ", " to "] {
            if let Some(position) = object.find(marker) {
                let head = object[..position].trim();
                if head.split_whitespace().count() >= 3 {
                    return (normalize(head), Some(object[position..].trim().to_string()));
                }
            }
        }
    }
    (normalize(object), None)
}

/// T1.19a env gate for tail-relation inheritance (default on).
fn tail_enabled() -> bool {
    std::env::var("AXIOM_V2_TAIL").map(|v| v != "0").unwrap_or(true)
}

/// Is this a location-producing relation (object is a place)?
fn is_location_relation(relation: &str) -> bool {
    matches!(
        relation,
        "located_in" | "located_at" | "located_near" | "part_of" | "from"
            | "capital_of" | "born_in" | "died_in" | "lived_in" | "founded_in"
            | "took_place_in" | "occurred_in" | "happened_in" | "released_in"
            | "published_in" | "created_in" | "developed_in"
    )
}

/// Is the subject a plausible entity (not a long lowercase descriptive phrase)?
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
    // T1.19a clause-level continuation: the object of the previous LOCATION
    // fact, so a following bare proper-noun clause ("X is a village in Dumfries
    // and Galloway, Scotland" → clause "Scotland") can be attached as the next
    // hop of the location chain (LESSONS §2.2). env AXIOM_V2_TAIL (default on).
    let mut last_location_object: Option<String> = None;

    for clause in clauses {
        let Some((position, relation, object)) = find_predicate(&clause) else {
            // Bare continuation clause (no predicate): if the previous clause
            // was a location fact, this is the tail of the chain.
            if tail_enabled() {
                if let Some(prev) = last_location_object.as_ref() {
                    let entity = normalize(&clause);
                    if !entity.is_empty()
                        && is_entity_like(&entity)
                        && entity.split_whitespace().count() <= 3
                        && entity != *prev
                    {
                        let f = DecomposedFact {
                            subject: prev.clone(),
                            relation: "located_in".to_string(),
                            object: entity.clone(),
                        };
                        if is_fact_worthy(&f) {
                            facts.push(f);
                            last_location_object = Some(entity);
                        }
                    }
                }
            }
            continue;
        };
        let raw_object = object.clone();
        let mut subject = normalize(&clause[..position]);
        let subject_canonical = canonical_subject(&subject);
        // Whether the subject came from the clause (derived) or was inherited /
        // the fallback page title (trusted — always a valid entity).
        let mut trusted_subject = false;

        // A clause-subject that is a copula fragment ("is a ballet composed by
        // X") has no real subject — the leading word is a copula and the rest
        // is a descriptive complement with no proper noun. Inherit the previous
        // subject ("Swan Lake") so the strong created_by link anchors correctly.
        // Only treat as fragment when there is no capitalised proper noun after
        // the copula — "Was President Kennedy the..." keeps its real subject.
        let subj_first = subject_canonical.split_whitespace().next().unwrap_or("");
        let leading_copula = matches!(subj_first.to_lowercase().as_str(),
            "is" | "was" | "are" | "were" | "has" | "have" | "had" | "been" | "being");
        let rest_has_proper = subject_canonical.split_whitespace().skip(1)
            .any(|w| w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false));
        let copula_fragment = leading_copula && !rest_has_proper;

        if subject_canonical.is_empty() || copula_fragment {
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

        // Truncate subject at commas for clean entity boundaries.
        // "Chicago, Illinois, 17 mi northwest" → "Chicago".
        if let Some(pos) = subject.find(',') {
            let head = subject[..pos].trim();
            if !head.is_empty() { subject = head.to_string(); }
        }
        // T1.19c: If the subject contains a mid-clause copula (e.g. "Jurassic Park is a 1993 film"
        // before predicate "directed by"), trim to the pre-copula entity ("Jurassic Park").
        for copula in [" is ", " was ", " are ", " were "] {
            if let Some(pos) = subject.find(copula) {
                let head = subject[..pos].trim();
                let head_canonical = canonical_subject(head);
                if !head_canonical.is_empty()
                    && !is_discardable(&head_canonical.to_lowercase())
                    && is_entity_like(&head_canonical)
                {
                    subject = head_canonical;
                    break;
                }
            }
        }
        // Strip a trailing copula verb from the subject (T1.10d subject
        // resolution): "Zadok the Priest were composed by ..." → subject
        // "Zadok the Priest were" must become "Zadok the Priest". Without this
        // the entity "Zadok the Priest" never gets its strong created_by link
        // and sinks to deep rank.
        let mut subj_tokens: Vec<&str> = subject.split_whitespace().collect();
        while let Some(last) = subj_tokens.last() {
            if matches!(last.to_lowercase().as_str(),
                "is" | "was" | "are" | "were" | "been" | "being" | "had" | "has" | "have")
            {
                subj_tokens.pop();
            } else {
                break;
            }
        }
        subject = subj_tokens.join(" ");
        // For long subjects (5+ words), also cut at prepositions.
        // "the capital of France, Paris" trimmed at comma → "the capital of France"
        // then trimmed at " of " → "the capital" (exclude "of" for entity names).
        let subj_words: Vec<&str> = subject.split_whitespace().collect();
        if subj_words.len() >= 5 {
            for marker in [" in ", " on ", " at ", " for ", " from ", " by ", " to "] {
                if let Some(pos) = subject.find(marker) {
                    let head = subject[..pos].trim();
                    if head.split_whitespace().count() >= 2 {
                        subject = head.to_string();
                        break;
                    }
                }
            }
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
        let (object, tail) = truncate_object(&object);
        let _ = tail; // clause-level continuation handles location-chain tails
        if object.is_empty() {
            continue;
        }

        // Extract a year reference from the FULL pre-truncation object and
        // link it as a temporal anchor: (subject, happened_in, year).
        for word in raw_object.split_whitespace() {
            let cleaned = word.trim_matches(|c: char| !c.is_ascii_digit());
            if cleaned.len() == 4 && cleaned.starts_with('1') || cleaned.len() == 4 && cleaned.starts_with('2') {
                let f = DecomposedFact {
                    subject: subject.clone(),
                    relation: "happened_in".to_string(),
                    object: cleaned.to_string(),
                };
                if is_fact_worthy(&f) { facts.push(f); }
                break;
            }
        }

        let fact = DecomposedFact {
            subject: subject.clone(),
            relation: relation.to_string(),
            object: object.clone(),
        };
        if is_fact_worthy(&fact) {
            facts.push(fact.clone());
        }
        // T1.19a: track the object of a LOCATION fact so a following bare
        // proper-noun clause can extend the chain ("..., Scotland").
        if is_location_relation(relation) {
            last_location_object = Some(object.clone());
        } else {
            last_location_object = None;
        }

        // When a birth or parentage fact is extracted, scan the raw
        // (pre-truncation) object for " to <Name> and <Name>" patterns.
        // E.g. "born in Košice ... to Melanie Molitorová and Karol Hingis"
        // yields has_mother / has_father facts that compose multi-hop chains.
        if matches!(relation, "born_in" | "born_on" | "child_of" | "born_to") {
            for parent in extract_parent_names(&raw_object) {
                let p_fact = DecomposedFact {
                    subject: subject.clone(),
                    relation: "has_parent".to_string(),
                    object: parent,
                };
                if is_fact_worthy(&p_fact)
                    && !facts.iter().any(|f| f.subject == p_fact.subject && f.object == p_fact.object)
                {
                    facts.push(p_fact);
                }
            }
            // Also scan the full sentence: ellipsis/punctuation may split the
            // parent clause away from the birth clause across clause boundaries.
            for parent in extract_parent_names(sentence) {
                let p_fact = DecomposedFact {
                    subject: subject.clone(),
                    relation: "has_parent".to_string(),
                    object: parent,
                };
                if is_fact_worthy(&p_fact)
                    && !facts.iter().any(|f| f.subject == p_fact.subject && f.object == p_fact.object)
                {
                    facts.push(p_fact);
                }
            }
        }

        // Surface embedded proper-noun phrases from the object as standalone
        // entities. E.g. "the inventor of the lightweight baby buggy" yields
        // the additional fact (subject, is_related_to, Baby Buggy). TriviaQA
        // answers are often capitalized phrases buried inside long objects.
        if object.split_whitespace().count() <= 40 {
            for phrase in extract_proper_nouns(&object) {
                if !facts.iter().any(|f| f.subject == subject && f.object == phrase) {
                    let f = DecomposedFact { subject: subject.clone(), relation: "is_related_to".to_string(), object: phrase };
                    if is_fact_worthy(&f) { facts.push(f); }
                }
            }
        }

        // Tail entity recovery: after truncation at prepositions/clauses,
        // the tail often contains specific noun phrases that are the answer.
        if let Some(ref tail_text) = tail {
            if tail_text.split_whitespace().count() <= 30 {
                // When the head fact is a location relation ("village in
                // Dumfries and Galloway, Scotland"), proper nouns in the tail
                // ("Scotland") inherit the SAME strong relation — they are
                // appositive continuations of the location, not mere mentions.
                let tail_rel = if matches!(relation, "located_in" | "located_at"
                    | "capital_of" | "part_of" | "located_near" | "born_in"
                    | "lived_in" | "died_in" | "took_place_in" | "occurred_in"
                    | "from") { relation } else { "is_related_to" };
                for phrase in extract_proper_nouns(tail_text) {
                    if !facts.iter().any(|f| f.subject == subject && f.object == phrase) {
                        let f = DecomposedFact { subject: subject.clone(), relation: tail_rel.to_string(), object: phrase };
                        if is_fact_worthy(&f) { facts.push(f); }
                    }
                }
                // Tight lowercase noun-phrase extraction: 3+ word all-lowercase
                // phrases that are specific enough to be answers ("collapsible
                // support assembly").  Only from the tail (post-preposition),
                // gated by a dedicated lowercase pass.
                for phrase in extract_lowercase_noun_phrases(tail_text) {
                    if !facts.iter().any(|f| f.subject == subject && f.object == phrase) {
                        let f = DecomposedFact { subject: subject.clone(), relation: "is_related_to".to_string(), object: phrase };
                        if is_fact_worthy(&f) { facts.push(f); }
                    }
                }
            }
        }
    }

    facts
}

/// Extract proper-noun entities from an entire evidence sentence and emit
/// them as `(fallback_subject, mentions, EntityName)` facts.
///
/// This is a high-recall safety net: even when decomposition fails to
/// extract the relation connecting subject to answer, the answer entity
/// (which is usually a capitalized proper noun in the evidence text)
/// still enters the knowledge graph.
pub fn extract_sentence_entities(sentence: &str, fallback_subject: &str) -> Vec<DecomposedFact> {
    let mut facts: Vec<DecomposedFact> = Vec::new();
    for phrase in extract_proper_nouns(sentence) {
        if !facts.iter().any(|f: &DecomposedFact| f.subject == fallback_subject && f.object.as_str() == phrase.as_str()) {
            let f = DecomposedFact {
                subject: fallback_subject.to_string(),
                relation: "mentions".to_string(),
                object: phrase,
            };
            if is_fact_worthy(&f) { facts.push(f); }
        }
    }
    facts
}

/// Extract capitalized proper-noun phrases from a string.
///
/// Looks for runs of tokens where at least the first is capitalized and not at
/// sentence start (heuristically: the token before is lowercase or a
/// preposition). Returns the longest capitalized phrase per window, stopping
/// the phrase at clause/preposition/coordinator boundaries so names stay
/// clean: "present-day Switzerland by Count Radbot" keeps only "Switzerland"
/// (not "Switzerland by Count Radbot"), and "Chicago, Illinois, 17 mi northwest"
/// yields "Chicago" and "Illinois" (not a single comma-junk entity).
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
            let mut comma_terminated = false;
            // Stop the phrase at relative pronouns, prepositions, coordinators,
            // numbers, and punctuation.  A trailing comma (",", ";") closes the
            // phrase but keeps the word before it — "Chicago," → "Chicago".
            while j < words.len() {
                let token = words[j];
                let lower = token.to_lowercase();
                let starts_digit = token.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false);
                let ends_comma = token.ends_with(',') || token.ends_with(';');
                if starts_digit
                    || matches!(lower.as_str(),
                        "who" | "which" | "that" | "where" | "with" | "in" | "on" | "at"
                            | "of" | "to" | "from" | "and" | "but" | "for" | "the" | "a"
                            | "an" | "his" | "her" | "its" | "was" | "is" | "are" | "were"
                            | "by" | "or" | "as" | "alongside" | "named" | "such" | "like"
                            | "including" | "between" | "after" | "before" | "through"
                            | "under" | "over" | "during" | "within" | "about" | "around")
                {
                    break;
                }
                j += 1;
                if ends_comma {
                    comma_terminated = true;
                    break;
                }
            }
            // Trim trailing commas/semicolons/periods from the last word so
            // "Chicago," becomes "Chicago", "Airport," becomes "Airport", and
            // "Loop." stays "Loop" (still filtered below as a common noun).
            let phrase_words: Vec<&str> = words[i..j]
                .iter()
                .map(|w| w.trim_end_matches(|c: char| c == ',' || c == ';' || c == '.'))
                .filter(|w| !w.is_empty())
                .collect();
            let phrase_len = phrase_words.len();
            // Admit single capitalized words only when they are genuine proper
            // nouns, not sentence-initial or article-headed common nouns:
            //   1. A comma/semicolon terminated them — apposition/list pattern
            //      ("Chicago, Illinois").
            //   2. The preceding token is lowercase — a mid-sentence proper
            //      noun ("present-day Switzerland", "of Alaska").
            //   3. Not preceded by an article — "the Loop" is a common-noun
            //      phrase, "Loop" alone is not a proper noun.
            //   4. The word is not a discardable function word ("He", "Located").
            let is_appositive = comma_terminated && phrase_len == 1;
            let preceded_lower = phrase_len == 1
                && i > 0
                && words[i - 1]
                    .chars()
                    .next()
                    .map(|c| c.is_lowercase())
                    .unwrap_or(false);
            let preceded_article = phrase_len == 1
                && i > 0
                && matches!(
                    words[i - 1].to_lowercase().as_str(),
                    "a" | "an" | "the"
                );
            let single_ok = phrase_len == 1
                && !is_discardable(&phrase_words[0].to_lowercase())
                && phrase_words[0].len() >= 3;
            if (phrase_len >= 2 && phrase_len <= 5)
                || (is_appositive || preceded_lower) && !preceded_article && single_ok
            {
                let first = phrase_words[0].to_lowercase();
                if !matches!(first.as_str(), "a" | "an" | "the" | "his" | "her" | "its") {
                    let phrase = phrase_words.join(" ");
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

/// Extract tight lowercase noun phrases (3-4 words) that are specific enough
/// to be answers.  Only from tail text (post-preposition), skips stopwords
/// and articles so "collapsible support assembly" survives but "the sky is"
/// does not.
fn extract_lowercase_noun_phrases(text: &str) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let first = words[i].to_lowercase();
        if matches!(first.as_str(), "a" | "an" | "the" | "and" | "or" | "but" | "of"
            | "with" | "from" | "that" | "this" | "which" | "who" | "to" | "in" | "on"
            | "at" | "for" | "by" | "as" | "is" | "are" | "was" | "were" | "has" | "have"
            | "it" | "its" | "there" | "such" | "including" | "also" | "most" | "more")
            || first.chars().any(|c| c.is_uppercase())
        {
            i += 1;
            continue;
        }
        // Try the longest window (4 words) first, require all-lowercase.
        for len in (3..=4).rev() {
            if i + len > words.len() { continue; }
            let window = &words[i..i + len];
            let all_lower = window.iter().all(|w| w.chars().all(|c| !c.is_uppercase()));
            if !all_lower { continue; }
            let joined = window.join(" ");
            let has_stop = window.iter().any(|w| matches!(w.to_lowercase().as_str(),
                "and" | "or" | "but" | "with" | "from" | "that" | "which" | "who" | "to"
                | "in" | "on" | "at" | "for" | "by" | "as" | "is" | "are" | "was" | "were"
                | "has" | "have" | "it" | "its" | "there" | "such" | "including" | "also"));
            if !has_stop {
                out.push(joined);
                i += len;
                break;
            }
        }
        i += 1;
    }
    out
}

/// Scan the raw (pre-truncation) object of a birth/location fact for
/// parent names: "… to Melanie Molitorová and Karol Hingis" yields two
/// `has_parent` entities.
fn extract_parent_names(raw_object: &str) -> Vec<String> {
    let lower = raw_object.to_lowercase();
    let to_marker = " to ";
    let Some(to_pos) = lower.find(to_marker) else { return vec![] };
    // Ensure "to" is a whole word on the left side.
    if to_pos > 0 {
        let before = raw_object.as_bytes()[to_pos - 1];
        if before.is_ascii_alphanumeric() { return vec![]; }
    }
    let after = &raw_object[to_pos + to_marker.len()..];
    let names: Vec<&str> = after.split(" and ")
        .map(|s| s.trim().trim_matches(|c: char| c == '.' || c == ','))
        .filter(|s| !s.is_empty())
        .collect();
    let mut out: Vec<String> = Vec::new();
    for name in names {
        let words: Vec<&str> = name.split_whitespace().collect();
        if words.is_empty() || words.len() > 3 { continue; }
        let starts_cap = words[0].chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
        if !starts_cap { continue; }
        out.push(name.to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_relations_extracts_intent() {
        let r = query_relations("Who composed the ballet Swan Lake?");
        assert!(r.contains(&"composed".to_string()) || r.contains(&"created_by".to_string()), "got {r:?}");
        let r2 = query_relations("Who was the founder of Microsoft?");
        assert!(r2.iter().any(|x| x == "founded_by" || x == "founder_of"), "got {r2:?}");
        let r3 = query_relations("How many episodes were made?");
        assert!(r3.is_empty(), "no relational phrase expected, got {r3:?}");
    }

    #[test]
    fn tail_inheritance_restores_location_chain() {
        // "X ... in Dumfries and Galloway, Scotland" — the bare continuation
        // clause "Scotland" must attach as (Dumfries, located_in, Scotland)
        // so transitivity can chain X → Scotland (LESSONS §2.2).
        let facts = decompose_sentence(
            "Wanlockhead is a village in Dumfries and Galloway, Scotland.",
            "",
        );
        let has_chain = facts.iter().any(|f| {
            f.relation == "located_in"
                && f.object == "Scotland"
                && f.subject == "Dumfries"
        }) || facts.iter().any(|f| {
            f.relation == "located_in" && f.object == "Scotland" && f.subject == "Dumfries and Galloway"
        });
        assert!(has_chain, "expected a Dumfries→Scotland located_in edge, got {facts:?}");
    }

    #[test]
    fn decomposes_relative_clause_into_chain() {
        let facts = decompose_sentence(
            "Martina Hingis is a Swiss tennis player who won the Australian Open in 1997.",
            "Martina Hingis",
        );
        // Quality gate rejects descriptive copulas ("is a Swiss tennis player").
        // Only the won + happened_in facts should survive.
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
    fn subject_resolution_recovers_creator_relation() {
        // "Swan Lake, Op. 20, is a ballet composed by Pyotr Ilyich Tchaikovsky"
        // must yield (Swan Lake, created_by, Tchaikovsky), not a junk subject
        // "is a ballet" that leaves the composer as weak mentions (deep rank).
        let facts = decompose_sentence(
            "Swan Lake, Op. 20, is a ballet composed by Pyotr Ilyich Tchaikovsky in 1875-76.",
            "Swan Lake",
        );
        assert!(
            facts.iter().any(|f| f.subject == "Swan Lake"
                && f.relation == "created_by"
                && f.object == "Pyotr Ilyich Tchaikovsky"),
            "expected (Swan Lake, created_by, Tchaikovsky), got {:?}",
            facts.iter().map(|f| (f.subject.as_str(), f.relation.as_str(), f.object.as_str())).collect::<Vec<_>>()
        );
    }

    #[test]
    fn subject_resolution_strips_trailing_copula() {
        // "and Zadok the Priest were composed by George Frideric Handel"
        // must yield subject "Zadok the Priest", not "Zadok the Priest were".
        let facts = decompose_sentence(
            "The four anthems The King Shall Rejoice, and Zadok the Priest were composed by George Frideric Handel for the coronation.",
            "Zadok the Priest",
        );
        assert!(
            facts.iter().any(|f| f.subject == "Zadok the Priest"
                && f.relation == "created_by"
                && f.object == "George Frideric Handel"),
            "expected (Zadok the Priest, created_by, Handel), got {:?}",
            facts.iter().map(|f| (f.subject.as_str(), f.relation.as_str(), f.object.as_str())).collect::<Vec<_>>()
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

    #[test]
    fn extracts_parents_from_born_to_pattern() {
        let facts = decompose_sentence(
            "Hingis was born in Košice ... to Melanie Molitorová and Karol Hingis",
            "Hingis",
        );
        assert!(
            facts.iter().any(|f| f.relation == "has_parent" && f.object == "Melanie Molitorová"),
            "expected has_parent for Melanie Molitorová, got {:?}",
            facts.iter().filter(|f| f.relation == "has_parent").map(|f| f.object.clone()).collect::<Vec<_>>()
        );
        assert!(
            facts.iter().any(|f| f.relation == "has_parent" && f.object == "Karol Hingis"),
            "expected has_parent for Karol Hingis, got {:?}",
            facts.iter().filter(|f| f.relation == "has_parent").map(|f| f.object.clone()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn copula_filter_rejects_descriptive_complement() {
        // "a Swiss tennis player" is a description, not an entity.
        let f = DecomposedFact {
            subject: "Hingis".into(),
            relation: "is".into(),
            object: "a Swiss tennis player".into(),
        };
        assert!(!is_fact_worthy(&f));
    }

    #[test]
    fn copula_filter_keeps_entity_complement() {
        // "the capital of France" contains "France" (proper noun) — keep it.
        let f = DecomposedFact {
            subject: "Paris".into(),
            relation: "is".into(),
            object: "the capital of France".into(),
        };
        assert!(is_fact_worthy(&f));
    }

    #[test]
    fn copula_filter_rejects_long_article_phrase_without_proper_noun() {
        // No capitalised mid-word → descriptive junk.
        let f = DecomposedFact {
            subject: "Hingis".into(),
            relation: "is".into(),
            object: "a former world number one tennis player".into(),
        };
        assert!(!is_fact_worthy(&f));
    }

    #[test]
    fn proper_nouns_stop_at_connectors_and_prepositions() {
        // "by" terminates the phrase: "present-day Switzerland by Count Radbot"
        // must yield "Switzerland", not "Switzerland by Count Radbot".
        let nouns = extract_proper_nouns(
            "a fortress built in present-day Switzerland by Count Radbot of Klettgau",
        );
        assert!(
            nouns.iter().any(|n| n == "Switzerland"),
            "expected Switzerland, got {:?}",
            nouns
        );
        assert!(
            nouns.iter().any(|n| n == "Count Radbot"),
            "expected Count Radbot, got {:?}",
            nouns
        );
        assert!(
            !nouns.iter().any(|n| n.contains("by")),
            "phrase must not swallow the 'by' connector: {:?}",
            nouns
        );
    }

    #[test]
    fn proper_nouns_split_on_commas_and_numbers() {
        // "Chicago, Illinois, 17 mi northwest" must yield clean "Chicago" and
        // "Illinois", never the comma-junk entity "Chicago, Illinois, 17 mi northwest".
        let nouns = extract_proper_nouns(
            "located on the Far Northwest Side of Chicago, Illinois, 17 mi northwest of the Loop",
        );
        assert!(
            nouns.iter().any(|n| n == "Chicago"),
            "expected clean Chicago, got {:?}",
            nouns
        );
        assert!(
            nouns.iter().any(|n| n == "Illinois"),
            "expected clean Illinois, got {:?}",
            nouns
        );
        assert!(
            !nouns.iter().any(|n| n.contains(',')),
            "no phrase may retain a comma: {:?}",
            nouns
        );
        assert!(
            !nouns.iter().any(|n| n.contains("northwest")),
            "number-preposition junk must not survive: {:?}",
            nouns
        );
    }

    #[test]
    fn proper_nouns_reject_article_headed_common_nouns() {
        // "the Loop" is a common-noun phrase — "Loop" alone is not an entity.
        let nouns = extract_proper_nouns("located on the Loop");
        assert!(
            !nouns.iter().any(|n| n == "Loop"),
            "article-headed common noun must not be an entity: {:?}",
            nouns
        );
    }

    #[test]
    fn proper_nouns_reject_sentence_initial_common_word() {
        // A bare capitalized function word at sentence start is grammar, not
        // an entity — unless it forms a 2+ word phrase.
        let nouns = extract_proper_nouns("Located in the Alps, the valley is deep");
        assert!(
            !nouns.iter().any(|n| n == "Located"),
            "sentence-initial 'Located' is not an entity: {:?}",
            nouns
        );
    }

    #[test]
    fn proper_nouns_surface_appositive_single_word() {
        // Comma-terminated single proper nouns (apposition/list) are real
        // entities: "Paris, France" → Paris and France.
        let nouns = extract_proper_nouns("the city of Paris, France, is a capital");
        assert!(
            nouns.iter().any(|n| n == "Paris"),
            "expected Paris, got {:?}",
            nouns
        );
        assert!(
            nouns.iter().any(|n| n == "France"),
            "expected France, got {:?}",
            nouns
        );
    }

    #[test]
    fn subject_resolution_handles_mid_clause_copulas() {
        let facts = decompose_sentence(
            "Jurassic Park is a 1993 American science fiction adventure film directed by Steven Spielberg.",
            "Jurassic Park",
        );
        assert!(
            facts.iter().any(|f| f.subject == "Jurassic Park"
                && f.relation == "directed_by"
                && f.object == "Steven Spielberg"),
            "expected (Jurassic Park, directed_by, Steven Spielberg), got {:?}",
            facts.iter().map(|f| (f.subject.as_str(), f.relation.as_str(), f.object.as_str())).collect::<Vec<_>>()
        );

        let facts_hp = decompose_sentence(
            "Harry Potter is a series of fantasy novels written by J. K. Rowling.",
            "Harry Potter",
        );
        assert!(
            facts_hp.iter().any(|f| f.subject == "Harry Potter"
                && f.relation == "written_by"
                && f.object.contains("Rowling")),
            "expected (Harry Potter, written_by, J. K. Rowling), got {:?}",
            facts_hp.iter().map(|f| (f.subject.as_str(), f.relation.as_str(), f.object.as_str())).collect::<Vec<_>>()
        );
    }
}
