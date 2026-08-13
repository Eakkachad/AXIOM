//! Linearization: Convert knowledge graph paths to natural language sentences.
//!
//! Implements a template-based linearization system that:
//! 1. Classifies the query intent (Why/What/How/Where/When/Declarative)
//! 2. Maps relations to natural language templates
//! 3. Joins triples with intent-appropriate connectives
//! 4. Applies article insertion and capitalization

use std::collections::HashSet;
use crate::templates::TemplateBank;

/// The communicative intent detected from the query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intent {
    Why,
    What,
    Who,
    How,
    Where,
    When,
    Declarative,
}

/// Classify the intent of a query string.
///
/// Simple keyword-based classification for the question type.
pub fn classify_intent(query: &str) -> Intent {
    let lower = query.to_lowercase();
    if lower.starts_with("why") || lower.contains("why ") {
        Intent::Why
    } else if lower.starts_with("what") || lower.contains("what ") {
        Intent::What
    } else if lower.starts_with("who") || lower.contains("who ") {
        Intent::Who
    } else if lower.starts_with("how") || lower.contains("how ") {
        Intent::How
    } else if lower.starts_with("where") || lower.contains("where ") {
        Intent::Where
    } else if lower.starts_with("when") || lower.contains("when ") {
        Intent::When
    } else {
        Intent::Declarative
    }
}

/// Get the connective word for joining clauses based on intent.
fn connective_for_intent(intent: Intent) -> &'static str {
    match intent {
        Intent::Why => "because",
        Intent::How => "which",
        Intent::What | Intent::Who => "that is",
        Intent::Where => "where",
        Intent::When => "when",
        Intent::Declarative => "and",
    }
}

/// Convert a relation name to a natural language verb phrase.
///
/// Maps common relation patterns to readable English.
fn relation_to_template(relation: &str) -> &str {
    match relation {
        "is" => "is",
        "is_a" => "is a",
        "are" => "are",
        "has" => "has",
        "has_a" => "has a",
        "have" => "have",
        "can" => "can",
        "contains" => "contains",
        "causes" => "causes",
        "caused_by" => "is caused by",
        "scatters" => "scatters",
        "produces" => "produces",
        "makes" => "makes",
        "creates" => "creates",
        "leads to" => "leads to",
        "comes from" => "comes from",
        "results in" => "results in",
        "part_of" => "is part of",
        "made_of" => "is made of",
        "located_in" => "is located in",
        "occurs_at" => "occurs at",
        "relates_to" => "relates to",
        "enables" => "enables",
        "requires" => "requires",
        "used_for" => "is used for",
        "defined_as" => "is defined as",
        // T1.21: relations the decomposer produces that were printed raw
        "capital_of" => "is the capital of",
        "orbits" => "orbits",
        "founded" => "founded",
        "founded_by" => "was founded by",
        "created_by" => "was created by",
        "written_by" => "was written by",
        "directed_by" => "was directed by",
        "composed_by" => "was composed by",
        "painted_by" => "was painted by",
        "built_by" => "was built by",
        "invented_by" => "was invented by",
        "discovered_by" => "was discovered by",
        "wrote" => "wrote",
        "directed" => "directed",
        "composed" => "composed",
        "painted" => "painted",
        "invented" => "invented",
        "discovered" => "discovered",
        "designed" => "designed",
        "built" => "built",
        "developed" => "developed",
        "won" => "won",
        "starred" => "starred in",
        "played" => "played",
        "played_for" => "played for",
        "born_in" => "was born in",
        "died_in" => "died in",
        "lived_in" => "lived in",
        "married_to" => "married",
        "named_after" => "is named after",
        "known_for" => "is known for",
        "known_as" => "is known as",
        "based_on" => "is based on",
        "set_in" => "is set in",
        "happened_in" => "happened in",
        "occurred_in" => "occurred in",
        "took_place_in" => "took place in",
        "led_by" => "was led by",
        "first_appeared_in" => "first appeared in",
        "featured_in" => "featured in",
        "hosted" => "hosted",
        "produced" => "produced",
        "led" => "led",
        "owned" => "owned",
        "served_as" => "served as",
        "happened" => "happened",
        _ => relation,
    }
}

/// Convert an entity name to a readable form.
///
/// Replaces underscores with spaces for display.
fn entity_to_text(entity: &str) -> String {
    entity.replace('_', " ")
}

/// Determine whether to use "a" or "an" before a word.
fn article_for(word: &str) -> &'static str {
    let first_char = word.chars().next().unwrap_or('x').to_ascii_lowercase();
    match first_char {
        'a' | 'e' | 'i' | 'o' | 'u' => "an",
        _ => "a",
    }
}

/// Words that never take an indefinite article: mass nouns and
/// adjective/color complements ("the sky is blue", "water is liquid").
fn is_no_article(text: &str) -> bool {
    matches!(
        text,
        "evaporation" | "water" | "information" | "knowledge" | "software"
            | "equipment" | "research" | "blue" | "red" | "green" | "yellow"
            | "black" | "white" | "purple" | "orange" | "pink" | "brown"
            | "grey" | "gray" | "liquid" | "solid" | "gas" | "alive" | "dead"
            | "hot" | "cold" | "warm" | "wet" | "dry" | "soft" | "hard"
            | "heavy" | "light" | "old" | "new" | "big" | "small" | "large"
    )
}

/// Known proper nouns (no indefinite article, keep capitalization mid-sentence):
/// countries, continents, planets, the sun/moon. H1 grammar finisher.
fn is_proper_noun(text: &str) -> bool {
    const PROPER: &[&str] = &[
        "france", "england", "scotland", "wales", "ireland", "germany", "spain", "italy",
        "portugal", "switzerland", "austria", "belgium", "netherlands", "greece", "russia",
        "china", "japan", "india", "canada", "america", "australia", "brazil", "mexico",
        "europe", "asia", "africa", "america", "antarctica", "earth", "mars", "venus",
        "jupiter", "saturn", "uranus", "neptune", "mercury", "pluto", "sun", "moon",
        "london", "paris", "berlin", "rome", "madrid", "washington", "new york", "tokyo",
        "beijing", "moscow", "thailand", "bangkok",
    ];
    PROPER.contains(&text)
}

/// Does the entity text already begin with an article ("a/an/the")?
fn starts_with_article(text: &str) -> bool {
    text.split_whitespace()
        .next()
        .map(|w| matches!(w, "a" | "an" | "the" | "A" | "An" | "The"))
        .unwrap_or(false)
}

/// H1 grammar finisher: lowercase a clause-initial word after a comma unless
/// it is a proper noun or a short connective ("Cats are animals, and Animals
/// have hearts" → "…and animals have hearts").
fn finish_grammar(s: &str) -> String {
    let words: Vec<&str> = s.split_whitespace().collect();
    let mut out = String::new();
    for (i, w) in words.iter().enumerate() {
        // a comma in the previous 1-2 tokens (", and ", ", because ") marks a
        // new clause → lowercase its first word unless proper/connective
        let has_comma_before = i > 0 && (words[i - 1].ends_with(',') || words[i - 1] == "and" && i > 1 && words[i - 2].ends_with(','));
        let lower_first = if has_comma_before {
            let bare = w.trim_matches(|c: char| !c.is_alphabetic()).to_lowercase();
            let connective = matches!(
                bare.as_str(),
                "and" | "or" | "but" | "because" | "that" | "which" | "while"
                    | "when" | "where" | "so" | "yet" | "then" | "however"
            );
            if !connective
                && !is_proper_noun(&bare)
                && w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
            {
                let mut c = w.chars();
                let first = c.next().unwrap().to_lowercase().collect::<String>();
                Some(first + c.as_str())
            } else {
                None
            }
        } else {
            None
        };
        out.push_str(lower_first.as_deref().unwrap_or(w));
        if i < words.len() - 1 {
            out.push(' ');
        }
    }
    out
}

/// Insert an appropriate article before a noun if it's first occurrence.
fn with_article(entity: &str, seen: &mut HashSet<String>) -> String {
    let text = entity_to_text(entity);

    // Mass nouns, adjectives/colors, and known proper nouns get no article.
    if is_no_article(&text) || is_proper_noun(&text.to_lowercase()) {
        seen.insert(text.clone());
        return text;
    }

    // Skip articles for:
    // - Entities that already start with an article
    // - Plural nouns (ending in 's')
    // - Proper-looking nouns (original starts with uppercase — but we lowercased)
    // - Mass nouns and multi-word phrases
    let words: Vec<&str> = text.split_whitespace().collect();
    let starts_with_article = matches!(words.first(), Some(&"a") | Some(&"an") | Some(&"the"));
    let is_plural = text.ends_with('s') && !text.ends_with("ss");
    let is_multi_word = words.len() > 2;

    if starts_with_article || is_multi_word {
        // Already has article or is a phrase — use as-is
        if seen.contains(&text) {
            return text;
        }
        seen.insert(text.clone());
        return text;
    }

    if seen.contains(&text) {
        format!("the {}", text)
    } else {
        seen.insert(text.clone());
        if is_plural {
            text // plurals don't need articles: "clouds", "animals"
        } else {
            let article = article_for(&text);
            format!("{} {}", article, text)
        }
    }
}

/// Capitalize the first letter of a string.
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
    }
}

/// Linearize a path of triples into a natural language sentence.
///
/// The algorithm:
/// 1. For each triple, produce "subject relation object" using templates
/// 2. Join consecutive triple phrases with intent-appropriate connectives
/// 3. Apply article insertion (a/an for first mention, the for subsequent)
/// 4. Capitalize the first letter
pub fn linearize(
    path_triples: &[crate::graph::Triple],
    entities: &[String],
    relations: &[String],
    intent: Intent,
) -> String {
    if path_triples.is_empty() {
        return String::new();
    }

    let connective = connective_for_intent(intent);
    let mut seen_entities: HashSet<String> = HashSet::new();
    let mut clauses: Vec<String> = Vec::new();

    for (ci, triple) in path_triples.iter().enumerate() {
        let subject = &entities[triple.subject_id];
        let relation = &relations[triple.relation_id];
        let object = &entities[triple.object_id];

        // T1.21: the FIRST clause's subject is the topic we asked about →
        // definite ("the sky is blue"), not indefinite ("a sky is blue").
        // Skip plurals ("cats") and article-less words ("water").
        let mut subj_text = with_article(subject, &mut seen_entities);
        if ci == 0 && !is_no_article(&entity_to_text(subject)) && !starts_with_article(&subj_text) {
            let w = entity_to_text(subject);
            let plural = w.ends_with('s') && !w.ends_with("ss");
            if !plural {
                subj_text = format!("the {}", subj_text);
            }
        }
        let rel_text = relation_to_template(relation);
        let obj_text = with_article(object, &mut seen_entities);

        clauses.push(format!("{} {} {}", subj_text, rel_text, obj_text));
    }

    // Join clauses with connective
    let joined = if clauses.len() == 1 {
        clauses[0].clone()
    } else {
        // First clause stands alone, subsequent clauses are joined with connective
        let mut result = clauses[0].clone();
        for clause in &clauses[1..] {
            result = format!("{}, {} {}", result, connective, clause);
        }
        result
    };

    // Capitalize and add period (H1: fix clause-initial capitalization first).
    let sentence = capitalize_first(&finish_grammar(&joined));
    if sentence.ends_with('.') || sentence.ends_with('?') || sentence.ends_with('!') {
        sentence
    } else {
        format!("{}.", sentence)
    }
}

/// Linearize using corpus-derived templates when one matches the relation.
pub fn linearize_with_templates(
    path_triples: &[crate::graph::Triple],
    entities: &[String],
    relations: &[String],
    intent: Intent,
    templates: &TemplateBank,
) -> String {
    if path_triples.is_empty() {
        return String::new();
    }

    let mut clauses = Vec::with_capacity(path_triples.len());
    for triple in path_triples {
        let subject = &entities[triple.subject_id];
        let relation = &relations[triple.relation_id];
        let object = &entities[triple.object_id];
        let clause = templates.generate(subject, relation, object).unwrap_or_else(|| {
            linearize(std::slice::from_ref(triple), entities, relations, Intent::Declarative)
                .trim_end_matches('.').to_string()
        });
        clauses.push(clause);
    }

    let connective = connective_for_intent(intent);
    let mut sentence = clauses[0].clone();
    for clause in &clauses[1..] {
        sentence = format!("{}, {} {}", sentence, connective, clause);
    }
    let sentence = capitalize_first(&finish_grammar(&sentence));
    if sentence.ends_with('.') { sentence } else { format!("{}.", sentence) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{KnowledgeGraph, Triple};

    #[test]
    fn test_classify_intent_why() {
        assert_eq!(classify_intent("why is the sky blue?"), Intent::Why);
        assert_eq!(classify_intent("Why does it rain?"), Intent::Why);
    }

    #[test]
    fn test_classify_intent_what() {
        assert_eq!(classify_intent("what is water?"), Intent::What);
        assert_eq!(classify_intent("What causes rain?"), Intent::What);
    }

    #[test]
    fn test_classify_intent_how() {
        assert_eq!(classify_intent("how does it work?"), Intent::How);
    }

    #[test]
    fn test_classify_intent_declarative() {
        assert_eq!(classify_intent("tell me about sky"), Intent::Declarative);
    }

    #[test]
    fn test_linearize_single_triple() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple("sky", "is", "blue");

        let result = linearize(
            &kg.triples,
            &kg.entities,
            &kg.relations,
            Intent::Declarative,
        );

        assert!(!result.is_empty());
        assert!(result.starts_with(|c: char| c.is_uppercase()));
        assert!(result.ends_with('.'));
        assert!(result.to_lowercase().contains("sky"));
        assert!(result.to_lowercase().contains("blue"));
    }

    #[test]
    fn test_linearize_multi_triple() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple("sky", "is", "blue");
        kg.add_triple("blue", "has", "short_wavelength");

        let result = linearize(
            &kg.triples,
            &kg.entities,
            &kg.relations,
            Intent::Why,
        );

        assert!(result.contains("because"), "Why-intent should use 'because': {}", result);
        assert!(result.to_lowercase().contains("sky"));
        assert!(result.to_lowercase().contains("short wavelength"));
    }

    #[test]
    fn test_linearize_articles() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple("sky", "is", "blue");
        kg.add_triple("blue", "has", "color");

        let result = linearize(
            &kg.triples,
            &kg.entities,
            &kg.relations,
            Intent::Declarative,
        );

        // "blue" appears twice — first time "a blue", second time "the blue"
        let lower = result.to_lowercase();
        assert!(lower.contains("a sky") || lower.contains("an sky") || lower.contains("the sky"));
    }

    #[test]
    fn test_entity_to_text() {
        assert_eq!(entity_to_text("short_wavelength"), "short wavelength");
        assert_eq!(entity_to_text("blue"), "blue");
    }

    #[test]
    fn test_article_for() {
        assert_eq!(article_for("apple"), "an");
        assert_eq!(article_for("blue"), "a");
        assert_eq!(article_for("orange"), "an");
    }

    #[test]
    fn test_mass_noun_has_no_indefinite_article() {
        let mut seen = HashSet::new();
        assert_eq!(with_article("evaporation", &mut seen), "evaporation");
    }

    #[test]
    fn test_linearize_empty_path() {
        let triples: Vec<Triple> = vec![];
        let entities: Vec<String> = vec![];
        let relations: Vec<String> = vec![];
        let result = linearize(&triples, &entities, &relations, Intent::Declarative);
        assert!(result.is_empty());
    }

    #[test]
    fn test_linearize_with_templates() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple("cat", "is", "animal");
        let mut bank = TemplateBank::new();
        bank.add_template(crate::templates::Template {
            pattern: "[SUBJ] is an [OBJ]".to_string(),
            frequency: 10,
            num_slots: 2,
            relation_hint: Some("is".to_string()),
        });
        let result = linearize_with_templates(&kg.triples, &kg.entities, &kg.relations, Intent::What, &bank);
        assert_eq!(result, "Cat is an animal.");
    }
}
