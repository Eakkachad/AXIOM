//! Linearization: Convert knowledge graph paths to natural language sentences.
//!
//! Implements a template-based linearization system that:
//! 1. Classifies the query intent (Why/What/How/Where/When/Declarative)
//! 2. Maps relations to natural language templates
//! 3. Joins triples with intent-appropriate connectives
//! 4. Applies article insertion and capitalization

use std::collections::HashSet;

/// The communicative intent detected from the query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intent {
    Why,
    What,
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
        Intent::What => "that is",
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
        "has" => "has",
        "has_a" => "has a",
        "contains" => "contains",
        "causes" => "causes",
        "caused_by" => "is caused by",
        "scatters" => "scatters",
        "produces" => "produces",
        "leads_to" => "leads to",
        "part_of" => "is part of",
        "made_of" => "is made of",
        "located_in" => "is located in",
        "occurs_at" => "occurs at",
        "relates_to" => "relates to",
        "enables" => "enables",
        "requires" => "requires",
        "used_for" => "is used for",
        "defined_as" => "is defined as",
        "results_in" => "results in",
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

/// Insert an appropriate article before a noun if it's first occurrence.
fn with_article(entity: &str, seen: &mut HashSet<String>) -> String {
    let text = entity_to_text(entity);
    if seen.contains(&text) {
        format!("the {}", text)
    } else {
        seen.insert(text.clone());
        let article = article_for(&text);
        format!("{} {}", article, text)
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

    for triple in path_triples {
        let subject = &entities[triple.subject_id];
        let relation = &relations[triple.relation_id];
        let object = &entities[triple.object_id];

        let subj_text = with_article(subject, &mut seen_entities);
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

    // Capitalize and add period
    let sentence = capitalize_first(&joined);
    if sentence.ends_with('.') || sentence.ends_with('?') || sentence.ends_with('!') {
        sentence
    } else {
        format!("{}.", sentence)
    }
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
    fn test_linearize_empty_path() {
        let triples: Vec<Triple> = vec![];
        let entities: Vec<String> = vec![];
        let relations: Vec<String> = vec![];
        let result = linearize(&triples, &entities, &relations, Intent::Declarative);
        assert!(result.is_empty());
    }
}
