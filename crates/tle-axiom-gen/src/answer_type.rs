//! Answer-type prediction + typed final-hop expansion (T1.18c, OPI-style).
//!
//! The structural blind spot (Mode C, ~20% of AXIOM's selection failures) is a
//! gold answer that IS in the graph but has zero direct connectivity to the
//! query entities (conn=0), so it sits deep in the ranking. The verified fix
//! (OPI arXiv:2606.28076: +4.6/+8.9 Hit@1) is to widen — but ONLY along
//! **answer-type-compatible final hops**: predict the question's expected
//! answer type, then enumerate candidates whose connecting relation's tail
//! type matches it. Blind widening adds noise (QASA arXiv:2606.30133: T=4
//! degrades); typed widening stays precise.
//!
//! All deterministic, zero-training, CPU-only.

use crate::linearize::Intent;

/// Predicted answer type from question intent + surface words.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnswerType {
    Entity,
    Person,
    Place,
    Temporal,
    Number,
}

/// Predict the answer type from intent + question text (word rules only).
pub fn predict_answer_type(intent: Intent, query: &str) -> AnswerType {
    let q = query.to_lowercase();
    if intent == Intent::Where {
        return AnswerType::Place;
    }
    if intent == Intent::Who {
        return AnswerType::Person;
    }
    // "how many/how much" → Number (cardinality or count)
    if q.contains("how many") || q.contains("how much") || q.contains(" number ") {
        return AnswerType::Number;
    }
    // "in what year / which year / when / what date" → Temporal
    if q.contains("what year")
        || q.contains("which year")
        || q.contains("in what year")
        || q.contains("what date")
        || q.contains("when was")
        || q.contains("when did")
        || intent == Intent::When
    {
        return AnswerType::Temporal;
    }
    if q.contains("where") || q.contains(" in which city") || q.contains(" in which country") {
        return AnswerType::Place;
    }
    if q.contains("who") || q.contains("whose") || q.contains("which person") {
        return AnswerType::Person;
    }
    AnswerType::Entity
}

/// Type of the OBJECT of a relation (what kind of node the relation produces
/// as its tail). Place/Person are the discriminative kinds; the rest are
/// Entity (ambiguous).
pub fn relation_tail_kind(relation: &str) -> AnswerType {
    match relation {
        "located_in" | "located_at" | "located_near" | "capital_of" | "part_of"
        | "home_to" | "born_in" | "born_on" | "lived_in" | "died_in"
        | "took_place_in" | "occurred_in" | "founded_in" | "created_in"
        | "developed_in" | "released_in" | "published_in" | "from" | "happened_in" => {
            AnswerType::Place
        }
        "president_of" | "founder_of" | "leader_of" | "author_of" | "director_of"
        | "has_mother" | "has_father" | "has_parent" | "daughter_of" | "son_of"
        | "wife_of" | "husband_of" | "sister_of" | "brother_of" | "married_to"
        | "child_of" | "played_for" | "wrote" | "directed" | "painted"
        | "discovered" | "invented" | "founded" | "designed" | "built" | "created"
        | "developed" | "composed" | "starred" | "won" | "released" | "published"
        | "written_by" | "created_by" | "directed_by" | "played_by" | "portrayed_by"
        | "founded_by" | "invented_by" | "discovered_by" | "painted_by" | "built_by" => {
            AnswerType::Person
        }
        _ => AnswerType::Entity,
    }
}

/// Relations whose object is typically a YEAR (temporal attribute value).
pub fn relation_is_temporal(relation: &str) -> bool {
    matches!(
        relation,
        "released" | "published" | "founded" | "won" | "born_in" | "born_on"
            | "died_in" | "happened_in" | "founded_in" | "took_place_in" | "occurred_in"
            | "released_in" | "published_in" | "created_in" | "developed_in"
    )
}

/// Does this string look like a number / 4-digit year?
pub fn is_numeric_value(name: &str) -> bool {
    let cleaned: String = name.chars().filter(|c| c.is_ascii_digit()).collect();
    if cleaned.is_empty() {
        return false;
    }
    // allow pure digits (1-4 digits), or digit groups like "18th"/"1900s"
    let stripped: String = name
        .chars()
        .filter(|c| c.is_ascii_digit() || matches!(c, 't' | 'h' | 's' | ' ' | '-'))
        .collect();
    let digits = cleaned.len();
    let alpha_digitish = stripped.len() >= digits;
    alpha_digitish && digits >= 1 && digits <= 4
}

/// For the person-producing relations, which side is the person?
/// `Some(true)` = the person is the SUBJECT (e.g. "Tchaikovsky wrote X"),
/// `Some(false)` = the person is the OBJECT (e.g. "X has author Y").
/// `None` = not a person-producing relation.
pub fn relation_person_side(relation: &str) -> Option<bool> {
    match relation {
        "wrote" | "directed" | "painted" | "discovered" | "invented" | "founded"
        | "designed" | "built" | "created" | "developed" | "composed" | "starred"
        | "won" | "released" | "published" | "written_by" | "created_by"
        | "directed_by" | "played_by" | "portrayed_by" | "founded_by"
        | "invented_by" | "discovered_by" | "painted_by" | "built_by" => Some(true),
        "president_of" | "founder_of" | "leader_of" | "author_of" | "director_of"
        | "has_mother" | "has_father" | "has_parent" | "daughter_of" | "son_of"
        | "wife_of" | "husband_of" | "sister_of" | "brother_of" | "married_to"
        | "child_of" | "played_for" => Some(false),
        _ => None,
    }
}

/// Type match: does a candidate (with name `node_name`, reached via relation
/// `relation`) plausibly satisfy the predicted answer type?
pub fn matches_answer_type(
    predicted: AnswerType,
    relation: &str,
    node_name: &str,
) -> bool {
    match predicted {
        AnswerType::Place => {
            relation_tail_kind(relation) == AnswerType::Place
        }
        AnswerType::Person => {
            relation_tail_kind(relation) == AnswerType::Person
        }
        AnswerType::Number => is_numeric_value(node_name),
        AnswerType::Temporal => is_numeric_value(node_name) && relation_is_temporal(relation),
        AnswerType::Entity => true,
    }
}

/// Orientation-aware type match: `node_is_object` says whether the candidate
/// is the OBJECT of the triple (bridge, relation, candidate) or the SUBJECT
/// (candidate, relation, bridge). Fixes the `*_by` / "wrote" Person case where
/// the person is the SUBJECT, and Place where the place is the OBJECT. Returns
/// `false` for the non-discriminative Entity prediction (no signal — blanket
/// type-compatible boosts regress).
pub fn matches_answer_type_oriented(
    predicted: AnswerType,
    relation: &str,
    node_is_object: bool,
    node_name: &str,
) -> bool {
    match predicted {
        AnswerType::Place => {
            node_is_object && relation_tail_kind(relation) == AnswerType::Place
        }
        AnswerType::Person => match relation_person_side(relation) {
            Some(person_is_subject) => person_is_subject != node_is_object,
            None => false,
        },
        AnswerType::Number => is_numeric_value(node_name),
        AnswerType::Temporal => is_numeric_value(node_name) && relation_is_temporal(relation),
        AnswerType::Entity => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_type_from_intent_and_words() {
        assert_eq!(predict_answer_type(Intent::Where, "In which country is X?"), AnswerType::Place);
        assert_eq!(predict_answer_type(Intent::Who, "Who composed X?"), AnswerType::Person);
        assert_eq!(predict_answer_type(Intent::What, "How many teams end in United?"), AnswerType::Number);
        assert_eq!(predict_answer_type(Intent::What, "In what year was X founded?"), AnswerType::Temporal);
        assert_eq!(predict_answer_type(Intent::What, "What is the capital of X?"), AnswerType::Entity);
    }

    #[test]
    fn tail_kind_maps_relations() {
        assert_eq!(relation_tail_kind("located_in"), AnswerType::Place);
        assert_eq!(relation_tail_kind("capital_of"), AnswerType::Place);
        assert_eq!(relation_tail_kind("born_in"), AnswerType::Place);
        assert_eq!(relation_tail_kind("wrote"), AnswerType::Person);
        assert_eq!(relation_tail_kind("directed_by"), AnswerType::Person);
        assert_eq!(relation_tail_kind("has_relation"), AnswerType::Entity);
    }

    #[test]
    fn numeric_and_temporal_matching() {
        assert!(is_numeric_value("1998"));
        assert!(is_numeric_value("18th"));
        assert!(!is_numeric_value("France"));
        assert!(matches_answer_type(AnswerType::Number, "won", "3"));
        assert!(matches_answer_type(AnswerType::Temporal, "released", "1998"));
        assert!(!matches_answer_type(AnswerType::Temporal, "released", "France"));
        assert!(matches_answer_type(AnswerType::Place, "capital_of", "France"));
        assert!(!matches_answer_type(AnswerType::Place, "wrote", "France"));
    }
}
