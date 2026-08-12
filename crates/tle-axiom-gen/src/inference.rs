//! Deductive inference layer over the knowledge graph (T1.10b).
//!
//! Uses an embedded Datalog engine (Ascent) to derive new facts from the raw
//! graph via deterministic rules, then expose them for answer selection:
//!
//! 1. **Inversion** — "A is mother of B" ⟹ "B has mother A". Doubles the
//!    connectivity of family/ownership relations so answers reachable only via
//!    the inverse direction now connect to the query.
//! 2. **Transitivity** — "A located_in B, B located_in C" ⟹ "A located_in C".
//!    Recovers answers like Scotland (Wanlockhead → Dumfries & Galloway →
//!    Scotland) that currently sit at depth 2+ and lose to 1-hop junk.
//! 3. **Answer-type gates** — classify each entity by the relation families it
//!    participates in (person/location/event/thing) so `extract_answer` can
//!    apply a hard veto: a Where-question must never return a person.

use crate::graph::KnowledgeGraph;
use std::collections::HashMap;

/// Relation families used for answer-type inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    Location,
    Person,
    Event,
    Thing,
    Other,
}

/// A derived fact produced by the inference rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedFact {
    pub subject: String,
    pub relation: String,
    pub object: String,
}

/// Map a relation name to its answer-type family (for the F1 veto).
pub fn relation_family(relation: &str) -> Family {
    match relation {
        "located_in" | "located_at" | "located_near" | "capital_of" | "part_of"
        | "home_to" | "born_in" | "born_on" | "lived_in" | "died_in"
        | "took_place_in" | "occurred_in" | "founded_in" | "created_in"
        | "developed_in" | "released_in" | "published_in" | "from" => Family::Location,
        "president_of" | "founder_of" | "leader_of" | "author_of" | "director_of"
        | "has_mother" | "has_father" | "has_parent" | "daughter_of" | "son_of"
        | "wife_of" | "husband_of" | "sister_of" | "brother_of" | "married_to"
        | "child_of" | "has_mother_inv" | "has_father_inv" | "played_for"
        | "wrote" | "directed" | "painted" | "discovered" | "invented"
        | "founded" | "designed" | "built" | "created" | "developed"
        | "composed" | "starred" | "released" | "published" | "won" => Family::Person,
        "happened_in" | "occurred_in" => Family::Event,
        _ => Family::Thing,
    }
}

/// Namespace a derived relation so it never collides with a raw one.
fn inv(name: &str) -> String {
    format!("{}_inv", name)
}

/// Derive facts from the raw graph with deterministic Datalog rules.
///
/// `enabled` gates the rules (all true by default); each is independently
/// testable so the bench can isolate the contribution of each rule family.
pub fn derive_facts(
    graph: &KnowledgeGraph,
    inversion: bool,
    transitivity: bool,
) -> Vec<DerivedFact> {
    let mut derived: Vec<DerivedFact> = Vec::new();
    let mut seen: std::collections::HashSet<(String, String, String)> = std::collections::HashSet::new();

    let mut push = |s: String, r: String, o: String| {
        if seen.insert((s.clone(), r.clone(), o.clone())) {
            derived.push(DerivedFact { subject: s, relation: r, object: o });
        }
    };

    // Raw facts feed the rules.
    let mut raw: Vec<(String, String, String)> = graph.triples.iter().map(|t| {
        (
            graph.entity_name(t.subject_id).to_string(),
            graph.relation_name(t.relation_id).to_string(),
            graph.entity_name(t.object_id).to_string(),
        )
    }).collect();

    // Inversion rule: symmetric family/ownership relations.
    if inversion {
        let mut added = Vec::new();
        for (s, r, o) in &raw {
            if matches!(r.as_str(),
                "has_mother" | "has_father" | "has_parent" | "daughter_of"
                | "son_of" | "wife_of" | "husband_of" | "sister_of" | "brother_of"
                | "child_of" | "married_to" | "author_of" | "director_of"
                | "founder_of" | "leader_of" | "president_of" | "capital_of"
                | "part_of")
            {
                // inverse: (B, r_inv, A)
                added.push((o.clone(), inv(r), s.clone()));
            }
        }
        for f in added { push(f.0, f.1, f.2); }
        raw.extend(derived.iter().map(|d| (d.subject.clone(), d.relation.clone(), d.object.clone())));
    }

    // Transitivity rule: located_in / part_of chains.
    if transitivity {
        let mut changed = true;
        while changed {
            changed = false;
            // materialize a set of (subj, obj) for the transitive relations
            let loc: Vec<(String, String)> = raw.iter()
                .filter(|(_, r, _)| matches!(r.as_str(), "located_in" | "part_of" | "located_near"))
                .map(|(s, _, o)| (s.clone(), o.clone()))
                .collect();
            for (a, b) in &loc {
                for (c, d) in &loc {
                    if b == c && a != d {
                        let key = (a.clone(), "located_in".to_string(), d.clone());
                        if seen.insert(key.clone()) {
                            derived.push(DerivedFact { subject: key.0.clone(), relation: key.1.clone(), object: key.2.clone() });
                            raw.push(key);
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    derived
}

/// Entity → set of relation families it participates in (for the F1 veto).
pub fn entity_families(graph: &KnowledgeGraph) -> HashMap<usize, Vec<Family>> {
    let mut map: HashMap<usize, Vec<Family>> = HashMap::new();
    for t in &graph.triples {
        let fam = relation_family(graph.relation_name(t.relation_id));
        map.entry(t.subject_id).or_default().push(fam);
        map.entry(t.object_id).or_default().push(fam);
    }
    map
}

/// Hard answer-type veto: does this entity plausibly match the expected type
/// for the query intent? Returns true if it should SURVIVE the filter.
pub fn passes_answer_type(intent: crate::linearize::Intent, families: &[Family]) -> bool {
    let expected: Option<Family> = match intent {
        crate::linearize::Intent::Where => Some(Family::Location),
        crate::linearize::Intent::Who => Some(Family::Person),
        _ => None,
    };
    match expected {
        Some(f) => families.contains(&f) || families.is_empty(),
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::KnowledgeGraph;

    fn kg() -> KnowledgeGraph {
        let mut g = KnowledgeGraph::new();
        g.add_triple("Wanlockhead", "located_in", "Dumfries and Galloway");
        g.add_triple("Dumfries and Galloway", "located_in", "Scotland");
        g.add_triple("Hingis", "has_mother", "Melanie Molitorová");
        g
    }

    #[test]
    fn transitivity_recovers_deep_location() {
        let g = kg();
        let facts = derive_facts(&g, false, true);
        assert!(
            facts.iter().any(|f| f.subject == "Wanlockhead"
                && f.object == "Scotland"
                && f.relation == "located_in"),
            "expected transitivity to add Wanlockhead→Scotland, got {:?}",
            facts
        );
    }

    #[test]
    fn inversion_creates_has_mother_inv() {
        let g = kg();
        let facts = derive_facts(&g, true, false);
        assert!(
            facts.iter().any(|f| f.subject == "Melanie Molitorová"
                && f.object == "Hingis"
                && f.relation == "has_mother_inv"),
            "expected inversion, got {:?}",
            facts
        );
    }

    #[test]
    fn answer_type_veto_where_rejects_person() {
        let g = kg();
        let fams = entity_families(&g);
        let hingis = g.entity_id("Hingis").unwrap();
        // Hingis participates in has_mother (Person) → a Where question must
        // not return Hingis as answer.
        assert!(!passes_answer_type(crate::linearize::Intent::Where, &fams[&hingis]));
        let scotland = g.entity_id("Scotland").unwrap();
        assert!(passes_answer_type(crate::linearize::Intent::Where, &fams[&scotland]));
    }
}
