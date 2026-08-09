//! KnowledgePrior: steer VSA-LM generation toward fact-consistent text.
//!
//! The knowledge prior wraps a fact store `(subject, relation, object)` and,
//! given the recently generated context, boosts candidate tokens that are
//! *fact-connected* to entities mentioned in the context.
//!
//! **Entity-level matching:** a fact only fires when its FULL subject (or
//! object) entity appears in the context — not a partial word. This prevents
//! `wavelength` (part of `short_wavelength`) from falsely triggering facts
//! about `short_wavelength`.
//!
//! Example:
//! - context contains `cat` and fact `(cat, is, animal)` → boost `animal`
//! - then context contains `animal`, fact `(animal, has, heart)` → boost
//!   `heart` → **knowledge-guided multi-hop chaining**

use std::collections::HashMap;

/// A fact-aware candidate scorer for the VSA-LM.
#[derive(Debug, Clone, Default)]
pub struct KnowledgePrior {
    /// Full subject entity → list of (relation, object words)
    forward: Vec<(Vec<String>, String, Vec<String>)>,
    /// Full object entity → list of (relation, subject words)
    reverse: Vec<(Vec<String>, String, Vec<String>)>,
    /// Number of facts stored.
    pub facts: usize,
}

impl KnowledgePrior {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a fact triple. Entity names are split on spaces/underscores so
    /// `short_wavelength` becomes the words `["short", "wavelength"]` that the
    /// word-tokenizer can actually generate. Matching, however, requires the
    /// full word sequence of the entity to be present in the context.
    pub fn add_fact(&mut self, subject: &str, relation: &str, object: &str) {
        let subj_words = split_entity(subject);
        let obj_words = split_entity(object);
        if subj_words.is_empty() || obj_words.is_empty() {
            return;
        }
        self.forward
            .push((subj_words.clone(), relation.to_string(), obj_words.clone()));
        self.reverse
            .push((obj_words.clone(), relation.to_string(), subj_words.clone()));
        self.facts += 1;
    }

    /// Fact-connected candidate words for the given context words, with boost
    /// scores. Returns `(word, score)` pairs, highest score first.
    ///
    /// - A context that contains a full subject entity boosts its object words.
    /// - A context that contains a full object entity boosts its subject words
    ///   (handles "Who ..." / subject-answer questions).
    /// - A candidate word already present in the context is NOT re-boosted.
    pub fn candidates(&self, context: &[String]) -> Vec<(String, f32)> {
        let mut scores: HashMap<String, f32> = HashMap::new();
        for (subj_words, _, obj_words) in &self.forward {
            if !contains_all(context, subj_words) {
                continue;
            }
            for obj in obj_words {
                if !context.contains(obj) {
                    *scores.entry(obj.clone()).or_insert(0.0) += 2.0;
                }
            }
        }
        for (obj_words, _, subj_words) in &self.reverse {
            if !contains_all(context, obj_words) {
                continue;
            }
            for subj in subj_words {
                if !context.contains(subj) {
                    *scores.entry(subj.clone()).or_insert(0.0) += 1.5;
                }
            }
        }
        let mut ranked: Vec<(String, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Boost score for a specific candidate word given the context.
    pub fn score(&self, context: &[String], candidate: &str) -> f32 {
        let candidate_lower = candidate.to_lowercase();
        self.candidates(context)
            .into_iter()
            .find(|(w, _)| w == &candidate_lower)
            .map(|(_, s)| s)
            .unwrap_or(0.0)
    }
}

/// Does the context contain every word of the entity, consecutively?
fn contains_all(context: &[String], entity_words: &[String]) -> bool {
    if entity_words.is_empty() || entity_words.len() > context.len() {
        return false;
    }
    // Consecutive subsequence match.
    context.windows(entity_words.len()).any(|win| win == entity_words)
}

/// Split an entity name into lowercase word tokens on spaces and underscores.
fn split_entity(name: &str) -> Vec<String> {
    name.split(|c: char| c == ' ' || c == '_' || c == '-')
        .map(|w| w.to_lowercase())
        .filter(|w| !w.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_prior_boosts_object() {
        let mut kp = KnowledgePrior::new();
        kp.add_fact("sky", "is", "blue");
        kp.add_fact("blue", "has", "short_wavelength");

        let ctx = vec!["sky".to_string(), "is".to_string()];
        assert!(kp.score(&ctx, "blue") > 0.0);

        // Full entity "short wavelength" must be present to trigger the
        // short_wavelength fact — "blue" alone already fired blue's fact.
        let ctx2 = vec!["blue".to_string()];
        assert!(kp.score(&ctx2, "wavelength") > 0.0);

        let ctx3 = vec!["blue".to_string(), "has".to_string(), "short".to_string()];
        assert!(kp.score(&ctx3, "wavelength") > 0.0);
    }

    #[test]
    fn test_partial_word_does_not_trigger_fact() {
        let mut kp = KnowledgePrior::new();
        kp.add_fact("short_wavelength", "scatters", "in_atmosphere");
        // Only "wavelength" present — NOT the full "short wavelength" entity.
        let ctx = vec!["wavelength".to_string()];
        assert_eq!(kp.score(&ctx, "in"), 0.0, "partial entity must not fire the fact");
        assert_eq!(kp.score(&ctx, "atmosphere"), 0.0);
        // Full entity present → fires.
        let ctx2 = vec!["short".to_string(), "wavelength".to_string()];
        assert!(kp.score(&ctx2, "in") > 0.0);
    }

    #[test]
    fn test_knowledge_prior_reverse_handles_subject_answer() {
        let mut kp = KnowledgePrior::new();
        kp.add_fact("Einstein", "developed", "relativity");
        let ctx = vec!["relativity".to_string()];
        assert!(kp.score(&ctx, "einstein") > 0.0);
    }

    #[test]
    fn test_knowledge_prior_candidates_ranking() {
        let mut kp = KnowledgePrior::new();
        kp.add_fact("sky", "is", "blue");
        kp.add_fact("sky", "is", "gray");
        let ctx = vec!["sky".to_string()];
        let cands = kp.candidates(&ctx);
        assert!(!cands.is_empty());
        assert!(cands.iter().all(|(_, s)| *s > 0.0));
    }

    #[test]
    fn test_knowledge_prior_split_underscore_entities() {
        let mut kp = KnowledgePrior::new();
        kp.add_fact("short_wavelength", "scatters", "blue_light");
        let ctx = vec!["short".to_string(), "wavelength".to_string()];
        assert!(kp.score(&ctx, "light") > 0.0);
    }
}
