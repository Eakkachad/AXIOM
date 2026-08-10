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
//! **O(1) hash index:** facts are indexed by their first entity word for
//! sub-linear lookup instead of O(F) linear scan.  Scales to 100K+ facts.

use std::collections::HashMap;

/// A fact-aware candidate scorer for the VSA-LM.
#[derive(Debug, Clone, Default)]
pub struct KnowledgePrior {
    /// Full subject entity → (relation, object words) indexed by first word.
    forward: Vec<(Vec<String>, String, Vec<String>)>,
    forward_index: HashMap<String, Vec<usize>>,
    /// Full object entity → (relation, subject words) indexed by first word.
    reverse: Vec<(Vec<String>, String, Vec<String>)>,
    reverse_index: HashMap<String, Vec<usize>>,
    /// Number of facts stored.
    pub facts: usize,
}

impl KnowledgePrior {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a fact triple. Entity names are split on spaces/underscores so
    /// `short_wavelength` becomes the words `["short", "wavelength"]` that the
    /// word-tokenizer can actually generate.
    pub fn add_fact(&mut self, subject: &str, relation: &str, object: &str) {
        let subj_words = split_entity(subject);
        let obj_words = split_entity(object);
        if subj_words.is_empty() || obj_words.is_empty() { return; }
        let idx = self.forward.len();
        self.forward.push((subj_words.clone(), relation.to_string(), obj_words.clone()));
        self.forward_index.entry(subj_words[0].clone()).or_default().push(idx);
        self.reverse.push((obj_words.clone(), relation.to_string(), subj_words.clone()));
        self.reverse_index.entry(obj_words[0].clone()).or_default().push(idx);
        self.facts += 1;
    }

    /// Fact-connected candidate words for the given context words, with boost
    /// scores. Uses first-word hash index for O(context_words × facts_per_word)
    /// lookup instead of O(F) linear scan.
    pub fn candidates(&self, context: &[String]) -> Vec<(String, f32)> {
        let mut scores: HashMap<String, f32> = HashMap::new();
        for ctx_word in context {
            if let Some(idxs) = self.forward_index.get(ctx_word) {
                for &idx in idxs {
                    let (subj_words, _, obj_words) = &self.forward[idx];
                    if !contains_all(context, subj_words) { continue; }
                    for obj in obj_words {
                        if !context.contains(obj) {
                            *scores.entry(obj.clone()).or_insert(0.0) += 2.0;
                        }
                    }
                }
            }
            if let Some(idxs) = self.reverse_index.get(ctx_word) {
                for &idx in idxs {
                    let (obj_words, _, subj_words) = &self.reverse[idx];
                    if !contains_all(context, obj_words) { continue; }
                    for subj in subj_words {
                        if !context.contains(subj) {
                            *scores.entry(subj.clone()).or_insert(0.0) += 1.5;
                        }
                    }
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
    if entity_words.is_empty() || entity_words.len() > context.len() { return false; }
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
        let ctx2 = vec!["blue".to_string()];
        assert!(kp.score(&ctx2, "wavelength") > 0.0);
        let ctx3 = vec!["blue".to_string(), "has".to_string(), "short".to_string()];
        assert!(kp.score(&ctx3, "wavelength") > 0.0);
    }

    #[test]
    fn test_partial_word_does_not_trigger_fact() {
        let mut kp = KnowledgePrior::new();
        kp.add_fact("short_wavelength", "scatters", "in_atmosphere");
        let ctx = vec!["wavelength".to_string()];
        assert_eq!(kp.score(&ctx, "in"), 0.0);
        assert_eq!(kp.score(&ctx, "atmosphere"), 0.0);
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
