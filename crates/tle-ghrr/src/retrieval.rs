//! Calibrated path scoring + relation-schema IDF (PathHD Eqs 5-6).
//!
//! `s(z) = sim(v_q, v_z) + α·IDF(z) − β·λ^|z|` with λ<1, so longer paths are
//! penalized LESS (counteracts accumulated binding noise). IDF is computed
//! training-free over the evidence graph's relation/schema frequency.

use std::collections::HashMap;

/// Training-free IDF over relation and ordered relation-sequence frequency.
#[derive(Debug, Clone, Default)]
pub struct RelationSchemaIndex {
    rel_freq: HashMap<String, usize>,
    seq_freq: HashMap<String, usize>,
    total: usize,
}

impl RelationSchemaIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Count occurrences of a single relation (for 1-hop paths).
    pub fn count(&mut self, relation: &str) {
        *self.rel_freq.entry(relation.to_string()).or_insert(0) += 1;
        self.total += 1;
    }

    /// Count occurrences of an ordered relation sequence (for 2-hop paths).
    pub fn count_seq(&mut self, r1: &str, r2: &str) {
        let key = format!("{r1}|{r2}");
        *self.seq_freq.entry(key).or_insert(0) += 1;
    }

    /// `idf(r) = log(1 + total / (1 + freq(r)))` — rare schemas get a bonus.
    pub fn idf(&self, relation: &str) -> f32 {
        let f = self.rel_freq.get(relation).copied().unwrap_or(0);
        (1.0 + self.total as f32 / (1.0 + f as f32)).ln()
    }

    /// `idf` for an ordered 2-hop sequence.
    pub fn idf_seq(&self, r1: &str, r2: &str) -> f32 {
        let key = format!("{r1}|{r2}");
        let f = self.seq_freq.get(&key).copied().unwrap_or(0);
        (1.0 + self.total as f32 / (1.0 + f as f32)).ln()
    }
}

/// Calibrated score `sim + α·idf − β·λ^|z|` (Table-11 defaults: α=0.2, β=0.1,
/// λ=0.8).
pub fn calibrated_score(sim: f32, idf: f32, path_len: usize, alpha: f32, beta: f32, lambda: f32) -> f32 {
    sim + alpha * idf - beta * lambda.powi(path_len as i32)
}

/// Map a set of per-path scores to a single entity-level score: the max
/// calibrated score over all the entity's candidate paths.
pub fn path_scores_to_entity(scores: impl IntoIterator<Item = f32>) -> f32 {
    scores.into_iter().fold(f32::NEG_INFINITY, |a, b| a.max(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idf_rewards_rare_relations() {
        let mut idx = RelationSchemaIndex::new();
        for _ in 0..20 {
            idx.count("mentions");
        }
        idx.count("capital_of");
        assert!(idx.idf("capital_of") > idx.idf("mentions"));
    }

    #[test]
    fn calibrated_length_term() {
        // λ<1 ⇒ longer path penalty is SMALLER (0.8^2 < 0.8^1)
        let s1 = calibrated_score(0.5, 0.5, 1, 0.2, 0.1, 0.8);
        let s2 = calibrated_score(0.5, 0.5, 2, 0.2, 0.1, 0.8);
        assert!(s2 > s1, "2-hop must be penalized less: {s1} vs {s2}");
    }

    #[test]
    fn entity_score_is_max() {
        let v = path_scores_to_entity(vec![0.1, 0.7, 0.3]);
        assert!((v - 0.7).abs() < 1e-6);
    }
}
