//! Multi-hop transition node — transitive reasoning via chained VSA transitions.
//!
//! Implements: T(A→B→C) = T(A→B) ⊗ π(T(B→C))
//!
//! This enables algebraic inference chains:
//!   "cat is_a animal" + "animal has heart" → "cat has heart"
//!
//! The key insight: by composing transitions with permutation,
//! we can represent multi-step reasoning paths in a single vector operation.

use tle_vsa::{bind, cosine_similarity, HyperVector};

use crate::node::{FlowNode, FlowState};

/// Multi-hop transition node.
///
/// Given multiple Transition Memories (one per relation type),
/// scores candidates by how well they complete a multi-hop chain
/// from the current context.
///
/// Example:
///   TM_is_a encodes "cat → animal", "dog → animal", ...
///   TM_has encodes "animal → heart", "animal → brain", ...
///   Query: "cat has ?" → chain is_a then has → "heart"
///
/// Math:
///   hop1 = π(current) ⊗ TM_is_a     → retrieve "animal"
///   hop2 = π(hop1_result) ⊗ TM_has   → retrieve "heart"
///   Score candidates by similarity to hop2
pub struct MultiHopNode {
    /// Transition memories for each hop (ordered: hop1, hop2, ...)
    pub hop_memories: Vec<HyperVector>,
    /// Weight multiplier for this node's score contribution.
    pub weight: f32,
    /// Maximum hops to attempt.
    pub max_hops: usize,
    /// Codebook vectors for intermediate resolution (optional).
    /// If provided, after each hop the result is cleaned up against these.
    pub codebook: Option<Vec<HyperVector>>,
}

impl MultiHopNode {
    /// Create a new multi-hop node with the given transition memories.
    ///
    /// Each TM represents a different relation type.
    /// The chain is: current → TM[0] → TM[1] → ... → candidates
    pub fn new(hop_memories: Vec<HyperVector>, weight: f32) -> Self {
        let max_hops = hop_memories.len();
        Self {
            hop_memories,
            weight,
            max_hops,
            codebook: None,
        }
    }

    /// Create with codebook for intermediate cleanup.
    pub fn with_codebook(
        hop_memories: Vec<HyperVector>,
        weight: f32,
        codebook: Vec<HyperVector>,
    ) -> Self {
        let max_hops = hop_memories.len();
        Self {
            hop_memories,
            weight,
            max_hops,
            codebook: Some(codebook),
        }
    }

    /// Compose a 2-hop transition: T(A→B→C) = T(A→B) ⊗ π(T(B→C))
    ///
    /// This is the core algebraic composition that enables transitive reasoning.
    pub fn compose_two_hop(tm_ab: &HyperVector, tm_bc: &HyperVector) -> HyperVector {
        let shifted_bc = tm_bc.permute(1);
        bind(tm_ab, &shifted_bc)
    }

    /// Resolve intermediate vector against codebook (nearest neighbor cleanup).
    fn cleanup(&self, noisy: &HyperVector) -> HyperVector {
        match &self.codebook {
            Some(codebook) => {
                let mut best_sim = f32::NEG_INFINITY;
                let mut best_vec = noisy.clone();

                for vec in codebook {
                    let sim = cosine_similarity(noisy, vec);
                    if sim > best_sim {
                        best_sim = sim;
                        best_vec = vec.clone();
                    }
                }
                best_vec
            }
            None => noisy.clone(),
        }
    }
}

impl FlowNode for MultiHopNode {
    fn transform(&self, mut state: FlowState) -> FlowState {
        if self.hop_memories.is_empty() {
            return state;
        }

        // Start from current vector
        let mut current = state.current.clone();

        // Chain through each hop
        for hop_tm in &self.hop_memories {
            // Unbind: π(current) ⊗ TM → predicted next
            let shifted = current.permute(1);
            let predicted = bind(&shifted, hop_tm);

            // Optional cleanup against codebook
            current = self.cleanup(&predicted);
        }

        // Score candidates by similarity to the final hop result
        for (i, candidate) in state.candidates.iter().enumerate() {
            let sim = cosine_similarity(&current, candidate);
            state.scores[i] += self.weight * sim;
        }

        state
    }
}

/// Build a composed multi-hop transition memory from individual hop TMs.
///
/// Given TM1 (A→B relations) and TM2 (B→C relations),
/// produces a single vector that directly encodes A→C.
///
/// This is the algebraic "shortcut" — instead of chaining at inference time,
/// pre-compute the composed transition once.
pub fn compose_transition_chain(tms: &[&HyperVector]) -> HyperVector {
    if tms.is_empty() {
        panic!("Cannot compose empty chain");
    }
    if tms.len() == 1 {
        return tms[0].clone();
    }

    let mut composed = tms[0].clone();
    for tm in &tms[1..] {
        // Compose: existing ⊗ π(next_tm)
        let shifted = tm.permute(1);
        composed = bind(&composed, &shifted);
    }
    composed
}

#[cfg(test)]
mod tests {
    use super::*;
    use tle_vsa::HyperVector;

    #[test]
    fn test_two_hop_composition() {
        let dim = 2048;

        // Create entities
        let cat = HyperVector::random_bipolar(dim, 1);
        let animal = HyperVector::random_bipolar(dim, 2);
        let heart = HyperVector::random_bipolar(dim, 3);
        let rock = HyperVector::random_bipolar(dim, 99); // distractor

        // Build TM_is_a: cat → animal
        let tm_isa = bind(&cat.permute(1), &animal);

        // Build TM_has: animal → heart
        let tm_has = bind(&animal.permute(1), &heart);

        // Multi-hop: cat → ? → ? (should reach heart)
        let node = MultiHopNode::new(vec![tm_isa, tm_has], 1.0);

        let mut state = FlowState::new(dim);
        state.current = cat.clone();
        state.candidates = vec![heart.clone(), rock.clone(), animal.clone()];
        state.scores = vec![0.0, 0.0, 0.0];

        let result = node.transform(state);

        // "heart" should score highest (cat is_a animal, animal has heart)
        assert!(
            result.scores[0] > result.scores[1],
            "heart ({}) should score higher than rock ({})",
            result.scores[0], result.scores[1]
        );
    }

    #[test]
    fn test_compose_transition_chain() {
        let dim = 2048;

        let a = HyperVector::random_bipolar(dim, 10);
        let b = HyperVector::random_bipolar(dim, 20);
        let c = HyperVector::random_bipolar(dim, 30);

        // A→B
        let tm_ab = bind(&a.permute(1), &b);
        // B→C
        let tm_bc = bind(&b.permute(1), &c);

        // Compose A→C
        let composed = compose_transition_chain(&[&tm_ab, &tm_bc]);

        // The composed vector should exist and have correct dimension
        assert_eq!(composed.dim(), dim);
        // It should not be zero
        assert!(composed.norm() > 0.0);
    }

    #[test]
    fn test_single_hop_equivalent_to_transition() {
        let dim = 2048;

        let current = HyperVector::random_bipolar(dim, 1);
        let target = HyperVector::random_bipolar(dim, 2);
        let distractor = HyperVector::random_bipolar(dim, 3);

        // Single transition: current → target
        let tm = bind(&current.permute(1), &target);

        // Use MultiHopNode with 1 hop — should behave like TransitionScoreNode
        let node = MultiHopNode::new(vec![tm], 1.0);

        let mut state = FlowState::new(dim);
        state.current = current;
        state.candidates = vec![target.clone(), distractor.clone()];
        state.scores = vec![0.0, 0.0];

        let result = node.transform(state);

        assert!(
            result.scores[0] > result.scores[1],
            "target ({}) should score higher than distractor ({})",
            result.scores[0], result.scores[1]
        );
    }
}
