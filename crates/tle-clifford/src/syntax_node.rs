//! Syntax Node: Processes linguistic structure using geometric algebra.
//!
//! A syntax node receives hypervectors representing words/phrases and
//! computes their geometric relationships to determine syntactic structure.

use tle_vsa::HyperVector;
use crate::multivector::MultiVector;

/// Types of syntactic relations that can be detected.
#[derive(Clone, Debug, PartialEq)]
pub enum SyntaxRelation {
    /// Subject-Verb relation (directed: subject acts via verb)
    SubjectVerb,
    /// Verb-Object relation (directed: verb acts on object)
    VerbObject,
    /// Subject-Verb-Object complete clause
    Clause,
    /// Modifier relation (adjective-noun, adverb-verb)
    Modifier,
    /// Coordination (and, or, but)
    Coordination,
    /// Subordination (because, although, when)
    Subordination,
    /// Unknown/unclassified relation
    Unknown,
}

/// A deterministic syntax processing node.
///
/// Replaces learned attention heads with geometric algebra operations.
/// The node takes role-bound hypervectors and determines their
/// syntactic relationships using wedge products and geometric classification.
#[derive(Clone)]
pub struct SyntaxNode {
    /// Minimum bivector magnitude to classify as a valid relation.
    pub relation_threshold: f32,
    /// Minimum trivector magnitude for clause detection.
    pub clause_threshold: f32,
}

impl SyntaxNode {
    /// Create a new syntax node with default thresholds.
    pub fn new() -> Self {
        Self {
            relation_threshold: 0.3,
            clause_threshold: 0.2,
        }
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(relation_threshold: f32, clause_threshold: f32) -> Self {
        Self {
            relation_threshold,
            clause_threshold,
        }
    }

    /// Detect the syntactic relation between two hypervectors.
    ///
    /// Projects both vectors into Cl(3,0) and computes their wedge product.
    /// The resulting bivector's orientation determines the relation type.
    pub fn detect_relation(&self, a: &HyperVector, b: &HyperVector) -> (SyntaxRelation, f32) {
        let mv_a = MultiVector::from_hypervector(a);
        let mv_b = MultiVector::from_hypervector(b);

        // Compute wedge product (oriented area between vectors)
        let wedge = mv_a.wedge(&mv_b);

        // Extract bivector components
        let e12 = wedge.components[4]; // Subject-Verb plane
        let e13 = wedge.components[5]; // Subject-Object plane
        let e23 = wedge.components[6]; // Verb-Object plane

        let mag = (e12*e12 + e13*e13 + e23*e23).sqrt();

        if mag < self.relation_threshold {
            return (SyntaxRelation::Unknown, mag);
        }

        // Classify based on dominant bivector component
        let abs_e12 = e12.abs();
        let abs_e13 = e13.abs();
        let abs_e23 = e23.abs();

        let relation = if abs_e12 >= abs_e13 && abs_e12 >= abs_e23 {
            SyntaxRelation::SubjectVerb
        } else if abs_e23 >= abs_e12 && abs_e23 >= abs_e13 {
            SyntaxRelation::VerbObject
        } else {
            SyntaxRelation::Modifier
        };

        (relation, mag)
    }

    /// Detect a full clause (Subject-Verb-Object) from three hypervectors.
    ///
    /// Computes the trivector a ∧ b ∧ c. A non-zero result indicates
    /// the three concepts span a full clause volume.
    pub fn detect_clause(
        &self,
        subject: &HyperVector,
        verb: &HyperVector,
        object: &HyperVector,
    ) -> (bool, f32) {
        let mv_s = MultiVector::from_hypervector(subject);
        let mv_v = MultiVector::from_hypervector(verb);
        let mv_o = MultiVector::from_hypervector(object);

        // Compute S ∧ V
        let sv = mv_s.wedge(&mv_v);
        // Compute (S ∧ V) ∧ O
        let svo = sv.wedge(&mv_o);

        // Trivector magnitude
        let trivector_mag = svo.components[7].abs();
        let is_clause = trivector_mag > self.clause_threshold;

        (is_clause, trivector_mag)
    }

    /// Parse a sequence of hypervectors into syntactic structure.
    ///
    /// Returns pairs of (relation_type, strength) for consecutive elements.
    /// This is the deterministic replacement for attention scoring.
    pub fn parse_sequence(&self, vectors: &[&HyperVector]) -> Vec<(SyntaxRelation, f32)> {
        if vectors.len() < 2 {
            return Vec::new();
        }

        vectors
            .windows(2)
            .map(|pair| self.detect_relation(pair[0], pair[1]))
            .collect()
    }

    /// Compute the "syntactic energy" of a sequence.
    ///
    /// Higher energy = more well-formed syntactic structure.
    /// This is analogous to attention entropy but computed geometrically.
    pub fn syntactic_energy(&self, vectors: &[&HyperVector]) -> f32 {
        let relations = self.parse_sequence(vectors);
        if relations.is_empty() {
            return 0.0;
        }

        // Energy = sum of relation magnitudes weighted by type coherence
        let mut energy = 0.0f32;
        for (relation, mag) in &relations {
            let type_weight = match relation {
                SyntaxRelation::SubjectVerb => 1.5,
                SyntaxRelation::VerbObject => 1.5,
                SyntaxRelation::Clause => 2.0,
                SyntaxRelation::Modifier => 1.0,
                SyntaxRelation::Coordination => 0.8,
                SyntaxRelation::Subordination => 1.2,
                SyntaxRelation::Unknown => 0.1,
            };
            energy += mag * type_weight;
        }

        energy / relations.len() as f32
    }
}

impl Default for SyntaxNode {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tle_vsa::DEFAULT_DIM;

    #[test]
    fn test_different_vectors_have_relation() {
        let a = HyperVector::random_bipolar(DEFAULT_DIM, 100);
        let b = HyperVector::random_bipolar(DEFAULT_DIM, 200);

        let node = SyntaxNode::new();
        let (relation, mag) = node.detect_relation(&a, &b);

        // Two random vectors should produce some geometric relationship
        // (their projections into 3D won't be parallel)
        assert!(mag > 0.0, "Random vectors should have non-zero wedge product");
        assert_ne!(relation, SyntaxRelation::Unknown);
    }

    #[test]
    fn test_clause_detection() {
        let s = HyperVector::random_bipolar(DEFAULT_DIM, 10);
        let v = HyperVector::random_bipolar(DEFAULT_DIM, 20);
        let o = HyperVector::random_bipolar(DEFAULT_DIM, 30);

        let node = SyntaxNode::new();
        let (is_clause, vol) = node.detect_clause(&s, &v, &o);

        // Three independent random vectors should span a volume
        assert!(vol > 0.0);
        // Whether it passes threshold depends on projection magnitudes
        println!("Clause volume: {}, is_clause: {}", vol, is_clause);
    }

    #[test]
    fn test_deterministic_parsing() {
        let vectors: Vec<HyperVector> = (0..5)
            .map(|i| HyperVector::random_bipolar(DEFAULT_DIM, i * 100))
            .collect();
        let refs: Vec<&HyperVector> = vectors.iter().collect();

        let node = SyntaxNode::new();
        let result1 = node.parse_sequence(&refs);
        let result2 = node.parse_sequence(&refs);

        // Must be deterministic
        assert_eq!(result1.len(), result2.len());
        for (r1, r2) in result1.iter().zip(result2.iter()) {
            assert_eq!(r1.0, r2.0);
            assert!((r1.1 - r2.1).abs() < 1e-7);
        }
    }
}
