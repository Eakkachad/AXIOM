//! # Clifford Algebra Syntax Engine
//!
//! Maps linguistic hypervectors into a geometric multivector space
//! for deterministic syntactic transformations.
//!
//! ## Mathematical Foundation
//!
//! We use Clifford algebra Cl(3,0) (Projective Geometric Algebra)
//! where elements are multivectors: scalars + vectors + bivectors + trivectors.
//!
//! Linguistic relationships are modeled as geometric transformations:
//! - **Subject-Verb**: Directed wedge product (x ∧ y) creates oriented plane
//! - **Verb-Object**: Another wedge product, creating a complementary plane
//! - **SVO structure**: Full trivector (x ∧ y ∧ z) = oriented volume
//! - **Transformations**: Versors (rotors) apply syntactic rewriting rules
//!
//! This replaces trained attention weights with deterministic geometric operations.

pub mod multivector;
pub mod syntax_node;
pub mod transformations;

pub use multivector::MultiVector;
pub use syntax_node::{SyntaxNode, SyntaxRelation};
pub use transformations::{Rotor, apply_transformation};
