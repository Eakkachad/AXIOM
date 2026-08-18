//! Clifford Geometric Algebra Cl(3,0) for Syntax-Preserving Non-Commutative Token Composition.
//!
//! Provides multivectors, grade projections, reversion involutions, and Rotor Sandwich
//! transformations R v R† on the Lie group Spin(3).
//!
//! Key Mathematical Guarantees:
//! - Norm Invariance: ||R v R†|| == ||v|| (Exact energy preservation).
//! - Grade Preservation: Scalar, Vector, Bivector, Pseudoscalar subspaces remain unmixed.
//! - Non-Commutativity: R_subj(R_verb(v)) != R_verb(R_subj(v)) (Strict syntax order preservation).

use std::f32::consts::PI;

/// Clifford Cl(3,0) Multivector with 8 basis components:
/// [1, e1, e2, e3, e12, e13, e23, e123]
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clifford3D {
    pub s: f32,       // Grade 0: Scalar (1)
    pub e1: f32,      // Grade 1: Vector X
    pub e2: f32,      // Grade 1: Vector Y
    pub e3: f32,      // Grade 1: Vector Z
    pub e12: f32,     // Grade 2: Bivector XY
    pub e13: f32,     // Grade 2: Bivector XZ
    pub e23: f32,     // Grade 2: Bivector YZ
    pub e123: f32,    // Grade 3: Pseudoscalar XYZ
}

impl Clifford3D {
    pub const ZERO: Self = Self {
        s: 0.0,
        e1: 0.0,
        e2: 0.0,
        e3: 0.0,
        e12: 0.0,
        e13: 0.0,
        e23: 0.0,
        e123: 0.0,
    };

    pub const IDENTITY: Self = Self {
        s: 1.0,
        e1: 0.0,
        e2: 0.0,
        e3: 0.0,
        e12: 0.0,
        e13: 0.0,
        e23: 0.0,
        e123: 0.0,
    };

    /// Construct a pure Grade-1 vector (x e1 + y e2 + z e3).
    #[inline(always)]
    pub fn new_vector(x: f32, y: f32, z: f32) -> Self {
        Self {
            s: 0.0,
            e1: x,
            e2: y,
            e3: z,
            e12: 0.0,
            e13: 0.0,
            e23: 0.0,
            e123: 0.0,
        }
    }

    /// Construct a pure Grade-2 bivector (b12 e12 + b13 e13 + b23 e23).
    #[inline(always)]
    pub fn new_bivector(b12: f32, b13: f32, b23: f32) -> Self {
        Self {
            s: 0.0,
            e1: 0.0,
            e2: 0.0,
            e3: 0.0,
            e12: b12,
            e13: b13,
            e23: b23,
            e123: 0.0,
        }
    }

    /// Full Geometric Product in Cl(3,0): u * v = u . v + u ^ v.
    #[inline]
    pub fn geometric_product(&self, rhs: &Self) -> Self {
        Self {
            s: self.s * rhs.s + self.e1 * rhs.e1 + self.e2 * rhs.e2 + self.e3 * rhs.e3
                - self.e12 * rhs.e12
                - self.e13 * rhs.e13
                - self.e23 * rhs.e23
                - self.e123 * rhs.e123,
            e1: self.s * rhs.e1 + self.e1 * rhs.s - self.e2 * rhs.e12 - self.e3 * rhs.e13
                + self.e12 * rhs.e2
                + self.e13 * rhs.e3
                - self.e23 * rhs.e123
                - self.e123 * rhs.e23,
            e2: self.s * rhs.e2 + self.e1 * rhs.e12 + self.e2 * rhs.s - self.e3 * rhs.e23
                - self.e12 * rhs.e1
                + self.e13 * rhs.e123
                + self.e23 * rhs.e3
                + self.e123 * rhs.e13,
            e3: self.s * rhs.e3 + self.e1 * rhs.e13 + self.e2 * rhs.e23 + self.e3 * rhs.s
                - self.e12 * rhs.e123
                - self.e13 * rhs.e1
                - self.e23 * rhs.e2
                - self.e123 * rhs.e12,
            e12: self.s * rhs.e12 + self.e1 * rhs.e2 - self.e2 * rhs.e1 + self.e3 * rhs.e123
                + self.e12 * rhs.s
                - self.e13 * rhs.e23
                + self.e23 * rhs.e13
                + self.e123 * rhs.e3,
            e13: self.s * rhs.e13 + self.e1 * rhs.e3 - self.e2 * rhs.e123 - self.e3 * rhs.e1
                + self.e12 * rhs.e23
                + self.e13 * rhs.s
                - self.e23 * rhs.e12
                + self.e123 * rhs.e2,
            e23: self.s * rhs.e23 + self.e1 * rhs.e123 + self.e2 * rhs.e3 - self.e3 * rhs.e2
                - self.e12 * rhs.e13
                + self.e13 * rhs.e12
                + self.e23 * rhs.s
                - self.e123 * rhs.e1,
            e123: self.s * rhs.e123 + self.e1 * rhs.e23 - self.e2 * rhs.e13 + self.e3 * rhs.e12
                + self.e12 * rhs.e3
                - self.e13 * rhs.e2
                + self.e23 * rhs.e1
                + self.e123 * rhs.s,
        }
    }

    /// Reversion Involution: M† (Grade 0, 1 unchanged; Grade 2, 3 negated).
    #[inline(always)]
    pub fn reverse(&self) -> Self {
        Self {
            s: self.s,
            e1: self.e1,
            e2: self.e2,
            e3: self.e3,
            e12: -self.e12,
            e13: -self.e13,
            e23: -self.e23,
            e123: -self.e123,
        }
    }

    /// Construct a Rotor R = exp(-B/2) from a unit bivector B and rotation angle theta.
    pub fn from_rotor_bivector(b12: f32, b13: f32, b23: f32, theta: f32) -> Self {
        let norm = (b12 * b12 + b13 * b13 + b23 * b23).sqrt();
        if norm < 1e-7 {
            return Self::IDENTITY;
        }
        let half_theta = theta * 0.5;
        let s = half_theta.cos();
        let scale = -half_theta.sin() / norm;
        Self {
            s,
            e1: 0.0,
            e2: 0.0,
            e3: 0.0,
            e12: b12 * scale,
            e13: b13 * scale,
            e23: b23 * scale,
            e123: 0.0,
        }
    }

    /// Rotor Sandwich Action: R * v * R† (Structure-preserving, grade-preserving, norm-preserving rotation).
    #[inline]
    pub fn rotate_sandwich(&self, v: &Self) -> Self {
        let r_v = self.geometric_product(v);
        r_v.geometric_product(&self.reverse())
    }

    /// Multivector Euclidean Squared Norm: ||M||² = Σ_i m_i².
    #[inline(always)]
    pub fn norm_squared(&self) -> f32 {
        self.s * self.s
            + self.e1 * self.e1
            + self.e2 * self.e2
            + self.e3 * self.e3
            + self.e12 * self.e12
            + self.e13 * self.e13
            + self.e23 * self.e23
            + self.e123 * self.e123
    }

    /// Inner Product (Scalar part of M * N†): <M N†>_0.
    #[inline(always)]
    pub fn inner_product(&self, other: &Self) -> f32 {
        self.s * other.s
            + self.e1 * other.e1
            + self.e2 * other.e2
            + self.e3 * other.e3
            + self.e12 * other.e12
            + self.e13 * other.e13
            + self.e23 * other.e23
            + self.e123 * other.e123
    }
}

/// Syntactic SVO Role Conjugation in Cl(3,0).
pub struct SyntacticRotorCodebook {
    pub subject_rotor: Clifford3D,
    pub verb_rotor: Clifford3D,
    pub object_rotor: Clifford3D,
}

impl SyntacticRotorCodebook {
    pub fn default_roles() -> Self {
        Self {
            // Three mutually orthogonal bivector planes in Cl(3,0)
            subject_rotor: Clifford3D::from_rotor_bivector(1.0, 0.0, 0.0, PI * 0.5),
            verb_rotor: Clifford3D::from_rotor_bivector(0.0, 1.0, 0.0, PI * 0.5),
            object_rotor: Clifford3D::from_rotor_bivector(0.0, 0.0, 1.0, PI * 0.5),
        }
    }

    /// Compose a directed Subject-Verb-Object triple into a single multivector.
    pub fn compose_svo(
        &self,
        subject: &Clifford3D,
        verb: &Clifford3D,
        object: &Clifford3D,
    ) -> Clifford3D {
        let s_bound = self.subject_rotor.rotate_sandwich(subject);
        let v_bound = self.verb_rotor.rotate_sandwich(verb);
        let o_bound = self.object_rotor.rotate_sandwich(object);

        s_bound
            .geometric_product(&v_bound)
            .geometric_product(&o_bound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clifford_geometric_product_and_reversion() {
        let e1 = Clifford3D::new_vector(1.0, 0.0, 0.0);
        let e2 = Clifford3D::new_vector(0.0, 1.0, 0.0);

        // e1 * e2 = e12 (Bivector)
        let e12 = e1.geometric_product(&e2);
        assert_eq!(e12.e12, 1.0);
        assert_eq!(e12.s, 0.0);

        // e2 * e1 = -e12 (Anticommutative)
        let e21 = e2.geometric_product(&e1);
        assert_eq!(e21.e12, -1.0);

        // Reversion of e12 is -e12
        let rev_e12 = e12.reverse();
        assert_eq!(rev_e12.e12, -1.0);
    }

    #[test]
    fn test_clifford_rotor_sandwich_preserves_norm_and_grade() {
        let rotor = Clifford3D::from_rotor_bivector(1.0, 1.0, 0.0, PI * 0.33);
        let v = Clifford3D::new_vector(3.0, 4.0, 5.0);

        let v_rotated = rotor.rotate_sandwich(&v);

        // Norm must be exactly preserved
        let diff_norm = (v_rotated.norm_squared() - v.norm_squared()).abs();
        assert!(
            diff_norm < 1e-5,
            "Rotor sandwich must preserve vector norm! Got diff {}",
            diff_norm
        );

        // Must remain pure Grade-1 vector within f32 numerical tolerance
        assert!(v_rotated.s.abs() < 1e-6);
        assert!(v_rotated.e12.abs() < 1e-6);
        assert!(v_rotated.e13.abs() < 1e-6);
        assert!(v_rotated.e23.abs() < 1e-6);
        assert!(v_rotated.e123.abs() < 1e-6);
    }

    #[test]
    fn test_clifford_syntactic_non_commutativity() {
        let codebook = SyntacticRotorCodebook::default_roles();
        let alice = Clifford3D::new_vector(1.0, 0.0, 0.0);
        let loves = Clifford3D::new_vector(0.0, 1.0, 0.0);
        let bob = Clifford3D::new_vector(0.0, 0.0, 1.0);

        // "Alice loves Bob"
        let alice_loves_bob = codebook.compose_svo(&alice, &loves, &bob);
        // "Bob loves Alice"
        let bob_loves_alice = codebook.compose_svo(&bob, &loves, &alice);

        // Must not be identical (Strict syntax directionality)
        let sim = alice_loves_bob.inner_product(&bob_loves_alice)
            / (alice_loves_bob.norm_squared().sqrt() * bob_loves_alice.norm_squared().sqrt());
        assert!(
            (sim - 1.0).abs() > 0.1,
            "SVO composition must distinguish Subject and Object roles! Similarity: {}",
            sim
        );
    }
}
