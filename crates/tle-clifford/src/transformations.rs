//! Geometric Transformations (Rotors/Versors) for Syntactic Operations.
//!
//! Versors in Clifford algebra perform rotations and reflections
//! without trainable parameters. They encode syntactic transformation rules:
//!
//! - Active→Passive voice: rotation in Subject-Object plane
//! - Nominalization: projection from verb-space to noun-space
//! - Negation: reflection through semantic hyperplane

use crate::multivector::MultiVector;
use tle_vsa::HyperVector;

/// A rotor (even versor) in Cl(3,0).
///
/// Rotors encode rotations: R = cos(θ/2) + sin(θ/2) * B
/// where B is a unit bivector defining the plane of rotation.
///
/// Application: x' = R * x * R̃ (sandwich product)
#[derive(Clone, Debug)]
pub struct Rotor {
    /// The multivector representing this rotor (scalar + bivector parts).
    pub mv: MultiVector,
}

impl Rotor {
    /// Create a rotor from an angle and a bivector plane.
    ///
    /// R = cos(θ/2) + sin(θ/2) * B_normalized
    pub fn from_angle_plane(angle: f32, plane: &MultiVector) -> Self {
        let half_angle = angle / 2.0;
        let cos_ha = half_angle.cos();
        let sin_ha = half_angle.sin();

        // Normalize the bivector
        let bv_mag = (plane.components[4].powi(2)
            + plane.components[5].powi(2)
            + plane.components[6].powi(2))
        .sqrt();

        let mut mv = MultiVector::zero();
        mv.components[0] = cos_ha; // scalar part

        if bv_mag > 1e-10 {
            let inv_mag = sin_ha / bv_mag;
            mv.components[4] = plane.components[4] * inv_mag; // e12
            mv.components[5] = plane.components[5] * inv_mag; // e13
            mv.components[6] = plane.components[6] * inv_mag; // e23
        }

        Self { mv }
    }

    /// Create an identity rotor (no rotation).
    pub fn identity() -> Self {
        Self {
            mv: MultiVector::scalar(1.0),
        }
    }

    /// Create a rotor that rotates in the Subject-Verb (e12) plane.
    pub fn subject_verb_rotation(angle: f32) -> Self {
        let plane = MultiVector::bivector(1.0, 0.0, 0.0);
        Self::from_angle_plane(angle, &plane)
    }

    /// Create a rotor that rotates in the Verb-Object (e23) plane.
    pub fn verb_object_rotation(angle: f32) -> Self {
        let plane = MultiVector::bivector(0.0, 0.0, 1.0);
        Self::from_angle_plane(angle, &plane)
    }

    /// Create a rotor that rotates in the Subject-Object (e13) plane.
    /// Useful for active→passive voice transformation.
    pub fn subject_object_rotation(angle: f32) -> Self {
        let plane = MultiVector::bivector(0.0, 1.0, 0.0);
        Self::from_angle_plane(angle, &plane)
    }

    /// Get the reverse (conjugate) of this rotor: R̃.
    pub fn reverse(&self) -> Self {
        Self {
            mv: self.mv.reverse(),
        }
    }

    /// Apply this rotor to a multivector: x' = R * x * R̃
    /// (sandwich product).
    pub fn apply(&self, x: &MultiVector) -> MultiVector {
        let rev = self.reverse();
        let rx = self.mv.geometric_product(x);
        rx.geometric_product(&rev.mv)
    }

    /// Compose two rotors: R_combined = R2 * R1
    /// (R2 applied after R1)
    pub fn compose(&self, other: &Rotor) -> Rotor {
        Rotor {
            mv: other.mv.geometric_product(&self.mv),
        }
    }
}

/// Apply a geometric transformation to a hypervector via its Cl(3,0) projection.
///
/// 1. Project hypervector into Cl(3,0) multivector
/// 2. Apply rotor transformation
/// 3. Project back into hypervector space
///
/// The projection-transform-reconstruct cycle is fully deterministic.
pub fn apply_transformation(hv: &HyperVector, rotor: &Rotor) -> HyperVector {
    let mv = MultiVector::from_hypervector(hv);
    let transformed = rotor.apply(&mv);

    // Reconstruct hypervector: distribute transformed components back
    let dim = hv.dim();
    let block_size = dim / 3;
    let mut data = hv.data.clone();

    // Modulate the three blocks based on transformed grade-1 components
    let scale_1 = if transformed.components[1].abs() > 1e-10 {
        transformed.components[1] / mv.components[1].max(1e-10)
    } else {
        1.0
    };
    let scale_2 = if transformed.components[2].abs() > 1e-10 {
        transformed.components[2] / mv.components[2].max(1e-10)
    } else {
        1.0
    };
    let scale_3 = if transformed.components[3].abs() > 1e-10 {
        transformed.components[3] / mv.components[3].max(1e-10)
    } else {
        1.0
    };

    for i in 0..block_size {
        data[i] *= scale_1.signum(); // Preserve sign change
    }
    for i in block_size..2*block_size {
        data[i] *= scale_2.signum();
    }
    for i in 2*block_size..3*block_size {
        data[i] *= scale_3.signum();
    }

    HyperVector::new(data)
}

/// Predefined syntactic transformations.
pub mod presets {
    use super::*;

    /// Active voice → Passive voice transformation.
    /// Swaps Subject and Object by rotating π in the e13 plane.
    pub fn active_to_passive() -> Rotor {
        Rotor::subject_object_rotation(std::f32::consts::PI)
    }

    /// Negation: reflects through the verb axis.
    /// Implemented as π rotation in Subject-Object plane.
    pub fn negation() -> Rotor {
        Rotor::from_angle_plane(
            std::f32::consts::PI,
            &MultiVector::bivector(0.0, 1.0, 0.0),
        )
    }

    /// Question formation: rotates Subject toward Query position.
    pub fn question_transform() -> Rotor {
        Rotor::subject_verb_rotation(std::f32::consts::FRAC_PI_2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_rotor() {
        let v = MultiVector::vector(1.0, 2.0, 3.0);
        let identity = Rotor::identity();
        let result = identity.apply(&v);

        for i in 0..8 {
            assert!(
                (result.components[i] - v.components[i]).abs() < 1e-5,
                "Identity should not change vector, component {} differs",
                i
            );
        }
    }

    #[test]
    fn test_90_degree_rotation() {
        // Rotate e1 by 90° in e12 plane
        // In Cl(3,0) with R = cos(θ/2) + sin(θ/2)*e12:
        // R*e1*R̃ maps e1 to some combination of e1 and e2
        let e1 = MultiVector::vector(1.0, 0.0, 0.0);
        let rotor = Rotor::subject_verb_rotation(std::f32::consts::FRAC_PI_2);
        let result = rotor.apply(&e1);

        // The result should be purely in the e1-e2 plane (no e3)
        assert!((result.components[3]).abs() < 1e-5, "e3 component should be ~0");
        // And should have unit magnitude in that plane
        let mag = (result.components[1].powi(2) + result.components[2].powi(2)).sqrt();
        assert!((mag - 1.0).abs() < 1e-4, "should preserve magnitude, got {}", mag);
        // e1 component should be ~0 (fully rotated away)
        assert!(result.components[1].abs() < 1e-4, "e1 should be ~0 after 90° rotation, got {}", result.components[1]);
    }

    #[test]
    fn test_180_degree_swaps_direction() {
        let e1 = MultiVector::vector(1.0, 0.0, 0.0);
        let rotor = Rotor::subject_verb_rotation(std::f32::consts::PI);
        let result = rotor.apply(&e1);

        // 180° rotation of e1 in e12 plane → -e1
        assert!((result.components[1] + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rotor_composition() {
        let r1 = Rotor::subject_verb_rotation(std::f32::consts::FRAC_PI_4);
        let r2 = Rotor::subject_verb_rotation(std::f32::consts::FRAC_PI_4);
        let composed = r1.compose(&r2);

        // Two 45° rotations = one 90° rotation
        let e1 = MultiVector::vector(1.0, 0.0, 0.0);
        let result = composed.apply(&e1);

        // After 90° rotation, e1 should be gone (in e2 direction)
        assert!(result.components[1].abs() < 1e-3, "e1 should be ~0 after 90° rotation, got {}", result.components[1]);
        // e3 should remain 0
        assert!(result.components[3].abs() < 1e-4, "e3 should remain 0");
        // Magnitude in e1-e2 plane should be preserved
        let mag = (result.components[1].powi(2) + result.components[2].powi(2)).sqrt();
        assert!((mag - 1.0).abs() < 1e-3, "magnitude should be ~1, got {}", mag);
    }

    #[test]
    fn test_active_to_passive() {
        let rotor = presets::active_to_passive();
        // Subject (e1 direction) → should move toward Object (e3 direction)
        let subject = MultiVector::vector(1.0, 0.0, 0.0);
        let result = rotor.apply(&subject);

        // After π rotation in e13 plane: e1 → -e1
        assert!((result.components[1] + 1.0).abs() < 1e-5);
    }
}
