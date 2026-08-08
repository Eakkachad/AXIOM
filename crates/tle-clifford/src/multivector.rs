//! Multivector representation for Clifford algebra Cl(3,0).
//!
//! A multivector in Cl(3,0) has 8 components:
//! - 1 scalar (grade 0)
//! - 3 vectors e1, e2, e3 (grade 1)
//! - 3 bivectors e12, e13, e23 (grade 2)
//! - 1 trivector e123 (grade 3)
//!
//! For linguistic applications, we extend to Cl(n,0) where n is derived
//! from the dimensionality of the VSA space (projected down).

use tle_vsa::HyperVector;

/// A multivector in Clifford algebra Cl(3,0).
///
/// Represents geometric objects that encode linguistic relationships.
/// The 8 components correspond to the full basis of Cl(3,0):
///
/// | Index | Basis    | Grade | Linguistic Interpretation |
/// |-------|----------|-------|---------------------------|
/// | 0     | 1        | 0     | Scalar (certainty weight) |
/// | 1     | e1       | 1     | Subject direction         |
/// | 2     | e2       | 1     | Verb direction            |
/// | 3     | e3       | 1     | Object direction          |
/// | 4     | e12      | 2     | Subject-Verb plane        |
/// | 5     | e13      | 2     | Subject-Object plane      |
/// | 6     | e23      | 2     | Verb-Object plane         |
/// | 7     | e123     | 3     | Full SVO volume           |
#[derive(Clone, Debug, PartialEq)]
pub struct MultiVector {
    /// The 8 components: [scalar, e1, e2, e3, e12, e13, e23, e123]
    pub components: [f32; 8],
}

impl MultiVector {
    /// Create a zero multivector.
    pub fn zero() -> Self {
        Self { components: [0.0; 8] }
    }

    /// Create a scalar multivector.
    pub fn scalar(s: f32) -> Self {
        let mut mv = Self::zero();
        mv.components[0] = s;
        mv
    }

    /// Create a grade-1 vector (e1, e2, e3).
    pub fn vector(e1: f32, e2: f32, e3: f32) -> Self {
        Self {
            components: [0.0, e1, e2, e3, 0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Create a bivector (e12, e13, e23).
    pub fn bivector(e12: f32, e13: f32, e23: f32) -> Self {
        Self {
            components: [0.0, 0.0, 0.0, 0.0, e12, e13, e23, 0.0],
        }
    }

    /// Create a trivector (pseudoscalar e123).
    pub fn trivector(e123: f32) -> Self {
        let mut mv = Self::zero();
        mv.components[7] = e123;
        mv
    }

    /// Geometric product of two multivectors in Cl(3,0).
    ///
    /// The geometric product combines the inner (dot) and outer (wedge) products.
    /// For basis vectors: e_i * e_i = 1 (positive definite metric)
    ///                     e_i * e_j = -e_j * e_i for i ≠ j
    pub fn geometric_product(&self, other: &Self) -> Self {
        let a = &self.components;
        let b = &other.components;

        // Full Cl(3,0) geometric product via multiplication table
        // Basis: {1, e1, e2, e3, e12, e13, e23, e123}
        let mut result = [0.0f32; 8];

        // scalar component
        result[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
                  - a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7];

        // e1 component
        result[1] = a[0]*b[1] + a[1]*b[0] - a[2]*b[4] - a[3]*b[5]
                  + a[4]*b[2] + a[5]*b[3] - a[6]*b[7] - a[7]*b[6];

        // e2 component
        result[2] = a[0]*b[2] + a[1]*b[4] + a[2]*b[0] - a[3]*b[6]
                  - a[4]*b[1] + a[5]*b[7] + a[6]*b[3] + a[7]*b[5];

        // e3 component
        result[3] = a[0]*b[3] + a[1]*b[5] + a[2]*b[6] + a[3]*b[0]
                  - a[4]*b[7] - a[5]*b[1] - a[6]*b[2] - a[7]*b[4];

        // e12 component
        result[4] = a[0]*b[4] + a[1]*b[2] - a[2]*b[1] + a[3]*b[7]
                  + a[4]*b[0] - a[5]*b[6] + a[6]*b[5] + a[7]*b[3];

        // e13 component
        result[5] = a[0]*b[5] + a[1]*b[3] - a[2]*b[7] - a[3]*b[1]
                  + a[4]*b[6] + a[5]*b[0] - a[6]*b[4] - a[7]*b[2];

        // e23 component
        result[6] = a[0]*b[6] + a[1]*b[7] + a[2]*b[3] - a[3]*b[2]
                  - a[4]*b[5] + a[5]*b[4] + a[6]*b[0] + a[7]*b[1];

        // e123 component (pseudoscalar)
        result[7] = a[0]*b[7] + a[1]*b[6] - a[2]*b[5] + a[3]*b[4]
                  + a[4]*b[3] - a[5]*b[2] + a[6]*b[1] + a[7]*b[0];

        Self { components: result }
    }

    /// Wedge (exterior) product: extracts only the grade-raising part.
    ///
    /// For two grade-1 vectors a and b:
    /// a ∧ b = a*b - a·b (the bivector part of the geometric product)
    ///
    /// Linguistically: a ∧ b represents the directed relationship
    /// between concepts a and b (e.g., Subject→Verb).
    pub fn wedge(&self, other: &Self) -> Self {
        // For general multivectors, wedge product keeps only
        // components whose grade = grade(a) + grade(b)
        let product = self.geometric_product(other);

        let grade_self = self.dominant_grade();
        let grade_other = other.dominant_grade();
        let target_grade = grade_self + grade_other;

        product.extract_grade(target_grade)
    }

    /// Inner (dot/contraction) product: extracts grade-lowering part.
    pub fn inner(&self, other: &Self) -> Self {
        let product = self.geometric_product(other);
        let grade_self = self.dominant_grade();
        let grade_other = other.dominant_grade();

        if grade_self == 0 || grade_other == 0 {
            return Self::zero();
        }

        let target_grade = (grade_self as i32 - grade_other as i32).unsigned_abs() as usize;
        product.extract_grade(target_grade)
    }

    /// Extract components of a specific grade.
    pub fn extract_grade(&self, grade: usize) -> Self {
        let mut result = Self::zero();
        match grade {
            0 => result.components[0] = self.components[0],
            1 => {
                result.components[1] = self.components[1];
                result.components[2] = self.components[2];
                result.components[3] = self.components[3];
            }
            2 => {
                result.components[4] = self.components[4];
                result.components[5] = self.components[5];
                result.components[6] = self.components[6];
            }
            3 => result.components[7] = self.components[7],
            _ => {} // Higher grades don't exist in Cl(3,0)
        }
        result
    }

    /// Determine the dominant grade (grade with largest magnitude).
    pub fn dominant_grade(&self) -> usize {
        let grade_mags = [
            self.components[0].abs(), // grade 0
            (self.components[1].powi(2) + self.components[2].powi(2) + self.components[3].powi(2)).sqrt(), // grade 1
            (self.components[4].powi(2) + self.components[5].powi(2) + self.components[6].powi(2)).sqrt(), // grade 2
            self.components[7].abs(), // grade 3
        ];

        grade_mags
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Reverse operation: reverses the order of basis vectors in each blade.
    /// For a k-blade: rev(B_k) = (-1)^{k(k-1)/2} * B_k
    pub fn reverse(&self) -> Self {
        Self {
            components: [
                self.components[0],    // grade 0: +1
                self.components[1],    // grade 1: +1
                self.components[2],
                self.components[3],
                -self.components[4],   // grade 2: -1
                -self.components[5],
                -self.components[6],
                -self.components[7],   // grade 3: -1
            ],
        }
    }

    /// Norm squared: M * M̃ (geometric product with reverse).
    pub fn norm_squared(&self) -> f32 {
        let rev = self.reverse();
        let product = self.geometric_product(&rev);
        product.components[0] // Scalar part of M * M̃
    }

    /// Magnitude (norm).
    pub fn magnitude(&self) -> f32 {
        self.norm_squared().abs().sqrt()
    }

    /// Convert from a hypervector by projection onto 3D subspace.
    /// Uses the first 3 principal components of the hypervector
    /// as the grade-1 vector components.
    pub fn from_hypervector(hv: &HyperVector) -> Self {
        let dim = hv.dim();
        // Project: sum blocks of D/3 dimensions into 3 components
        let block_size = dim / 3;
        let e1: f32 = hv.data[..block_size].iter().sum::<f32>() / (block_size as f32).sqrt();
        let e2: f32 = hv.data[block_size..2*block_size].iter().sum::<f32>() / (block_size as f32).sqrt();
        let e3: f32 = hv.data[2*block_size..3*block_size].iter().sum::<f32>() / (block_size as f32).sqrt();

        Self::vector(e1, e2, e3)
    }

    /// Add two multivectors.
    pub fn add(&self, other: &Self) -> Self {
        let mut result = [0.0f32; 8];
        for i in 0..8 {
            result[i] = self.components[i] + other.components[i];
        }
        Self { components: result }
    }

    /// Scale by a scalar.
    pub fn scale(&self, s: f32) -> Self {
        let mut result = [0.0f32; 8];
        for i in 0..8 {
            result[i] = self.components[i] * s;
        }
        Self { components: result }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basis_vector_squares() {
        // In Cl(3,0): e_i * e_i = 1
        let e1 = MultiVector::vector(1.0, 0.0, 0.0);
        let e1_sq = e1.geometric_product(&e1);
        assert!((e1_sq.components[0] - 1.0).abs() < 1e-6); // scalar = 1

        let e2 = MultiVector::vector(0.0, 1.0, 0.0);
        let e2_sq = e2.geometric_product(&e2);
        assert!((e2_sq.components[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_anticommutativity() {
        // e1 * e2 = -e2 * e1 = e12
        let e1 = MultiVector::vector(1.0, 0.0, 0.0);
        let e2 = MultiVector::vector(0.0, 1.0, 0.0);

        let e1e2 = e1.geometric_product(&e2);
        let e2e1 = e2.geometric_product(&e1);

        // e12 component should be +1 for e1*e2
        assert!((e1e2.components[4] - 1.0).abs() < 1e-6);
        // e12 component should be -1 for e2*e1
        assert!((e2e1.components[4] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_wedge_product_creates_bivector() {
        let a = MultiVector::vector(1.0, 0.0, 0.0); // e1
        let b = MultiVector::vector(0.0, 1.0, 0.0); // e2

        let ab = a.wedge(&b);
        // Should be pure bivector e12
        assert!((ab.components[4] - 1.0).abs() < 1e-6); // e12 = 1
        assert!(ab.components[0].abs() < 1e-6); // no scalar
        assert!(ab.components[1].abs() < 1e-6); // no e1
    }

    #[test]
    fn test_svo_trivector() {
        // Subject ∧ Verb ∧ Object = full oriented volume
        let s = MultiVector::vector(1.0, 0.0, 0.0);
        let v = MultiVector::vector(0.0, 1.0, 0.0);
        let o = MultiVector::vector(0.0, 0.0, 1.0);

        let sv = s.wedge(&v);
        let svo = sv.wedge(&o);

        // Should produce trivector e123
        assert!((svo.components[7] - 1.0).abs() < 1e-6);
    }
}
