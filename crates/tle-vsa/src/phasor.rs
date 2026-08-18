//! Continuous Phasor VSA (Fourier Holographic Reduced Representation / FHRR on Torus T^D).
//!
//! Replaces discrete bipolar hypervectors with continuous phase angles θ_d ∈ [-π, π).
//! Key Mathematical Guarantees:
//! - Exact Unitary Invertibility: z* ⊙ (z ⊙ w) ≡ w with ZERO quantization distortion.
//! - Continuous Fractional Shift: z^α parameterizes continuous group homomorphisms.
//! - Smooth Rayleigh phase noise distribution under circular mean bundling.

use std::f32::consts::PI;

/// Continuous Phasor Hypervector on the Torus T^D = (S^1)^D.
/// Each component is a pure phase angle θ_d ∈ [-π, π), representing unit complex number e^{i θ_d}.
#[derive(Debug, Clone, PartialEq)]
pub struct PhasorVector {
    pub dim: usize,
    pub phases: Vec<f32>,
}

impl PhasorVector {
    /// Create a zero-phase identity vector (all θ_d = 0.0, representing 1 + 0i).
    pub fn identity(dim: usize) -> Self {
        Self {
            dim,
            phases: vec![0.0; dim],
        }
    }

    /// Create from a slice of phase angles, normalizing to [-π, π).
    pub fn from_phases(phases: &[f32]) -> Self {
        let dim = phases.len();
        let normalized = phases.iter().map(|&p| Self::normalize_angle(p)).collect();
        Self {
            dim,
            phases: normalized,
        }
    }

    /// Deterministic pseudo-random phasor vector generation from a 64-bit seed (SplitMix64).
    pub fn random(dim: usize, seed: u64) -> Self {
        let mut phases = Vec::with_capacity(dim);
        let mut state = seed;
        for _ in 0..dim {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z = z ^ (z >> 31);
            let norm_val = (z as f64) / (u64::MAX as f64);
            phases.push(((norm_val * 2.0 - 1.0) * PI as f64) as f32);
        }
        Self { dim, phases }
    }

    /// Normalizes any angle to [-π, π).
    #[inline(always)]
    pub fn normalize_angle(angle: f32) -> f32 {
        let mut a = angle.rem_euclid(2.0 * PI);
        if a > PI {
            a -= 2.0 * PI;
        }
        a
    }

    /// Phasor Binding (⊗): Elementwise complex multiplication on unit circle.
    /// (u ⊙ v)_d = (θ_u,d + θ_v,d) mod 2π.
    #[inline]
    pub fn bind(&self, other: &Self) -> Self {
        assert_eq!(self.dim, other.dim, "Dimension mismatch in Phasor bind");
        let mut out = Vec::with_capacity(self.dim);
        for d in 0..self.dim {
            out.push(Self::normalize_angle(self.phases[d] + other.phases[d]));
        }
        Self {
            dim: self.dim,
            phases: out,
        }
    }

    /// Exact Unitary Unbinding (⊗⁻¹): Conjugate multiplication.
    /// (u* ⊙ w)_d = (θ_w,d - θ_u,d) mod 2π.
    /// Identically recovers v from (u ⊙ v) with 0.0 error.
    #[inline]
    pub fn unbind(&self, bound: &Self) -> Self {
        assert_eq!(self.dim, bound.dim, "Dimension mismatch in Phasor unbind");
        let mut out = Vec::with_capacity(self.dim);
        for d in 0..self.dim {
            out.push(Self::normalize_angle(bound.phases[d] - self.phases[d]));
        }
        Self {
            dim: self.dim,
            phases: out,
        }
    }

    /// Continuous Fractional Power Transformation: z^α.
    /// Maps θ_d -> (α * θ_d) mod 2π.
    /// Enables continuous position encoding p(τ) = p_0^τ with exact homomorphism p(τ1 + τ2) = p(τ1) ⊙ p(τ2).
    #[inline]
    pub fn fractional_shift(&self, alpha: f32) -> Self {
        let mut out = Vec::with_capacity(self.dim);
        for d in 0..self.dim {
            out.push(Self::normalize_angle(self.phases[d] * alpha));
        }
        Self {
            dim: self.dim,
            phases: out,
        }
    }

    /// Normalized Circular Mean Bundling (+): Superposition of multiple phasors.
    /// Evaluates Arg(Σ_k e^{i θ_k,d}) for each dimension d.
    pub fn bundle(vectors: &[&Self]) -> Self {
        if vectors.is_empty() {
            panic!("Cannot bundle empty vector list");
        }
        let dim = vectors[0].dim;
        let mut out = Vec::with_capacity(dim);

        for d in 0..dim {
            let mut re_sum = 0.0f32;
            let mut im_sum = 0.0f32;
            for v in vectors {
                let (im, re) = v.phases[d].sin_cos();
                re_sum += re;
                im_sum += im;
            }
            out.push(im_sum.atan2(re_sum));
        }
        Self { dim, phases: out }
    }

    /// Hermitian Cosine Similarity on T^D: (1/D) * Σ_d cos(θ_u,d - θ_v,d).
    /// Range: [-1.0, 1.0]. Returns 1.0 for identical vectors, ~0.0 for orthogonal vectors.
    #[inline]
    pub fn similarity(&self, other: &Self) -> f32 {
        assert_eq!(self.dim, other.dim, "Dimension mismatch in Phasor similarity");
        let mut sum = 0.0f32;
        for d in 0..self.dim {
            sum += (self.phases[d] - other.phases[d]).cos();
        }
        sum / (self.dim as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phasor_exact_unitary_unbinding() {
        let dim = 1024;
        let u = PhasorVector::random(dim, 42);
        let v = PhasorVector::random(dim, 99);

        // Bind u and v
        let bound = u.bind(&v);
        // Unbind u from bound to recover v
        let recovered_v = u.unbind(&bound);

        let sim = recovered_v.similarity(&v);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Phasor unbinding must be exact unitary recovery! Got similarity {}",
            sim
        );
    }

    #[test]
    fn test_phasor_fractional_shift_homomorphism() {
        let dim = 512;
        let p0 = PhasorVector::random(dim, 1337);

        let p_tau1 = p0.fractional_shift(1.5);
        let p_tau2 = p0.fractional_shift(2.5);

        // p(1.5 + 2.5) = p(4.0)
        let p_tau1_plus_tau2 = p0.fractional_shift(4.0);
        // p(1.5) ⊙ p(2.5)
        let p_bound = p_tau1.bind(&p_tau2);

        let sim = p_tau1_plus_tau2.similarity(&p_bound);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Fractional shift must satisfy group homomorphism! Got similarity {}",
            sim
        );
    }

    #[test]
    fn test_phasor_bundling_similarity() {
        let dim = 1024;
        let v1 = PhasorVector::random(dim, 101);
        let v2 = PhasorVector::random(dim, 102);
        let v3 = PhasorVector::random(dim, 103);

        let bundled = PhasorVector::bundle(&[&v1, &v2, &v3]);

        assert!(bundled.similarity(&v1) > 0.45);
        assert!(bundled.similarity(&v2) > 0.45);
        assert!(bundled.similarity(&v3) > 0.45);

        let orthogonal = PhasorVector::random(dim, 999);
        assert!(bundled.similarity(&orthogonal).abs() < 0.12);
    }
}
