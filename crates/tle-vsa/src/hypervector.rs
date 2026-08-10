//! HyperVector: The fundamental data type for VSA computation.
//!
//! A hypervector is a high-dimensional real-valued vector (D > 10,000)
//! that represents a concept in the latent space. The high dimensionality
//! ensures that randomly generated vectors are quasi-orthogonal with
//! probability approaching 1.

use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::DEFAULT_DIM;

/// A high-dimensional vector representing a concept in latent space.
///
/// # Mathematical Properties
/// - Random hypervectors are quasi-orthogonal: E[cos(v_i, v_j)] ≈ 0 for i ≠ j
/// - Variance of random dot product: Var[v_i · v_j] = D·σ⁴ for i ≠ j
/// - Self-similarity: cos(v, v) = 1.0
#[derive(Clone, Serialize, Deserialize)]
pub struct HyperVector {
    /// The raw vector components. Length = D (default 10,240).
    pub data: Vec<f32>,
    /// Bit-packed bipolar representation (each u64 = 64 dimensions).
    /// Populated lazily via `pack()` for fast cosine/binding.
    #[serde(skip, default = "Option::default")]
    pub packed: Option<Vec<u64>>,
}

impl HyperVector {
    /// Create a new hypervector with the given data.
    #[inline]
    pub fn new(data: Vec<f32>) -> Self {
        Self { data, packed: None }
    }

    /// Create a zero vector of given dimension.
    #[inline]
    pub fn zeros(dim: usize) -> Self {
        Self {
            data: vec![0.0; dim],
            packed: None,
        }
    }

    /// Build the bit-packed representation from the f32 data.
    /// Must only be called on bipolar vectors (all values ±1).
    pub fn pack(&mut self) {
        if self.packed.is_some() { return; }
        let num_words = self.data.len().div_ceil(64);
        let mut bits = vec![0u64; num_words];
        for (i, &val) in self.data.iter().enumerate() {
            if val > 0.0 {
                bits[i / 64] |= 1u64 << (i % 64);
            }
        }
        self.packed = Some(bits);
    }

    /// Generate a random bipolar hypervector ({-1, +1}^D) with auto-packing.
    pub fn random_bipolar(dim: usize, seed: u64) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let num_words = dim.div_ceil(64);
        let mut bits = vec![0u64; num_words];
        let data: Vec<f32> = (0..dim)
            .map(|i| {
                if rng.gen_bool(0.5) {
                    bits[i / 64] |= 1u64 << (i % 64);
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        Self { data, packed: Some(bits) }
    }

    /// Generate a random real-valued hypervector from N(0, 1/√D).
    /// Normalized so that E[||v||²] = 1.
    pub fn random_gaussian(dim: usize, seed: u64) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let scale = 1.0 / (dim as f32).sqrt();
        let data: Vec<f32> = (0..dim)
            .map(|_| {
                // Box-Muller transform for Gaussian samples
                let u1: f32 = rng.gen_range(0.0001f32..1.0);
                let u2: f32 = rng.gen::<f32>() * std::f32::consts::TAU;
                (-2.0 * u1.ln()).sqrt() * u2.cos() * scale
            })
            .collect();
        Self { data, packed: None }
    }

    /// Dimension of this vector.
    #[inline]
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// L2 norm (Euclidean length).
    #[inline]
    pub fn norm(&self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Dot product with another vector.
    #[inline]
    pub fn dot(&self, other: &Self) -> f32 {
        debug_assert_eq!(self.dim(), other.dim(), "Dimension mismatch in dot product");
        // SIMD-friendly: process in chunks of 8
        let mut sum = 0.0f32;
        let chunks = self.data.len() / 8;
        for i in 0..chunks {
            let base = i * 8;
            // Manual unroll for auto-vectorization
            sum += self.data[base] * other.data[base];
            sum += self.data[base + 1] * other.data[base + 1];
            sum += self.data[base + 2] * other.data[base + 2];
            sum += self.data[base + 3] * other.data[base + 3];
            sum += self.data[base + 4] * other.data[base + 4];
            sum += self.data[base + 5] * other.data[base + 5];
            sum += self.data[base + 6] * other.data[base + 6];
            sum += self.data[base + 7] * other.data[base + 7];
        }
        // Handle remainder
        for i in (chunks * 8)..self.data.len() {
            sum += self.data[i] * other.data[i];
        }
        sum
    }

    /// Normalize to unit length. Returns zero vector if norm is 0.
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n < 1e-10 {
            return Self::zeros(self.dim());
        }
        let inv_n = 1.0 / n;
        Self {
            data: self.data.iter().map(|&x| x * inv_n).collect(),
            packed: None,
        }
    }

    /// Element-wise sign function: maps to bipolar {-1, +1}.
    /// Used as a hard cleanup rule for resonator networks.
    pub fn sign(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| if x >= 0.0 { 1.0 } else { -1.0 }).collect();
        // sign() always produces bipolar → auto-pack
        let num_words = data.len().div_ceil(64);
        let mut bits = vec![0u64; num_words];
        for (i, &val) in data.iter().enumerate() {
            if val > 0.0 { bits[i / 64] |= 1u64 << (i % 64); }
        }
        Self { data, packed: Some(bits) }
    }

    /// Scale all components by a scalar.
    #[inline]
    pub fn scale(&self, s: f32) -> Self {
        Self {
            data: self.data.iter().map(|&x| x * s).collect(),
            packed: None,
        }
    }

    /// Element-wise addition (bundling primitive).
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        debug_assert_eq!(self.dim(), other.dim());
        Self {
            data: self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a + b).collect(),
            packed: None,
        }
    }

    /// Element-wise subtraction.
    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        debug_assert_eq!(self.dim(), other.dim());
        Self {
            data: self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a - b).collect(),
            packed: None,
        }
    }

    /// Element-wise multiplication (Hadamard product / binding primitive).
    #[inline]
    pub fn hadamard(&self, other: &Self) -> Self {
        debug_assert_eq!(self.dim(), other.dim());
        let data: Vec<f32> = self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a * b).collect();
        // Hadamard of two bipolar vectors is bipolar — auto-pack if both were.
        let packed = if self.packed.is_some() && other.packed.is_some() {
            let num_words = data.len().div_ceil(64);
            let mut bits = vec![0u64; num_words];
            for (i, &val) in data.iter().enumerate() {
                if val > 0.0 { bits[i / 64] |= 1u64 << (i % 64); }
            }
            Some(bits)
        } else { None };
        Self { data, packed }
    }

    /// Circular permutation (shift) by `amount` positions.
    /// Used for encoding positional/sequential information.
    pub fn permute(&self, amount: i32) -> Self {
        let d = self.dim();
        let shift = ((amount % d as i32) + d as i32) as usize % d;
        let mut new_data = vec![0.0f32; d];
        for i in 0..d {
            new_data[(i + shift) % d] = self.data[i];
        }
        let packed = if self.packed.is_some() {
            let mut hv = Self { data: new_data.clone(), packed: None };
            hv.pack();
            hv.packed
        } else { None };
        Self { data: new_data, packed }
    }

    /// Inverse permutation (undo a permute).
    #[inline]
    pub fn inv_permute(&self, amount: i32) -> Self {
        self.permute(-amount)
    }
}

impl fmt::Debug for HyperVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let preview: Vec<String> = self.data.iter().take(5).map(|x| format!("{:.3}", x)).collect();
        write!(f, "HV[D={}, [{}, ...]]", self.dim(), preview.join(", "))
    }
}

impl PartialEq for HyperVector {
    fn eq(&self, other: &Self) -> bool {
        self.data.len() == other.data.len()
            && self.data.iter().zip(other.data.iter()).all(|(a, b)| (a - b).abs() < 1e-7)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bipolar_properties() {
        let v = HyperVector::random_bipolar(DEFAULT_DIM, 42);
        assert_eq!(v.dim(), DEFAULT_DIM);
        for &x in &v.data {
            assert!(x == 1.0 || x == -1.0);
        }
        let norm = v.norm();
        assert!((norm - (DEFAULT_DIM as f32).sqrt()).abs() < 0.01);
    }

    #[test]
    fn test_packed_cosine_matches_f32() {
        let v1 = HyperVector::random_bipolar(10_240, 42);
        let v2 = HyperVector::random_bipolar(10_240, 99);
        let cos_fast = crate::similarity::cosine_similarity(&v1, &v2);
        let v1_np = HyperVector { data: v1.data.clone(), packed: None };
        let v2_np = HyperVector { data: v2.data.clone(), packed: None };
        let cos_slow = crate::similarity::cosine_similarity(&v1_np, &v2_np);
        assert!((cos_fast - cos_slow).abs() < 1e-5, "{} vs {}", cos_fast, cos_slow);
    }

    #[test]
    fn test_deterministic_generation() {
        let v1 = HyperVector::random_bipolar(DEFAULT_DIM, 42);
        let v2 = HyperVector::random_bipolar(DEFAULT_DIM, 42);
        assert_eq!(v1, v2, "Same seed must produce identical vectors");
    }

    #[test]
    fn test_quasi_orthogonality() {
        let v1 = HyperVector::random_bipolar(DEFAULT_DIM, 1);
        let v2 = HyperVector::random_bipolar(DEFAULT_DIM, 2);
        // Expected: |cos(v1, v2)| ≈ 0, with std ≈ 1/√D ≈ 0.01
        let cos = v1.dot(&v2) / (v1.norm() * v2.norm());
        assert!(
            cos.abs() < 0.05,
            "Random vectors should be quasi-orthogonal, got cos={}",
            cos
        );
    }

    #[test]
    fn test_hadamard_self_inverse() {
        let v = HyperVector::random_bipolar(DEFAULT_DIM, 42);
        // For bipolar: v ⊙ v = 1 (all ones vector)
        let result = v.hadamard(&v);
        for &x in &result.data {
            assert_eq!(x, 1.0);
        }
    }

    #[test]
    fn test_permute_inverse() {
        let v = HyperVector::random_bipolar(DEFAULT_DIM, 99);
        let shifted = v.permute(17);
        let recovered = shifted.inv_permute(17);
        assert_eq!(v, recovered);
    }
}
