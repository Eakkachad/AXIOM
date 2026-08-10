//! HyperVector: The fundamental data type for VSA computation.
//!
//! A hypervector is a high-dimensional real-valued vector (D > 10,000)
//! that represents a concept in the latent space. The high dimensionality
//! ensures that randomly generated vectors are quasi-orthogonal with
//! probability approaching 1.
//!
//! # Memory optimisation
//! Bipolar vectors (±1) from `random_bipolar()` are stored only as packed
//! u64 bits (32× smaller than f32).  The f32 data is lazily decompressed
//! on first access via `as_slice()`.

use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::cell::UnsafeCell;
use std::fmt;

use crate::DEFAULT_DIM;

/// Pack `dim` bipolar values into a `Vec<u64>` (1 bit per dimension).
fn pack_from_slice(data: &[f32]) -> Vec<u64> {
    let num_words = data.len().div_ceil(64);
    let mut bits = vec![0u64; num_words];
    for (i, &val) in data.iter().enumerate() {
        if val > 0.0 { bits[i / 64] |= 1u64 << (i % 64); }
    }
    bits
}

pub struct HyperVector {
    /// The raw vector components. For bipolar vectors this may be empty
    /// until first access (decompressed lazily from packed bits).
    data: UnsafeCell<Vec<f32>>,
    /// Bit-packed bipolar representation (each u64 = 64 dimensions).
    pub packed: Option<Vec<u64>>,
    pub dim: usize,
}

impl Clone for HyperVector {
    fn clone(&self) -> Self {
        let data_clone = unsafe { (*self.data.get()).clone() };
        Self { data: UnsafeCell::new(data_clone), packed: self.packed.clone(), dim: self.dim }
    }
}

// HyperVector contains no internal references and UnsafeCell is Send.
unsafe impl Send for HyperVector {}
unsafe impl Sync for HyperVector {}

impl Serialize for HyperVector {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("HyperVector", 2)?;
        st.serialize_field("dim", &self.dim)?;
        st.serialize_field("packed", &self.packed)?;
        st.end()
    }
}

impl<'de> Deserialize<'de> for HyperVector {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct HvRepr { dim: usize, packed: Option<Vec<u64>> }
        let r = HvRepr::deserialize(d)?;
        Ok(Self { data: UnsafeCell::new(Vec::new()), packed: r.packed, dim: r.dim })
    }
}

impl HyperVector {
    #[inline]
    pub fn new(data: Vec<f32>) -> Self {
        let dim = data.len();
        Self { data: UnsafeCell::new(data), packed: None, dim }
    }

    #[inline]
    pub fn zeros(dim: usize) -> Self {
        Self { data: UnsafeCell::new(vec![0.0; dim]), packed: None, dim }
    }

    /// Ensure f32 data is available, decompressing from packed if needed.
    /// SAFETY: must not be called concurrently with write access.
    #[inline]
    fn ensure_data(&self) {
        // Fast path: data already populated.
        let ptr = self.data.get();
        if unsafe { (*ptr).is_empty() } {
            let packed = self.packed.as_ref().expect("compressed vector without packed bits");
            let mut decompressed = vec![0.0f32; self.dim];
            for (i, val) in decompressed.iter_mut().enumerate() {
                *val = if (packed[i / 64] >> (i % 64)) & 1 == 1 { 1.0 } else { -1.0 };
            }
            unsafe { *ptr = decompressed; }
        }
    }

    /// Borrow the f32 data slice. Decompresses from packed on first access.
    /// The returned reference is valid for the lifetime of &self.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        self.ensure_data();
        unsafe { &*self.data.get() }
    }

    /// Drop the f32 data, keeping only the packed representation.
    /// Memory: 40KB → 1.25KB per vector (32× reduction).
    pub fn compress(&self) {
        if self.packed.is_none() { return; }
        unsafe { *self.data.get() = Vec::new(); }
    }

    /// Generate a random bipolar hypervector ({-1, +1}^D) — compressed.
    pub fn random_bipolar(dim: usize, seed: u64) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let num_words = dim.div_ceil(64);
        let mut bits = vec![0u64; num_words];
        for i in 0..dim {
            if rng.gen_bool(0.5) {
                bits[i / 64] |= 1u64 << (i % 64);
            }
        }
        Self { data: UnsafeCell::new(Vec::new()), packed: Some(bits), dim }
    }

    /// Generate a random real-valued hypervector from N(0, 1/√D).
    pub fn random_gaussian(dim: usize, seed: u64) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let scale = 1.0 / (dim as f32).sqrt();
        let data: Vec<f32> = (0..dim)
            .map(|_| {
                let u1: f32 = rng.gen_range(0.0001f32..1.0);
                let u2: f32 = rng.gen::<f32>() * std::f32::consts::TAU;
                (-2.0 * u1.ln()).sqrt() * u2.cos() * scale
            })
            .collect();
        Self { data: UnsafeCell::new(data), packed: None, dim }
    }

    #[inline] pub fn dim(&self) -> usize { self.dim }
    #[inline] pub fn norm(&self) -> f32 { self.dot(self).sqrt() }

    #[inline]
    pub fn dot(&self, other: &Self) -> f32 {
        debug_assert_eq!(self.dim, other.dim);
        let d1 = self.as_slice();
        let d2 = other.as_slice();
        let mut sum = 0.0f32;
        let chunks = self.dim / 8;
        for i in 0..chunks {
            let base = i * 8;
            sum += d1[base] * d2[base];
            sum += d1[base + 1] * d2[base + 1];
            sum += d1[base + 2] * d2[base + 2];
            sum += d1[base + 3] * d2[base + 3];
            sum += d1[base + 4] * d2[base + 4];
            sum += d1[base + 5] * d2[base + 5];
            sum += d1[base + 6] * d2[base + 6];
            sum += d1[base + 7] * d2[base + 7];
        }
        for i in (chunks * 8)..self.dim { sum += d1[i] * d2[i]; }
        sum
    }

    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n < 1e-10 { return Self::zeros(self.dim); }
        let inv_n = 1.0 / n;
        Self { data: UnsafeCell::new(self.as_slice().iter().map(|&x| x * inv_n).collect()), packed: None, dim: self.dim }
    }

    pub fn sign(&self) -> Self {
        let d = self.as_slice();
        let data: Vec<f32> = d.iter().map(|&x| if x >= 0.0 { 1.0 } else { -1.0 }).collect();
        let bits = pack_from_slice(&data);
        Self { data: UnsafeCell::new(data), packed: Some(bits), dim: self.dim }
    }

    #[inline]
    pub fn scale(&self, s: f32) -> Self {
        Self { data: UnsafeCell::new(self.as_slice().iter().map(|&x| x * s).collect()), packed: None, dim: self.dim }
    }

    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        debug_assert_eq!(self.dim, other.dim);
        let d1 = self.as_slice();
        let d2 = other.as_slice();
        Self { data: UnsafeCell::new(d1.iter().zip(d2.iter()).map(|(&a, &b)| a + b).collect()), packed: None, dim: self.dim }
    }

    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        debug_assert_eq!(self.dim, other.dim);
        let d1 = self.as_slice();
        let d2 = other.as_slice();
        Self { data: UnsafeCell::new(d1.iter().zip(d2.iter()).map(|(&a, &b)| a - b).collect()), packed: None, dim: self.dim }
    }

    #[inline]
    pub fn hadamard(&self, other: &Self) -> Self {
        debug_assert_eq!(self.dim, other.dim);
        let d1 = self.as_slice();
        let d2 = other.as_slice();
        let data: Vec<f32> = d1.iter().zip(d2.iter()).map(|(&a, &b)| a * b).collect();
        let packed = if self.packed.is_some() && other.packed.is_some() {
            Some(pack_from_slice(&data))
        } else { None };
        Self { data: UnsafeCell::new(data), packed, dim: self.dim }
    }

    pub fn permute(&self, amount: i32) -> Self {
        let shift = ((amount % self.dim as i32) + self.dim as i32) as usize % self.dim;
        let d = self.as_slice();
        let mut new_data = vec![0.0f32; self.dim];
        for i in 0..self.dim { new_data[(i + shift) % self.dim] = d[i]; }
        let packed = if self.packed.is_some() {
            Some(pack_from_slice(&new_data))
        } else { None };
        Self { data: UnsafeCell::new(new_data), packed, dim: self.dim }
    }

    #[inline] pub fn inv_permute(&self, amount: i32) -> Self { self.permute(-amount) }
}

impl fmt::Debug for HyperVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let d = self.as_slice();
        let preview: Vec<String> = d.iter().take(5).map(|x| format!("{:.3}", x)).collect();
        write!(f, "HV[D={}, [{}, ...]]", self.dim, preview.join(", "))
    }
}

impl PartialEq for HyperVector {
    fn eq(&self, other: &Self) -> bool {
        if self.dim != other.dim { return false; }
        let d1 = self.as_slice();
        let d2 = other.as_slice();
        d1.iter().zip(d2.iter()).all(|(a, b)| (a - b).abs() < 1e-7)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bipolar_properties() {
        let v = HyperVector::random_bipolar(DEFAULT_DIM, 42);
        assert_eq!(v.dim(), DEFAULT_DIM);
        for &x in v.as_slice() { assert!(x == 1.0 || x == -1.0); }
        let norm = v.norm();
        assert!((norm - (DEFAULT_DIM as f32).sqrt()).abs() < 0.01);
    }

    #[test]
    fn test_packed_cosine_matches_f32() {
        let v1 = HyperVector::random_bipolar(10_240, 42);
        let v2 = HyperVector::random_bipolar(10_240, 99);
        let cos_fast = crate::similarity::cosine_similarity(&v1, &v2);
        let cos_slow = v1.dot(&v2) / (v1.norm() * v2.norm());
        assert!((cos_fast - cos_slow).abs() < 1e-5, "{} vs {}", cos_fast, cos_slow);
    }

    #[test]
    fn test_deterministic_generation() {
        let v1 = HyperVector::random_bipolar(DEFAULT_DIM, 42);
        let v2 = HyperVector::random_bipolar(DEFAULT_DIM, 42);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_quasi_orthogonality() {
        let v1 = HyperVector::random_bipolar(DEFAULT_DIM, 1);
        let v2 = HyperVector::random_bipolar(DEFAULT_DIM, 2);
        let cos = v1.dot(&v2) / (v1.norm() * v2.norm());
        assert!(cos.abs() < 0.05);
    }

    #[test]
    fn test_hadamard_self_inverse() {
        let v = HyperVector::random_bipolar(DEFAULT_DIM, 42);
        let result = v.hadamard(&v);
        for &x in result.as_slice() { assert_eq!(x, 1.0); }
    }

    #[test]
    fn test_permute_inverse() {
        let v = HyperVector::random_bipolar(DEFAULT_DIM, 99);
        let shifted = v.permute(17);
        let recovered = shifted.inv_permute(17);
        assert_eq!(v, recovered);
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        let v = HyperVector::random_bipolar(DEFAULT_DIM, 123);
        assert!(v.packed.is_some());
        let dot_before = v.dot(&v);
        v.compress();
        let dot_after = v.dot(&v);
        assert!((dot_before - dot_after).abs() < 1e-5);
    }
}
