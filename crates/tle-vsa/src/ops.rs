//! Core VSA Operations: Binding, Unbinding, and Bundling.
//!
//! ## Mathematical Specification
//!
//! Given role vectors R_i and filler vectors F_i:
//!
//! - **Bind**: B_i = R_i ⊗ F_i (Hadamard product for bipolar)
//! - **Bundle**: S = Σ B_i (element-wise sum of bound pairs)
//! - **Unbind**: F̂_i = R_i ⊗ S (retrieves F_i + noise from other bindings)
//!
//! The signal-to-noise ratio for k items in superposition:
//!   SNR = √(D) / √(k - 1) ≈ √(D/k)
//!
//! For D=10,240 and k=10: SNR ≈ 32 (excellent retrieval)
//! For D=10,240 and k=100: SNR ≈ 10 (needs cleanup)
//! For D=10,240 and k=1000: SNR ≈ 3.2 (requires resonator cleanup)

use crate::HyperVector;

/// Bind two hypervectors using Hadamard product (element-wise multiplication).
///
/// For bipolar vectors, this is equivalent to XOR in binary VSA.
/// The result is quasi-orthogonal to both inputs.
///
/// # Properties
/// - Associative: (A ⊗ B) ⊗ C = A ⊗ (B ⊗ C)
/// - Commutative: A ⊗ B = B ⊗ A
/// - Self-inverse for bipolar: A ⊗ A = 1 (identity)
/// - Preserves norm: ||A ⊗ B|| = ||A|| · ||B|| for bipolar
#[inline]
pub fn bind(a: &HyperVector, b: &HyperVector) -> HyperVector {
    a.hadamard(b)
}

/// Unbind (retrieve) a vector from a binding.
///
/// For bipolar Hadamard binding, unbinding is the same operation as binding
/// because each element is its own inverse: (-1)² = 1, (1)² = 1.
///
/// To retrieve filler F from bound pair B = R ⊗ F:
///   F̂ = R ⊗ B = R ⊗ (R ⊗ F) = (R ⊗ R) ⊗ F = 1 ⊗ F = F
///
/// When applied to a bundle S = Σ(R_i ⊗ F_i):
///   F̂_j = R_j ⊗ S = F_j + Σ_{i≠j}(R_j ⊗ R_i ⊗ F_i)
///   The second term is noise (crosstalk) with magnitude ~√(k-1)/√D
#[inline]
pub fn unbind(role: &HyperVector, composite: &HyperVector) -> HyperVector {
    role.hadamard(composite)
}

/// Bundle multiple hypervectors into a single superposition vector.
///
/// This is simple element-wise addition. The resulting vector is
/// similar to ALL its constituents (within noise bounds).
///
/// # Capacity
/// For reliable retrieval, the number of items k must satisfy:
///   k < D / (2 * ln(N)) where N is codebook size
///
/// For D=10,240 and N=50,000: k < 10,240 / (2 * 10.82) ≈ 473 items
pub fn bundle(vectors: &[&HyperVector]) -> HyperVector {
    assert!(!vectors.is_empty(), "Cannot bundle empty set");
    let dim = vectors[0].dim();

    let mut result = vec![0.0f32; dim];
    for v in vectors {
        debug_assert_eq!(v.dim(), dim, "All vectors must have same dimension");
        let d = v.as_slice();
        let chunks = dim / 8;
        for i in 0..chunks {
            let base = i * 8;
            result[base] += d[base];
            result[base + 1] += d[base + 1];
            result[base + 2] += d[base + 2];
            result[base + 3] += d[base + 3];
            result[base + 4] += d[base + 4];
            result[base + 5] += d[base + 5];
            result[base + 6] += d[base + 6];
            result[base + 7] += d[base + 7];
        }
        for i in (chunks * 8)..dim {
            result[i] += d[i];
        }
    }

    HyperVector::new(result)
}

/// Bundle with weighting factors.
/// Useful for attention-like mechanisms where some bindings are more important.
pub fn weighted_bundle(vectors: &[(&HyperVector, f32)]) -> HyperVector {
    assert!(!vectors.is_empty(), "Cannot bundle empty set");
    let dim = vectors[0].0.dim();

    let mut result = vec![0.0f32; dim];
    for (v, weight) in vectors {
        debug_assert_eq!(v.dim(), dim);
        let d = v.as_slice();
        for i in 0..dim {
            result[i] += d[i] * weight;
        }
    }

    HyperVector::new(result)
}

/// Bind with positional encoding using circular permutation.
///
/// Encodes sequence position by permuting the filler before binding:
///   B_pos = R ⊗ ρ^pos(F)
///
/// This allows retrieval by position:
///   F̂ = ρ^{-pos}(R ⊗ B_pos) = F
pub fn bind_with_position(role: &HyperVector, filler: &HyperVector, position: i32) -> HyperVector {
    let permuted = filler.permute(position);
    bind(role, &permuted)
}

/// Unbind with positional decoding.
pub fn unbind_with_position(
    role: &HyperVector,
    composite: &HyperVector,
    position: i32,
) -> HyperVector {
    let unbound = unbind(role, composite);
    unbound.inv_permute(position)
}

/// Compute the theoretical Signal-to-Noise Ratio for k items in D dimensions.
///
/// SNR = √(D / (k - 1))
///
/// Returns infinity if k <= 1.
pub fn theoretical_snr(dim: usize, k: usize) -> f32 {
    if k <= 1 {
        return f32::INFINITY;
    }
    ((dim as f32) / ((k - 1) as f32)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_DIM;

    #[test]
    fn test_bind_unbind_exact() {
        let role = HyperVector::random_bipolar(DEFAULT_DIM, 100);
        let filler = HyperVector::random_bipolar(DEFAULT_DIM, 200);

        let bound = bind(&role, &filler);
        let recovered = unbind(&role, &bound);

        // For single binding with bipolar: exact recovery
        assert_eq!(recovered, filler);
    }

    #[test]
    fn test_bundle_retrieval_with_noise() {
        let dim = DEFAULT_DIM;
        let role_a = HyperVector::random_bipolar(dim, 10);
        let role_b = HyperVector::random_bipolar(dim, 20);
        let role_c = HyperVector::random_bipolar(dim, 30);

        let filler_a = HyperVector::random_bipolar(dim, 110);
        let filler_b = HyperVector::random_bipolar(dim, 120);
        let filler_c = HyperVector::random_bipolar(dim, 130);

        // Bind and bundle
        let ba = bind(&role_a, &filler_a);
        let bb = bind(&role_b, &filler_b);
        let bc = bind(&role_c, &filler_c);
        let superposition = bundle(&[&ba, &bb, &bc]);

        // Unbind role_a: should recover filler_a + noise
        let recovered_a = unbind(&role_a, &superposition);

        // Check similarity to original filler_a
        let cos_sim = recovered_a.dot(&filler_a) / (recovered_a.norm() * filler_a.norm());

        // With D=10240 and k=3: SNR = √(10240/2) ≈ 71.6
        // Expected cos_sim ≈ 1/√(1 + (k-1)/D) ≈ 0.999+
        assert!(
            cos_sim > 0.5,
            "Retrieval should have high similarity, got {}",
            cos_sim
        );
    }

    #[test]
    fn test_positional_binding() {
        let role = HyperVector::random_bipolar(DEFAULT_DIM, 50);
        let filler = HyperVector::random_bipolar(DEFAULT_DIM, 60);

        let bound = bind_with_position(&role, &filler, 3);
        let recovered = unbind_with_position(&role, &bound, 3);

        assert_eq!(recovered, filler);
    }

    #[test]
    fn test_snr_calculation() {
        let snr = theoretical_snr(10_240, 10);
        // √(10240/9) ≈ 33.7
        assert!((snr - 33.7).abs() < 1.0);

        let snr_100 = theoretical_snr(10_240, 100);
        // √(10240/99) ≈ 10.2
        assert!((snr_100 - 10.2).abs() < 0.5);
    }
}
