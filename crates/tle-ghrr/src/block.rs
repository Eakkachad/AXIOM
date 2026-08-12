//! GHRR block-unitary primitives (PathHD, arXiv:2512.09369).
//!
//! A symbol (relation) is a block vector `v = [A_1; …; A_D]` where each
//! `A_j ∈ O(4)` is a real orthogonal block. Blocks are built deterministically
//! from a seed as a product of TWO Householder reflections — genuinely
//! non-commuting (the paper's `diag(e^{iφ})` family is a commuting special
//! case, a bug in the paper; we use the Householder alternative it suggests).
//!
//! Binding is blockwise matrix product: `X ⊛ Y = [X_1Y_1; …; X_DY_D]`. Products
//! of orthogonal blocks stay orthogonal, so per-block Frobenius norms are
//! preserved under binding depth (no variance blow-up, unlike circular
//! convolution), and `r1→r2 ≠ r2→r1` (order- and direction-sensitive).

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// Block size m = 4 (paper §I.2: "fix a small block size m=4").
pub const M: usize = 4;
/// Block dimension = d / m² = 2048 / 16.
pub const D_BLOCKS: usize = 128;
/// Flattened vector dimension.
pub const DIM: usize = D_BLOCKS * M * M;

/// A single real orthogonal block (O(4)).
pub type UnitaryBlock = [[f32; M]; M];

/// Deterministic O(4) block from a seed: product of two Householder
/// reflections. Orthogonal ⇒ ‖A‖_F = √m = 2 exactly.
pub fn random_orthogonal_block(seed: u64) -> UnitaryBlock {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let h = |rng: &mut ChaCha20Rng| -> UnitaryBlock {
        // random normal vector u ~ N(0, I_4), then H(u) = I − 2uuᵀ/‖u‖²
        let uvals: Vec<f32> = (0..M)
            .map(|_| {
                let u1: f32 = rng.gen_range(0.0001f32..1.0);
                let u2: f32 = rng.gen::<f32>() * std::f32::consts::TAU;
                (-2.0 * u1.ln()).sqrt() * u2.cos()
            })
            .collect();
        let mut u = [0.0f32; M];
        u.copy_from_slice(&uvals);
        let mut a = [[0.0f32; M]; M];
        let norm2: f32 = u.iter().map(|x| x * x).sum();
        for i in 0..M {
            for j in 0..M {
                a[i][j] = if i == j { 1.0 } else { 0.0 } - 2.0 * u[i] * u[j] / norm2;
            }
        }
        a
    };
    mat_mul(&h(&mut rng), &h(&mut rng))
}

/// Blockwise matrix product `A·B`.
pub fn mat_mul(a: &UnitaryBlock, b: &UnitaryBlock) -> UnitaryBlock {
    let mut c = [[0.0f32; M]; M];
    for i in 0..M {
        for j in 0..M {
            let mut s = 0.0f32;
            for k in 0..M {
                s += a[i][k] * b[k][j];
            }
            c[i][j] = s;
        }
    }
    c
}

/// Frobenius inner product `tr(AᵀB)` (real GHRR: `Re⟨A,B⟩_F`).
pub fn frob_inner(a: &UnitaryBlock, b: &UnitaryBlock) -> f32 {
    let mut s = 0.0f32;
    for i in 0..M {
        for j in 0..M {
            s += a[i][j] * b[i][j];
        }
    }
    s
}

/// Frobenius norm.
pub fn frob_norm(a: &UnitaryBlock) -> f32 {
    frob_inner(a, a).sqrt()
}

/// Normalize a block to unit Frobenius norm (exact for orthogonal blocks).
pub fn normalize(a: &UnitaryBlock) -> UnitaryBlock {
    let n = frob_norm(a);
    let mut c = *a;
    if n > 1e-9 {
        for i in 0..M {
            for j in 0..M {
                c[i][j] /= n;
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_is_orthogonal() {
        let a = random_orthogonal_block(42);
        // AᵀA = I
        let at = transpose(&a);
        let prod = mat_mul(&at, &a);
        for i in 0..M {
            for j in 0..M {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((prod[i][j] - expect).abs() < 1e-4, "AᵀA[{i}][{j}] = {}", prod[i][j]);
            }
        }
        assert!((frob_norm(&a) - 2.0).abs() < 1e-4);
    }

    #[test]
    fn block_deterministic() {
        assert_eq!(random_orthogonal_block(7), random_orthogonal_block(7));
    }

    #[test]
    fn blocks_non_commuting() {
        let a = random_orthogonal_block(11);
        let b = random_orthogonal_block(22);
        let ab = mat_mul(&a, &b);
        let ba = mat_mul(&b, &a);
        let mut diff = 0.0f32;
        for i in 0..M {
            for j in 0..M {
                diff = diff.max((ab[i][j] - ba[i][j]).abs());
            }
        }
        assert!(diff > 1e-3, "A and B must not commute (max diff {diff})");
    }

    fn transpose(a: &UnitaryBlock) -> UnitaryBlock {
        let mut t = [[0.0f32; M]; M];
        for i in 0..M {
            for j in 0..M {
                t[i][j] = a[j][i];
            }
        }
        t
    }
}
