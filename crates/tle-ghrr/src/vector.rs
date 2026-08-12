//! GHRR vectors: block-unitary hypervectors and path binding.
//!
//! `GhrrVector` is a block vector `[A_1; …; A_D]`, each `A_j ∈ O(4)`.
//! Binding a path is the left-to-right blockwise product (Eq 3):
//! `v_z = v_{r1} ⊛ v_{r2} ⊛ … ⊛ v_{rℓ}`.
//! Similarity is the blockwise cosine (Eq 4), which for unit-Frobenius blocks
//! equals the flat cosine of the flattened vector.

use crate::block::{D_BLOCKS, DIM, UnitaryBlock, frob_inner, mat_mul, normalize};

/// A GHRR block vector (d = D·m² = 2048).
#[derive(Debug, Clone, PartialEq)]
pub struct GhrrVector {
    pub blocks: Vec<UnitaryBlock>,
}

impl GhrrVector {
    /// Build from a codebook of per-(symbol, block) matrices.
    pub fn from_blocks(blocks: Vec<UnitaryBlock>) -> Self {
        debug_assert_eq!(blocks.len(), D_BLOCKS);
        Self { blocks }
    }

    /// Flattened length (d = 2048).
    pub fn dim(&self) -> usize {
        DIM
    }

    /// Left-to-right blockwise binding of the relation vectors in order.
    /// `bind_path([r1, r2]) = r1 ⊛ r2`, which ≠ `r2 ⊛ r1` (non-commutative).
    pub fn bind_path(vecs: &[&GhrrVector]) -> GhrrVector {
        if vecs.is_empty() {
            return Self { blocks: vec![[[0.0f32; 4]; 4]; D_BLOCKS] };
        }
        let mut out: Vec<UnitaryBlock> = vecs[0].blocks.clone();
        for v in &vecs[1..] {
            for (j, b) in out.iter_mut().enumerate() {
                *b = mat_mul(b, &v.blocks[j]);
            }
        }
        for b in out.iter_mut() {
            *b = normalize(b);
        }
        Self { blocks: out }
    }

    /// Blockwise cosine (Eq 4): mean over D blocks of the normalized
    /// Frobenius inner product.
    pub fn blockwise_cosine(&self, other: &GhrrVector) -> f32 {
        let mut sum = 0.0f32;
        for j in 0..D_BLOCKS {
            sum += frob_inner(&self.blocks[j], &other.blocks[j])
                / (crate::block::frob_norm(&self.blocks[j])
                    * crate::block::frob_norm(&other.blocks[j]));
        }
        sum / D_BLOCKS as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::random_orthogonal_block;

    fn code(seed: u64) -> GhrrVector {
        let blocks: Vec<UnitaryBlock> = (0..D_BLOCKS)
            .map(|j| random_orthogonal_block(seed ^ (j as u64).wrapping_mul(0x9E3779B97F4A7C15)))
            .collect();
        GhrrVector::from_blocks(blocks)
    }

    #[test]
    fn self_similarity_is_one() {
        let v = code(1);
        assert!((v.blockwise_cosine(&v) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn different_symbols_quasi_orthogonal() {
        let a = code(1);
        let b = code(2);
        let c = a.blockwise_cosine(&b);
        assert!(c.abs() < 0.15, "cos={c}");
    }

    #[test]
    fn order_sensitive_binding() {
        let r1 = code(10);
        let r2 = code(20);
        let fwd = GhrrVector::bind_path(&[&r1, &r2]);
        let rev = GhrrVector::bind_path(&[&r2, &r1]);
        let c = fwd.blockwise_cosine(&rev);
        // m=4 has inherent correlation E[tr(ABAᵀBᵀ)/n]≈1/(n−1)≈0.33; the
        // discriminating margin is self=1.0 vs reversed≈0.23 vs random≈0.11.
        assert!(c < 0.45, "r1⊛r2 vs r2⊛r1 must be distinguishable, cos={c}");
        assert!(c > 0.05, "reversed product must not be perfectly orthogonal either, cos={c}");
    }

    #[test]
    fn binding_preserves_norm() {
        let r1 = code(10);
        let r2 = code(20);
        let b = GhrrVector::bind_path(&[&r1, &r2]);
        for blk in &b.blocks {
            let n = crate::block::frob_norm(blk);
            assert!((n - 1.0).abs() < 1e-3, "block norm {n} (unit after normalize)");
        }
    }

    #[test]
    fn deterministic_binding() {
        let r1 = code(10);
        let r2 = code(20);
        let a = GhrrVector::bind_path(&[&r1, &r2]);
        let b = GhrrVector::bind_path(&[&r1, &r2]);
        assert_eq!(a, b);
    }
}
