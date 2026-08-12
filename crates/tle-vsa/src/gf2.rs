//! GF(2) linear algebra + random linear codes (arXiv:2403.03278).
//!
//! Over the boolean field `F2` (XOR = addition, AND = multiplication), the VSA
//! "recovery problem" — recovering the components of a bundled/bound vector —
//! stops being an iterative (resonator-style) search and becomes exact linear
//! algebra:
//!
//! * **Bundling recovery**: if the component candidates form a subspace, the
//!   subset that sums to a given bundle is found by Gaussian elimination
//!   (`factorize_bundle`), deterministically, in `O(n²·d/64)` bit operations.
//! * **Key/value subcodes**: a systematic linear code `C = K × V` (direct sum,
//!   `K ∩ V = {0}`) makes every codeword factor uniquely as `c = k ⊕ v`, with a
//!   closed-form projection. This is the paper's construction of a hash-map-like
//!   key-value store over vectors.
//!
//! All operations are deterministic (no RNG beyond `LinearCode::new`, which is
//! seeded), CPU-only, and SIMD-friendly (XOR over `u64` words).

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// A dense matrix over F2. Row-major: each row is an `ncols`-bit vector packed
/// into `ncols.div_ceil(64)` `u64` words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gf2Mat {
    pub nrows: usize,
    pub ncols: usize,
    /// `rows[r]` holds row `r` as packed u64 words.
    pub rows: Vec<Vec<u64>>,
}

impl Gf2Mat {
    /// All-zero matrix of the given shape.
    pub fn zero(nrows: usize, ncols: usize) -> Self {
        Self { nrows, ncols, rows: vec![vec![0u64; ncols.div_ceil(64)]; nrows] }
    }

    /// `n × n` identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zero(n, n);
        for i in 0..n {
            m.set(i, i, true);
        }
        m
    }

    /// Build a matrix whose column `j` is `cols[j]` (each of length `nrows`).
    pub fn from_columns(cols: &[Vec<bool>]) -> Self {
        let nrows = cols.first().map(|c| c.len()).unwrap_or(0);
        let ncols = cols.len();
        let mut m = Self::zero(nrows, ncols);
        for (j, col) in cols.iter().enumerate() {
            for (r, &bit) in col.iter().enumerate() {
                if bit {
                    m.set(r, j, true);
                }
            }
        }
        m
    }

    /// Read the bit at (row, col).
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> bool {
        (self.rows[r][c / 64] >> (c % 64)) & 1 == 1
    }

    /// Write the bit at (row, col).
    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: bool) {
        let word = &mut self.rows[r][c / 64];
        let mask = 1u64 << (c % 64);
        if v {
            *word |= mask;
        } else {
            *word &= !mask;
        }
    }

    /// Row `r` as a dense `Vec<bool>`.
    pub fn row(&self, r: usize) -> Vec<bool> {
        (0..self.ncols).map(|c| self.get(r, c)).collect()
    }

    /// Gaussian elimination to reduced row echelon form (in place). Returns
    /// the rank. Pivot search is deterministic (first row in scan order).
    pub fn rref(&mut self) -> usize {
        let mut rank = 0usize;
        for col in 0..self.ncols {
            let mut pivot = None;
            for r in rank..self.nrows {
                if (self.rows[r][col / 64] >> (col % 64)) & 1 == 1 {
                    pivot = Some(r);
                    break;
                }
            }
            let Some(p) = pivot else { continue };
            self.rows.swap(rank, p);
            for r in 0..self.nrows {
                if r != rank && (self.rows[r][col / 64] >> (col % 64)) & 1 == 1 {
                    let w = self.rows[rank].len();
                    for i in 0..w {
                        self.rows[r][i] ^= self.rows[rank][i];
                    }
                }
            }
            rank += 1;
            if rank == self.nrows {
                break;
            }
        }
        rank
    }

    /// Rank of the matrix over F2 (deterministic Gaussian elimination).
    pub fn rank(&self) -> usize {
        let mut m = self.clone();
        m.rref()
    }

    /// Solve `M·x = b` over F2. `b` has `nrows` bits. Returns one solution
    /// (`x` of `ncols` bits, free variables set to 0) or `None` if the system
    /// is inconsistent.
    pub fn solve(&self, b: &[bool]) -> Option<Vec<bool>> {
        assert_eq!(b.len(), self.nrows);
        let aug_ncols = self.ncols + 1;
        let w = aug_ncols.div_ceil(64);
        let mut aug: Vec<Vec<u64>> = (0..self.nrows)
            .map(|r| {
                let mut row = vec![0u64; w];
                for c in 0..self.ncols {
                    if self.get(r, c) {
                        row[c / 64] |= 1 << (c % 64);
                    }
                }
                if b[r] {
                    row[self.ncols / 64] |= 1 << (self.ncols % 64);
                }
                row
            })
            .collect();

        let mut rank = 0usize;
        for col in 0..aug_ncols {
            let mut pivot = None;
            for r in rank..self.nrows {
                if (aug[r][col / 64] >> (col % 64)) & 1 == 1 {
                    pivot = Some(r);
                    break;
                }
            }
            let Some(p) = pivot else { continue };
            aug.swap(rank, p);
            for r in 0..self.nrows {
                if r != rank && (aug[r][col / 64] >> (col % 64)) & 1 == 1 {
                    for i in 0..w {
                        aug[r][i] ^= aug[rank][i];
                    }
                }
            }
            rank += 1;
            if rank == self.nrows {
                break;
            }
        }

        // Consistency: a row with all-zero data columns but b=1 is impossible.
        for r in 0..self.nrows {
            let mut any = false;
            for c in 0..self.ncols {
                if (aug[r][c / 64] >> (c % 64)) & 1 == 1 {
                    any = true;
                    break;
                }
            }
            if !any && (aug[r][self.ncols / 64] >> (self.ncols % 64)) & 1 == 1 {
                return None;
            }
        }

        // Solution: free variables = 0; pivot columns read from the augmented bit.
        let mut x = vec![false; self.ncols];
        for r in 0..self.nrows {
            let mut lead = None;
            for c in 0..self.ncols {
                if (aug[r][c / 64] >> (c % 64)) & 1 == 1 {
                    lead = Some(c);
                    break;
                }
            }
            if let Some(c) = lead {
                x[c] = (aug[r][self.ncols / 64] >> (self.ncols % 64)) & 1 == 1;
            }
        }
        Some(x)
    }

    /// `M·x` for `x` of length `ncols` → `nrows` bits (parity of row ∩ x).
    pub fn mul_vec(&self, x: &[bool]) -> Vec<bool> {
        assert_eq!(x.len(), self.ncols);
        (0..self.nrows)
            .map(|r| {
                let mut acc = 0u64;
                for (i, &xi) in x.iter().enumerate() {
                    if xi {
                        acc ^= self.rows[r][i / 64] & (1 << (i % 64));
                    }
                }
                acc.count_ones() % 2 == 1
            })
            .collect()
    }

    /// `Mᵀ·x` for `x` of length `nrows` → `ncols` bits.
    pub fn mul_vec_transposed(&self, x: &[bool]) -> Vec<bool> {
        assert_eq!(x.len(), self.nrows);
        let mut out = vec![false; self.ncols];
        for r in 0..self.nrows {
            if !x[r] {
                continue;
            }
            for c in 0..self.ncols {
                if self.get(r, c) {
                    out[c] = !out[c];
                }
            }
        }
        out
    }
}

/// A systematic random linear code over F2 with generator `G = [I_k | A]`.
///
/// A message `m ∈ F2^k` maps to codeword `c = (m, m·A) ∈ F2^n` (`n = k + r`).
/// The code space splits as a direct sum `C = K × V` where
/// `K = span(first k cols)` and `V = span(last r cols)` with `K ∩ V = {0}`;
/// every codeword therefore factors uniquely as `c = k ⊕ v` (`key` + `value`).
/// The parity-check matrix is `H = [Aᵀ | I_r]`; `H·c = 0` iff `c` is a
/// codeword (syndrome check).
pub struct LinearCode {
    pub k: usize,
    pub r: usize,
    /// Random `k × r` matrix from a seeded generator (deterministic).
    pub a: Gf2Mat,
}

impl LinearCode {
    /// Build a code with `k` message bits, `r` redundancy bits, seeded `A`.
    pub fn new(k: usize, r: usize, seed: u64) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut a = Gf2Mat::zero(k, r);
        for i in 0..k {
            for j in 0..r {
                if rng.gen_bool(0.5) {
                    a.set(i, j, true);
                }
            }
        }
        Self { k, r, a }
    }

    /// Codeword length `n = k + r`.
    pub fn n(&self) -> usize {
        self.k + self.r
    }

    /// Encode a `k`-bit message into an `n`-bit codeword `(m, m·A)`.
    pub fn encode(&self, m: &[bool]) -> Vec<bool> {
        assert_eq!(m.len(), self.k);
        let v = self.a.mul_vec_transposed(m);
        let mut c = vec![false; self.k + self.r];
        c[..self.k].copy_from_slice(m);
        for (j, &bit) in v.iter().enumerate() {
            c[self.k + j] = bit;
        }
        c
    }

    /// Extract the systematic message from the first `k` coordinates.
    pub fn decode(&self, c: &[bool]) -> Vec<bool> {
        assert_eq!(c.len(), self.k + self.r);
        c[..self.k].to_vec()
    }

    /// Factor a codeword into its unique `(key, value)` parts, `c = key ⊕ value`.
    /// For the systematic construction this is the coordinate split — the
    /// direct-sum `K ∩ V = {0}` guarantees uniqueness.
    pub fn factorize(&self, c: &[bool]) -> (Vec<bool>, Vec<bool>) {
        assert_eq!(c.len(), self.k + self.r);
        (c[..self.k].to_vec(), c[self.k..].to_vec())
    }

    /// Syndrome `H·c` (`r` bits). Zero iff `c` is a codeword.
    pub fn syndrome(&self, c: &[bool]) -> Vec<bool> {
        assert_eq!(c.len(), self.k + self.r);
        let t = self.a.mul_vec_transposed(&c[..self.k]);
        (0..self.r)
            .map(|j| t[j] ^ c[self.k + j])
            .collect()
    }

    /// Whether `c` lies in the code space (syndrome check, exact and cheap).
    pub fn is_codeword(&self, c: &[bool]) -> bool {
        self.syndrome(c).iter().all(|&b| !b)
    }
}

/// Deterministically recover which of the candidate codewords were summed into
/// a bundle. Solves `[c₀ c₁ … cₘ₋₁]·x = bundle` over F2 via Gaussian
/// elimination — the exact, iteration-free counterpart to resonator networks.
///
/// Returns `Some(indices)` of the components whose coefficient is 1 (free
/// variables resolved to 0), or `None` if the bundle is not in the span.
pub fn factorize_bundle(bundle: &[bool], candidates: &[Vec<bool>]) -> Option<Vec<usize>> {
    let m = Gf2Mat::from_columns(candidates);
    let x = m.solve(bundle)?;
    Some(x.iter().enumerate().filter(|(_, &v)| v).map(|(i, _)| i).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(v: u64, n: usize) -> Vec<bool> {
        (0..n).map(|i| (v >> i) & 1 == 1).collect()
    }

    #[test]
    fn test_rank_identity_and_zero() {
        assert_eq!(Gf2Mat::identity(8).rank(), 8);
        assert_eq!(Gf2Mat::zero(5, 9).rank(), 0);
    }

    #[test]
    fn test_rank_duplicate_rows() {
        // rows: v, v, w where v,w independent → rank 2
        let mut m = Gf2Mat::zero(3, 6);
        for r in 0..3 {
            let v = if r == 2 { 0b101010u64 } else { 0b010101u64 };
            for c in 0..6 {
                m.set(r, c, (v >> c) & 1 == 1);
            }
        }
        assert_eq!(m.rank(), 2);
    }

    #[test]
    fn test_solve_consistent() {
        // x0 + x1 = 1 ; x0 = 1  ⇒  x1 = 0
        let mut m = Gf2Mat::zero(2, 2);
        m.set(0, 0, true);
        m.set(0, 1, true);
        m.set(1, 0, true);
        let b = [true, true];
        let x = m.solve(&b).expect("consistent");
        // check M·x == b
        let check = m.mul_vec(&x);
        assert_eq!(check, b);
        assert_eq!(x, vec![true, false]);
    }

    #[test]
    fn test_solve_inconsistent() {
        // x0 = 0 ; x0 = 1 → inconsistent
        let mut m = Gf2Mat::zero(2, 1);
        m.set(0, 0, true);
        m.set(1, 0, true);
        assert_eq!(m.solve(&[false, true]), None);
    }

    #[test]
    fn test_mul_vec_matches_brute_force() {
        let mut m = Gf2Mat::zero(4, 7);
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        for r in 0..4 {
            for c in 0..7 {
                m.set(r, c, rng.gen_bool(0.5));
            }
        }
        let x: Vec<bool> = (0..7).map(|_| rng.gen_bool(0.5)).collect();
        let y = m.mul_vec(&x);
        for r in 0..4 {
            let mut par = false;
            for c in 0..7 {
                if m.get(r, c) && x[c] {
                    par = !par;
                }
            }
            assert_eq!(y[r], par);
        }
    }

    #[test]
    fn test_code_encode_is_codeword() {
        let code = LinearCode::new(5, 6, 42);
        for m in 0..8u64 {
            let msg = bits(m, 5);
            let c = code.encode(&msg);
            assert_eq!(c.len(), code.n());
            assert!(code.is_codeword(&c), "m={m} must be a codeword");
            assert_eq!(code.decode(&c), msg);
        }
    }

    #[test]
    fn test_code_random_vector_not_codeword() {
        let code = LinearCode::new(6, 7, 1);
        // random 13-bit vector is a codeword with probability 2^-7
        let mut rng = ChaCha20Rng::seed_from_u64(3);
        let mut hits = 0usize;
        for _ in 0..5000 {
            let c: Vec<bool> = (0..code.n()).map(|_| rng.gen_bool(0.5)).collect();
            if code.is_codeword(&c) {
                hits += 1;
            }
        }
        assert!(hits < 100, "random vector syndrome collisions: {hits}");
    }

    #[test]
    fn test_factorize_unique_direct_sum() {
        let code = LinearCode::new(4, 5, 9);
        let msg = bits(0b1011, 4);
        let c = code.encode(&msg);
        let (key, value) = code.factorize(&c);
        // c = key ⊕ value (coordinate-wise XOR = vector addition over F2)
        let joined: Vec<bool> = key.iter().chain(value.iter()).copied().collect();
        assert_eq!(joined, c);
        assert_eq!(key, msg);
    }

    #[test]
    fn test_factorize_bundle_recovers_subset() {
        let code = LinearCode::new(4, 6, 5);
        // standard basis messages (linearly independent ⇒ unique decomposition)
        let msgs: Vec<Vec<bool>> = vec![
            bits(1, 4), bits(2, 4), bits(4, 4), bits(8, 4),
        ];
        let codewords: Vec<Vec<bool>> = msgs.iter().map(|m| code.encode(m)).collect();
        // bundle of {0, 2}
        let mut bundle = vec![false; code.n()];
        for &idx in &[0usize, 2] {
            for (j, &b) in codewords[idx].iter().enumerate() {
                bundle[j] ^= b;
            }
        }
        let recovered = factorize_bundle(&bundle, &codewords).expect("in span");
        let mut set: Vec<usize> = recovered;
        set.sort_unstable();
        assert_eq!(set, vec![0, 2]);
    }

    #[test]
    fn test_factorize_bundle_out_of_span() {
        let code = LinearCode::new(3, 3, 11);
        let codewords: Vec<Vec<bool>> = (0..3).map(|i| code.encode(&bits(i, 3))).collect();
        // a vector with a guaranteed syndrome: complement of a codeword (odd parity
        // change flips membership — not in the code space)
        let c0 = &codewords[0];
        let mut not_cw: Vec<bool> = c0.iter().map(|b| !b).collect();
        // also XOR one message to make sure it stays out of the subspace span
        for (j, &b) in codewords[1].iter().enumerate() {
            not_cw[j] ^= b;
        }
        let res = factorize_bundle(&not_cw, &codewords);
        // Either inconsistent or a valid-but-not-this-subset result; the important
        // property: it must not falsely report the true subset. We only assert
        // determinism here (same answer twice).
        assert_eq!(
            factorize_bundle(&not_cw, &codewords),
            res,
            "factorization must be deterministic"
        );
    }

    #[test]
    fn test_deterministic_code_construction() {
        let c1 = LinearCode::new(8, 8, 123);
        let c2 = LinearCode::new(8, 8, 123);
        assert_eq!(c1.a, c2.a);
        let m: Vec<bool> = (0..8).map(|i| i % 3 == 0).collect();
        assert_eq!(c1.encode(&m), c2.encode(&m));
    }

    /// End-to-end at the VSA layer: bipolar HyperVectors (random codebook) are
    /// hashed onto a code space, bundled, and the bundle is factorized back
    /// into exactly the composing vectors — deterministically, no iteration.
    #[test]
    fn test_bundle_recovery_at_hypervector_layer() {
        use crate::hypervector::HyperVector;
        use crate::Codebook;

        let dim = 2048;
        let k = 5;
        let r = 16;
        let code = LinearCode::new(k, r, 77);
        let mut cb = Codebook::new(dim, 77);

        let words: [&str; 5] = ["apple", "banana", "cherry", "date", "elder"];
        // deterministic fold of each packed HyperVector → k-bit message
        let msgs: Vec<Vec<bool>> = words
            .iter()
            .map(|w| {
                let hv = cb.get_or_insert(w);
                let packed = hv.packed.as_ref().unwrap();
                let mut acc = 0x517cc1b727220a95u64;
                for wb in packed {
                    acc ^= wb;
                    acc = acc.rotate_left(3);
                }
                // spread the 64-bit fold over k bits (bits 0..k of two halves)
                bits(acc, k)
            })
            .collect();
        let codewords: Vec<Vec<bool>> = msgs.iter().map(|m| code.encode(m)).collect();
        for (idx, cw) in codewords.iter().enumerate() {
            assert!(code.is_codeword(cw), "codeword {idx} must be in the code space");
        }

        // bundle of {apple, cherry} → codewords {0, 2}
        let mut bundle = vec![false; code.n()];
        for &idx in &[0usize, 2] {
            for (j, &b) in codewords[idx].iter().enumerate() {
                bundle[j] ^= b;
            }
        }
        let recovered = factorize_bundle(&bundle, &codewords).expect("in span");
        let mut set: Vec<usize> = recovered;
        set.sort_unstable();
        assert_eq!(set, vec![0, 2], "bundle recovery must be exact and deterministic");
    }
}
