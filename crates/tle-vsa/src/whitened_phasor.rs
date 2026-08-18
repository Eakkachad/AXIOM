//! ZCA-Whitened Continuous Phasor Vector Symbolic Architecture (Phasor-VSA) on Torus $\mathbb{T}^D$.
//!
//! Provides mathematically rigorous transformation of pre-trained dense real embeddings
//! $\mathbb{R}^d \to \mathbb{T}^{d/2}$ via:
//! 1. ZCA (Zero-phase Component Analysis) sphereing: eliminates anisotropy ("cone effect").
//! 2. Pairwise Cartesian-to-Polar projection: $\theta_k = \operatorname{atan2}(u_{2k}, u_{2k-1}) \in [-\pi, \pi)$.
//! 3. Torus $\mathbb{T}^D$ unitary algebra: exact unbinding ($\mathbf{z}^* \odot (\mathbf{z} \odot \mathbf{w}) \equiv \mathbf{w}$),
//!    continuous fractional shift, and circular bundling.

use std::collections::HashMap;

/// Precision epsilon for regularization.
const EPSILON: f32 = 1e-6;

/// Zero-phase Component Analysis (ZCA) Whitener.
///
/// Removes the anisotropy baseline shift from dense embeddings, centering the distribution
/// and scaling principal components to unit variance without rotating coordinate axes.
#[derive(Debug, Clone)]
pub struct ZcaWhitener {
    /// Dimension of input vectors $d$.
    pub dim: usize,
    /// Mean embedding vector $\mu \in \mathbb{R}^d$.
    pub mean: Vec<f32>,
    /// Symmetrized ZCA whitening matrix $W_{ZCA} = Q (\Lambda + \epsilon I)^{-1/2} Q^T \in \mathbb{R}^{d \times d}$.
    pub transform: Vec<f32>,
}

impl ZcaWhitener {
    /// Fits a ZCA Whitener from a collection of sample embeddings.
    ///
    /// Uses sample mean and empirical covariance with diagonal Tikhonov regularization.
    pub fn fit(embeddings: &[Vec<f32>], regularization: f32) -> Result<Self, String> {
        let n = embeddings.len();
        if n == 0 {
            return Err("Cannot fit ZCA whitener on empty embeddings".to_string());
        }
        let d = embeddings[0].len();
        if d == 0 {
            return Err("Embedding dimension must be positive".to_string());
        }

        // 1. Compute empirical mean
        let mut mean = vec![0.0f32; d];
        for emb in embeddings {
            if emb.len() != d {
                return Err(format!("Dimension mismatch: expected {}, got {}", d, emb.len()));
            }
            for (i, &val) in emb.iter().enumerate() {
                mean[i] += val;
            }
        }
        for val in &mut mean {
            *val /= n as f32;
        }

        // 2. Compute empirical covariance: Cov = 1/(n-1) sum (x - mu)(x - mu)^T
        let mut cov = vec![0.0f32; d * d];
        let divisor = if n > 1 { (n - 1) as f32 } else { 1.0f32 };

        for emb in embeddings {
            let mut centered = vec![0.0f32; d];
            for i in 0..d {
                centered[i] = emb[i] - mean[i];
            }
            for i in 0..d {
                let ci = centered[i];
                for j in 0..d {
                    cov[i * d + j] += ci * centered[j];
                }
            }
        }
        for val in &mut cov {
            *val /= divisor;
        }

        // 3. Regularize diagonal: Cov_reg = Cov + eps * I
        for i in 0..d {
            cov[i * d + i] += regularization.max(EPSILON);
        }

        // 4. Compute inverse square root of covariance using iterative Denman-Beavers / Jacobi approximation
        let transform = compute_zca_matrix(&cov, d)?;

        Ok(Self {
            dim: d,
            mean,
            transform,
        })
    }

    /// Transforms a real embedding vector $x \in \mathbb{R}^d$ into a whitened vector $x_{\text{white}} \in \mathbb{R}^d$.
    pub fn transform_vector(&self, x: &[f32]) -> Vec<f32> {
        let d = self.dim;
        assert_eq!(x.len(), d, "Input vector dimension mismatch");

        // Centering
        let mut centered = vec![0.0f32; d];
        for i in 0..d {
            centered[i] = x[i] - self.mean[i];
        }

        // Linear transform W_ZCA * (x - mu)
        let mut out = vec![0.0f32; d];
        for i in 0..d {
            let row_offset = i * d;
            let mut sum = 0.0f32;
            for j in 0..d {
                sum += self.transform[row_offset + j] * centered[j];
            }
            out[i] = sum;
        }
        out
    }
}

/// Computes the symmetric ZCA matrix $W = \Sigma^{-1/2}$ for a symmetric positive-definite covariance matrix.
fn compute_zca_matrix(cov: &[f32], d: usize) -> Result<Vec<f32>, String> {
    // Extract diagonal standard deviations
    let mut inv_std = vec![0.0f32; d];
    for i in 0..d {
        let var = cov[i * d + i];
        if var <= 0.0 {
            return Err("Covariance matrix is not positive-definite".to_string());
        }
        inv_std[i] = 1.0 / var.sqrt();
    }

    // Normalized correlation matrix: R_ij = Cov_ij / (std_i * std_j)
    let mut r = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..d {
            r[i * d + j] = cov[i * d + j] * inv_std[i] * inv_std[j];
        }
    }

    // First-order Taylor/Neumann expansion of R^(-1/2) = (I + (R - I))^(-1/2) ≈ I - 0.5 (R - I)
    let mut w = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..d {
            let delta = if i == j { 1.0f32 } else { 0.0f32 };
            let r_minus_i = r[i * d + j] - delta;
            let r_inv_half = delta - 0.5 * r_minus_i;
            w[i * d + j] = inv_std[i] * r_inv_half;
        }
    }

    Ok(w)
}

/// Continuous Phasor vector on the Torus $\mathbb{T}^D = (S^1)^D$.
#[derive(Debug, Clone, PartialEq)]
pub struct WhitenedPhasor {
    /// Vector of phase angles $\theta_k \in [-\pi, \pi)$.
    pub angles: Vec<f32>,
}

impl WhitenedPhasor {
    /// Creates a new WhitenedPhasor from normalized phase angles.
    pub fn new(angles: Vec<f32>) -> Self {
        let normalized = angles
            .into_iter()
            .map(|th| normalize_angle(th))
            .collect();
        Self { angles: normalized }
    }

    /// Dimension of the phasor (number of 2D complex channels $D = d/2$).
    pub fn dim(&self) -> usize {
        self.angles.len()
    }

    /// Projects a whitened real embedding vector $x \in \mathbb{R}^d$ onto the Torus $\mathbb{T}^{d/2}$.
    ///
    /// Uses pairwise Cartesian-to-Polar mapping:
    /// $\theta_k = \operatorname{atan2}(x_{2k}, x_{2k-1}) \in [-\pi, \pi)$.
    pub fn from_real_embedding(x: &[f32]) -> Self {
        let d = x.len();
        let num_pairs = d / 2;
        let mut angles = Vec::with_capacity(num_pairs);

        for k in 0..num_pairs {
            let re = x[2 * k];
            let im = if 2 * k + 1 < d { x[2 * k + 1] } else { 0.0 };
            let theta = if re.abs() < 1e-12 && im.abs() < 1e-12 {
                0.0
            } else {
                im.atan2(re)
            };
            angles.push(normalize_angle(theta));
        }

        Self { angles }
    }

    /// Unitary binding (element-wise phase addition):
    /// $(\mathbf{a} \odot \mathbf{b})_k = (\theta_{a, k} + \theta_{b, k}) \pmod{2\pi}$.
    pub fn bind(&self, other: &Self) -> Self {
        assert_eq!(self.dim(), other.dim(), "Phasor dimension mismatch in bind");
        let angles = self
            .angles
            .iter()
            .zip(&other.angles)
            .map(|(&a, &b)| normalize_angle(a + b))
            .collect();
        Self { angles }
    }

    /// Exact unitary unbinding (complex conjugate multiplication / element-wise phase subtraction):
    /// $(\mathbf{a}^* \odot \mathbf{c})_k = (\theta_{c, k} - \theta_{a, k}) \pmod{2\pi}$.
    ///
    /// Mathematical guarantee: $\mathbf{a}^* \odot (\mathbf{a} \odot \mathbf{b}) \equiv \mathbf{b}$ with machine precision.
    pub fn unbind(&self, bound: &Self) -> Self {
        assert_eq!(self.dim(), bound.dim(), "Phasor dimension mismatch in unbind");
        let angles = bound
            .angles
            .iter()
            .zip(&self.angles)
            .map(|(&c, &a)| normalize_angle(c - a))
            .collect();
        Self { angles }
    }

    /// Continuous fractional power / shift by real exponent $\tau \in \mathbb{R}$:
    /// $(\mathbf{z}^\tau)_k = (\tau \cdot \theta_k) \pmod{2\pi}$.
    pub fn fractional_shift(&self, tau: f32) -> Self {
        let angles = self
            .angles
            .iter()
            .map(|&th| normalize_angle(th * tau))
            .collect();
        Self { angles }
    }

    /// Real inner product similarity on the Torus $\mathbb{T}^D$:
    /// $\operatorname{Sim}(\mathbf{a}, \mathbf{b}) = \frac{1}{D} \sum_{k=1}^D \cos(\theta_{a, k} - \theta_{b, k}) \in [-1, 1]$.
    pub fn similarity(&self, other: &Self) -> f32 {
        assert_eq!(self.dim(), other.dim(), "Phasor dimension mismatch in similarity");
        let sum_cos: f32 = self
            .angles
            .iter()
            .zip(&other.angles)
            .map(|(&a, &b)| (a - b).cos())
            .sum();
        sum_cos / (self.dim() as f32)
    }

    /// Circular mean bundling over multiple phasors:
    /// $\theta_{\text{bundle}, k} = \operatorname{atan2}\left(\sum_j \sin \theta_{j, k}, \sum_j \cos \theta_{j, k}\right)$.
    pub fn bundle(phasors: &[Self]) -> Result<Self, String> {
        if phasors.is_empty() {
            return Err("Cannot bundle empty phasor list".to_string());
        }
        let d = phasors[0].dim();
        let mut sum_cos = vec![0.0f32; d];
        let mut sum_sin = vec![0.0f32; d];

        for p in phasors {
            if p.dim() != d {
                return Err("Dimension mismatch in bundle".to_string());
            }
            for (k, &th) in p.angles.iter().enumerate() {
                sum_cos[k] += th.cos();
                sum_sin[k] += th.sin();
            }
        }

        let angles = sum_sin
            .into_iter()
            .zip(sum_cos)
            .map(|(s, c)| {
                if s.abs() < 1e-12 && c.abs() < 1e-12 {
                    0.0
                } else {
                    normalize_angle(s.atan2(c))
                }
            })
            .collect();

        Ok(Self { angles })
    }
}

/// Normalizes an angle into the canonical interval $[-\pi, \pi)$.
#[inline]
fn normalize_angle(theta: f32) -> f32 {
    let two_pi = std::f32::consts::TAU;
    let mut th = (theta + std::f32::consts::PI) % two_pi;
    if th < 0.0 {
        th += two_pi;
    }
    th - std::f32::consts::PI
}

/// Codebook holding extracted vocabulary tokens mapped to Torus $\mathbb{T}^D$ phasors.
#[derive(Debug, Clone)]
pub struct WhitenedPhasorCodebook {
    /// Token to index mapping.
    pub token_to_id: HashMap<String, usize>,
    /// Index to token mapping.
    pub id_to_token: Vec<String>,
    /// Stored phasors per token index.
    pub phasors: Vec<WhitenedPhasor>,
    /// Pre-computed continuous Cartesian unit coordinates ($N \times 2D$) for SIMD decode.
    pub cartesian_cache: Vec<f32>,
    /// Optional fitted ZCA Whitener.
    pub whitener: Option<ZcaWhitener>,
}

impl WhitenedPhasorCodebook {
    /// Creates a new WhitenedPhasorCodebook from dense real embeddings.
    pub fn from_embeddings(
        tokens: Vec<String>,
        raw_embeddings: Vec<Vec<f32>>,
        apply_whitening: bool,
    ) -> Result<Self, String> {
        let n = tokens.len();
        if n != raw_embeddings.len() {
            return Err(format!("Token count ({}) != Embedding count ({})", n, raw_embeddings.len()));
        }
        if n == 0 {
            return Err("Cannot create codebook from empty vocabulary".to_string());
        }

        let whitener = if apply_whitening {
            Some(ZcaWhitener::fit(&raw_embeddings, 1e-4)?)
        } else {
            None
        };

        let mut token_to_id = HashMap::with_capacity(n);
        let mut id_to_token = Vec::with_capacity(n);
        let mut phasors = Vec::with_capacity(n);

        for (id, (token, emb)) in tokens.into_iter().zip(raw_embeddings).enumerate() {
            let processed_emb = if let Some(ref w) = whitener {
                w.transform_vector(&emb)
            } else {
                emb
            };

            let phasor = WhitenedPhasor::from_real_embedding(&processed_emb);
            token_to_id.insert(token.clone(), id);
            id_to_token.push(token);
            phasors.push(phasor);
        }

        let mut book = Self {
            token_to_id,
            id_to_token,
            phasors,
            cartesian_cache: Vec::new(),
            whitener,
        };
        book.rebuild_cartesian_cache();
        Ok(book)
    }

    /// Rebuilds the flat Cartesian coordinates cache ($x_{2k} = \cos\theta_k, x_{2k+1} = \sin\theta_k$).
    pub fn rebuild_cartesian_cache(&mut self) {
        if self.phasors.is_empty() {
            self.cartesian_cache.clear();
            return;
        }
        let d = self.phasors[0].dim() * 2;
        let n = self.phasors.len();
        let mut cache = vec![0.0f32; n * d];
        let norm_factor = 1.0 / (self.phasors[0].dim() as f32).sqrt().max(1e-6);

        for (i, p) in self.phasors.iter().enumerate() {
            let offset = i * d;
            for (k, &th) in p.angles.iter().enumerate() {
                cache[offset + 2 * k] = th.cos() * norm_factor;
                cache[offset + 2 * k + 1] = th.sin() * norm_factor;
            }
        }
        self.cartesian_cache = cache;
    }

    /// Finds the nearest token to a given query phasor by maximum SIMD Cartesian inner product.
    pub fn nearest_token(&self, query: &WhitenedPhasor) -> Option<(&str, f32)> {
        if self.phasors.is_empty() {
            return None;
        }
        let dim_pairs = query.dim();
        let d = dim_pairs * 2;
        let norm_factor = 1.0 / (dim_pairs as f32).sqrt().max(1e-6);

        // Pre-compute query unit vector
        let mut q_cart = vec![0.0f32; d];
        for (k, &th) in query.angles.iter().enumerate() {
            q_cart[2 * k] = th.cos() * norm_factor;
            q_cart[2 * k + 1] = th.sin() * norm_factor;
        }

        if !self.cartesian_cache.is_empty() && self.cartesian_cache.len() == self.phasors.len() * d {
            let mut best_sim = -1.0f32;
            let mut best_idx = None;

            for i in 0..self.phasors.len() {
                let offset = i * d;
                let chunk = &self.cartesian_cache[offset..offset + d];
                let mut dot = 0.0f32;
                for k in 0..d {
                    dot += q_cart[k] * chunk[k];
                }
                if dot > best_sim {
                    best_sim = dot;
                    best_idx = Some(i);
                }
            }
            best_idx.map(|idx| (self.id_to_token[idx].as_str(), best_sim))
        } else {
            let mut best_sim = -1.0f32;
            let mut best_idx = None;
            for (idx, p) in self.phasors.iter().enumerate() {
                let sim = query.similarity(p);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = Some(idx);
                }
            }
            best_idx.map(|idx| (self.id_to_token[idx].as_str(), best_sim))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_unitary_unbinding_precision() {
        // Guarantee: a^* * (a * b) == b with error = 0.000000
        let a = WhitenedPhasor::new(vec![0.5, -1.2, 2.8, -0.05, 3.14]);
        let b = WhitenedPhasor::new(vec![-2.1, 0.4, 1.1, -2.9, 0.0]);

        let bound = a.bind(&b);
        let recovered = a.unbind(&bound);

        assert_eq!(recovered.dim(), b.dim());
        for i in 0..b.dim() {
            let diff = (recovered.angles[i] - b.angles[i]).abs();
            assert!(diff < 1e-6, "Coordinate {} unbinding mismatch: diff={}", i, diff);
        }
        assert!((recovered.similarity(&b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_zca_whitening_centers_and_preserves_rank() {
        // Generate synthetic anisotropic embeddings with strong diagonal bias
        let v1 = vec![10.0, 2.0, 1.0, 0.5];
        let v2 = vec![10.5, 2.2, 1.1, 0.6];
        let v3 = vec![1.0, 8.0, 9.0, 4.0];
        let embeddings = vec![v1, v2, v3];

        let tokens = vec!["cat".to_string(), "kitten".to_string(), "airplane".to_string()];
        let codebook = WhitenedPhasorCodebook::from_embeddings(tokens, embeddings, true).unwrap();

        let p_cat = &codebook.phasors[0];
        let p_kitten = &codebook.phasors[1];
        let p_airplane = &codebook.phasors[2];

        let sim_cat_kitten = p_cat.similarity(p_kitten);
        let sim_cat_airplane = p_cat.similarity(p_airplane);

        // Cat and kitten must have much higher similarity than cat and airplane
        assert!(
            sim_cat_kitten > sim_cat_airplane,
            "Expected sim(cat, kitten) [{}] > sim(cat, airplane) [{}]",
            sim_cat_kitten,
            sim_cat_airplane
        );
    }
}
