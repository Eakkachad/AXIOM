//! # Cellular Sheaf Context Layer (Deterministic / Non-Neural Attention Alternative)
//!
//! Replaces standard Transformer Self-Attention with Cellular Sheaf Diffusion:
//!   dx/dt = - L_F(t) x
//!
//! Key Mathematical Guarantees:
//! - Orthogonal Parallel Transport P_{i <- j} ∈ SO(d) evaluated in O(d) via Cayley-Woodbury formula.
//! - Non-trivial bundle curvature (holonomy) prevents oversmoothing (dim H^0 = 0).
//! - Deterministic topological routing without learnable projection weight matrices (W_Q, W_K, W_V).

/// Mode of orthogonal parallel transport map P_{i <- j} ∈ SO(d) / O(d).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotorType {
    /// Rank-2 Cayley transform with Sherman-Morrison-Woodbury O(d) action.
    CayleyWoodbury,
    /// Direct Clifford planar rotor in span(u, v).
    CliffordPlanar,
    /// Block-diagonal SO(2)^(d/2) phase rotor (Topological RoPE).
    BlockPlanar,
    /// Trivial identity map (Standard Attention baseline for A/B testing).
    TrivialIdentity,
}

/// Routing topology and diffusion configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SheafConfig {
    pub stalk_dim: usize,
    pub diffusion_steps: usize,
    pub step_size: f32,
    pub rotor_type: RotorType,
    pub causal: bool,
    pub similarity_threshold: f32,
    pub kernel_power: f32,
}

impl Default for SheafConfig {
    fn default() -> Self {
        Self {
            stalk_dim: 64,
            diffusion_steps: 2,
            step_size: 0.5,
            rotor_type: RotorType::CayleyWoodbury,
            causal: false,
            similarity_threshold: 0.0,
            kernel_power: 2.0,
        }
    }
}

/// Computes inner product between two d-dimensional vectors.
#[inline(always)]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Computes Euclidean norm of a vector.
#[inline(always)]
pub fn norm_l2(a: &[f32]) -> f32 {
    dot_product(a, a).sqrt().max(1e-12)
}

/// Applies Cayley parallel transport P_{u <- v} x in exact O(d) operations.
///
/// P = (I - 1/2 A)^(-1) (I + 1/2 A) where A = (u v^T - v u^T) / (||u|| ||v|| + eps).
/// Uses Sherman-Morrison-Woodbury rank-2 formula.
pub fn apply_cayley_transport(u: &[f32], v: &[f32], x: &[f32], out: &mut [f32]) {
    let d = u.len();
    debug_assert_eq!(v.len(), d);
    debug_assert_eq!(x.len(), d);
    debug_assert_eq!(out.len(), d);

    let norm_u = norm_l2(u);
    let norm_v = norm_l2(v);
    let scale = 1.0 / (norm_u * norm_v + 1e-8);

    // Compute scalar projections
    let u_dot_v = dot_product(u, v) * scale;
    let u_norm_sq = dot_product(u, u) * scale;
    let v_norm_sq = dot_product(v, v) * scale;

    let v_dot_x = dot_product(v, x) * scale;
    let u_dot_x = dot_product(u, x) * scale;

    // First step: w = (I + 1/2 A) x = x + 0.5 * (u (v^T x) - v (u^T x))
    let mut w = vec![0.0f32; d];
    for i in 0..d {
        w[i] = x[i] + 0.5 * (u[i] * v_dot_x - v[i] * u_dot_x);
    }

    // Second step: solve (I - 1/2 A) y = w using Woodbury identity
    let v_dot_w = dot_product(v, &w) * scale;
    let u_dot_w = dot_product(u, &w) * scale;

    let m11 = 1.0 - 0.5 * u_dot_v;
    let m12 = 0.5 * v_norm_sq;
    let m21 = -0.5 * u_norm_sq;
    let m22 = 1.0 + 0.5 * u_dot_v;

    let det_m = m11 * m22 - m12 * m21;
    let inv_det = 1.0 / det_m.max(1e-12);

    // M^(-1) * [v^T w; u^T w]
    let z1 = inv_det * (m22 * v_dot_w - m12 * u_dot_w);
    let z2 = inv_det * (-m21 * v_dot_w + m11 * u_dot_w);

    // out = w + 1/2 U * z = w + 0.5 * (u * z1 - v * z2)
    for i in 0..d {
        out[i] = w[i] + 0.5 * (u[i] * z1 - v[i] * z2);
    }
}

/// Applies Clifford planar rotor in span(u, v) in O(d) operations.
pub fn apply_clifford_transport(u: &[f32], v: &[f32], x: &[f32], out: &mut [f32]) {
    let d = u.len();
    let norm_u = norm_l2(u);
    let norm_v = norm_l2(v);

    let mut e1 = vec![0.0f32; d];
    for i in 0..d {
        e1[i] = u[i] / norm_u;
    }

    let u_dot_v = dot_product(&e1, v);
    let mut e2 = vec![0.0f32; d];
    for i in 0..d {
        e2[i] = v[i] - u_dot_v * e1[i];
    }
    let norm_e2 = norm_l2(&e2);

    if norm_e2 < 1e-6 {
        out.copy_from_slice(x);
        return;
    }

    for i in 0..d {
        e2[i] /= norm_e2;
    }

    let cos_theta = (dot_product(u, v) / (norm_u * norm_v)).clamp(-1.0, 1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

    let e1_dot_x = dot_product(&e1, x);
    let e2_dot_x = dot_product(&e2, x);

    for i in 0..d {
        out[i] = x[i]
            + (cos_theta - 1.0) * (e1_dot_x * e1[i] + e2_dot_x * e2[i])
            + sin_theta * (e1_dot_x * e2[i] - e2_dot_x * e1[i]);
    }
}

/// Applies block-planar SO(2)^(d/2) rotor in O(d) operations.
pub fn apply_block_planar_transport(u: &[f32], v: &[f32], x: &[f32], out: &mut [f32]) {
    let d = u.len();
    let pairs = d / 2;

    for k in 0..pairs {
        let idx = 2 * k;
        let phi_u = u[idx + 1].atan2(u[idx]);
        let phi_v = v[idx + 1].atan2(v[idx]);
        let dphi = phi_u - phi_v;

        let cos_phi = dphi.cos();
        let sin_phi = dphi.sin();

        out[idx] = cos_phi * x[idx] - sin_phi * x[idx + 1];
        out[idx + 1] = sin_phi * x[idx] + cos_phi * x[idx + 1];
    }

    if d % 2 == 1 {
        out[d - 1] = x[d - 1];
    }
}

/// Dispatches parallel transport based on rotor configuration.
pub fn parallel_transport(rotor: RotorType, u: &[f32], v: &[f32], x: &[f32], out: &mut [f32]) {
    match rotor {
        RotorType::CayleyWoodbury => apply_cayley_transport(u, v, x, out),
        RotorType::CliffordPlanar => apply_clifford_transport(u, v, x, out),
        RotorType::BlockPlanar => apply_block_planar_transport(u, v, x, out),
        RotorType::TrivialIdentity => out.copy_from_slice(x),
    }
}

/// Multi-token Cellular Sheaf Context Layer.
pub struct SheafContextLayer {
    pub config: SheafConfig,
}

impl SheafContextLayer {
    pub fn new(config: SheafConfig) -> Self {
        Self { config }
    }

    /// Executes multi-step Sheaf Diffusion across a sequence of token stalks.
    /// Tokens shape: [N, D] where N = sequence length, D = stalk dimension.
    pub fn forward(&self, tokens: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = tokens.len();
        if n == 0 {
            return Vec::new();
        }
        let d = self.config.stalk_dim;

        let mut current_state = tokens.to_vec();
        let mut next_state = vec![vec![0.0f32; d]; n];
        let mut transported_buf = vec![0.0f32; d];

        for _step in 0..self.config.diffusion_steps {
            for i in 0..n {
                let mut weight_sum = 0.0f32;
                let mut accumulated = vec![0.0f32; d];

                for j in 0..n {
                    if self.config.causal && j > i {
                        continue;
                    }

                    // Compute topological similarity kernel
                    let norm_i = norm_l2(&current_state[i]);
                    let norm_j = norm_l2(&current_state[j]);
                    let cos_sim = (dot_product(&current_state[i], &current_state[j])
                        / (norm_i * norm_j))
                        .clamp(-1.0, 1.0);

                    let routed_sim = (cos_sim - self.config.similarity_threshold).max(0.0);
                    let weight = routed_sim.powf(self.config.kernel_power);

                    if weight > 1e-7 {
                        parallel_transport(
                            self.config.rotor_type,
                            &current_state[i],
                            &current_state[j],
                            &current_state[j],
                            &mut transported_buf,
                        );

                        for k in 0..d {
                            accumulated[k] += weight * transported_buf[k];
                        }
                        weight_sum += weight;
                    }
                }

                // Discretized Sheaf Diffusion Step: x_i^(t+1) = (1 - tau) x_i^(t) + tau * (Aggregated)
                let inv_weight = if weight_sum > 1e-8 {
                    1.0 / weight_sum
                } else {
                    0.0
                };
                for k in 0..d {
                    let diffused = accumulated[k] * inv_weight;
                    next_state[i][k] = (1.0 - self.config.step_size) * current_state[i][k]
                        + self.config.step_size * diffused;
                }
            }
            current_state.clone_from(&next_state);
        }

        current_state
    }

    /// Computes the Sheaf Dirichlet Energy: E_F(x) = 1/2 sum_{i,j} w_ij ||x_i - P_{i <- j} x_j||^2.
    pub fn compute_dirichlet_energy(&self, tokens: &[Vec<f32>]) -> f32 {
        let n = tokens.len();
        let d = self.config.stalk_dim;
        let mut total_energy = 0.0f32;
        let mut transported = vec![0.0f32; d];

        for i in 0..n {
            for j in 0..n {
                if self.config.causal && j > i {
                    continue;
                }
                let norm_i = norm_l2(&tokens[i]);
                let norm_j = norm_l2(&tokens[j]);
                let cos_sim =
                    (dot_product(&tokens[i], &tokens[j]) / (norm_i * norm_j)).clamp(-1.0, 1.0);
                let weight = (cos_sim - self.config.similarity_threshold)
                    .max(0.0)
                    .powf(self.config.kernel_power);

                if weight > 1e-7 {
                    parallel_transport(
                        self.config.rotor_type,
                        &tokens[i],
                        &tokens[j],
                        &tokens[j],
                        &mut transported,
                    );

                    let mut diff_sq = 0.0f32;
                    for k in 0..d {
                        let diff = tokens[i][k] - transported[k];
                        diff_sq += diff * diff;
                    }
                    total_energy += 0.5 * weight * diff_sq;
                }
            }
        }
        total_energy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cayley_transport_preserves_norm() {
        let d = 32;
        let u: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();
        let v: Vec<f32> = (0..d).map(|i| (i as f32 * 0.2 + 0.5).cos()).collect();
        let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.3 + 1.0).sin()).collect();

        let mut px = vec![0.0f32; d];
        apply_cayley_transport(&u, &v, &x, &mut px);

        let initial_norm = norm_l2(&x);
        let transported_norm = norm_l2(&px);

        assert!(
            (initial_norm - transported_norm).abs() < 1e-4,
            "Cayley transport must be orthogonal and preserve norm! Got {} vs {}",
            initial_norm,
            transported_norm
        );
    }

    #[test]
    fn test_sheaf_diffusion_prevents_oversmoothing() {
        let n = 8;
        let d = 16;
        let initial_tokens: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..d).map(|j| ((i * d + j) as f32 * 0.17).sin()).collect())
            .collect();

        // 1. Trivial Sheaf (Standard Attention)
        let config_trivial = SheafConfig {
            stalk_dim: d,
            diffusion_steps: 10,
            step_size: 0.8,
            rotor_type: RotorType::TrivialIdentity,
            causal: false,
            similarity_threshold: -1.0,
            kernel_power: 1.0,
        };
        let trivial_layer = SheafContextLayer::new(config_trivial);
        let trivial_out = trivial_layer.forward(&initial_tokens);

        let trivial_var = compute_inter_token_variance(&trivial_out);

        // 2. Non-Trivial Sheaf Diffusion (Cayley Rotor)
        let config_sheaf = SheafConfig {
            stalk_dim: d,
            diffusion_steps: 10,
            step_size: 0.8,
            rotor_type: RotorType::CayleyWoodbury,
            causal: false,
            similarity_threshold: -1.0,
            kernel_power: 1.0,
        };
        let sheaf_layer = SheafContextLayer::new(config_sheaf);
        let sheaf_out = sheaf_layer.forward(&initial_tokens);

        let sheaf_var = compute_inter_token_variance(&sheaf_out);

        assert!(
            sheaf_var > trivial_var * 1.2,
            "Sheaf Diffusion must preserve feature variance against oversmoothing! (sheaf: {}, trivial: {})",
            sheaf_var, trivial_var
        );
    }

    fn compute_inter_token_variance(tokens: &[Vec<f32>]) -> f32 {
        let n = tokens.len();
        let d = tokens[0].len();
        let mut mean = vec![0.0f32; d];
        for t in tokens {
            for k in 0..d {
                mean[k] += t[k] / n as f32;
            }
        }
        let mut total_var = 0.0f32;
        for t in tokens {
            for k in 0..d {
                let diff = t[k] - mean[k];
                total_var += diff * diff;
            }
        }
        total_var / n as f32
    }
}
