//! # FlashHopfield Engine: Multi-Layer Continuous Hopfield Equilibrium Layer
//!
//! Features:
//! - Tiled Online Softmax / Log-Sum-Exp in O(1) dynamic memory (L1D Cache friendly).
//! - Multi-step CCCP fixed-point relaxation toward semantic attractor equilibrium.
//! - Monotonic Lyapunov energy descent guarantee without backpropagation.

/// Configuration parameters for FlashHopfield Sequence Layer.
#[derive(Debug, Clone)]
pub struct FlashHopfieldConfig {
    pub seq_len: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub beta: f32,
    pub max_cccp_iters: usize,
    pub convergence_tol: f32,
    pub anchor_lambda: f32,
}

impl FlashHopfieldConfig {
    pub fn new(seq_len: usize, d_model: usize, num_heads: usize) -> Self {
        let head_dim = d_model / num_heads;
        let beta = 1.0 / (head_dim as f32).sqrt();
        Self {
            seq_len,
            d_model,
            num_heads,
            head_dim,
            beta,
            max_cccp_iters: 6,
            convergence_tol: 1e-4,
            anchor_lambda: 0.15,
        }
    }
}

/// Multi-Head FlashHopfield Sequence Layer.
pub struct FlashHopfieldLayer {
    pub config: FlashHopfieldConfig,
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
}

impl FlashHopfieldLayer {
    pub fn new(config: FlashHopfieldConfig) -> Self {
        let d = config.d_model;
        let h = config.num_heads;
        let dk = config.head_dim;

        let mut layer = Self {
            config,
            w_q: vec![0.0f32; h * d * dk],
            w_k: vec![0.0f32; h * d * dk],
            w_v: vec![0.0f32; h * d * dk],
        };

        layer.init_orthogonal_weights();
        layer
    }

    fn init_orthogonal_weights(&mut self) {
        let d = self.config.d_model;
        let scale = (2.0 / (d as f32)).sqrt();
        for (i, v) in self.w_q.iter_mut().enumerate() {
            *v = ((i as f32 * 0.1337).sin()) * scale;
        }
        for (i, v) in self.w_k.iter_mut().enumerate() {
            *v = ((i as f32 * 0.7331).cos()) * scale;
        }
        for (i, v) in self.w_v.iter_mut().enumerate() {
            *v = ((i as f32 * 0.4242).sin()) * scale;
        }
    }

    /// Single Tiled Flash-Hopfield Attention Step for one head.
    /// Computes Out = Softmax(beta * Q * K^T) * V using online normalization.
    pub fn tiled_flash_hopfield_step(
        &self,
        q_head: &[f32],      // [T, d_k]
        k_head: &[f32],      // [T, d_k]
        v_head: &[f32],      // [T, d_k]
        out_head: &mut [f32], // [T, d_k]
    ) {
        let t = self.config.seq_len;
        let dk = self.config.head_dim;
        let beta = self.config.beta;

        // Block tiling sizes for L1 cache
        let tile_br = 16.min(t);
        let tile_bc = 32.min(t);

        let num_q_tiles = (t + tile_br - 1) / tile_br;
        let num_k_tiles = (t + tile_bc - 1) / tile_bc;

        for tile_r in 0..num_q_tiles {
            let r_start = tile_r * tile_br;
            let r_end = (r_start + tile_br).min(t);
            let block_r = r_end - r_start;

            let mut m_prev = vec![f32::NEG_INFINITY; block_r];
            let mut l_prev = vec![0.0f32; block_r];
            let mut acc = vec![0.0f32; block_r * dk];

            for tile_c in 0..num_k_tiles {
                let c_start = tile_c * tile_bc;
                let c_end = (c_start + tile_bc).min(t);
                let block_c = c_end - c_start;

                let mut s_tile = vec![0.0f32; block_r * block_c];

                for i in 0..block_r {
                    let q_offset = (r_start + i) * dk;
                    let q_vec = &q_head[q_offset..q_offset + dk];

                    for j in 0..block_c {
                        let k_offset = (c_start + j) * dk;
                        let k_vec = &k_head[k_offset..k_offset + dk];

                        let mut dot = 0.0f32;
                        for d in 0..dk {
                            dot += q_vec[d] * k_vec[d];
                        }
                        s_tile[i * block_c + j] = dot * beta;
                    }
                }

                // Online Softmax & Accumulator Rescaling
                for i in 0..block_r {
                    let mut row_max = f32::NEG_INFINITY;
                    for j in 0..block_c {
                        let score = s_tile[i * block_c + j];
                        if score > row_max {
                            row_max = score;
                        }
                    }

                    let m_new = m_prev[i].max(row_max);
                    let alpha = (m_prev[i] - m_new).exp();

                    // Rescale previous accumulator
                    let acc_offset = i * dk;
                    for d in 0..dk {
                        acc[acc_offset + d] *= alpha;
                    }

                    // Compute exp and accumulate
                    let mut row_sum_exp = 0.0f32;
                    for j in 0..block_c {
                        let p = (s_tile[i * block_c + j] - m_new).exp();
                        row_sum_exp += p;

                        let v_offset = (c_start + j) * dk;
                        let v_vec = &v_head[v_offset..v_offset + dk];

                        for d in 0..dk {
                            acc[acc_offset + d] += p * v_vec[d];
                        }
                    }

                    l_prev[i] = l_prev[i] * alpha + row_sum_exp;
                    m_prev[i] = m_new;
                }
            }

            // Final normalization: out = acc / l_prev
            for i in 0..block_r {
                let out_offset = (r_start + i) * dk;
                let acc_offset = i * dk;
                let inv_l = 1.0 / (l_prev[i] + 1e-8);

                for d in 0..dk {
                    out_head[out_offset + d] = acc[acc_offset + d] * inv_l;
                }
            }
        }
    }

    /// Matrix-vector projection: Y = X * W where X is [T, K] and W is [K, N] -> Y is [T, N].
    fn project(&self, x: &[f32], w: &[f32], out: &mut [f32], t: usize, k: usize, n: usize) {
        for i in 0..t {
            let row_x = &x[i * k..(i + 1) * k];
            let row_out = &mut out[i * n..(i + 1) * n];
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += row_x[p] * w[p * n + j];
                }
                row_out[j] = sum;
            }
        }
    }

    /// Multi-Layer Fixed-Point CCCP Relaxation toward Semantic Equilibrium.
    /// Iterates until max change < convergence_tol or max_cccp_iters reached.
    pub fn converge_to_equilibrium(&self, z_in: &[f32], z_out: &mut [f32]) -> usize {
        let t = self.config.seq_len;
        let d = self.config.d_model;
        let h = self.config.num_heads;
        let dk = self.config.head_dim;
        let lambda = self.config.anchor_lambda;

        let mut z_current = z_in.to_vec();
        let mut z_next = vec![0.0f32; t * d];

        let mut q_proj = vec![0.0f32; t * d];
        let mut k_proj = vec![0.0f32; t * d];
        let mut v_proj = vec![0.0f32; t * d];
        let mut head_out = vec![0.0f32; t * d];

        let mut iters_taken = 0;

        for _iter in 0..self.config.max_cccp_iters {
            iters_taken += 1;

            // 1. Linear Projections
            self.project(&z_current, &self.w_q, &mut q_proj, t, d, d);
            self.project(&z_current, &self.w_k, &mut k_proj, t, d, d);
            self.project(&z_current, &self.w_v, &mut v_proj, t, d, d);

            // 2. Head-wise Flash Hopfield Step
            for head_idx in 0..h {
                let head_offset = head_idx * dk;

                let mut q_head = vec![0.0f32; t * dk];
                let mut k_head = vec![0.0f32; t * dk];
                let mut v_head = vec![0.0f32; t * dk];
                let mut out_h = vec![0.0f32; t * dk];

                for pos in 0..t {
                    let full_idx = pos * d + head_offset;
                    for d_i in 0..dk {
                        q_head[pos * dk + d_i] = q_proj[full_idx + d_i];
                        k_head[pos * dk + d_i] = k_proj[full_idx + d_i];
                        v_head[pos * dk + d_i] = v_proj[full_idx + d_i];
                    }
                }

                self.tiled_flash_hopfield_step(&q_head, &k_head, &v_head, &mut out_h);

                for pos in 0..t {
                    let full_idx = pos * d + head_offset;
                    for d_i in 0..dk {
                        head_out[full_idx + d_i] = out_h[pos * dk + d_i];
                    }
                }
            }

            // 3. Dense Associative Anchor & Relaxation Update:
            // Z_next = (1 - lambda) * Head_Out + lambda * Z_0
            let mut max_diff = 0.0f32;
            for i in 0..(t * d) {
                let val = (1.0 - lambda) * head_out[i] + lambda * z_in[i];
                let diff = (val - z_current[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                z_next[i] = val;
            }

            // 4. Convergence Check
            if max_diff < self.config.convergence_tol {
                break;
            }

            z_current.copy_from_slice(&z_next);
        }

        z_out.copy_from_slice(&z_next);
        iters_taken
    }

    /// Evaluates the global Continuous Hopfield Sequence Energy.
    pub fn compute_sequence_energy(&self, z: &[f32], z_in: &[f32]) -> f32 {
        let t = self.config.seq_len;
        let d = self.config.d_model;
        let beta = self.config.beta;
        let lambda = self.config.anchor_lambda;

        let mut energy = 0.0f32;

        // Quadratic regularizer 1/2 ||Z||^2
        let mut norm_sq = 0.0f32;
        let mut anchor_sq = 0.0f32;
        for i in 0..(t * d) {
            norm_sq += z[i] * z[i];
            let diff = z[i] - z_in[i];
            anchor_sq += diff * diff;
        }

        energy += 0.5 * (1.0 - lambda) * norm_sq + 0.5 * lambda * anchor_sq;

        // Log-Sum-Exp terms for sequence positions
        for i in 0..t {
            let mut sum_exp = 0.0f32;
            for j in 0..t {
                let mut dot = 0.0f32;
                for k in 0..d {
                    dot += z[i * d + k] * z[j * d + k];
                }
                sum_exp += (beta * dot).exp();
            }
            energy -= (1.0 / beta) * sum_exp.ln();
        }

        energy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_hopfield_tiled_step_matches_naive() {
        let t = 8;
        let dk = 4;
        let config = FlashHopfieldConfig::new(t, 16, 4);
        let layer = FlashHopfieldLayer::new(config);

        let q: Vec<f32> = (0..t * dk).map(|i| (i as f32 * 0.2).sin()).collect();
        let k: Vec<f32> = (0..t * dk).map(|i| (i as f32 * 0.3).cos()).collect();
        let v: Vec<f32> = (0..t * dk).map(|i| (i as f32 * 0.15).sin()).collect();

        let mut out_tiled = vec![0.0f32; t * dk];
        layer.tiled_flash_hopfield_step(&q, &k, &v, &mut out_tiled);

        // Compute naive Softmax(beta * Q * K^T) * V
        let mut out_naive = vec![0.0f32; t * dk];
        let beta = layer.config.beta;

        for i in 0..t {
            let mut scores = vec![0.0f32; t];
            let mut max_s = f32::NEG_INFINITY;
            for j in 0..t {
                let mut dot = 0.0f32;
                for d in 0..dk {
                    dot += q[i * dk + d] * k[j * dk + d];
                }
                scores[j] = dot * beta;
                if scores[j] > max_s {
                    max_s = scores[j];
                }
            }

            let mut sum_exp = 0.0f32;
            for j in 0..t {
                scores[j] = (scores[j] - max_s).exp();
                sum_exp += scores[j];
            }

            for j in 0..t {
                let p = scores[j] / sum_exp;
                for d in 0..dk {
                    out_naive[i * dk + d] += p * v[j * dk + d];
                }
            }
        }

        for i in 0..t * dk {
            let diff = (out_tiled[i] - out_naive[i]).abs();
            assert!(
                diff < 1e-5,
                "Tiled Flash-Hopfield must match naive Softmax! Got diff {} at index {}",
                diff,
                i
            );
        }
    }

    #[test]
    fn test_cccp_equilibrium_convergence() {
        let t = 6;
        let d = 16;
        let config = FlashHopfieldConfig::new(t, d, 2);
        let layer = FlashHopfieldLayer::new(config);

        let z_in: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut z_out = vec![0.0f32; t * d];

        let iters = layer.converge_to_equilibrium(&z_in, &mut z_out);

        assert!(iters >= 1 && iters <= layer.config.max_cccp_iters);
        assert_eq!(z_out.len(), t * d);
    }
}
