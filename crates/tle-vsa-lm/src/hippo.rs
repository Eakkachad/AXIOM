//! # HiPPO-LegS Continuous State-Space Polynomial Memory (O(1) Step SSM)
//!
//! Projects continuous streaming sequence features onto orthogonal Shifted Legendre Polynomials.
//! Key Mathematical Guarantees:
//! - Strictly O(1) time and O(1) memory per token step (No KV-cache growth).
//! - Continuous-time signal reconstruction at any historical point τ ∈ [0, 1].
//! - Exact Bilinear (Tustin) discretization preserving Lyapunov stability.

/// HiPPO-LegS Discretized Continuous Memory Accumulator.
/// Maintains polynomial coefficients c_n(t) for N Legendre basis orders across D feature channels.
#[derive(Debug, Clone)]
pub struct HippoLegSMemory {
    pub order: usize,
    pub dim: usize,
    /// Discretized transition matrix A_bar: [ORDER, ORDER]
    pub a_disc: Vec<f32>,
    /// Discretized input projection vector B_bar: [ORDER]
    pub b_disc: Vec<f32>,
    /// Polynomial state coefficients: [ORDER, DIM]
    pub state: Vec<f32>,
    pub step_count: usize,
}

impl HippoLegSMemory {
    /// Constructs a new HiPPO-LegS state space accumulator with sampling period delta_t.
    pub fn new(order: usize, dim: usize, delta_t: f32) -> Self {
        // 1. Construct continuous A and B matrices for HiPPO-LegS
        let mut a_cont = vec![0.0f32; order * order];
        let mut b_cont = vec![0.0f32; order];

        for n in 0..order {
            let coef_n = ((2 * n + 1) as f32).sqrt();
            b_cont[n] = coef_n;
            for k in 0..order {
                if n > k {
                    let coef_k = ((2 * k + 1) as f32).sqrt();
                    a_cont[n * order + k] = coef_n * coef_k;
                } else if n == k {
                    a_cont[n * order + k] = (n + 1) as f32;
                } else {
                    a_cont[n * order + k] = 0.0;
                }
            }
        }

        // 2. Bilinear Discretization:
        // A_bar = (I - dt/2 * A)^(-1) * (I + dt/2 * A)
        // Since A is lower-triangular, (I - dt/2 * A) is lower-triangular and inverted in O(N^2).
        let half_dt = delta_t * 0.5;
        let mut m_minus = vec![0.0f32; order * order];
        let mut m_plus = vec![0.0f32; order * order];

        for i in 0..order {
            for j in 0..order {
                let id = if i == j { 1.0 } else { 0.0 };
                let a_val = a_cont[i * order + j];
                m_minus[i * order + j] = id + half_dt * a_val;
                m_plus[i * order + j] = id - half_dt * a_val;
            }
        }

        // Invert m_minus via forward substitution (lower-triangular)
        let mut inv_minus = vec![0.0f32; order * order];
        for j in 0..order {
            for i in j..order {
                let mut sum = if i == j { 1.0 } else { 0.0 };
                for k in j..i {
                    sum -= m_minus[i * order + k] * inv_minus[k * order + j];
                }
                inv_minus[i * order + j] = sum / m_minus[i * order + i].max(1e-12);
            }
        }

        // A_bar = inv_minus * m_plus
        let mut a_disc = vec![0.0f32; order * order];
        for i in 0..order {
            for j in 0..order {
                let mut s = 0.0f32;
                for k in 0..order {
                    s += inv_minus[i * order + k] * m_plus[k * order + j];
                }
                a_disc[i * order + j] = s;
            }
        }

        // B_bar = inv_minus * (dt * B)
        let mut b_disc = vec![0.0f32; order];
        for i in 0..order {
            let mut s = 0.0f32;
            for k in 0..order {
                s += inv_minus[i * order + k] * (delta_t * b_cont[k]);
            }
            b_disc[i] = s;
        }

        Self {
            order,
            dim,
            a_disc,
            b_disc,
            state: vec![0.0f32; order * dim],
            step_count: 0,
        }
    }

    /// O(1) Constant-Time Streaming Step Update: c_{k+1} = A_bar * c_k + B_bar * f_k.
    #[inline]
    pub fn update_step(&mut self, input_features: &[f32]) {
        assert_eq!(input_features.len(), self.dim, "Feature dimension mismatch");

        let mut new_state = vec![0.0f32; self.order * self.dim];

        for n in 0..self.order {
            let b_n = self.b_disc[n];
            for d in 0..self.dim {
                let mut acc = b_n * input_features[d];
                for k in 0..self.order {
                    acc += self.a_disc[n * self.order + k] * self.state[k * self.dim + d];
                }
                new_state[n * self.dim + d] = acc;
            }
        }

        self.state = new_state;
        self.step_count += 1;
    }

    /// Continuous-Time Signal Reconstruction at normalized historical position tau ∈ [0, 1].
    /// Evaluates: \hat{f}(tau) = sum_{n=0}^{ORDER-1} c_n * (2n+1)^{1/2} * P_n(2 * tau - 1)
    pub fn reconstruct_at(&self, tau: f32) -> Vec<f32> {
        let x = (2.0 * tau - 1.0).clamp(-1.0, 1.0); // Map [0, 1] -> [-1, 1] for Legendre

        // Evaluate Shifted Legendre polynomials up to order - 1
        let mut p_vals = vec![0.0f32; self.order];
        if self.order > 0 {
            p_vals[0] = 1.0;
        }
        if self.order > 1 {
            p_vals[1] = x;
        }
        for n in 2..self.order {
            let nf = n as f32;
            p_vals[n] = ((2.0 * nf - 1.0) * x * p_vals[n - 1] - (nf - 1.0) * p_vals[n - 2]) / nf;
        }

        let mut out = vec![0.0f32; self.dim];
        for n in 0..self.order {
            let weight = ((2 * n + 1) as f32).sqrt() * p_vals[n];
            for d in 0..self.dim {
                out[d] += self.state[n * self.dim + d] * weight;
            }
        }
        out
    }

    /// Reset state buffer to zeros.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.step_count = 0;
    }
}

/// HiPPO-LegT Sliding-Window Continuous Memory Accumulator.
/// Maintains orthogonal projection over a fixed sliding window of continuous duration theta.
#[derive(Debug, Clone)]
pub struct HippoLegTMemory {
    pub order: usize,
    pub dim: usize,
    pub a_disc: Vec<f32>,
    pub b_disc: Vec<f32>,
    pub state: Vec<f32>,
    pub step_count: usize,
}

impl HippoLegTMemory {
    pub fn new(order: usize, dim: usize, window_theta: f32, delta_t: f32) -> Self {
        let mut a_cont = vec![0.0f32; order * order];
        let mut b_cont = vec![0.0f32; order];

        for n in 0..order {
            let coef_n = ((2 * n + 1) as f32).sqrt();
            let sign_n = if n % 2 == 0 { 1.0 } else { -1.0 };
            b_cont[n] = (1.0 / window_theta) * coef_n * sign_n;

            for k in 0..order {
                let coef_k = ((2 * k + 1) as f32).sqrt();
                if n > k {
                    a_cont[n * order + k] = (1.0 / window_theta) * coef_n * coef_k;
                } else if n == k {
                    a_cont[n * order + k] = (1.0 / window_theta) * (n + 1) as f32;
                } else {
                    let sign_nk = if (n + k) % 2 == 0 { 1.0 } else { -1.0 };
                    a_cont[n * order + k] = (1.0 / window_theta) * sign_nk * coef_n * coef_k;
                }
            }
        }

        // Bilinear Discretization: (I - dt/2 * A)^(-1) * (I + dt/2 * A)
        let half_dt = delta_t * 0.5;
        let mut m_minus = vec![0.0f32; order * order];
        let mut m_plus = vec![0.0f32; order * order];

        for i in 0..order {
            for j in 0..order {
                let id = if i == j { 1.0 } else { 0.0 };
                let a_val = a_cont[i * order + j];
                m_minus[i * order + j] = id + half_dt * a_val;
                m_plus[i * order + j] = id - half_dt * a_val;
            }
        }

        // Invert m_minus via Gauss-Jordan elimination
        let inv_minus = invert_matrix(&m_minus, order);

        let mut a_disc = vec![0.0f32; order * order];
        for i in 0..order {
            for j in 0..order {
                let mut s = 0.0f32;
                for k in 0..order {
                    s += inv_minus[i * order + k] * m_plus[k * order + j];
                }
                a_disc[i * order + j] = s;
            }
        }

        let mut b_disc = vec![0.0f32; order];
        for i in 0..order {
            let mut s = 0.0f32;
            for k in 0..order {
                s += inv_minus[i * order + k] * (delta_t * b_cont[k]);
            }
            b_disc[i] = s;
        }

        Self {
            order,
            dim,
            a_disc,
            b_disc,
            state: vec![0.0f32; order * dim],
            step_count: 0,
        }
    }

    #[inline]
    pub fn update_step(&mut self, input_features: &[f32]) {
        assert_eq!(input_features.len(), self.dim);
        let mut new_state = vec![0.0f32; self.order * self.dim];

        for n in 0..self.order {
            let b_n = self.b_disc[n];
            for d in 0..self.dim {
                let mut acc = b_n * input_features[d];
                for k in 0..self.order {
                    acc += self.a_disc[n * self.order + k] * self.state[k * self.dim + d];
                }
                new_state[n * self.dim + d] = acc;
            }
        }

        self.state = new_state;
        self.step_count += 1;
    }
}

/// Gauss-Jordan matrix inversion for dense matrices.
fn invert_matrix(a: &[f32], n: usize) -> Vec<f32> {
    let mut aug = vec![0.0f32; n * (2 * n)];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = a[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }

    for i in 0..n {
        let mut pivot_row = i;
        let mut max_val = aug[i * 2 * n + i].abs();
        for k in (i + 1)..n {
            let val = aug[k * 2 * n + i].abs();
            if val > max_val {
                max_val = val;
                pivot_row = k;
            }
        }

        if pivot_row != i {
            for j in 0..2 * n {
                aug.swap(i * 2 * n + j, pivot_row * 2 * n + j);
            }
        }

        let pivot = aug[i * 2 * n + i];
        let inv_pivot = 1.0 / pivot.max(1e-12);
        for j in 0..2 * n {
            aug[i * 2 * n + j] *= inv_pivot;
        }

        for k in 0..n {
            if k != i {
                let factor = aug[k * 2 * n + i];
                for j in 0..2 * n {
                    aug[k * 2 * n + j] -= factor * aug[i * 2 * n + j];
                }
            }
        }
    }

    let mut inv = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hippo_legs_constant_step_reconstruction() {
        let order = 8;
        let dim = 4;
        let mut hippo = HippoLegSMemory::new(order, dim, 0.05);

        let signal = vec![1.0f32, -0.5f32, 0.25f32, 0.8f32];
        for _ in 0..40 {
            hippo.update_step(&signal);
        }

        let recon = hippo.reconstruct_at(0.9);
        for d in 0..dim {
            assert!(
                (recon[d] - signal[d]).abs() < 0.2,
                "HiPPO Legendre reconstruction must track input signal! Got {} vs expected {}",
                recon[d],
                signal[d]
            );
        }
    }

    #[test]
    fn test_hippo_legt_sliding_window_runs() {
        let order = 6;
        let dim = 2;
        let mut hippo = HippoLegTMemory::new(order, dim, 1.0, 0.05);

        let signal = vec![0.5f32, -0.5f32];
        for _ in 0..20 {
            hippo.update_step(&signal);
        }

        assert_eq!(hippo.step_count, 20);
    }
}
