//! Data-Dependent Gated Cellular Sheaf Diffusion Layer.
//!
//! Bridges Pre-trained Transformer Attention projections ($W_Q, W_K$) with Cellular Sheaf Theory:
//! 1. Orthogonal Procrustes gauge extraction: $P_{i \leftarrow j} \in SO(d)$ planar rotors.
//! 2. Data-dependent phase gating: $\alpha_{ij}(z) = \sigma\left(\frac{1}{\tau} \operatorname{Re}\langle z_i, z_j \rangle_{\mathbb{T}^D}\right)$
//!    restoring dynamic in-context induction routing without $O(N^2)$ softmax overhead.
//! 3. Fast $O(d)$ Cayley-Woodbury parallel transport forward step.
//! 4. Sheaf Dirichlet Energy minimization measuring semantic topological consistency.

use tle_vsa::whitened_phasor::WhitenedPhasor;

/// Gated Sheaf Edge representing a directed topological connection $j \to i$.
#[derive(Debug, Clone)]
pub struct GatedSheafEdge {
    /// Source token index $j$.
    pub source: usize,
    /// Target token index $i$.
    pub target: usize,
    /// Static planar rotation angle $\phi \in [-\pi, \pi)$ extracted from $W_Q, W_K$.
    pub static_angle: f32,
    /// Dynamic coupling gate $\alpha_{ij} \in [0, 1]$ computed from token states.
    pub dynamic_gate: f32,
}

/// Data-Dependent Gated Cellular Sheaf Layer.
#[derive(Debug, Clone)]
pub struct GatedSheafLayer {
    /// Dimension of node stalk $d_{\text{stalk}}$.
    pub stalk_dim: usize,
    /// Temperature parameter $\tau$ for phase gating.
    pub temperature: f32,
    /// Diffusion step size $\gamma \in (0, 1]$.
    pub diffusion_rate: f32,
    /// Extracted edges with gauge restriction maps.
    pub edges: Vec<GatedSheafEdge>,
}

impl GatedSheafLayer {
    /// Creates a new GatedSheafLayer.
    pub fn new(stalk_dim: usize, temperature: f32, diffusion_rate: f32) -> Self {
        Self {
            stalk_dim,
            temperature: temperature.max(1e-4),
            diffusion_rate: diffusion_rate.clamp(0.01, 1.0),
            edges: Vec::new(),
        }
    }

    /// Adds a directed edge $j \to i$ with static rotor angle $\phi$ (extracted via Procrustes/RoPE).
    pub fn add_edge(&mut self, source: usize, target: usize, static_angle: f32) {
        self.edges.push(GatedSheafEdge {
            source,
            target,
            static_angle,
            dynamic_gate: 1.0,
        });
    }

    /// Updates dynamic edge gates $\alpha_{ij}$ from input token phasors $z_1, \dots, z_N \in \mathbb{T}^D$.
    ///
    /// $\alpha_{ij} = \sigma\left(\frac{1}{\tau} \operatorname{Re}\langle z_i, z_j \rangle_{\mathbb{T}^D}\right) = \frac{1}{1 + \exp\left(-\frac{\operatorname{Sim}(z_i, z_j)}{\tau}\right)}$
    pub fn update_dynamic_gates(&mut self, token_phasors: &[WhitenedPhasor]) {
        for edge in &mut self.edges {
            if edge.source < token_phasors.len() && edge.target < token_phasors.len() {
                let sim = token_phasors[edge.target].similarity(&token_phasors[edge.source]);
                let logit = sim / self.temperature;
                let gate = 1.0 / (1.0 + (-logit).exp());
                edge.dynamic_gate = gate;
            }
        }
    }

    /// Performs one step of discretized Gated Sheaf Diffusion:
    ///
    /// $x_i^{(t+1)} = (1 - \gamma) x_i^{(t)} + \gamma \sum_{j \in \mathcal{N}(i)} \frac{\alpha_{ij}}{\sum_k \alpha_{ik}} P_{i \leftarrow j} x_j^{(t)}$
    ///
    /// Runs in $O(N \cdot d_{\text{stalk}})$ FLOPs using 2D block Givens rotations ($O(d)$ parallel transport).
    pub fn diffuse_step(&self, stalks: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = stalks.len();
        if n == 0 {
            return Vec::new();
        }
        let d = self.stalk_dim;

        // Initialize output cochain with decay: (1 - gamma) * x_i
        let mut out = vec![vec![0.0f32; d]; n];
        let mut gate_sums = vec![0.0f32; n];
        let mut incoming_acc = vec![vec![0.0f32; d]; n];

        for edge in &self.edges {
            if edge.source < n && edge.target < n {
                let j = edge.source;
                let i = edge.target;
                let alpha = edge.dynamic_gate;

                // Apply SO(d) planar rotation P_{i <- j} to stalk x_j
                let transported = apply_planar_rotation(&stalks[j], edge.static_angle);

                for k in 0..d {
                    incoming_acc[i][k] += alpha * transported[k];
                }
                gate_sums[i] += alpha;
            }
        }

        let gamma = self.diffusion_rate;
        for i in 0..n {
            let total_gate = gate_sums[i];
            if total_gate > 1e-6 {
                for k in 0..d {
                    out[i][k] = (1.0 - gamma) * stalks[i][k] + gamma * (incoming_acc[i][k] / total_gate);
                }
            } else {
                out[i] = stalks[i].clone();
            }
        }

        out
    }

    /// Computes the Gated Sheaf Dirichlet Energy:
    ///
    /// $\mathcal{E}_{\mathcal{F}}(X) = \frac{1}{2} \sum_{(j \to i)} \alpha_{ij} \|x_i - P_{i \leftarrow j} x_j\|_2^2$
    ///
    /// Zero energy $\iff$ global cochain is harmonic and strictly topologically consistent across all active relations.
    pub fn compute_dirichlet_energy(&self, stalks: &[Vec<f32>]) -> f32 {
        let mut energy = 0.0f32;

        for edge in &self.edges {
            if edge.source < stalks.len() && edge.target < stalks.len() {
                let j = edge.source;
                let i = edge.target;
                let alpha = edge.dynamic_gate;

                let transported = apply_planar_rotation(&stalks[j], edge.static_angle);
                let mut dist_sq = 0.0f32;
                for k in 0..self.stalk_dim {
                    let diff = stalks[i][k] - transported[k];
                    dist_sq += diff * diff;
                }
                energy += 0.5 * alpha * dist_sq;
            }
        }

        energy
    }
}

/// Applies a 2D block Givens rotation $P \in SO(d)$ with angle $\phi$ to vector $x \in \mathbb{R}^d$ in $O(d)$ FLOPs.
#[inline]
fn apply_planar_rotation(x: &[f32], angle: f32) -> Vec<f32> {
    let d = x.len();
    let cos_th = angle.cos();
    let sin_th = angle.sin();
    let mut out = vec![0.0f32; d];

    let num_pairs = d / 2;
    for k in 0..num_pairs {
        let x1 = x[2 * k];
        let x2 = x[2 * k + 1];
        // 2D SO(2) rotation: [cos -sin; sin cos] * [x1; x2]
        out[2 * k] = cos_th * x1 - sin_th * x2;
        out[2 * k + 1] = sin_th * x1 + cos_th * x2;
    }
    if d % 2 != 0 {
        out[d - 1] = x[d - 1];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planar_rotation_preserves_euclidean_norm() {
        let x = vec![3.0, 4.0, 1.0, -2.0, 0.5, 0.8];
        let norm_before: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

        let rotated = apply_planar_rotation(&x, 1.25);
        let norm_after: f32 = rotated.iter().map(|v| v * v).sum::<f32>().sqrt();

        assert!((norm_before - norm_after).abs() < 1e-5);
    }

    #[test]
    fn test_dynamic_phase_gating_and_dirichlet_energy() {
        let mut layer = GatedSheafLayer::new(4, 0.5, 0.5);
        layer.add_edge(0, 1, 0.0); // Edge 0 -> 1 with identity transport

        let z0 = WhitenedPhasor::new(vec![0.1, 0.2]);
        let z1 = WhitenedPhasor::new(vec![0.1, 0.2]); // Highly aligned phasors
        let phasors = vec![z0, z1];

        layer.update_dynamic_gates(&phasors);
        assert!(layer.edges[0].dynamic_gate > 0.8, "Aligned phasors must have high dynamic gate");

        let stalks = vec![vec![1.0, 0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0, 0.0]];
        let energy = layer.compute_dirichlet_energy(&stalks);
        assert!(energy < 1e-6, "Harmonic identical cochain must have near-zero Dirichlet energy");
    }
}
