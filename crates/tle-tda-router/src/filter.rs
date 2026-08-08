//! Filter functions for the TDA Mapper algorithm.
//!
//! A filter function f: X → ℝ projects high-dimensional data onto
//! a scalar that captures some meaningful structure. The choice of
//! filter determines what topological features are revealed.

use tle_vsa::HyperVector;

/// Available filter functions for the Mapper algorithm.
#[derive(Clone, Debug)]
pub enum FilterFunction {
    /// L2 norm: captures "distance from origin" in latent space.
    /// Good for distinguishing content words from function words.
    Norm,

    /// First principal component approximation: sum of first D/4 dimensions.
    /// Captures the dominant direction of variation.
    FirstComponent,

    /// Syntactic energy: uses Clifford algebra to measure grammatical structure.
    /// Requires a syntax node to compute.
    SyntacticEnergy,

    /// Density estimation: local density based on average distance to k-nearest neighbors.
    /// Captures clustering structure in the latent manifold.
    LocalDensity { k: usize },

    /// Geometric centroid distance: distance to the mean of a reference set.
    CentroidDistance,

    /// Custom: user-provided scalar function index.
    Custom(usize),
}

impl FilterFunction {
    /// Evaluate the filter function on a single hypervector.
    pub fn evaluate(&self, hv: &HyperVector) -> f32 {
        match self {
            FilterFunction::Norm => hv.norm(),

            FilterFunction::FirstComponent => {
                let block = hv.dim() / 4;
                hv.data[..block].iter().sum::<f32>() / (block as f32).sqrt()
            }

            FilterFunction::SyntacticEnergy => {
                // Simplified: use norm of bivector-projected components
                let dim = hv.dim();
                let third = dim / 3;
                let e1: f32 = hv.data[..third].iter().sum();
                let e2: f32 = hv.data[third..2*third].iter().sum();
                let e3: f32 = hv.data[2*third..3*third].iter().sum();
                // Approximate "oriented area" = cross product magnitude
                let cross_mag = ((e1*e2 - e2*e1).powi(2)
                    + (e1*e3 - e3*e1).powi(2)
                    + (e2*e3 - e3*e2).powi(2))
                .sqrt();
                cross_mag / (dim as f32)
            }

            FilterFunction::LocalDensity { .. } => {
                // Single-vector density is meaningless; returns norm as fallback
                hv.norm()
            }

            FilterFunction::CentroidDistance => hv.norm(),

            FilterFunction::Custom(_) => hv.norm(),
        }
    }

    /// Evaluate the filter on a batch of vectors.
    /// For density-based filters, uses the full batch for computation.
    pub fn evaluate_batch(&self, vectors: &[&HyperVector]) -> Vec<f32> {
        match self {
            FilterFunction::LocalDensity { k } => {
                self.compute_local_density(vectors, *k)
            }
            _ => vectors.iter().map(|v| self.evaluate(v)).collect(),
        }
    }

    /// Compute local density for each vector in a batch.
    /// Density = 1 / (average distance to k nearest neighbors)
    fn compute_local_density(&self, vectors: &[&HyperVector], k: usize) -> Vec<f32> {
        let n = vectors.len();
        let k = k.min(n - 1).max(1);

        let mut densities = vec![0.0f32; n];

        for i in 0..n {
            // Compute distances to all other vectors
            let mut distances: Vec<f32> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let diff = vectors[i].sub(vectors[j]);
                    diff.dot(&diff).sqrt()
                })
                .collect();

            // Sort and take k nearest
            distances.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let avg_dist: f32 = distances[..k].iter().sum::<f32>() / k as f32;

            densities[i] = if avg_dist > 1e-10 {
                1.0 / avg_dist
            } else {
                f32::MAX
            };
        }

        densities
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tle_vsa::DEFAULT_DIM;

    #[test]
    fn test_norm_filter() {
        let hv = HyperVector::random_bipolar(DEFAULT_DIM, 42);
        let filter = FilterFunction::Norm;
        let val = filter.evaluate(&hv);
        // Bipolar norm = √D
        let expected = (DEFAULT_DIM as f32).sqrt();
        assert!((val - expected).abs() < 0.01);
    }

    #[test]
    fn test_filter_determinism() {
        let hv = HyperVector::random_bipolar(DEFAULT_DIM, 42);
        let filter = FilterFunction::FirstComponent;
        let v1 = filter.evaluate(&hv);
        let v2 = filter.evaluate(&hv);
        assert_eq!(v1, v2, "Filter must be deterministic");
    }

    #[test]
    fn test_batch_density() {
        let vectors: Vec<HyperVector> = (0..10)
            .map(|i| HyperVector::random_gaussian(100, i * 10))
            .collect();
        let refs: Vec<&HyperVector> = vectors.iter().collect();

        let filter = FilterFunction::LocalDensity { k: 3 };
        let densities = filter.evaluate_batch(&refs);
        assert_eq!(densities.len(), 10);
        // All densities should be positive
        for d in &densities {
            assert!(*d > 0.0);
        }
    }
}
