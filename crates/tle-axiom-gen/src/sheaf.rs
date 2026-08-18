//! Cellular Sheaf Consistency & Zero-Backpropagation Harmonic Extension Engine
//! Implements Cellular Sheaves, Sheaf Laplacians (L_F = δ^T δ), Dirichlet Energy,
//! and Harmonic Extension deduction based on Bodnar et al. (ICLR 2022) and Gebhart et al. (AISTATS 2023).

use std::collections::HashMap;

/// Dense vector representation for stalk states.
#[derive(Clone, Debug, PartialEq)]
pub struct StalkVector(pub Vec<f64>);

impl StalkVector {
    pub fn zeros(len: usize) -> Self {
        StalkVector(vec![0.0; len])
    }

    pub fn from_slice(slice: &[f64]) -> Self {
        StalkVector(slice.to_vec())
    }

    pub fn dot(&self, other: &StalkVector) -> f64 {
        assert_eq!(self.0.len(), other.0.len(), "Dimension mismatch in dot product");
        self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum()
    }

    pub fn norm_sq(&self) -> f64 {
        self.dot(self)
    }

    pub fn norm(&self) -> f64 {
        self.norm_sq().sqrt()
    }

    pub fn axpy(&self, alpha: f64, x: &StalkVector) -> StalkVector {
        assert_eq!(self.0.len(), x.0.len());
        StalkVector(self.0.iter().zip(x.0.iter()).map(|(y, xi)| y + alpha * xi).collect())
    }

    pub fn sub(&self, other: &StalkVector) -> StalkVector {
        assert_eq!(self.0.len(), other.0.len());
        StalkVector(self.0.iter().zip(other.0.iter()).map(|(a, b)| a - b).collect())
    }
}

/// Dense matrix representation for linear restriction maps.
#[derive(Clone, Debug, PartialEq)]
pub struct StalkMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>, // Row-major
}

impl StalkMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        StalkMatrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn identity(dim: usize) -> Self {
        let mut mat = StalkMatrix::zeros(dim, dim);
        for i in 0..dim {
            mat.data[i * dim + i] = 1.0;
        }
        mat
    }

    pub fn rotation_2d(theta_rad: f64) -> Self {
        let c = theta_rad.cos();
        let s = theta_rad.sin();
        StalkMatrix {
            rows: 2,
            cols: 2,
            data: vec![c, -s, s, c],
        }
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] = val;
    }

    pub fn transpose(&self) -> StalkMatrix {
        let mut res = StalkMatrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                res.set(c, r, self.get(r, c));
            }
        }
        res
    }

    pub fn matmul(&self, other: &StalkMatrix) -> StalkMatrix {
        assert_eq!(self.cols, other.rows, "Matrix dimension mismatch in multiplication");
        let mut res = StalkMatrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let aik = self.get(i, k);
                for j in 0..other.cols {
                    let old = res.get(i, j);
                    res.set(i, j, old + aik * other.get(k, j));
                }
            }
        }
        res
    }

    pub fn matvec(&self, v: &StalkVector) -> StalkVector {
        assert_eq!(self.cols, v.0.len(), "Matrix-Vector dimension mismatch");
        let mut res = vec![0.0; self.rows];
        for r in 0..self.rows {
            let mut sum = 0.0;
            let offset = r * self.cols;
            for c in 0..self.cols {
                sum += self.data[offset + c] * v.0[c];
            }
            res[r] = sum;
        }
        StalkVector(res)
    }
}

pub type NodeId = usize;
pub type EdgeId = usize;

/// A relational directed edge in the Cellular Sheaf.
#[derive(Clone, Debug)]
pub struct SheafEdge {
    pub id: EdgeId,
    pub src: NodeId,
    pub tgt: NodeId,
    pub relation: String,
    pub stalk_dim: usize,
    pub src_restriction: StalkMatrix, // F_{src ⊴ e}: F(src) -> F(e)
    pub tgt_restriction: StalkMatrix, // F_{tgt ⊴ e}: F(tgt) -> F(e)
}

/// The Cellular Sheaf data structure over a local Knowledge Graph complex.
pub struct CellularSheaf {
    pub node_dims: HashMap<NodeId, usize>,
    pub edges: Vec<SheafEdge>,
    pub node_offsets: HashMap<NodeId, usize>,
    pub total_c0_dim: usize,
}

impl CellularSheaf {
    pub fn new(node_dims: HashMap<NodeId, usize>) -> Self {
        let mut sorted_nodes: Vec<_> = node_dims.keys().cloned().collect();
        sorted_nodes.sort();

        let mut node_offsets = HashMap::new();
        let mut current_offset = 0;
        for node in &sorted_nodes {
            node_offsets.insert(*node, current_offset);
            current_offset += node_dims[node];
        }

        CellularSheaf {
            node_dims,
            edges: Vec::new(),
            node_offsets,
            total_c0_dim: current_offset,
        }
    }

    pub fn add_edge(
        &mut self,
        id: EdgeId,
        src: NodeId,
        tgt: NodeId,
        relation: &str,
        stalk_dim: usize,
        src_restriction: StalkMatrix,
        tgt_restriction: StalkMatrix,
    ) {
        let d_src = self.node_dims[&src];
        let d_tgt = self.node_dims[&tgt];
        assert_eq!(src_restriction.rows, stalk_dim);
        assert_eq!(src_restriction.cols, d_src);
        assert_eq!(tgt_restriction.rows, stalk_dim);
        assert_eq!(tgt_restriction.cols, d_tgt);

        self.edges.push(SheafEdge {
            id,
            src,
            tgt,
            relation: relation.to_string(),
            stalk_dim,
            src_restriction,
            tgt_restriction,
        });
    }

    /// Assembles the global Sheaf Laplacian Matrix L_F = δ^T δ
    pub fn assemble_sheaf_laplacian(&self) -> StalkMatrix {
        let mut laplacian = StalkMatrix::zeros(self.total_c0_dim, self.total_c0_dim);

        for edge in &self.edges {
            let u = edge.src;
            let v = edge.tgt;
            let offset_u = self.node_offsets[&u];
            let offset_v = self.node_offsets[&v];

            let f_u = &edge.src_restriction;
            let f_v = &edge.tgt_restriction;

            let f_u_t = f_u.transpose();
            let f_v_t = f_v.transpose();

            // Diagonal block (u, u): + F_u^T * F_u
            let l_uu = f_u_t.matmul(f_u);
            for r in 0..l_uu.rows {
                for c in 0..l_uu.cols {
                    let cur = laplacian.get(offset_u + r, offset_u + c);
                    laplacian.set(offset_u + r, offset_u + c, cur + l_uu.get(r, c));
                }
            }

            // Diagonal block (v, v): + F_v^T * F_v
            let l_vv = f_v_t.matmul(f_v);
            for r in 0..l_vv.rows {
                for c in 0..l_vv.cols {
                    let cur = laplacian.get(offset_v + r, offset_v + c);
                    laplacian.set(offset_v + r, offset_v + c, cur + l_vv.get(r, c));
                }
            }

            // Off-diagonal block (u, v): - F_u^T * F_v
            let l_uv = f_u_t.matmul(f_v);
            for r in 0..l_uv.rows {
                for c in 0..l_uv.cols {
                    let cur = laplacian.get(offset_u + r, offset_v + c);
                    laplacian.set(offset_u + r, offset_v + c, cur - l_uv.get(r, c));
                }
            }

            // Off-diagonal block (v, u): - F_v^T * F_u
            let l_vu = f_v_t.matmul(f_u);
            for r in 0..l_vu.rows {
                for c in 0..l_vu.cols {
                    let cur = laplacian.get(offset_v + r, offset_u + c);
                    laplacian.set(offset_v + r, offset_u + c, cur - l_vu.get(r, c));
                }
            }
        }

        laplacian
    }

    /// Evaluates the Coboundary (δx)_e for every edge:
    /// (δx)_e = F_{tgt} x_tgt - F_{src} x_src
    pub fn compute_coboundary(&self, x: &HashMap<NodeId, StalkVector>) -> HashMap<EdgeId, StalkVector> {
        let mut delta = HashMap::new();
        for edge in &self.edges {
            let x_src = &x[&edge.src];
            let x_tgt = &x[&edge.tgt];

            let trans_src = edge.src_restriction.matvec(x_src);
            let trans_tgt = edge.tgt_restriction.matvec(x_tgt);

            let discrepancy = trans_tgt.sub(&trans_src);
            delta.insert(edge.id, discrepancy);
        }
        delta
    }

    /// Computes the Global Sheaf Dirichlet Energy: E(x) = 0.5 * ||δx||^2
    pub fn compute_dirichlet_energy(&self, x: &HashMap<NodeId, StalkVector>) -> f64 {
        let delta = self.compute_coboundary(x);
        let mut total_energy = 0.0;
        for (_, disc) in delta {
            total_energy += 0.5 * disc.norm_sq();
        }
        total_energy
    }

    /// Deduce unobserved node states x_U given boundary premise facts x_B = g
    /// via closed-form Harmonic Extension: L_UU x_U = - L_UB g
    pub fn solve_harmonic_extension(
        &self,
        boundary: &HashMap<NodeId, StalkVector>,
    ) -> HashMap<NodeId, StalkVector> {
        let mut unobserved_nodes: Vec<NodeId> = self
            .node_dims
            .keys()
            .cloned()
            .filter(|n| !boundary.contains_key(n))
            .collect();
        unobserved_nodes.sort();

        if unobserved_nodes.is_empty() {
            return boundary.clone();
        }

        let mut u_offsets = HashMap::new();
        let mut total_u_dim = 0;
        for node in &unobserved_nodes {
            u_offsets.insert(*node, total_u_dim);
            total_u_dim += self.node_dims[node];
        }

        let mut l_uu = StalkMatrix::zeros(total_u_dim, total_u_dim);
        let mut rhs = vec![0.0; total_u_dim];

        let full_laplacian = self.assemble_sheaf_laplacian();

        for &u_node in &unobserved_nodes {
            let u_orig_off = self.node_offsets[&u_node];
            let u_dim = self.node_dims[&u_node];
            let u_sub_off = u_offsets[&u_node];

            for &v_node in &unobserved_nodes {
                let v_orig_off = self.node_offsets[&v_node];
                let v_dim = self.node_dims[&v_node];
                let v_sub_off = u_offsets[&v_node];

                for r in 0..u_dim {
                    for c in 0..v_dim {
                        let val = full_laplacian.get(u_orig_off + r, v_orig_off + c);
                        l_uu.set(u_sub_off + r, v_sub_off + c, val);
                    }
                }
            }

            for (b_node, g_b) in boundary {
                let b_orig_off = self.node_offsets[b_node];
                let b_dim = self.node_dims[b_node];

                for r in 0..u_dim {
                    for c in 0..b_dim {
                        let l_ub = full_laplacian.get(u_orig_off + r, b_orig_off + c);
                        rhs[u_sub_off + r] -= l_ub * g_b.0[c];
                    }
                }
            }
        }

        // Solve L_UU * x_U = rhs via Conjugate Gradient
        let x_u_sol = conjugate_gradient(&l_uu, &StalkVector(rhs), 500, 1e-10);

        let mut full_state = boundary.clone();
        for node in &unobserved_nodes {
            let off = u_offsets[node];
            let dim = self.node_dims[node];
            let state_vec = StalkVector(x_u_sol.0[off..off + dim].to_vec());
            full_state.insert(*node, state_vec);
        }

        full_state
    }
}

/// Conjugate Gradient solver for symmetric positive definite systems.
fn conjugate_gradient(a: &StalkMatrix, b: &StalkVector, max_iter: usize, tol: f64) -> StalkVector {
    let n = b.0.len();
    let mut x = StalkVector::zeros(n);
    let mut r = b.sub(&a.matvec(&x));
    let mut p = r.clone();
    let mut rs_old = r.norm_sq();

    if rs_old.sqrt() < tol {
        return x;
    }

    for _ in 0..max_iter {
        let ap = a.matvec(&p);
        let p_ap = p.dot(&ap);
        if p_ap.abs() < 1e-18 {
            break;
        }
        let alpha = rs_old / p_ap;
        x = x.axpy(alpha, &p);
        r = r.axpy(-alpha, &ap);

        let rs_new = r.norm_sq();
        if rs_new.sqrt() < tol {
            break;
        }
        let beta = rs_new / rs_old;
        p = r.axpy(beta, &p);
        rs_old = rs_new;
    }

    x
}

/// Deterministically maps a semantic relation to a rotation angle in radians.
pub fn relation_to_rotation(relation: &str) -> f64 {
    let lower = relation.to_lowercase();
    match lower.as_str() {
        // Location / Spatial hierarchy (π/4)
        "located_in" | "located_at" | "located_near" | "part_of" | "from" | "capital_of"
        | "born_in" | "died_in" | "lived_in" => std::f64::consts::PI / 4.0,

        // Temporal / Event progression (π/3)
        "happened_in" | "took_place_in" | "occurred_in" | "released_in" | "published_in"
        | "founded_in" | "created_in" => std::f64::consts::PI / 3.0,

        // Creator / Authorship / Agency (π/6)
        "created_by" | "written_by" | "directed_by" | "composed_by" | "painted_by"
        | "invented_by" | "discovered_by" | "built_by" | "produced_by" | "written"
        | "directed" | "composed" | "painted" | "invented" | "discovered" | "built" => {
            std::f64::consts::PI / 6.0
        }

        // Kinship / Social connection (5π/12)
        "child_of" | "married_to" | "has_mother" | "has_father" | "has_parent"
        | "sister_of" | "brother_of" | "son_of" | "daughter_of" => {
            5.0 * std::f64::consts::PI / 12.0
        }

        // Default: deterministic angle based on bytes
        _ => {
            let hash: u64 = lower.bytes().fold(5381, |acc, b| ((acc << 5).wrapping_add(acc)).wrapping_add(b as u64));
            let deg = (hash % 180) as f64 + 10.0;
            deg * std::f64::consts::PI / 180.0
        }
    }
}

/// Evaluates the consistency of candidate paths connecting query entities to candidate answer.
/// Returns Dirichlet Energy E(x) >= 0.0. (Lower energy = higher deductive consistency).
pub fn evaluate_subgraph_consistency(
    triples: &[(usize, String, usize)],
    query_nodes: &[usize],
    _target_node: usize,
) -> f64 {
    if triples.is_empty() || query_nodes.is_empty() {
        return 0.0;
    }

    let mut node_set = std::collections::HashSet::new();
    for &(src, _, tgt) in triples {
        node_set.insert(src);
        node_set.insert(tgt);
    }
    for &q in query_nodes {
        node_set.insert(q);
    }

    let mut dims = HashMap::new();
    for &n in &node_set {
        dims.insert(n, 2);
    }

    let mut sheaf = CellularSheaf::new(dims);
    let id2 = StalkMatrix::identity(2);

    for (edge_idx, &(src, ref rel, tgt)) in triples.iter().enumerate() {
        let theta = relation_to_rotation(rel);
        let r_mat = StalkMatrix::rotation_2d(theta);
        sheaf.add_edge(edge_idx, src, tgt, rel, 2, r_mat, id2.clone());
    }

    let mut boundary = HashMap::new();
    for &q in query_nodes {
        boundary.insert(q, StalkVector(vec![1.0, 0.0]));
    }

    let solution = sheaf.solve_harmonic_extension(&boundary);
    sheaf.compute_dirichlet_energy(&solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sheaf_harmonic_extension_2hop() {
        // Alice (0) --[born_in +pi/4]--> Kyoto (1) --[located_in +pi/4]--> Japan (2)
        let mut dims = HashMap::new();
        dims.insert(0, 2);
        dims.insert(1, 2);
        dims.insert(2, 2);

        let mut sheaf = CellularSheaf::new(dims);
        let r1 = StalkMatrix::rotation_2d(std::f64::consts::PI / 4.0);
        let r2 = StalkMatrix::rotation_2d(std::f64::consts::PI / 4.0);
        let id2 = StalkMatrix::identity(2);

        sheaf.add_edge(1, 0, 1, "born_in", 2, r1, id2.clone());
        sheaf.add_edge(2, 1, 2, "located_in", 2, r2, id2);

        let mut boundary = HashMap::new();
        boundary.insert(0, StalkVector(vec![1.0, 0.0]));

        let solution = sheaf.solve_harmonic_extension(&boundary);
        let japan_state = &solution[&2];

        // Alice (1, 0) rotated by 90 deg -> (0, 1)
        assert!((japan_state.0[0] - 0.0).abs() < 1e-5);
        assert!((japan_state.0[1] - 1.0).abs() < 1e-5);

        let energy = sheaf.compute_dirichlet_energy(&solution);
        assert!(energy < 1e-8, "Dirichlet energy should be zero for consistent reasoning");
    }
}
