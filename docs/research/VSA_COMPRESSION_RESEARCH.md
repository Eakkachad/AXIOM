# VSA Transition Memory Compression Research

**Problem**: AXIOM's per-word Transition Binding Algebra stores `V × D × 4` bytes of transition patterns. With V=50,000 words at D=10,240, the codebook alone costs 2GB, and the TransitionMemory adds another 2GB. The TrigramMemory is potentially V². This is the critical scaling bottleneck.

**Goal**: Compress from O(V×D) to something tractable (tens to hundreds of MB), while preserving the algebraic VSA scoring that drives next-token prediction.

---

## 1. LLM Knowledge Density: How Do Transformers Do It?

### The Numbers

Roberts et al. (2020, "How Much Knowledge Can You Pack Into the Parameters of a Language Model?") demonstrated that T5 models store approximately **2 bits per parameter** of factual knowledge. Morris et al. (2025, "How Much Do Language Models Memorize?") estimated α ≈ **3.64 bits-per-parameter** for GPT-scale models. A 7B-parameter model at fp16 uses:

- Storage: 7B × 2 bytes = **14 GB**
- Effective knowledge capacity: 7B × 2 bits ≈ **14 billion bits ≈ 1.75 GB** of pure fact content
- Equivalent to storing ~175M facts (at ~10 bytes/fact) in 14GB

Compare to AXIOM's flat codebook: 50K words × 10,240 dims × 4 bytes = **2.048 GB** for _just the vocabulary vectors_, storing essentially zero facts — just word identities. That's **1,000× less efficient** than an LLM's parameter utilization.

### Why Are LLMs So Dense? Superposition

The key mechanism is **superposition** (Elhage et al., 2022, "Toy Models of Superposition"): in a neural network, each parameter simultaneously participates in storing _thousands_ of independent facts through the associative property of matrix multiplication followed by nonlinearity:

```
y = σ(W₂ · σ(W₁x + b₁) + b₂)
```

Each element of W₁, W₂ contributes to all outputs. A single float32 weight stores fractional contributions to arbitrarily many independent features. By contrast, AXIOM's codebook uses each float32 to store exactly one bit (a±1) of exactly one word's representation — a 32-fold waste even before considering superposition.

**Key takeaway**: LLMs compress via _overlapping, learned representations_. AXIOM needs overlapping, learned (or at least structured) representations too.

### 1.58-bit LLMs: Proof That Extreme Quantization Works

Ma et al. (2024, "The Era of 1-bit LLMs: All Large Language Models Are in 1.58 Bits") showed that ternary {-1, 0, 1} weights match fp16 Transformer performance. This proves that reducing precision from 32 bits to 1.58 bits per scalar costs negligible accuracy _if the architecture is designed for it_. For VSA, this means bipolar {-1, +1} storage (1 bit) should be the _baseline_, not an optimization.

---

## 2. Technique 1: Binary-Packed Bipolar Vectors (Baseline)

**This is the minimum viable fix.** The AXIOM spec already defines hypervectors as `v ∈ {-1,+1}^d`. The implementation uses `Vec<f32>`, wasting 32 bits per dimension.

### Formula

```
Bits per vector = D (not 32×D)
Storage = V × D / 8 bytes
```

### Compression Ratio

| Storage | f32 (current) | bitpacked |
|---------|---------------|-----------|
| Per vector (D=10,240) | 40,960 bytes | 1,280 bytes |
| Codebook (V=50,000) | 2.048 GB | 64 MB |
| TransitionMemory (V=50,000) | 2.048 GB | 64 MB |
| **Total** | **~4 GB** | **~128 MB** |

**32× compression.** Fundamental limit: information-theoretic minimum for bipolar vectors.

### Rust Implementation Plan

Replace `HyperVector.data: Vec<f32>` with `Vec<u64>` where each u64 packs 64 bipolar dimensions as bits. Cosine similarity becomes popcount-based:

```rust
pub struct BipolarVector {
    pub data: Vec<u64>,  // each u64 = 64 signed dimensions, bit=1 means +1, bit=0 means -1
}

impl BipolarVector {
    fn cosine_similarity(&self, other: &Self) -> f32 {
        let same = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a ^ b).count_ones() as u32)
            .sum::<u32>();
        let diff = (self.data.len() * 64) as u32 - same;
        (same as f32 - diff as f32) / (self.data.len() * 64) as f32
    }
}
```

**Feasibility**: Trivial. Core VSA operations (bind = XOR, bundle = majority vote with counters, similarity = popcount) all map to fast bitwise SIMD.

### Risk

- **Bundling overflow**: Adding N bipolar vectors produces values in [-N, N]. Storing as i16 (or even i8 for small N) solves this. Or use stochastic rounding: `sgn(sum)` after bundling.
- **Cosine similarity noise**: With bipolar vectors, cosine similarity between N bundled items and a query has SNR ≈ 1/√N (same as f32). No accuracy loss.

---

## 3. Technique 2: Sparse Hypervectors

### Concept

Instead of dense D-dimensional vectors, use vectors where only ~d positions are nonzero (±1), d << D. A 1% sparse D=10,240 vector has only ~102 active positions. Store as `Vec<(u32, bool)>` (index, sign pairs).

### Formula

```
Bytes per vector = 2 × d_active × (4 + 1)  ≈ 10 × d_active
```

For d_active = 102 (1% density):
```
Per vector = 102 × 5 = 510 bytes
Codebook = 50,000 × 510 = 25.5 MB
```

### Compression Ratio

**~80×** vs f32 baseline. But only ~2.5× better than dense bitpacked.

### Key Property: Sparse VSA Operations

- Binding (XOR): Only XOR indices present in both vectors → O(d_active) instead of O(D)
- Bundling: Union of index sets with counter for repeats
- Similarity: Jaccard / overlap count → O(d_active₁ + d_active₂)

### Rust Feasibility

Moderate. Need COO or CSR sparse representation. `nalgebra_sparse` or hand-rolled. The key win is not just memory but _computational cost_ — similarity against a sparse bundle is O(d_active) instead of O(D), which matters for the O(V·D) full-vocabulary sweep on cold contexts (currently a fallback path in `predict_next_fast`).

### Known Work

- SparseHD (Imani et al., FCCM 2019): Sparse HD with 10% density, hardware-efficient
- Kanerva's original SDM uses sparse addresses (~√D active bits)

---

## 4. Technique 3: Product Quantization (PQ) for VSA Vectors

### Concept

Split each D-dimensional vector into M equal-length subvectors (e.g., M=256, each subvector of length D/M=40). Learn k-means codebooks per subspace (k=256 centroids). Store each word not as its full vector, but as M byte-sized indices (one per subspace). To reconstruct: look up M subvectors and concatenate.

### Formula (Jégou et al., 2011)

```
Codebook size = M × k × (D/M) × bytes_per_dim
Code size     = V × M × ceil(log₂(k) / 8)

With bipolar (1 bit per dim):
  Codebook = 256 × 256 × 40 / 8  = 327,680 bytes = 320 KB
  Codes    = 50,000 × 256 × 1   = 12,800,000 bytes = 12.8 MB
  Total    = ~13.1 MB
```

### Cosine Similarity with PQ

Reconstruct subvectors and compute cosine. Pre-computable: for each query, compute dot products with all k centroids per subspace (M×k operations), then any candidate's dot product is a sum over M lookups:

```
dot(q, reconstruct(code_i)) = Σ_{m=1..M} dot(q_m, C_m[code_i[m]])
```

This is O(M) per candidate instead of O(D). For M=256 vs D=10,240 that's a **40× speedup** in the scoring loop (critical for the O(V) cold-context fallback).

### Compression Ratio

**150×** vs f32 baseline, **~5×** vs dense bitpacked.

### Rust Implementation

Use `linfa-clustering` for k-means codebook training (offline, corpus-specific). The reconstruction is pure lookup tables. Store codes as `Vec<u8>` with M elements.

### Risk

PQ is _lossy_ — reconstruction error ≈ quantization distortion. For random vectors, PQ with M=256 subspaces and k=256 centroids preserves cosine similarity rank ordering well (subspace dimension 40 is well below the D/√M bound for subspace independence). For bundled vectors (sums of many word vectors), the reconstruction may degrade. Mitigation: train PQ on the _bundled_ vectors directly, not individual word vectors.

### Optimization: OPQ (Optimized Product Quantization)

Learn an orthonormal rotation matrix R ∈ R^{D×D} applied before PQ to decorrelate dimensions, improving reconstruction quality by 20-40%. Pre-multiply all vectors by R before encoding.

---

## 5. Technique 4: Learned Low-Dimensional Projections

### Concept

Train (or derive) a projection matrix W ∈ R^{D×d} where d << D, then store all transition patterns in the d-dimensional space. At scoring time, project candidate vectors through the same W and compute cosine similarity in the low-dimensional space.

### Formula

```
W_proj ∈ R^{D×d}   (project down)
W_back ∈ R^{d×D}   (approximate inverse, W_proj^T for JL-optimal)

Storage per transition: d × bytes_per_dim
For d=512, i16: 512 × 2 = 1024 bytes per vector
```

### Compression Ratio

| Scheme | Per Transition | Total (V=50K) |
|--------|---------------|---------------|
| f32 (D=10240) | 40,960 B | 2 GB |
| i16 (d=512) | 1,024 B | 51.2 MB |
| i16 (d=256) | 512 B | 25.6 MB |

**~40-80×** vs f32 baseline. But only ~1.25-2.5× better than dense bitpacked _for the codebook_. The real win here is for the **TransitionMemory** itself: we don't need to store full D-dimensional bundles.

### Johnson-Lindenstrauss Guarantee

A random Gaussian projection to d ≥ O(log(V)/ε²) preserves pairwise distances with distortion ≤ ε. For V=50,000, ε=0.1: d ≳ (8 × log(50000)) / 0.01 ≈ 8 × 10.8 / 0.01 ≈ **8,650**. This is close to the original D=10,240 — i.e., random projection alone won't compress much while preserving all pairwise similarities.

### Learned Projection (Supervised)

Train W_proj on the specific task of next-token prediction: minimize MSE between scores computed in full D-space and in projected d-space. For d=256-1024, this beats random projection by 3-10× in effective compression for the same accuracy.

### VSA Operation Commutation Problem

The fundamental issue: VSA binding ⊙ does _not_ commute with linear projection unless the projection is designed for it. If we store `proj(bundle(C(next1), C(next2), ...))`, we can't reconstruct the bundle and then unbind. Either:
- Store bundles in the projected space and score directly (loses unbinding path)
- Use a nonlinear projection (e.g., learned with a reconstruction loss that preserves Hadamard product)

### Rust Feasibility

Use `ndarray` for matrix ops, or `candle` for GPU-accelerated training of W. Once trained, projection is a simple matrix-vector multiply.

---

## 6. Technique 5: Sparse Distributed Memory (SDM) for Transition Storage

### Concept (Kanerva, 1988)

Instead of one vector per vocabulary word, allocate M fixed **hard locations** (e.g., M=100,000), each with:
- A D-dimensional address vector (random, fixed)
- A D-dimensional data counter (accumulator for stored patterns)

To **write** a pattern p addressed by a:
1. Find all hard locations whose address is within Hamming radius r of a (typically ~√D)
2. Add p to their data counters

To **read** at address a:
1. Find locations within radius r of a
2. Sum their data counters → accumulated pattern
3. The retrieved pattern ≈ the weighted sum of everything ever stored near a

### Why This Compresses Transition Memory

For the TBA, `TransitionMemory` stores `per_word: HashMap<usize, HyperVector>` — one vector per source word. With SDM:
- Allocate M fixed slots (e.g., M=100,000)
- The "address" for transitions out of word w is `C(w)` (the codebook vector)
- Write next-token bundles using `C(w)` as address → distributed across ~K hard locations near `C(w)`
- Read back by querying `C(w)` → gets weighted sum of stored patterns

### Compression Ratio

| Storage | Per-Word TBA | SDM (M=100K) |
|---------|-------------|--------------|
| Vectors stored | V ≈ 50,000 | M = 100,000 |
| Actually used | Only dominant words | All M always |
| **Key difference** | V grows with vocab | M is fixed |

With M=100K, SDM actually costs _more_ storage than per-word (100K vs 50K vectors). **The compression is not from reducing M, but from a different mechanism — automatic generalization**: words with similar addresses (nearby in Hamming space) share transition patterns, enabling zero-shot prediction for rare words.

### SDM for Trigram Compression (Where It Really Wins)

For trigrams, per-pair storage is V² (2.5B entries). SDM with M=1M hashes each (prev, current) pair to an address and distributes the transition pattern across ~K locations. M=1M is a tiny fraction of V²=2.5B.

### Rust Implementation

```rust
pub struct SdmTransitionMemory {
    addresses: Vec<Vec<u64>>,    // M bipolar address vectors (bitpacked)
    counters: Vec<Vec<f32>>,     // M accumulator vectors (or i16 for bundles)
    radius: u32,                 // Hamming radius for activation
    dim: usize,
}

impl SdmTransitionMemory {
    pub fn write(&mut self, addr: &[u64], pattern: &[f32]) {
        for (i, hard_addr) in self.addresses.iter().enumerate() {
            if hamming_distance(addr, hard_addr) <= self.radius {
                for j in 0..self.counters[i].len() {
                    self.counters[i][j] += pattern[j];
                }
            }
        }
    }
}
```

**Critical optimization**: Brute-force scanning M hard locations is O(M×D). Use **approximate nearest neighbor** (locality-sensitive hashing on the bitpacked addresses) to find activated locations in O(D + K×log M) instead of O(M×D). The `hnsw_rs` or `annoy-rs` crates handle this.

### Feasibility

**High** for a research prototype. The main implementation cost is the ANN index. Performance degrades gracefully with M: smaller M = more overlap = noisier retrieval, but VSA's high dimensionality means new random vectors are quasi-orthogonal to all stored patterns, so crosstalk is bounded.

---

## 7. Technique 6: Hierarchical/Compositional VSA Codebook

### Concept

Instead of V independent random vectors, construct a word's vector as the _binding of codebook vectors from a hierarchical factorization_. Example:

```
C("cat") = C_class("animal") ⊙ C_subclass("feline") ⊙ C_word("cat")
          = H₁[0] ⊙ H₂[3] ⊙ H₃[127]
```

With three codebooks of size C₁=50 (word classes), C₂=400 (subclasses), C₃=20,000 (words), the total number of stored vectors is 20,450 instead of 50,000 — but you can represent up to C₁ × C₂ × C₃ = 400 million unique words.

### Formula

```
Codebooks stored = Σ_{level} |C_level|
Max representable = Π_{level} |C_level|

Hierarchical: C("cat") = H_class ⊙ H_genus ⊙ H_species ⊙ H_word
```

### Resonator Network Decoding

Given a composite vector (e.g., from unbinding a transition prediction), the resonator network (Frady et al., 2020; Kent et al., 2020) factorizes it back into individual codebook atoms by iterative cleanup:

```
for each level l:
    unbind all OTHER levels → project onto C_l → pick best match → rebind
repeat until convergence
```

This replaces the O(V) vocabulary sweep with O(Σ|C_l|) factor lookups — which becomes O(log V) with a balanced hierarchy.

### Compression Ratio

| Scheme | Vectors Stored | Space (bitpacked, D=10240) |
|--------|---------------|---------------------------|
| Flat codebook | 50,000 | 64 MB |
| 3-level hierarchy (50+400+20000) | 20,450 | 26.1 MB |
| 4-level hierarchy (10+50+100+10000) | 10,160 | 13.0 MB |

Only **2.4-4.9×** compression for the codebook alone. The real win is for **trigram memory**: instead of V²=2.5B entries, you store per-class transitions (|C₁|²=2,500) and per-word transitions within class — a better decomposition.

### Rust Implementation

The resonator network requires implementing the iterative cleanup loop. A 3-level resonator might run 3-10 iterations per decode. Each iteration is O(|C₁|+|C₂|+|C₃|) dot products = ~20,450 operations — comparable to the current O(V) sweep for small V, but it scales as O(log V) for large V.

### Risk

Resonator networks can get stuck in local minima. Mitigation: stochastic cleanup (sample rather than argmax) or multi-start.

---

## 8. Technique 7: Eliminate Per-Word Transition Storage Entirely

### The Idea

The current codebase already has a **two-stage decode** (`predict_next_fast`, `lib.rs:231`):
1. **Engram** generates a candidate short-list from hash-addressed n-gram counts — O(K·log N) via hash table, independent of V
2. **TBA** only scores those K candidates — O(K·D) instead of O(V·D)

The problem is the TBA's _storage_, not its compute. If TBA were eliminated, only the Engram remains. The Engram already uses `HashMap<u64, HashMap<usize, u64>>` — its size is bounded by the number of _unique n-gram contexts_ seen in the corpus, not by vocabulary size.

For a 10M-word corpus: unique bigrams ≈ 500K-2M (Zipfian), each storing ~2-5 next-token counts → roughly 2M × 5 × 8 bytes ≈ **80 MB** for the Engram. That's already **50× smaller** than the f32 codebook alone.

### But What Does the TBA Add?

The TBA provides **VSA algebraic composition**: the bundled superposition captures _distributional similarity_ between words. Two words that appear in similar contexts produce similar transition bundles. This is the only learned/distributed aspect of the system — everything else is literal hash lookup.

### Hybrid: Engram + Small VSA Codebook

Keep the codebook lightweight (bitpacked, 64 MB) for VSA unbinding/analogy operations, and use the Engram for actual next-token probability. The TBA becomes an _optional_ signal blended with the Engram scores (weight w_tba, currently 1.0, could be reduced to 0.1 if TBA storage is reduced). With PQ-encoded transition bundles at 13 MB, the VSA signal costs almost nothing.

---

## 9. Technique 8: Federated / Incremental Compression

### Online Dictionary Learning

For incremental vocabulary growth, use **online k-means** or **Randomized SVD** to maintain a fixed-size codebook of basis vectors. New words are encoded as sparse linear combinations:

```
C("new_word") = sgn(Σ α_i · B_i)
```

where B_i are M fixed basis vectors (M << V). Storage = M × D/8 bytes for basis + V × M × bits for sparse coefficients.

For M=1000 basis vectors and 4-bit coefficients:
```
Basis = 1000 × 10240/8 = 1.28 MB
Coefficients = 50000 × 1000 × 0.5 = 25 MB
Total = ~26 MB
```

### Streaming VSA Encoding

Thomas et al. (2022, "Streaming Encoding Algorithms for Scalable Hyperdimensional Computing") use sparse Johnson-Lindenstrauss transforms for online encoding — the projection matrix doesn't need to be stored (it's a hash function), so new vocabulary items generate vectors on the fly with zero memory growth beyond the hash function seed.

### Application to AXIOM

The codebook's `get_or_insert` already generates vectors from a hash seed — it doesn't need pre-storage. The _only_ thing that needs storage is the **TransitionMemory** (bundles of observed transitions). If we can make the transition memory itself sparse/streaming, the total storage for new words is zero.

---

## 10. Summary: Recommended Compression Stack

| Layer | Technique | Memory (V=50K, D=10240) | Ratio vs f32 |
|-------|-----------|------------------------|--------------|
| **Codebook** | Bitpacked bipolar (intrinsic) | 0 MB (hash-generated) | ∞ |
| **Codebook (cached)** | Lazy bitpacked cache | 64 MB | 32× |
| **TransitionMemory** | PQ (M=256, k=256) | 13 MB | 160× |
| **TrigramMemory** | SDM (M=1M) or PQ on top-K pairs | ~100 MB | ~50-100× |
| **Engram (already OK)** | Hashmap of counts (existing) | ~80 MB | N/A |
| **Decoding** | Two-stage (Engram shortlist + PQ decoding), O(log V) | N/A | N/A |

### Total: ~160 MB vs current ~4 GB = **25× compression**

### Priority Implementation Order (Easiest → Hardest, Highest ROI First)

1. **Bitpacked HyperVector** (1 day): Change `Vec<f32>` to `Vec<u64>` bitpacked. 32× compression, zero accuracy loss for bipolar VSA. Enables the rest.

2. **Product-quantized TransitionMemory** (2-3 days): Train PQ on bundled transition vectors, store M-byte codes instead of D-bit vectors. 80-160× compression on transitions. Cosine similarity computed via precomputed lookup tables, which also gives 40× speedup in scoring.

3. **SDM-based TrigramMemory** (3-5 days): Replace `HashMap<(usize,usize), HyperVector>` with M fixed hard locations + ANN index. Compresses V²→M, enables zero-shot trigram prediction through address similarity.

4. **Hierarchical codebook + resonator** (5-10 days, research-heavy): Factorize vocabulary into 3-4 level product codebooks. Replaces O(V) decode sweep with O(Σ|C_l|) factor lookup. Critical for scaling beyond 100K words.

5. **TBA elimination / weight reduction** (tuning): Profile how much accuracy the TBA signal actually adds vs pure Engram. If Engram alone achieves 90%+ of combined accuracy on next-token prediction, reduce w_tba to 0.1 and store only PQ-encoded transitions as a lightweight boost.

---

## 11. Concrete Rust Implementation Template: BitpackedBipolarVector

```rust
/// Bitpacked bipolar hypervector: D dimensions in D/64 u64s.
/// Bit=1 means +1, bit=0 means -1.
#[derive(Clone, Serialize, Deserialize)]
pub struct BipolarVector {
    data: Box<[u64]>,
}

impl BipolarVector {
    pub fn random_bipolar(dim: usize, seed: u64) -> Self {
        let words = dim.div_ceil(64);
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut data = vec![0u64; words];
        for i in 0..dim {
            if rng.gen_bool(0.5) {
                data[i / 64] |= 1 << (i % 64);
            }
        }
        Self { data: data.into_boxed_slice() }
    }

    /// Cosine similarity via popcount. 
    /// cos(a,b) = (same - diff) / D = (2*same - D) / D
    pub fn cosine(&self, other: &Self) -> f32 {
        let same: u32 = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| (!(a ^ b)).count_ones())
            .sum();
        (2.0 * same as f32 - self.dim() as f32) / self.dim() as f32
    }

    /// Bind = XOR (Hadamard product for bipolar {-1,+1} where * maps to XOR)
    pub fn bind(&self, other: &Self) -> Self {
        Self { data: self.data.iter().zip(other.data.iter())
            .map(|(a, b)| a ^ b).collect::<Vec<_>>().into_boxed_slice() }
    }

    /// Bundle N vectors. The result is a vector of i16 counters,
    /// not bipolar — requires thresholding/conversion back to bipolar.
    /// For a transition bundle, store the counters directly (i16 vector).
    pub fn bundle(vectors: &[&Self]) -> CountVector {
        let dim = vectors[0].dim();
        let words = dim.div_ceil(64);
        let mut counters = vec![vec![0i16; 64]; words];
        for v in vectors {
            for (w, &word) in v.data.iter().enumerate() {
                for b in 0..64 {
                    counters[w][b] += if (word >> b) & 1 == 1 { 1 } else { -1 };
                }
            }
        }
        CountVector { counters, dim }
    }
}

/// i16 accumulator vector for bundling — not bit-packed.
pub struct CountVector {
    counters: Vec<[i16; 64]>,
    dim: usize,
}

impl CountVector {
    /// Cosine similarity with a bipolar query vector.
    /// For bundled vectors, cos(bundle, query) ≈ (1/N) * Σ cos(v_i, query).
    pub fn cosine(&self, query: &BipolarVector) -> f32 {
        let dot: i64 = self.counters.iter().enumerate()
            .flat_map(|(w, chunk)| {
                let qword = query.data[w];
                (0..64).map(move |b| {
                    chunk[b] as i64 * if (qword >> b) & 1 == 1 { 1 } else { -1 }
                })
            })
            .sum();
        dot as f32 / (self.norm() * query.norm())
    }

    fn norm(&self) -> f32 {
        let sum_sq: i64 = self.counters.iter()
            .flat_map(|chunk| chunk.iter().map(|&c| (c as i64) * (c as i64)))
            .sum();
        (sum_sq as f32).sqrt()
    }
}
```

---

## References

1. Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.
2. Jégou, H., Douze, M., & Schmid, C. (2011). "Product Quantization for Nearest Neighbor Search." *IEEE TPAMI*.
3. Roberts, A., Raffel, C., & Shazeer, N. (2020). "How Much Knowledge Can You Pack Into the Parameters of a Language Model?" *EMNLP*.
4. Morris, J.X., et al. (2025). "How Much Do Language Models Memorize?" *arXiv:2504.19427*.
5. Ma, S., et al. (2024). "The Era of 1-bit LLMs: All Large Language Models Are in 1.58 Bits." *arXiv:2402.17764*.
6. Imani, M., et al. (2019). "SparseHD: Algorithm-Hardware Co-Optimization for Efficient High-Dimensional Computing." *FCCM*.
7. Imani, M., et al. (2019). "QuantHD: A Quantization Framework for Hyperdimensional Computing." *IEEE TCAD*.
8. Frady, E.P., Kent, S.J., Olshausen, B.A., & Sommer, F.T. (2020). "Resonator Networks, 1: An Efficient Solution for Factoring High-Dimensional, Distributed Representations of Data Structures." *Neural Computation*.
9. Thomas, A., et al. (2022). "Streaming Encoding Algorithms for Scalable Hyperdimensional Computing." *arXiv*.
10. Elhage, N., et al. (2022). "Toy Models of Superposition." *Transformer Circuits Thread*.
11. Pandey, N.P., et al. (2025). "DPQ-HD: Post-Training Compression for Ultra-Low Power Hyperdimensional Computing." *ACM*.
12. Ge, L., & Parhi, K.K. (2020). "Classification Using Hyperdimensional Computing: A Review." *IEEE CAS Magazine*.
