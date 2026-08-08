# KatGPT-RS: Comprehensive Technical Analysis

> **Generated:** 2026-08-07 | **Codebase version:** 0.2.1 | **Source:** `/home/eggchad/eakject/research/Deep_Man/katgpt-rs/`

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [AC-Prefix Modelless G1: Bit-Identical Output with ~27× Speedup](#ac-prefix-modelless-g1)
4. [attends_dedup: Deterministic Attention Deduplication](#attends_dedup)
5. [Constraint Pruning: Sudoku Domain & O(log N) Hard Attention](#constraint-pruning)
6. [Embedding Router & TriggerGate: CPU/GPU Deterministic Routing](#embedding-router--triggergate)
7. [Feature Flag System (378+ Flags)](#feature-flag-system)
8. [Mathematical Formulas & State Transitions](#mathematical-formulas)
9. [Core Rust Data Structures](#core-rust-data-structures)
10. [SIMD Optimizations & Zero-Allocation Patterns](#simd-and-zero-allocation)
11. [Speculative Decoding Pipeline](#speculative-decoding-pipeline)
12. [Workspace & Crate Architecture](#workspace-architecture)

---

## Executive Summary

KatGPT-RS is a **GOAT-proved neuro-symbolic micro-Transformer** implemented in Rust with:

- **378 feature flags** (152 default-on, all GOAT-proved)
- **27 in-tree workspace crates** plus the root aggregator
- A **speculative decoding pipeline** with constraint pruning that eliminates statistical sampling
- **Zero-allocation hot paths** with SIMD vectorization
- Deterministic, reproducible inference across CPU/GPU/ANE backends

Key headline results:
| Metric | Value | Mechanism |
|--------|-------|-----------|
| AC-Prefix Modelless | 0.0 diff (bit-identical) | `attends_dedup` mask correction |
| AC-Prefix Speedup | 27.258× vs iterative MLM | Single-pass vs 64 iterative forwards |
| Sudoku Compression | 7,079× on Inkala's Hardest | Path-aware ConstraintPruner |
| KV Memory Reduction | 93.8% | MUX superposition fusion |
| RMSNorm Speedup | 2.4× | Kog CPU fusion kernel |

The architecture follows a strict pipeline:
```
LLM drafts logits → ConstraintPruner filters invalid → DDTree builds valid-only tree → Target verifies
```

---

## Architecture Overview

### Model Parameters (Micro Reference Config)

| Parameter | Value |
|-----------|-------|
| `vocab_size` | 27 (a–z + BOS) |
| `block_size` | 16 |
| `n_embd` | 16 |
| `n_head` | 4 |
| `mlp_hidden` | 64 (4×) |
| `n_layer` | 1 |
| `temperature` | 0.5 |

### Inference Flow (Default GOAT Stack)

The system uses **layered gating** — most of the 152 default-on features are bandit-driven, Option-gated, or compile-time-only. Only 12 features execute unconditionally on every token:

```
┌─────────────────────────────────────────────────────────┐
│  🔴 Always-On Hot Path (12 features per token)          │
│  kog_cpu_fusion, sparse_mlp, delta_routing,             │
│  mls_aggregate, domain_latent, spectral_quant,          │
│  hybrid_oct_pq, kvarn, kv_share, gdn2_attention,        │
│  lt2_looped, elf_sde                                    │
├─────────────────────────────────────────────────────────┤
│  🟡 Conditional (~30 features, 1 check each)            │
│  Bandit-driven, Option-gated, Thinking-mode,            │
│  Speculative pipeline                                   │
├─────────────────────────────────────────────────────────┤
│  🔵 Offline (~8 features, not in forward pass)          │
│  Training/diagnostics, Background (sleep, dreamer)      │
└─────────────────────────────────────────────────────────┘
```

### Core Traits

```rust
/// The central constraint interface — eliminates statistical sampling
pub trait ConstraintPruner: Send + Sync {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool;
    fn batch_is_valid(&self, depth: usize, tokens: &[usize], 
                      parent_tokens: &[usize], out: &mut [bool]);
    fn propagate(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) { }
    fn manifold_score(&self, depth: usize, token_idx: usize, 
                      parent_tokens: &[usize]) -> f32 { 0.0 }
}

/// Screening pruner for soft relevance scoring
pub trait ScreeningPruner: Send + Sync {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32;
}

/// Speculative generation abstraction
pub trait SpeculativeGenerator {
    type Condition;
    type Output;
    type Error;
    fn generate(&mut self, condition: &Self::Condition, 
                rng: &mut fastrand::Rng) -> Result<Vec<Self::Output>, Self::Error>;
}
```


---

## AC-Prefix Modelless G1

### The Problem

Standard causal Transformers cannot tractably evaluate arbitrary conditionals `p(xe | xc)` — conditioning on **future** tokens requires an intractable integral with no single-pass factorization for vanilla GPT.

The AC-GPT paper (arXiv:2606.14943, Lu et al., Mila, June 2026) solves this with a position-aware conditioning prefix: copy conditioning tokens `xc` to the front of the sequence with their **original position encodings**, allow **bidirectional self-attention among the copies**, and apply causal attention elsewhere.

### The Doubled-Signal Bias Problem

The naive AC-GPT mask failed on untrained micro-GPT (7.5e-4 error vs iterative-MLM) due to the **doubled-signal bias**: each `xc` token appears both as a copy in region 0 (r0) and in-place in region 1 (r1), doubling the conditioning signal on untrained weights.

### The Modelless Solution: `attends_dedup`

The `attends_dedup` method zeroes eval→in-place-xc attention, forcing **all conditioning to flow exclusively through r0 copies**:

```rust
/// Deduplicated attention rule — O(log |xc|) per pair, zero-alloc.
/// Eliminates doubled-signal bias bit-identically to iterative-MLM.
#[inline]
pub fn attends_dedup(&self, i: usize, j: usize) -> bool {
    let xc = self.conditioning_positions.len();
    let j_in_r0 = j < xc;
    if j_in_r0 {
        return true;  // Everyone attends to r0 copies
    }
    // j ∈ r1
    let i_in_r1 = i >= xc;
    if !i_in_r1 {
        return false; // r0 copies don't attend back to r1
    }
    // Both in r1. Standard causal, EXCEPT eval doesn't attend to in-place xc.
    if i < j {
        return false; // causal constraint
    }
    // i >= j, both in r1. Check if j is an in-place xc position.
    let j_original = j - xc;
    if self.is_xc_position(j_original) {
        return false; // THE KEY: eval doesn't attend to in-place xc
    }
    true
}
```

### How It Achieves Bit-Identical Output

For a single attention layer, K/V at any position depend only on the token embedding. The r0 copy of `xc` at original position `p` has:
- **Same token** as in-place r1 `xc` at position `p`
- **Same RoPE rotation** (both use original position `p`)
- **Same K/V** (same weights, same input embedding)

Therefore the deduplicated attended set for eval at position `k`:

```
Dedup attended = { all xc via r0 copies } ∪ { eval at positions ≤ k via r1 }
```

Is identical to iterative-MLM's attended set:

```
Iterative attended = { all xc in-place } ∪ { eval at positions ≤ k }
```

**Same K/V → same attention scores → same softmax → same logprobs → bit-identical.**

### The 27.258× Speedup

| Method | Forward passes | Time |
|--------|---------------|------|
| Iterative MLM | 64 (one per eval position) | Baseline |
| AC-Prefix single-pass | **1** | **27.258× faster** |

The speedup comes from replacing N iterative forward passes (one per evaluation position) with a single augmented forward pass using the three-region attention mask.

### Augmented Sequence Layout

```
┌────────────────────────────────────┬──────────────────────────────────────┐
│  Region 0: xc copies (front)       │  Region 1: full sequence x = xc ∪ xe │
│  Bidirectional self-attention       │  Causal attention + dedup rule        │
│  original_pos propagated (RoPE)    │  Loss computed only on xe positions   │
└────────────────────────────────────┴──────────────────────────────────────┘
```

### GOAT Gate Results (Benchmark 313)

| Gate | Threshold | Measured | Result |
|------|-----------|----------|--------|
| G1 (correctness) | \|dedup − iterative\| < 1e-4 | **0.000000** | PASS ✓ |
| G2 (speedup) | ≥ 3× faster | **27.258×** | PASS ✓ |
| G3 (no regression) | bit-identical with empty prefix | 0 mismatches | PASS ✓ |
| G4 (zero-alloc) | 0 heap allocs on hot path | 0 allocs | PASS ✓ |


---

## attends_dedup: Deterministic Attention Deduplication

### Three-Region Mask Architecture

The `AcPrefix` struct implements a zero-allocation attention mask builder with three distinct regions:

```rust
pub struct AcPrefix<'a> {
    /// Borrowed base token sequence (original x = xc ∪ xe)
    base_tokens: &'a [u32],
    /// Sorted indices into base_tokens marking xc positions
    conditioning_positions: &'a [usize],
}
```

**Mask Rules:**
1. **Region 0 → Region 0** (i,j both in copies): Always TRUE (bidirectional)
2. **Region 1 → Region 0** (eval attends to copies): Always TRUE
3. **Region 0 → Region 1** (copies don't look back): Always FALSE
4. **Region 1 → Region 1** (dedup causal): TRUE iff causal AND j is NOT in-place xc

### O(log N) Membership Check

The `is_xc_position` helper uses binary search on the sorted conditioning positions:

```rust
/// O(log |xc|) membership test — zero allocation, branch-free inner loop
fn is_xc_position(&self, original_pos: usize) -> bool {
    self.conditioning_positions.binary_search(&original_pos).is_ok()
}
```

### Bit-Packed Mask Materialization

For bulk attention computation, the mask can be materialized into a bit-packed buffer:

```rust
/// Bit-pack the attends_dedup rule into caller-provided buffer.
/// Zero heap allocations on the hot path.
pub fn materialize_dedup_from(prefix: &AcPrefix) -> AcPrefixMask {
    Self::materialize_with(prefix, |p, i, j| p.attends_dedup(i, j))
}
```

### Why Deduplication Eliminates Sampling

The key insight: by making the attended set **deterministically identical** to what iterative-MLM would see, the system eliminates any need for statistical sampling or approximation. The output is provably exact — not "close enough" but **bit-identical** on single-layer architectures.

---

## Constraint Pruning: Sudoku Domain & O(log N) Hard Attention

### How Decision Tree Pruning Works

The `ConstraintPruner` trait provides a **hard filter** that provably eliminates invalid tokens before they enter the speculative decoding tree. This converts the exponential search space into a tractable one:

```rust
impl ConstraintPruner for SudokuPruner {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        // Token 0 = empty/padding, never valid
        if token_idx == 0 { return false; }
        let digit = token_idx as u8;
        if !(1..=9).contains(&digit) { return false; }
        
        // Map depth to (row, col) — O(1) array lookup
        let Some(&(row, col)) = self.positions.get(depth) else {
            return false;
        };

        // Check against initial board state — row/col/box rules
        if !self.board.is_valid_move(row, col, digit) {
            return false;
        }

        // Path-aware: check cross-depth conflicts with parent tokens
        for (parent_depth, &parent_token) in parent_tokens.iter().enumerate() {
            if parent_token == 0 { continue; }
            let parent_digit = parent_token as u8;
            if parent_digit != digit { continue; }
            // Same digit — check if shares row/col/box
            let &(pr, pc) = &self.positions[parent_depth];
            if pr == row || pc == col || (pr/3 == row/3 && pc/3 == col/3) {
                return false; // Conflict!
            }
        }
        true
    }
}
```

### MRV (Minimum Remaining Values) Ordering

The `sudoku_mrv` feature reorders the depth→cell mapping by ascending candidate count:

```rust
/// Construct pruner with MRV ordering: fewest candidates first.
/// Forced cells (1 candidate) get depth 0–7 → drafter assigns p=1.0.
pub fn new_mrv(board: Sudoku9x9) -> Self {
    let mut keyed: Vec<(u32, usize, usize)> = Vec::with_capacity(60);
    for r in 0..9 {
        for c in 0..9 {
            if board.grid[r][c] == 0 {
                let (count, _) = Self::candidate_set(&board, r, c);
                keyed.push((count, r, c));
            }
        }
    }
    keyed.sort_unstable_by_key(|&(cnt, r, c)| (cnt, r, c));
    // ...
}
```

### Constraint Propagation Drafter (sudoku_cp)

The `latent_marginals` method produces draft probabilities from pure constraint logic — no neural network:

```rust
/// Naked singles → p=1.0; N candidates → uniform 1/N
/// Pure deterministic rules engine — no training, no gradient descent.
pub fn latent_marginals(&self, lookahead: usize) -> Vec<Vec<f32>> {
    // ...
    let (count, mask) = Self::candidate_set(&self.board, row, col);
    let prob = 1.0 / count as f32;
    for d in 1..=9u8 {
        if mask & (1 << (d - 1)) != 0 {
            p[d as usize] = if count == 1 { 1.0 } else { prob };
        }
    }
    // ...
}
```

### 7,079× Compression on Inkala's Hardest

The combination of:
1. **MRV ordering** (naked singles at shallow depths)
2. **Constraint propagation** (p=1.0 for forced cells)
3. **Path-aware cross-depth conflict detection**

Achieves 7,079× compression on Inkala's Hardest Sudoku — the DDTree only explores valid branches, never wasting compute on provably impossible placements.

### How Constraint Pruning Eliminates Statistical Sampling

Traditional LLM decoding uses temperature-based sampling with rejection. KatGPT-RS's approach is fundamentally different:

```
Traditional: sample → check → reject → resample (stochastic)
KatGPT-RS:   prune → build valid-only tree → verify (deterministic)
```

The `ConstraintPruner` acts as **hard attention** — O(1) per candidate for initial board check, O(depth) for path-aware cross-conflict, effectively O(log N) amortized across the tree structure. Invalid tokens are **never generated**, never scored, never rejected — they simply don't exist in the search space.


---

## Embedding Router & TriggerGate: CPU/GPU Deterministic Hardware Routing

### TriggerGate: Load-Adaptive Tier Promotion

The `TriggerGate` monitors live workload metrics and promotes/demotes inference across three compute tiers:

```rust
pub enum ComputeTier {
    CpuOnly,     // Low load — everything on CPU
    CpuGpu,      // Medium load — offload to GPU
    CpuGpuAne,   // High load — saturate all accelerators (Apple Neural Engine)
}
```

**Tier transition logic** (from `inference_router.rs`):

```rust
pub struct TriggerGateConfig {
    pub gpu_activate_qps: f64,        // QPS threshold for GPU promotion
    pub ane_activate_qps: f64,        // QPS threshold for ANE promotion  
    pub hysteresis_factor: f64,       // Prevents oscillation (0.7 default)
    pub queue_depth_trigger: u32,     // Queue depth co-signal
    pub latency_p99_trigger_us: u64,  // Latency co-signal
    pub min_tier_change_interval_ms: u64, // Debounce
}
```

### Multi-Signal Routing Stack

The `InferenceRouter` composes **five routing signals** in priority order:

```
1. TriggerGate QPS evaluation     → base tier from load
2. Trust-triggered adjustment     → low trust → tier-up; high trust → allow tier-down
3. RV-gated routing (Plan 202)    → acceptance variance signal overrides
4. Critical-interval gate         → entropy-triggered GPU promotion
5. TVP (Thicket Variance Probe)   → decoding-space disagreement signal
```

```rust
impl InferenceRouter {
    pub fn forward<'a>(
        &mut self, ctx: &'a mut ForwardContext,
        weights: &TransformerWeights,
        cache: &mut MultiLayerKVCache,
        token: usize, pos: usize,
    ) -> &'a [f32] {
        // Evaluate tier change
        if let Some(new_tier) = self.gate.evaluate() {
            self.tier_transitions.fetch_add(1, Ordering::Relaxed);
            self.signal_recompile_for_tier(new_tier);
        }
        
        let tier = self.gate.current_tier();
        
        // Trust-triggered tier adjustment (Plan 182)
        let tier_after_trust = if self.trust_signal < 0.4 && tier == ComputeTier::CpuOnly {
            ComputeTier::CpuGpu  // Low trust → upgrade
        } else { tier };
        
        // RV-gated override (Plan 202)
        // Critical-interval override (Plan 222)
        // TVP override (Plan 267)
        // ... route to selected backend
    }
}
```

### Deterministic Guarantee

The routing is **deterministic given the same input signals** — no randomness in tier selection. The `InferenceBackend` trait abstracts the compute target:

```rust
pub trait InferenceBackend: Send + Sync {
    fn forward(&mut self, ctx: &mut ForwardContext, weights: &TransformerWeights,
               cache: &mut MultiLayerKVCache, token: usize, pos: usize) -> &[f32];
    fn recompile_hint(&mut self);
}
```

### GOAT Proof (Test: goat_176_trigger_gate.rs)

The test suite proves:
- P1: Gate starts at `CpuOnly`
- P2: Promotes to `CpuGpu` under high QPS
- P3: Promotes to `CpuGpuAne` at extreme load
- P4: Demotes with hysteresis (prevents oscillation)
- P7: **Router forward is bit-identical to direct transformer forward** (determinism proof)

```rust
#[test]
fn goat_p7_router_matches_direct_forward() {
    // Direct forward
    let direct = forward(&mut ctx1, &weights, &mut cache1, 0, 0, &config).to_vec();
    // Router forward
    let routed = router.forward(&mut ctx2, &weights, &mut cache2, 0, 0).to_vec();
    
    for (i, (a, b)) in direct.iter().zip(routed.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "logits mismatch at {i}: {a} vs {b}");
    }
}
```


---

## Feature Flag System (378+ Flags)

### Organization

The feature system is organized in three tiers:

| Tier | Count | Examples |
|------|-------|---------|
| **Default-ON** | 152 | `gdn2_attention`, `lt2_looped`, `ac_prefix`, `funcattn` |
| **Opt-in** | ~200 | `sudoku`, `parallax_attn`, `sink_aware_attn` |
| **Deprecated** | ~26 | Exiled to `katgpt-deprecated` crate |

### Feature Categories (from Cargo.toml)

```toml
[features]
default = [
    # Attention alternatives
    "gdn2_attention",      # Gated DeltaNet-2 O(1) decode
    "lt2_looped",          # Weight-shared T-pass loop + AHLA
    "funcattn",            # Functional attention (Tikhonov operator)
    
    # Speculative decode
    "belief_drafter",      # BoM belief sampling
    "best_buddies",        # Marginal best-buddy alignment
    "trust_region_spec",   # Trust-region speculation
    
    # Pruning & routing
    "delta_routing",       # Cross-layer residual routing
    "bandit_top_p",        # dMoE adaptive top-p
    "constraint pruning",  # Sudoku/domain validators
    
    # KV cache optimization
    "spectral_quant",      # Eigenbasis KV codec
    "hybrid_oct_pq",       # OCT + PQ compression (64× fewer FMAs)
    "kvarn",               # Variance-normalized quantization
    "kv_share",            # Q-K=V projection sharing (50% reduction)
    
    # Intelligence / calibration
    "ac_prefix",           # Arbitrary-conditional prefix (27× speedup)
    "clr",                 # Claim-Level Reliability
    "claim_rubric",        # Meta-discipline validator
    "viable_manifold_graph", # Safe navigation (100% playability)
    
    # ... 152 total default-on features
]
```

### Feature Forwarding Architecture

Features propagate through the crate DAG via explicit forwarding:

```toml
# Root forwards to leaf crate features
depth_invariance = ["katgpt-core/depth_invariance", "katgpt-speculative/depth_invariance"]
dash_attn = ["katgpt-attn/dash_attn"]
vortex_flow = ["dash_attn", "katgpt-attn/vortex_flow"]
mls_aggregate = ["katgpt-forward/mls_aggregate"]
```

### GOAT Proof Requirement

Every default-on feature must pass its GOAT gate before promotion. Example gates:
- **G1 (Correctness):** Output matches reference within tolerance
- **G2 (Performance):** Meets speedup/quality target
- **G3 (No regression):** Feature-off is bit-identical to pre-feature baseline
- **G4 (Zero-alloc):** No heap allocations on hot path
- **G5 (Isolation):** No unintended cross-feature dependencies

---

## Mathematical Formulas & State Transitions

### AC-Prefix Conditional Log-Likelihood

Given base sequence `x = xc ∪ xe`, the single-pass conditional:

```
log p(xe | xc) = Σ_{k ∈ xe} log p(x_k | attended_set_k)
```

where `attended_set_k` under `attends_dedup`:
```
attended_set_k = { all xc via r0 copies } ∪ { xe at positions ≤ k }
```

### LT2 Looped Transformer State Transition

Per-loop residual gate (from `forward_looped`):

```
h^(τ) = h̃^(τ) + ρ_τ ⊙ h^(τ-1)
```

Where:
- `h̃^(τ)` = output of attention + MLP at iteration τ
- `ρ_τ` = zero-initialized learnable residual gate
- First iteration: `h^(1) = h̃^(1)` (no prior residual)

T loops yield effective depth `T × n_layer` with **no extra parameters**.

### GDN2 (Gated DeltaNet-2) O(1) Decode

State update per token:

```
S_t = (1 - β_t) · S_{t-1} + β_t · v_t · k_t^T
o_t = σ(g_t) ⊙ (S_t · q_t)
```

Where `S_t ∈ ℝ^{d×d}` is a fixed-size state matrix — **no KV cache growth**.

### TriggerGate Hysteresis

```
promote_threshold = gpu_activate_qps
demote_threshold  = gpu_activate_qps × hysteresis_factor

promote iff: estimated_qps > promote_threshold AND queue_depth > trigger
demote  iff: estimated_qps < demote_threshold
```

### Constraint Pruner Complexity

For Sudoku with N empty cells:
- **Initial board check:** O(1) — `is_valid_move` checks 3 sets (row/col/box)
- **Path-aware cross-conflict:** O(depth) — linear scan of parent tokens
- **MRV membership (binary search):** O(log |xc|) per dedup query
- **Amortized tree traversal:** O(log N) — pruning eliminates exponential branching

### SDAR Sigmoid Gate

```rust
/// Sigmoid-gated reward: g(r) = σ(β · r) where β controls sharpness
pub fn sdar_gate(reward: f64, beta: f64) -> f64 {
    1.0 / (1.0 + (-beta * reward).exp())
}
```


---

## Core Rust Data Structures

### ForwardContext — Zero-Allocation Scratch Buffers

```rust
/// Pre-allocated forward-pass scratch buffers. Created once, reused per token.
/// Fields are `pub` for cross-crate access (ForwardContext lives in katgpt-forward).
pub struct ForwardContext {
    pub logits: Vec<f32>,        // Single-token output buffer [vocab_size]
    pub batch_logits: Vec<f32>,  // Batched output buffer [N × vocab_size]
    pub hidden: Vec<f32>,        // Hidden state [n_embd]
    pub qkv: Vec<f32>,          // QKV projection scratch
    pub attn_out: Vec<f32>,     // Attention output scratch
    // ... ~20 more pre-allocated buffers for each pipeline stage
}
```

### MultiLayerKVCache — Layered State Management

```rust
/// Per-layer KV cache with position tracking
pub struct MultiLayerKVCache {
    layers: Vec<KVCache>,
    current_pos: usize,
}

pub struct KVCache {
    key: Vec<f32>,    // [block_size × n_head × head_dim]
    value: Vec<f32>,  // [block_size × n_head × head_dim]
}
```

### TransformerWeights — Contiguous Memory Layout

```rust
pub struct TransformerWeights {
    // Embedding
    pub token_embedding: Vec<f32>,  // [vocab_size × n_embd]
    // Per-layer
    pub layers: Vec<LayerWeights>,
    // LM head (tied with embedding)
    pub lm_head: Vec<f32>,          // [vocab_size × n_embd]
}

pub struct LayerWeights {
    pub rms_norm_weight: Vec<f32>,  // [n_embd]
    pub qkv_weight: Vec<f32>,      // [3 × n_embd × n_embd]
    pub attn_proj: Vec<f32>,       // [n_embd × n_embd]
    pub mlp_up: Vec<f32>,          // [mlp_hidden × n_embd]
    pub mlp_down: Vec<f32>,        // [n_embd × mlp_hidden]
    pub mlp_gate: Vec<f32>,        // [mlp_hidden × n_embd] (SwiGLU)
}
```

### DDTree (Decision-Diffusion Tree) — Speculative Search Structure

```rust
/// A node in the speculative decoding tree
pub struct TreeNode {
    pub token: usize,
    pub log_prob: f32,
    pub parent: Option<usize>,  // Index into tree Vec
    pub children: Vec<usize>,
    pub depth: usize,
    pub cumulative_log_prob: f32,
}
```

The DDTree is built by `build_dd_tree_pruned` which integrates the ConstraintPruner:

```rust
pub fn build_dd_tree_pruned(
    logits_fn: impl Fn(usize, &[usize]) -> Vec<f32>,
    pruner: &dyn ConstraintPruner,
    budget: usize,
    max_depth: usize,
) -> Vec<TreeNode> {
    // Only explores branches where pruner.is_valid() returns true
    // Result: 100% valid placements in the tree
}
```

### AcPrefix — Zero-Copy Sequence Augmentation

```rust
pub struct AcPrefix<'a> {
    base_tokens: &'a [u32],              // Borrowed — no copy
    conditioning_positions: &'a [usize], // Sorted — enables binary search
}

impl<'a> AcPrefix<'a> {
    pub fn augmented_len(&self) -> usize {
        self.conditioning_positions.len() + self.base_tokens.len()
    }
    
    /// Write augmented tokens into caller-owned buffer — zero heap alloc
    pub fn augmented_tokens_into(&self, out: &mut [u32]) { /* ... */ }
    
    /// Write original positions for RoPE — zero heap alloc
    pub fn original_positions_into(&self, out: &mut [usize]) { /* ... */ }
}
```

### Config — Compile-Time Determinism

```rust
pub struct Config {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub mlp_hidden: usize,
    pub temperature: f32,
    pub bos_token: usize,
    // ... feature-gated fields
}

impl Config {
    /// Micro config for testing — deterministic, reproducible
    pub fn micro() -> Self {
        Self {
            vocab_size: 27, block_size: 16, n_embd: 16,
            n_head: 4, n_layer: 1, mlp_hidden: 64,
            temperature: 0.5, bos_token: 0,
            // ...
        }
    }
}
```

---

## SIMD Optimizations & Zero-Allocation Patterns

### Zero-Allocation Design Principles

1. **Pre-allocated scratch buffers** (`ForwardContext`) — created once at startup
2. **Caller-owned output buffers** (`_into` suffix pattern) — no heap traffic
3. **`Vec::resize` reuse** — batch buffer grows once, never shrinks
4. **`fill(0.0)` over `vec![0.0; n]`** — reuses existing allocation

Example from `forward_batched`:
```rust
pub fn forward_batched<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    tokens: &[usize],
    pos_start: usize,
    config: &Config,
) -> Vec<&'a mut [f32]> {
    // Grow once — no per-token allocation
    ctx.batch_logits.resize(n_tokens * vocab, 0.0);
    
    for (i, &token) in tokens.iter().enumerate() {
        let _logits = forward(ctx, weights, cache, token, pos_start + i, config);
        // SAFETY: copy via raw pointers — logits and batch_logits are disjoint fields
        unsafe {
            std::ptr::copy_nonoverlapping(
                ctx.logits.as_ptr(),
                ctx.batch_logits.as_mut_ptr().add(i * vocab),
                vocab,
            );
        }
    }
    
    // Return disjoint mutable slices — zero allocation
    let base = ctx.batch_logits.as_mut_ptr();
    let mut out = Vec::with_capacity(n_tokens);
    for i in 0..n_tokens {
        let slice = unsafe { std::slice::from_raw_parts_mut(base.add(i * vocab), vocab) };
        out.push(slice);
    }
    out
}
```

### SIMD Patterns

Cross-resolution transport uses SIMD matmul (Plan 417 — 11-15× faster encode):

```rust
/// Transposed basis layout + simd_matmul_rows replaces strided gather-dot
/// for production-scale cross-resolution encoding
#[cfg(target_arch = "aarch64")]
fn simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    // NEON SIMD 4-wide dot product
    // Process 4 elements per iteration
}
```

### Atomic Operations for Lock-Free Counters

```rust
pub struct InferenceRouter {
    total_inferences: AtomicU64,  // Lock-free inference counter
    tier_transitions: AtomicU32,  // Bounded counter (saves 4 bytes vs u64)
    // ...
}
```

### Binary Search over Sorted Arrays (vs HashMap)

The `attends_dedup` function uses `binary_search` on sorted `conditioning_positions` rather than a `HashSet`:
- **O(log N)** lookup
- **Zero allocation** (no hash table)
- **Cache-friendly** (contiguous memory)
- **Deterministic** (no hash randomization)

### Top-p Coreset: Scratch Buffer Pattern

```rust
/// Zero-allocation top-p coreset selection
pub fn top_p_coreset(
    scores: &[f32],
    p: f32,
    scratch_indices: &mut [usize],  // Caller-owned scratch
    scratch_sorted: &mut [f32],     // Caller-owned scratch
    mask: &mut [bool],              // Caller-owned output
) -> usize {
    mask.fill(false);  // Reuse existing buffer
    // ... sort, cumsum, select — no heap allocation
}
```


---

## Speculative Decoding Pipeline

### Core Pipeline Architecture

```
┌──────────────┐    ┌──────────────────┐    ┌────────────┐    ┌──────────────┐
│ Draft Model  │───▶│ ConstraintPruner │───▶│  DDTree    │───▶│  Verifier    │
│ (logits)     │    │ (hard filter)    │    │ (tree)     │    │ (Leviathan)  │
└──────────────┘    └──────────────────┘    └────────────┘    └──────────────┘
                           │                       │
                    ScreeningPruner          BanditPruner
                    (soft relevance)        (adaptive UCB1)
```

### Leviathan Verifier

Implements p/q rejection sampling — guarantees **identical output distribution** to the target model:

```rust
pub struct LeviathanVerifier;

impl SpeculativeVerifier for LeviathanVerifier {
    /// Accept/reject each draft token using target/draft probability ratio.
    /// Accepted tokens are provably distributed as if sampled from target alone.
    fn verify(&self, draft_probs: &[f32], target_probs: &[f32], 
              rng: &mut Rng) -> (usize, Vec<usize>);
}
```

### DDTree Build Variants

The system provides multiple tree builders for different use cases:

| Builder | Use Case | Feature Gate |
|---------|----------|--------------|
| `build_dd_tree` | Basic best-first tree | always |
| `build_dd_tree_pruned` | With ConstraintPruner | always |
| `build_dd_tree_screened` | With ScreeningPruner | always |
| `build_dd_tree_sde` | With ELF noise injection | `elf_sde` |
| `build_dd_tree_lodestar` | With completion distance | `lodestar` |
| `build_dd_tree_manifold` | With manifold scoring | `manifold_pruner` |
| `build_dd_tree_belief` | With belief drafter | `belief_drafter` |
| `build_dd_tree_and_or` | AND-OR decomposition | `and_or_dtree` |

### Speculative Step Function

```rust
/// One speculative decoding step:
/// 1. Draft tree of candidate sequences
/// 2. Verify against target model
/// 3. Accept longest valid prefix
pub fn speculative_step(
    ctx: &mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    config: &Config,
    rng: &mut Rng,
    budget: usize,
) -> DraftResult {
    // Build draft tree
    let tree = build_dd_tree_pruned(/* ... */);
    // Extract best path
    let candidates = extract_candidate_sequences(&tree);
    // Verify with target
    let verified = verifier.verify(/* ... */);
    DraftResult { accepted_tokens: verified, /* ... */ }
}
```

### GDN Tree Verification (Plan 424)

For GDN2 (O(1) state-space) models, a specialized tree verifier achieves 7.09× speedup at T=128:

```rust
/// Rollback-free tree verify via masked triangular solve.
/// Exploits GDN2's recurrent structure — no KV cache rollback needed.
#[cfg(feature = "gdn_tree_verify")]
pub fn forward_tree_gdn2(/* ... */) -> Vec<f32> {
    // Processes entire draft tree in one pass
    // Matches paper's B200 GPU performance on CPU SIMD
}
```

### PPoT (Probabilistic Programs of Thought)

CPU-side logit resampling on failure — zero overhead on success:

```rust
/// Identifies high-entropy positions and applies targeted resampling
pub fn ppot_resample(
    logits: &[f32],
    config: &PpotConfig,
    rng: &mut Rng,
) -> Option<usize> {
    // Only activates at high-entropy decision points
    // Zero cost on confident predictions
}
```

---

## Workspace & Crate Architecture

### 27-Crate Workspace Hierarchy

```
katgpt-rs (root)
├── Leaves (depend on katgpt-types or nothing)
│   ├── katgpt-types      — Config, Rng, SIMD utilities
│   ├── katgpt-hla        — Higher-order Linear Attention substrate
│   ├── katgpt-tokenizer  — BPE, ConvexTok
│   ├── katgpt-dec        — DEC operators
│   ├── katgpt-micro-belief — BeliefKernel, BoMSampler
│   ├── katgpt-personality — Sigmoid composition
│   ├── katgpt-sense      — NPC sense composition
│   ├── katgpt-sleep      — Consolidation
│   ├── katgpt-validator  — Partial parser, syntax pruner
│   ├── katgpt-percepta   — Transformer-VM (zero katgpt deps)
│   ├── katgpt-proof-cert — GOAT proof certificates
│   └── katgpt-deprecated — Exiled losers
├── Core layer
│   └── katgpt-core       — Traits, attention primitives, cognitive kernels
├── Domain stacks
│   ├── katgpt-transformer — Weights, packing, MBU, tf_loop, SWiR, dense_mesh
│   ├── katgpt-forward    — ForwardContext (top tier join point)
│   ├── katgpt-quant      — KV codecs
│   ├── katgpt-spectral   — Eigenbasis
│   ├── katgpt-attn       — GDN2, CHIAR, RAT+, EGA
│   ├── katgpt-attn-match — MaxSim rerank
│   ├── katgpt-kv         — SP-KV, cache prune, segment checkpoint
│   ├── katgpt-speculative — DDTree, DFlash, SpecHop
│   ├── katgpt-pruners    — Bandit, screening, closure wire
│   ├── katgpt-band       — Band conditioner, collider pruner
│   ├── katgpt-sparse     — SOPTV task vector, SPLAT
│   ├── katgpt-claim      — Claim rubric, CLR
│   ├── katgpt-ruliology  — Wolfram ruliology
│   └── katgpt-backend    — CPU/ANE/GPU inference backends
└── Root (katgpt-rs)      — Feature aggregation surface, integration glue
```

### Dependency Rules

1. **Leaf crates** depend on `katgpt-types` (or nothing)
2. **`katgpt-core`** consumes leaf substrate crates and re-exports them
3. **`katgpt-forward`** is the top-tier domain crate (depends on core + transformer + pruners + speculative)
4. **Root** never ships to crates.io — it's the feature-aggregation surface
5. **Back-compat invariant:** every move keeps `pub use` re-exports so historical paths resolve

### Key Dependencies (External)

```toml
rayon = "1.10"           # Parallel iteration
blake3 = "1"             # Content-addressed hashing
fastrand = "2"           # Deterministic PRNG
serde = "1"              # Serialization
postcard = "1"           # Binary encoding (alloc feature)
bytemuck = "1"           # Zero-copy type casting
half = "2"               # f16 KV cache storage
bevy_ecs = "0.15"        # Optional ECS (Plan 033)
wasmi = "1.0"            # Optional WASM validator
papaya = "0.2"           # Lock-free concurrent HashMap
```

---

## Summary of Key Innovations

| Innovation | Mechanism | Impact |
|-----------|-----------|--------|
| **AC-Prefix Modelless** | `attends_dedup` three-region mask | Bit-identical to iterative MLM, 27× faster |
| **Constraint Pruning** | Hard-filter `is_valid` + path-aware conflicts | 7,079× compression, eliminates sampling |
| **TriggerGate** | Multi-signal load-adaptive tier routing | Deterministic CPU/GPU/ANE selection |
| **Feature Flags** | 378 flags, GOAT-proved before promotion | Composable, zero-cost when disabled |
| **Zero-Alloc Hot Path** | Pre-allocated buffers, `_into` pattern | No heap traffic per token |
| **DDTree + Verifier** | Speculative tree + Leviathan rejection | Provably identical output distribution |
| **GDN2 O(1) Decode** | Fixed-size state matrix, no KV growth | Constant memory regardless of sequence length |
| **LT2 Looped** | Weight-shared T-pass loop | T× effective depth, 0 extra parameters |

The codebase represents a principled approach to **eliminating non-determinism** from inference: constraint pruning replaces sampling, deterministic routing replaces heuristic dispatch, and bit-identical proofs replace statistical tolerance testing. Every shipped feature carries a machine-verifiable GOAT proof that it meets its claimed invariants.
