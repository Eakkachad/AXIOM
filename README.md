# Deep Man — Deterministic Engram-Addressed Memory with Algebraic Navigation

> Model-less, training-free, deterministic text generation via Vector Symbolic Architectures.

## What Is This?

A research system that generates text **without any neural network training** — using only:
- **Hyperdimensional computing** (Vector Symbolic Architecture)
- **Hash-based N-gram memory** (Engram)
- **Algebraic energy scoring** (composite energy minimization)
- **Transition Binding Algebra** (novel non-commutative VSA operation)

**Key properties:**
- ✅ 100% deterministic (same input → same output, always)
- ✅ Zero training (single-pass corpus ingestion only)
- ✅ CPU-only, no GPU required
- ✅ Incremental learning (add data → immediately smarter)
- ✅ Fully interpretable (trace every token selection)

## Architecture

```
Input: "the president of the"
  │
  ├─→ [LAYER 1: Engram]   O(1) hash lookup → candidates + confidence
  │     hit rate: 99.5%     (~6ms for 20 tokens)
  │
  ├─→ [LAYER 2: TBA]      VSA transition memory → fallback candidates
  │     (only on Engram miss)
  │
  └─→ [AFC Energy Scoring]
        E(token) = α·engram + β·transition - γ·repetition - δ·diversity
        argmax → next token (deterministic)
```

## Results

| Metric | Value |
|--------|-------|
| Generation speed | **1,531 tokens/sec** |
| Engram hit rate | 99.5% |
| Determinism | 100% (verified 100 runs) |
| Training | Zero |
| Memory | ~18 MB (264K N-gram contexts) |
| Corpus | WikiText-2 (2M tokens) |

## Crate Structure

```
crates/
├── tle-vsa/          Core VSA operations (D=10,240 bipolar hypervectors)
├── tle-afc/          Algebraic Flow Composition (composable generation pipelines)
├── tle-engram/       Multi-head N-gram hash table (Layer 1: O(1) lookup)
├── tle-deepman/      Unified orchestrator (Engram + TBA + AFC)
├── tle-transition/   Transition Binding Algebra (T(A→B) = π(A)⊗B)
├── tle-resonator/    Resonator Networks (iterative cleanup)
├── tle-clifford/     Clifford Algebra (syntactic transformations)
├── tle-tda-router/   Topological Data Analysis routing
├── tle-memory/       Persistent memory bank (role-filler bindings)
├── tle-decoder/      Token decoding & vocabulary management
├── tle-pipeline/     Full pipeline orchestration
├── tle-bench/        Benchmark suite
├── tle-chat/         Interactive chat interface
├── tle-reservoir/    Echo State Network experiments
└── tle-gen/          KN-5 language model (ppl=67.4)
```

## Quick Start

```bash
# Build everything
cargo build --release

# Run the unified Deep Man engine
cargo run --release -p tle-deepman

# Run the Engram demo
cargo run --release -p tle-engram

# Run TBA experiments
cargo run --release -p tle-transition

# Run tests
cargo test
```

### Data Setup

Download required data files into `data/`:
```bash
# WikiText-2 (required for tle-deepman and tle-engram)
# Place as: data/wiki_train.txt

# GloVe embeddings (optional, for experiments)
# Place as: data/glove.6B.50d.txt
```

## Research Paper

See `../RESEARCH_PAPER_DRAFT.md` for the full paper:
**"Transition Binding Algebra: Deterministic Sequential Generation via Energy-Minimizing Traversal in Hyperdimensional Space"**

## Novel Contributions

1. **Transition Binding Algebra** — `T(A→B) = π(A)⊗B`: non-commutative directional binding enabling VSA-based generation
2. **Algebraic Flow Composition** — composable, type-safe generation pipelines via `FlowNode` trait
3. **Multi-head Engram** — confidence-gated N-gram hash fusion with sigmoid weighting
4. **Two-layer confidence fallback** — Engram (fast) → TBA (algebraic) with automatic routing
5. **Hierarchical Transition Memory** — O(1) storage scaling proven

## License

MIT

## Authors

Deep_Man Research — August 2026
