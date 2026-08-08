# AXIOM — Algebraic neXt-token Inference On Memory

> Solve for X. No training required.

**AXIOM** treats text generation as an algebraic equation — given context, solve for the next token `X` using pure mathematics over hyperdimensional memory. No neural networks. No gradient descent. No GPU.

## What Is This?

A research system that generates text **without any neural network training** — using only:
- **Hyperdimensional computing** (Vector Symbolic Architecture, D=10,240)
- **Hash-based N-gram memory** (Engram — O(1) lookup)
- **Algebraic energy scoring** (composite energy minimization)
- **Transition Binding Algebra** (novel: `T(A→X) = π(A)⊗X`)

**Key properties:**
- ✅ 100% deterministic (same input → same output, always)
- ✅ Zero training (single-pass corpus ingestion only)
- ✅ CPU-only, no GPU required
- ✅ Incremental learning (add data → immediately smarter)
- ✅ Fully interpretable (trace every token selection)

## Architecture

```
Context: "the president of the"
                │
    ┌───────────┴───────────┐
    │                       │
    ▼                       ▼
[LAYER 1: Engram]     [LAYER 2: TBA]
 O(1) hash lookup      VSA algebra
 99.5% hit rate        fallback path
    │                       │
    └───────────┬───────────┘
                │
                ▼
    [AFC Energy Scoring]
    E(X) = α·engram(X) + β·transition(X) - γ·repeat(X) - δ·freq(X)
                │
                ▼
         argmax → X = "republic"
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
├── tle-deepman/      Unified AXIOM orchestrator (Engram + TBA + AFC)
├── tle-transition/   Transition Binding Algebra (T(A→X) = π(A)⊗X)
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

# Run AXIOM unified engine
cargo run --release -p tle-deepman

# Run the Engram standalone demo
cargo run --release -p tle-engram

# Run Transition Binding Algebra experiments
cargo run --release -p tle-transition

# Run all tests
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

## The Idea

Traditional LLMs: train billions of parameters with gradient descent → sample from learned distribution.

**AXIOM**: Encode corpus into algebraic memory → solve for X at inference time.

```
LLM:    P(X | context) ≈ softmax(W · h)        ← learned W
AXIOM:  X = argmax E(x) = α·sim(π(c)⊗TM, x)   ← algebraic, no W
```

The next token X is not "predicted" — it is **solved** from the transition algebra.

## Research

See `docs/` for full documentation:
- `RESEARCH_PAPER_DRAFT.md` — Full paper: "Transition Binding Algebra"
- `SYNTHESIS_PROPOSAL.md` — Architecture design & roadmap
- `KATGPT_ANALYSIS.md` — Prior art analysis
- `attention_decomposition_report.md` — Attention mechanism study

## Novel Contributions

1. **Transition Binding Algebra** — `T(A→X) = π(A)⊗X`: non-commutative directional binding for generation
2. **Algebraic Flow Composition** — composable, type-safe generation pipelines via `FlowNode` trait
3. **Multi-head Engram** — confidence-gated N-gram hash fusion with sigmoid weighting
4. **Two-layer confidence fallback** — Engram (fast) → TBA (algebraic) with automatic routing
5. **Hierarchical Transition Memory** — O(1) storage scaling proven

## License

MIT

## Authors

Deep_Man Research — August 2026
