# AXIOM — Algebraic neXt-token Inference On Memory

> Solve for X. No training required.

**AXIOM** is a novel AI system that generates text, answers questions, and reasons about knowledge — **without any neural network training**. It uses pure algebraic computation over hyperdimensional vectors.

## Demo

```
AXIOM> /teach elephants are large animals that live in Africa
AXIOM> /teach elephants can swim very well
AXIOM> /teach elephants have long trunks and big ears

AXIOM> tell me about elephants
  Elephants are large animals that live in africa.
  They have long trunks and big ears.
  They can swim very well. [155µs]

AXIOM> /teach cat is an animal
AXIOM> /teach animals have hearts

AXIOM> does cat have a heart?
  Yes! Because cat is an animal, and animals have hearts. [42µs]

AXIOM> /teach sky is blue
AXIOM> /teach blue has short wavelength

AXIOM> why is the sky blue?
  A sky is blue, because the blue has short wavelength. [633µs]
```

## What Makes AXIOM Different

| | AXIOM | ChatGPT/LLMs |
|---|:---:|:---:|
| Training required | **Zero** | Months + GPU cluster |
| Learn new facts | **Instantly** (µs) | Fine-tune (hours) |
| Speed | **22,000 tok/s** | ~50 tok/s |
| Hardware | **Any CPU** | GPU required |
| Deterministic | **100%** | No (sampling) |
| Interpretable | **Full reasoning trace** | Black box |
| Hallucination | **Never on taught facts** | Common |
| Memory | **18 MB** | Gigabytes |

## Capabilities

### 1. Teach → Remember → Answer (µs)
```
/teach Bangkok is the capital of Thailand
what is bangkok? → "Bangkok is the capital of thailand." [3µs]
```

### 2. Multi-Hop Reasoning (no training!)
```
/teach cat is an animal
/teach animals have hearts
does cat have a heart? → "Yes! Because cat is an animal, and animals have hearts." [42µs]
```

### 3. Compositional Generation (novel sentences never seen in corpus)
```
/teach sky is blue
/teach blue has short wavelength
why is the sky blue? → "A sky is blue, because the blue has short wavelength." [633µs]
```

### 4. Multi-Sentence Paragraphs
```
tell me about elephants →
  "Elephants are large animals. They have long trunks. They can swim." [155µs]
```

### 5. Analogical Reasoning
```
/teach cat is an animal
/teach cat has four legs
/teach bird is an animal
does bird have legs? → "Probably yes — by analogy with cat" [50µs]
```

### 6. Pronoun Resolution (conversation memory)
```
/teach elephants can swim
what are elephants? → "Elephants are large animals."
can they swim? → "Yes! Elephants can swim." (resolved "they" → "elephants")
```

### 7. Persistent Knowledge
```
/load knowledge.txt    → Learn 13,000 facts from file [68s]
/save state.json       → Save to disk [209µs]
/restore state.json    → Restore on restart
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AXIOM Engine                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input: "why is the sky blue?"                          │
│    │                                                    │
│    ├─→ [VSA Intent Detector] → "Why" (algebraic)       │
│    │                                                    │
│    ├─→ [AXIOM-Gen: Energy-Guided KG Beam Search]       │
│    │     path: sky→is→blue, blue→has→short_wavelength  │
│    │     E(path) = relevance + coherence + simplicity   │
│    │                                                    │
│    ├─→ [Linearizer + TemplateBank]                     │
│    │     "A sky is blue, because the blue has short     │
│    │      wavelength."                                  │
│    │                                                    │
│    └─→ Output [633µs, CPU, deterministic]              │
│                                                         │
│  Layers:                                                │
│    Layer 1: Engram (O(1) N-gram hash) — 99.5% hit      │
│    Layer 2: TBA (VSA transitions) — algebraic fallback  │
│    Layer 3: AXIOM-Gen (KG composition) — novel output   │
│    Layer 4: Reasoning (analogy + multi-hop + attractor) │
│                                                         │
│  Memory:                                                │
│    IncrementalStore — learn on every /teach             │
│    δ-Mem — conversation context + pronoun resolution    │
│    CKR — compressed VSA bundles for scale               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Performance

| Operation | Speed |
|-----------|:-----:|
| Fact recall (taught) | **2-11 µs** |
| Yes/No reasoning | **42-140 µs** |
| Paragraph (3 sentences) | **155-235 µs** |
| Compositional (multi-hop) | **350-633 µs** |
| N-gram generation | **22,000 tok/s** |
| /load 23K lines | 68 seconds |
| /teach 1 fact | **< 1 ms** |
| Memory usage | ~18 MB |

## Quick Start

```bash
# Build
cargo build --release

# Run interactive AXIOM
cargo run --release -p tle-deepman

# Commands:
#   /teach <fact>        Learn something
#   /load <file.txt>     Learn from file
#   /save <file.json>    Save knowledge
#   /restore <file.json> Restore knowledge
#   /stats               Show statistics
#   /quit                Exit
#   <anything>           Ask questions or chat
```

### Data (optional)

For N-gram generation from WikiText corpus, place `data/wiki_train.txt`.

## Crate Structure (17 crates, ~30,000 LOC)

```
crates/
├── tle-deepman/      AXIOM interactive REPL (the main binary)
├── tle-axiom-gen/    Compositional generation (KG beam search + templates)
├── tle-afc/          Core algorithms:
│   ├── FlowNode composition (7 nodes + 3 combinators)
│   ├── IncrementalStore (teach → learn instantly)
│   ├── DeltaMem (conversation context)
│   ├── MorphTokenizer (VSA subword — novel)
│   ├── VsaIntentDetector (algebraic intent — novel)
│   ├── AnalogicalEngine (structural inference)
│   ├── AttractorReasoner (iterative convergence)
│   └── ParagraphGenerator (multi-sentence)
├── tle-engram/       O(1) N-gram hash (sigmoid fusion, 5 heads)
├── tle-knowledge/    Compressed storage (Bloom + VSA bundles)
├── tle-vsa/          VSA math (bind, bundle, permute, cosine)
├── tle-transition/   Transition Binding Algebra
├── tle-resonator/    Resonator Networks
├── tle-clifford/     Clifford Algebra
├── tle-tda-router/   Topological routing
├── tle-memory/       Persistent memory bank
├── tle-decoder/      Token decoding
├── tle-pipeline/     Pipeline orchestration
├── tle-bench/        Benchmarks
├── tle-chat/         Original chatbot
├── tle-reservoir/    Echo State Network
└── tle-gen/          KN-5 model (ppl=67.4)
```

## Novel Contributions

1. **Transition Binding Algebra** — `T(A→X) = π(A)⊗X`: first non-commutative VSA operation for text generation
2. **AXIOM-Gen** — Energy-guided beam search over knowledge graphs with VSA scoring (generates novel sentences)
3. **VSA Morphological Tokenizer** — Subword composition via algebraic bundling (no BPE training)
4. **VSA Intent Detection** — Semantic intent matching without keywords or rules
5. **Attractor Reasoning** — Iterative convergence for concept refinement
6. **Multi-hop Composition** — `T(A→B→C) = T(A→B) ⊗ π(T(B→C))` for transitive inference
7. **Two-layer Confidence Fallback** — Engram (fast) → TBA (algebraic) automatic routing
8. **Compressed Knowledge Representation** — Bloom + exact + VSA bundles (O(√N) memory)

## Research Papers

- `docs/RESEARCH_PAPER_DRAFT.md` — Full paper on Transition Binding Algebra
- `docs/AXIOM_Gen_Algorithm.md` — AXIOM-Gen mathematical specification + proof
- `docs/SYNTHESIS_PROPOSAL.md` — Architecture design (3 approaches → unified)
- `docs/AXIOM_RESULTS.md` — Benchmarks and evaluation
- `docs/AGENT_HANDOFF.md` — Development plan and project status

## Honest Limitations

- **Quality gap vs LLMs** — Generated text is correct but formulaic. Not conversational-quality for open topics.
- **Knowledge must be taught** — Doesn't know anything unless you `/teach` it or `/load` a file.
- **English only** (Thai support planned)
- **No creative writing** — Can compose facts but can't write stories or poetry.
- **Single-sentence grammar still rough** — Articles and verb agreement need polish.

## License

MIT

## Author

Deep_Man Research — August 2026
