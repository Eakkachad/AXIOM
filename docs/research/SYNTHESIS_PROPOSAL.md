# Synthesis Proposal: Model-Less Text Generation System

**Date:** 2026-08-07  
**Status:** Actionable Research Proposal  
**Input:** 3 research tracks (generation mechanisms, weight extraction, katgpt-rs)

---

## TOP 3 CANDIDATE APPROACHES

### Approach #1: "Engram-VSA" — Hash-Addressed Hyperdimensional Memory + Energy Scoring

**Simple explanation:**  
Build a giant lookup table where every N-gram pattern (2-5 words) maps to a set of likely next tokens. The lookup uses hypervectors (very long random-ish vectors) so similar patterns can "blend" together. To pick which token comes next, we score candidates using simple energy functions (grammar correctness + topic relevance + n-gram probability) and pick the lowest-energy option.

Think of it like: a very smart autocomplete that combines dictionary lookup with grammar checking.

**How it works step by step:**
1. Input text → hash into hypervector query (multi-head N-gram hash, like katgpt-rs Engram)
2. Query hits VSA cleanup memory → retrieves bundle of candidate continuations
3. Each candidate scored by energy function (n-gram log-prob + coherence + constraint mask)
4. Lowest-energy candidate selected → becomes next token
5. Repeat

**What makes it special:**
- katgpt-rs PROVES Engram handles 80% of factual retrieval without neural forward pass
- Energy scoring is composable — add new constraints without retraining
- Hash lookup is O(1), scoring is O(vocabulary size) but prunable via SpecAsPruner bitmaps

| Metric | Rating |
|--------|--------|
| **Feasibility** | 8/10 — All components exist individually; integration is engineering |
| **Quality** | 4/10 — Fluent for short outputs, breaks down on long coherent text |

**What we need to build:**
- [ ] VSA cleanup memory populated from extracted LLM knowledge (SAE features → hypervectors)
- [ ] Multi-head N-gram hash encoder (port from katgpt-rs Engram design)
- [ ] Sigmoid-gated fusion layer (direct port from katgpt-rs)
- [ ] Energy function composer: n-gram scorer + grammar DFA + topic coherence via cosine similarity
- [ ] SpecAsPruner-style bitmap constraint system for format control
- [ ] Token selection via greedy energy minimization or light MCMC (3-5 steps)

---

### Approach #2: "Hopfield-SDM Hybrid" — Associative Memory with Galois-Field Addressing

**Simple explanation:**  
Store millions of text patterns (sentences, phrases, fact completions) as high-dimensional vectors in a Modern Hopfield Network. When given a prompt, the system "relaxes" to the nearest stored pattern — like how your brain completes "The capital of France is ___" automatically. For generation, chain multiple pattern completions together, using SDM's Galois-field math to navigate between related patterns deterministically.

Think of it like: a massive pattern-matching library that "auto-completes" by finding the best-matching stored pattern.

**How it works step by step:**
1. Input text → encode as hypervector using Clifford algebra rotors (position-aware)
2. Query Modern Hopfield Network → converge to nearest stored pattern (energy minimization)
3. Pattern contains "continuation" component → unbind to get next-segment candidates
4. Use SDM/VaCoAl path-dependent selection to choose which continuation branch to follow
5. Latent Field Steering (from katgpt-rs) adds style/topic bias to selection
6. Repeat with updated context

**What makes it special:**
- Modern Hopfield has EXPONENTIAL storage capacity (stores exp(d) patterns in d dimensions)
- VaCoAl (2026) proves deterministic semantic selection emerges without training
- NO hallucination on stored content — retrieval is exact
- Works natively in discrete space (avoids the geometric barrier for continuous→discrete)

| Metric | Rating |
|--------|--------|
| **Feasibility** | 6/10 — Hopfield + SDM integration is novel; corpus encoding is massive effort |
| **Quality** | 5/10 — Excellent recall, poor novelty; output feels "stitched together" |

**What we need to build:**
- [ ] Modern Hopfield Network implementation in Rust (exponential energy, softmax update rule)
- [ ] Corpus → pattern encoder (segment text into storable chunks + encode as hypervectors)
- [ ] SDM address space with Galois-field arithmetic (per VaCoAl paper)
- [ ] Continuation binding scheme: pattern ⊗ position_role → next_pattern_address
- [ ] Latent Field Steering integration (direct port from katgpt-rs)
- [ ] Clifford algebra rotor-based position encoding (from our existing infrastructure)
- [ ] Storage backend: Merkle-deduped chunks (adapt katgpt-rs Lore system)
- [ ] MASSIVE offline encoding pipeline (millions of patterns from corpus)

---

### Approach #3: "Extracted-Reservoir" — LLM Knowledge in Random Dynamics

**Simple explanation:**  
Take a trained LLM (like Llama-3.1-8B), extract its factual knowledge into a structured knowledge graph using SAE features + ROME analysis. Then feed queries through a random reservoir (untrained recurrent network) whose output is decoded by a simple linear layer trained on the extracted knowledge. The reservoir provides temporal dynamics (sentence flow), the knowledge graph provides facts.

Think of it like: Extract a textbook from a professor's brain, then use a simple machine to read from that textbook while maintaining conversation flow.

**How it works step by step:**
1. OFFLINE: Extract LLM knowledge via SAE → KG pipeline (Winnicki et al. 2026)
2. OFFLINE: Extract ROTATE vocabulary channels → token reachability maps
3. OFFLINE: Fit linear readout from reservoir states → token probabilities (closed-form, no backprop)
4. ONLINE: Input → reservoir dynamics → rich hidden state
5. Hidden state → linear readout → base token probabilities
6. Modulate by knowledge graph retrieval (VSA query → relevant facts → boost related tokens)
7. Constrain by vocabulary channel reachability → final token selection

**What makes it special:**
- Reservoir has NO trained weights (random, fixed) — "training" is only the linear readout
- The readout can be computed via pseudo-inverse (1 matrix operation, not gradient descent)
- ESN matches Transformers on grammaticality (2025 paper) — syntactic competence is "free"
- Knowledge graph provides factual grounding that raw reservoir lacks
- ROTATE extraction is data-free (pure weight analysis)

| Metric | Rating |
|--------|--------|
| **Feasibility** | 5/10 — Knowledge extraction pipeline is complex; reservoir scaling unclear |
| **Quality** | 6/10 — Best potential quality due to LLM knowledge transfer, but lossy |

**What we need to build:**
- [ ] SAE feature extraction pipeline (use existing Gemma Scope / Llama Scope)
- [ ] Winnicki et al. domain-filtered KG construction
- [ ] ROTATE weight decomposition → vocabulary channel maps
- [ ] Large Echo State Network (>100M parameters in reservoir)
- [ ] Linear readout fitting (pseudo-inverse on corpus statistics)
- [ ] VSA-encoded knowledge graph for fact injection
- [ ] Fusion mechanism: reservoir output + KG retrieval → final logits
- [ ] J-lens concept readout for intermediate reasoning verification

---

## RECOMMENDED ARCHITECTURE

### "Deep Man" — Deterministic Engram-Addressed Memory with Algebraic Navigation

This combines the BEST parts of each approach:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEEP MAN ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  INPUT: Token sequence                                            │
│    │                                                              │
│    ▼                                                              │
│  ┌───────────────────────────────────────┐                        │
│  │  ENCODER: Clifford Algebra Embedding   │                       │
│  │  - Tokens → multivectors (grade-0,1,2) │                      │
│  │  - Position via rotor multiplication    │                      │
│  │  - Context window as geometric product  │                      │
│  └───────────────┬───────────────────────┘                        │
│                  │                                                 │
│                  ▼                                                 │
│  ┌───────────────────────────────────────┐                        │
│  │  MEMORY LAYER 1: Engram (Fast Facts)   │  ← O(1) hash lookup  │
│  │  - Multi-head N-gram hash              │                       │
│  │  - Frozen table from LLM extraction    │                       │
│  │  - Sigmoid-gated fusion                │                       │
│  │  - Covers: common phrases, facts,      │                       │
│  │    collocations, idioms                 │                       │
│  └───────────────┬───────────────────────┘                        │
│                  │                                                 │
│                  ▼                                                 │
│  ┌───────────────────────────────────────┐                        │
│  │  MEMORY LAYER 2: VSA-Hopfield (Deep)   │  ← O(d) similarity   │
│  │  - Modern Hopfield with VSA patterns   │                       │
│  │  - Stores: complex associations,       │                       │
│  │    multi-hop relations, reasoning       │                       │
│  │    patterns extracted via SAE+J-lens    │                       │
│  │  - Convergence = "thinking"            │                       │
│  └───────────────┬───────────────────────┘                        │
│                  │                                                 │
│                  ▼                                                 │
│  ┌───────────────────────────────────────┐                        │
│  │  NAVIGATION: TDA-Guided Path Selection │                       │
│  │  - Persistent homology on concept      │                       │
│  │    neighborhoods                        │                       │
│  │  - Identifies "topological shortcuts"  │                       │
│  │    between related concept clusters     │                       │
│  │  - Prevents circular/dead-end paths    │                       │
│  └───────────────┬───────────────────────┘                        │
│                  │                                                 │
│                  ▼                                                 │
│  ┌───────────────────────────────────────┐                        │
│  │  STEERING: Latent Field Control        │                       │
│  │  - Style/tone via direction vectors    │                       │
│  │  - Topic focus via similarity gating   │                       │
│  │  - Format via SpecAsPruner DFA masks   │                       │
│  └───────────────┬───────────────────────┘                        │
│                  │                                                 │
│                  ▼                                                 │
│  ┌───────────────────────────────────────┐                        │
│  │  SCORING: Composite Energy Function    │                       │
│  │  E(token) = w1·E_ngram(token)          │                       │
│  │           + w2·E_coherence(token)       │                       │
│  │           + w3·E_grammar(token)         │                       │
│  │           + w4·E_topic(token)           │                       │
│  │           + w5·E_constraint(token)      │                       │
│  └───────────────┬───────────────────────┘                        │
│                  │                                                 │
│                  ▼                                                 │
│  OUTPUT: Selected token (lowest energy)                           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Combination Works:

| Component | Source | Role | Why It's Best |
|-----------|--------|------|---------------|
| Clifford Embedding | Our infrastructure | Position-aware encoding | Geometric algebra handles composition natively; rotor = rotation = position shift |
| Engram (Layer 1) | katgpt-rs | Fast factual recall | PROVEN: 80% of facts via O(1) hash; no neural forward pass needed |
| VSA-Hopfield (Layer 2) | Research track 1+2 | Deep association | Exponential capacity + convergence guarantee + no training |
| TDA Navigation | Our infrastructure | Path planning | Persistent homology finds "concept bridges" that prevent generation dead-ends |
| Latent Steering | katgpt-rs | Style/topic control | PROVEN: axis-independent control with no leakage |
| Energy Scoring | Research track 1 | Token selection | Composable, interpretable, no training for new constraints |
| SpecAsPruner | katgpt-rs | Format enforcement | PROVEN: O(1) DFA-based constraint; handles JSON/CSV/structured output |

### Knowledge Source (How to Fill the Memories):

**Primary path (best quality):**
1. Take Llama-3.1-8B (open weights)
2. Apply ROTATE decomposition → vocabulary channels (data-free, pure weight math)
3. Use existing Llama Scope SAEs → extract 128K features per layer
4. Run Winnicki et al. pipeline → domain knowledge graphs
5. Extract Linear Relational Embeddings → typed relation operators
6. Encode all of above into VSA format:
   - Vocabulary channels → Engram table entries
   - SAE features → Hopfield stored patterns (as hypervectors)
   - KG triples → bound hypervector associations
   - Relation operators → VSA binding matrices

**Secondary path (faster, lower quality):**
1. Prompt Llama-3.1-8B to generate millions of (subject, relation, object) triples
2. Encode directly into VSA memory
3. Build N-gram tables from large text corpus (Wikipedia, books)
4. This is "symbolic knowledge distillation" — proven to work (GPT-3 → ATOMIC, 6.5M triples)

### Integration with Our Existing Infrastructure:

| Existing Component | Role in Deep Man |
|---|---|
| **VSA/Hyperdimensional Computing** | Core memory format; all knowledge stored as hypervectors |
| **Clifford Algebra** | Token/position embedding via multivectors; binding via geometric product |
| **TDA (Persistent Homology)** | Navigate concept space; detect topological features of knowledge landscape |
| **Galois Fields** (if available) | SDM addressing arithmetic; deterministic navigation per VaCoAl |

---

## HONEST ASSESSMENT

### Realistic Best-Case Output Quality

**Short outputs (1-3 sentences, factual):** 60-70% as good as a 7B LLM  
- Factual recall will be accurate (if the fact was extracted)
- Grammar will be correct (DFA constraints + n-gram statistics)
- Will "sound right" for simple completions

**Medium outputs (1 paragraph, explanatory):** 30-50% as good as a 7B LLM  
- Coherence starts breaking down
- Repetition likely (Hopfield retrieval is "sticky")
- Topic drift without sophisticated steering
- Will feel "robotic" or "stitched together"

**Long outputs (multiple paragraphs, creative):** 10-20% as good as a 7B LLM  
- No compositional generalization = no true novelty
- Cannot reason about new combinations it hasn't seen
- Will be obviously mechanical to human readers

### Comparison to a 7B LLM

| Capability | Deep Man | 7B LLM (e.g., Llama-3.1-8B) |
|---|---|---|
| Factual recall (stored facts) | ✅ 90% accurate | ✅ 85% accurate (hallucination risk) |
| Grammar | ✅ 95% correct | ✅ 99% correct |
| Fluency / naturalness | ❌ 40% (robotic) | ✅ 95% (human-like) |
| Reasoning / multi-step | ❌ 20% (pattern match only) | ✅ 70% (chain-of-thought) |
| Creative generation | ❌ 10% (recombination only) | ✅ 80% (novel composition) |
| Speed (tokens/sec) | ✅ 10,000+ (hash + vector ops) | ⚠️ 50-200 (GPU forward pass) |
| Memory efficiency | ⚠️ Large (stores patterns) | ⚠️ Large (stores weights) |
| Interpretability | ✅ 100% traceable | ❌ 10% (black box) |
| Editability | ✅ Add/remove facts directly | ❌ Requires fine-tuning |
| Determinism | ✅ Same input → same output | ❌ Sampling randomness |
| No GPU needed | ✅ CPU + SIMD sufficient | ❌ Requires GPU |

### What Are the Dealbreakers?

1. **The Fluency Gap is Real**  
   LLMs achieve fluency through billions of parameter interactions during forward pass. No lookup/retrieval system can replicate this without some form of iterative refinement. The output will always feel "less human" for open-ended generation.

2. **Compositional Generalization is Missing**  
   "The purple elephant danced on the moon" — an LLM can generate this because it COMPOSES concepts. A retrieval system can only find stored patterns NEAR this. True novelty requires the kind of interpolation that comes from learned continuous representations.

3. **Scale of Extraction is Daunting**  
   A 7B LLM encodes millions of facts + billions of association patterns. Extracting even 10% into VSA format is a massive engineering effort (weeks of compute for SAE training + knowledge graph construction).

4. **The Geometric Barrier (arXiv 2606.30705)**  
   Smooth deterministic maps CANNOT resolve discrete branching choices. Language is full of branch points ("I went to the [store/park/office]"). Any continuous system mapped to discrete tokens faces this fundamental limit. We must work natively in discrete space — which we do (Galois fields, hash tables, bitmaps) — but this limits the "smoothness" of generation.

5. **Context Window Without Attention**  
   Transformers use attention to dynamically route information across ANY distance. Without it, our system relies on fixed-radius hashing (Engram) and pattern-level storage (Hopfield). Very long-range dependencies (reference something from 500 tokens ago) will be missed unless explicitly stored as a pattern.

### Is This Publishable Research?

**YES — but position it correctly.**

**DO publish as:**
- "Non-Parametric Text Generation via Hyperdimensional Associative Memory" — novel architecture paper
- "From Weights to Symbols: Extracting LLM Knowledge into Vector Symbolic Architectures" — knowledge distillation paper
- "Deterministic Language Generation with Engram-Addressed Hopfield Networks" — systems paper
- Connection to VaCoAl (2026), Versor (2025), and "GPT-2 Through VSA" (2024) makes this timely

**Strong publication angles:**
1. The VSA interpretation of transformer internals (builds on 2024 paper, adds extraction pipeline)
2. The Engram → VSA equivalence (novel theoretical contribution)
3. The TDA-guided navigation in concept space (novel application of persistent homology)
4. Clifford algebra as position encoding for non-neural sequence processing (novel)
5. Empirical comparison: what CAN you achieve without training? (valuable negative result + positive surprises)

**Target venues:**
- NeurIPS 2027 (main or workshop)
- ICML 2027
- ACL 2027 (if focused on language quality evaluation)
- AAAI 2027 (if focused on knowledge representation)
- arXiv immediately (establish priority)

**DON'T claim:**
- "This replaces LLMs" — it doesn't
- "Same quality as GPT" — it won't be
- "Training-free AI" — it still needs LLM weights as knowledge source (for the best version)

**DO claim:**
- "Interpretable, editable, deterministic text generation"
- "Novel bridge between mechanistic interpretability and hyperdimensional computing"
- "First system to run extracted LLM knowledge through a non-neural generation pipeline"
- "100× faster inference than equivalent-knowledge neural models"
- "Zero-GPU text generation with factual grounding"

---

## IMPLEMENTATION ROADMAP

### Phase 1: Proof of Concept (2-4 weeks)

**Goal:** Generate coherent 1-sentence completions for factual queries

1. Build Engram-style hash table from Wikipedia N-grams (Python prototype)
2. Implement basic VSA memory with 10K hypervector codebook
3. Simple energy function: n-gram log-probability + topic cosine similarity
4. SpecAsPruner-style DFA for basic grammar constraints
5. **Success metric:** "The capital of France is ___" → "Paris" reliably

### Phase 2: Knowledge Extraction (4-8 weeks)

**Goal:** Extract structured knowledge from Llama-3.1-8B into VSA format

1. Use existing Llama Scope SAEs (download from HuggingFace)
2. Implement ROTATE decomposition for vocabulary channels
3. Run Winnicki et al. pipeline on 2-3 target domains
4. Encode extracted KG into VSA-Hopfield memory
5. **Success metric:** Answer 500+ factual questions correctly from extracted knowledge

### Phase 3: Full System (8-12 weeks)

**Goal:** End-to-end generation with all components integrated

1. Port Engram + SpecAsPruner + Latent Steering to Rust (from katgpt-rs patterns)
2. Integrate Clifford algebra embedding
3. Add TDA navigation for multi-sentence coherence
4. Tune energy function weights on held-out text
5. **Success metric:** Generate coherent paragraphs that pass basic quality evaluation

### Phase 4: Paper & Evaluation (4 weeks)

**Goal:** Rigorous comparison and publication

1. Benchmark against GPT-2 (124M), Llama-3.1-8B on standard tasks
2. Measure: perplexity proxy, factual accuracy, grammar, human preference
3. Ablation study: contribution of each component
4. Write paper emphasizing the NOVEL contributions (VSA-transformer bridge, deterministic generation, interpretability)

---

## FINAL VERDICT

**Should we build this?** YES — as a research project, not a product.

**The realistic outcome:** A system that can do impressive factual retrieval and constrained generation (JSON output, form filling, factual Q&A) at extreme speed with full interpretability, but cannot match an LLM for open-ended creative writing or complex reasoning.

**The research value:** HIGH. This sits at the intersection of 5 hot fields (mechanistic interpretability, VSA/HDC, energy-based models, knowledge graphs, and alternative architectures). The negative results ("what can't you do without training?") are as valuable as the positive ones.

**The practical value:** NICHE BUT REAL. For applications needing:
- Deterministic output (same input → same output, always)
- Full auditability (trace exactly WHY each token was generated)
- Instant editability (add/remove facts without retraining)
- No GPU / edge deployment
- Structured output generation (JSON, forms, templates)

This system could outperform an LLM despite lower "general intelligence."

**One-line summary:** We're building a "deterministic brain" that remembers instead of thinks — fast, honest, and auditable, but not creative.
