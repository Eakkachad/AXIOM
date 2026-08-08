# AXIOM — Project Plan & Agent Handoff Document

> Last updated: 2026-08-08 23:35 ICT
> Status: Phase 1 PoC COMPLETE, entering Phase 2

---

## PROJECT IDENTITY

**Name:** AXIOM — Algebraic neXt-token Inference On Memory  
**Repo:** `/home/eggchad/eakject/research/Deep_Man/topological-latent-engine/`  
**GitHub:** `https://github.com/Eakkachad/AXIOM.git`  
**Language:** Rust (workspace, 16 crates)  
**Vision:** สร้าง AI แบบใหม่ที่เปลี่ยนโลก — ไม่ต้อง train, เร็วกว่า LLM 1000×, ทุกคนสร้างเองได้

---

## WHAT THIS IS

ระบบ AI ที่:
1. **ไม่ใช้ neural network** — ใช้พีชคณิต (VSA + Energy minimization + KG traversal)
2. **ไม่ train** — zero gradient, single-pass ingestion
3. **Compose ประโยคใหม่ได้** — ไม่ใช่แค่ recall
4. **เรียนรู้ทันที** — teach → know → answer (µs)
5. **Deterministic 100%** — same input → same output always
6. **CPU-only, <50MB** — ไม่ต้อง GPU

---

## CURRENT STATE (2026-08-08)

### What's Built:
- 16 Rust crates, ~27,000 LOC
- 143 tests passing
- 21 git commits

### What Works:
```
AXIOM> /teach sky is blue
AXIOM> /teach blue has short wavelength
AXIOM> why is the sky blue?
  A sky is blue, because the blue has short wavelength. [477µs]

AXIOM> /teach cat is an animal
AXIOM> /teach animals have hearts
AXIOM> does cat have a heart?
  Yes! Because cat is an animal, and animals have hearts. [42µs]
```

### Key Metrics:
| Metric | Value |
|--------|-------|
| Generation speed | 22,000 tok/s |
| Fact recall | 2-11 µs |
| Compositional generation | 350-500 µs |
| Accuracy (taught facts) | 91% (10/11) |
| Transitive reasoning | ✅ (multi-hop) |
| Analogical reasoning | ✅ (structural similarity) |
| Tests passing | 143 |
| Training required | ZERO |

### Crate Map:
```
crates/
├── tle-vsa/         ← Core math (bind, bundle, permute, cosine)
├── tle-afc/         ← Flow composition + IncrementalStore + DeltaMem + Analogy
├── tle-engram/      ← O(1) N-gram hash (Layer 1)
├── tle-axiom-gen/   ← Compositional generation engine (THE BREAKTHROUGH)
├── tle-deepman/     ← Unified REPL (interactive binary)
├── tle-transition/  ← Transition Binding Algebra (original research)
├── tle-resonator/   ← Iterative cleanup networks
├── tle-clifford/    ← Geometric algebra
├── tle-tda-router/  ← Topological routing
├── tle-memory/      ← Persistent memory bank
├── tle-decoder/     ← Token decoding
├── tle-pipeline/    ← Full pipeline orchestration
├── tle-bench/       ← Benchmarks
├── tle-chat/        ← Original chat interface (Day 1)
├── tle-reservoir/   ← Echo State Network experiments
└── tle-gen/         ← KN-5 language model (ppl=67.4)
```

---

## BUILD & RUN

```bash
cd /home/eggchad/eakject/research/Deep_Man/topological-latent-engine/

# Build everything
cargo build --release

# Run interactive AXIOM
cargo run --release -p tle-deepman

# Run tests
cargo test

# Run specific crate tests
cargo test -p tle-axiom-gen   # 37 tests - compositional generation
cargo test -p tle-afc         # 26 tests - AFC + incremental + analogy + delta_mem
cargo test -p tle-engram      # 19 tests - N-gram hash
```

---

## RESEARCH DOCUMENTS

| File | Content |
|------|---------|
| `docs/RESEARCH_PAPER_DRAFT.md` | Full paper: TBA + results + honest assessment |
| `docs/SYNTHESIS_PROPOSAL.md` | Architecture design (3 approaches → unified) |
| `docs/AXIOM_Gen_Algorithm.md` | AXIOM-Gen: full math spec + pseudocode + proof |
| `docs/AXIOM_RESULTS.md` | Benchmarks + demo transcript |
| `docs/KATGPT_ANALYSIS.md` | Prior art analysis from katgpt-rs |

---

## 12-WEEK DEVELOPMENT PLAN

### Phase 1: Knowledge Infrastructure (Week 1-3) ← START HERE

**Goal:** ระบบจัดการความรู้ที่ scale ได้ + auto-learn จาก internet

| Week | Task | Deliverable | Gate |
|:----:|------|-------------|------|
| 1 | **Compressed Knowledge Representation (CKR)** | `tle-knowledge` crate: hierarchical VSA bundles, O(√N) memory | Store 200K facts in 16MB |
| 2 | **Auto-Learn from Web** | `/learn-url` command: fetch → extract → compress | Learn 300+ facts from 1 Wikipedia page in <5s |
| 3 | **Knowledge Compaction** | Periodic merge/prune algorithm | 100K raw → 30K compacted, same coverage |

### Phase 2: Generation Quality (Week 4-7)

**Goal:** Fluent multi-sentence output ใกล้เคียง LLM

| Week | Task | Deliverable | Gate |
|:----:|------|-------------|------|
| 4 | **Template extraction** (10K+ from corpus) | Template bank + matcher | Generate varied sentence structures |
| 5 | **KN-5 fluency scoring** | E_fluency in energy function | Perplexity < 100 on generated text |
| 6 | **Multi-sentence generation** | Paragraph planner | Coherent 3-5 sentence responses |
| 7 | **Style adaptation** | Casual/formal/brief modes | User picks style |

### Phase 3: Intelligence Layer (Week 8-10)

**Goal:** Reasoning ที่ซับซ้อน + self-correction

| Week | Task | Deliverable | Gate |
|:----:|------|-------------|------|
| 8 | **Attractor reasoning** (iterative refinement) | Resonator-based multi-pass | Answer improves over 3-5 iterations |
| 9 | **PTG recursive composition** | Unbounded reasoning depth | Solve 10-hop inference chains |
| 10 | **Contradiction detection** | Conflict alert + resolution | Detect 90%+ contradictions |

### Phase 4: Scale & Deploy (Week 11-12)

**Goal:** พร้อมใช้จริง + publish

| Week | Task | Deliverable | Gate |
|:----:|------|-------------|------|
| 11 | **Background web learning** | Auto-fill knowledge gaps | AXIOM gets smarter daily |
| 12 | **Benchmark + paper** | TriviaQA 40%+, arXiv submission | Publishable results |

---

## TASK CHECKLIST (for current/next agent)

### ✅ COMPLETED (Phase 0 — PoC):
- [x] VSA core (bind, bundle, permute, cosine) — `tle-vsa`
- [x] Transition Binding Algebra — `tle-transition`
- [x] Algebraic Flow Composition (7 nodes + combinators) — `tle-afc`
- [x] Multi-head Engram (O(1) hash) — `tle-engram`
- [x] AXIOM unified engine (22K tok/s) — `tle-deepman`
- [x] Interactive REPL (/teach /ask /load /save)
- [x] Conversation mode (intent detection, Q&A)
- [x] Multi-hop transitive reasoning
- [x] Analogical reasoning (structural similarity)
- [x] δ-Mem (pronoun resolution, topic tracking)
- [x] AXIOM-Gen compositional generation — `tle-axiom-gen`
- [x] Wire AXIOM-Gen into REPL (end-to-end demo)
- [x] Grammar improvements
- [x] Sparse optimization (22K tok/s)
- [x] 91% accuracy evaluation

### 🔲 TODO (Phase 1 — Knowledge Infrastructure):
- [ ] **Week 1:** Design CKR (Compressed Knowledge Representation)
  - [ ] Hierarchical VSA bundle structure
  - [ ] Auto-split when bundle SNR drops
  - [ ] Bloom filter for fast "do I know this?" check
  - [ ] Create `tle-knowledge` crate
  - [ ] Test: 200K facts in <16MB
- [ ] **Week 2:** Auto-learn from web
  - [ ] HTTP fetch (reqwest or ureq)
  - [ ] HTML → text extraction
  - [ ] Sentence segmentation
  - [ ] Fact extraction (expanded patterns)
  - [ ] `/learn-url` command in REPL
  - [ ] Test: Wikipedia page → 300+ facts in <5s
- [ ] **Week 3:** Knowledge compaction
  - [ ] Topic clustering (cosine similarity)
  - [ ] Fact merging (shared subject → combine objects)
  - [ ] Stale fact pruning
  - [ ] Auto-trigger every 10K facts
  - [ ] Test: 100K → 30K, recall preserved

### 🔲 TODO (Phase 2 — Quality):
- [ ] Template bank extraction from WikiText
- [ ] KN-5 fluency as energy term
- [ ] Multi-sentence paragraph generation
- [ ] Style modes (casual/formal/brief)

### 🔲 TODO (Phase 3 — Intelligence):
- [ ] Attractor-based iterative reasoning
- [ ] PTG recursive composition
- [ ] Contradiction detection

### 🔲 TODO (Phase 4 — Deploy):
- [ ] Background web learning daemon
- [ ] TriviaQA benchmark
- [ ] arXiv paper submission

---

## KNOWN ISSUES

| Issue | Severity | Fix Plan |
|-------|:--------:|----------|
| KG cycle causes repetitive output | Medium | Add visited-set in beam search |
| "scatters" doesn't match `/teach` pattern | Low | Add to pattern list |
| Articles still imperfect ("a evaporation") | Low | Better plural/mass noun detection |
| Pronoun resolution sometimes picks verb as subject | Low | Add verb blacklist |
| AXIOM-Gen not triggered for "what is X?" (only why/how) | Medium | Expand intent → AXIOM-Gen routing |
| `/learn-url` not implemented yet | High | Phase 1 Week 2 |

---

## MATHEMATICAL FOUNDATIONS (for new agent context)

### Core Operations:
```
Bind:    A ⊙ B = element-wise multiply (Hadamard product)
Bundle:  A + B = element-wise addition
Permute: ρ(A) = circular shift by 1 position
Cosine:  cos(A, B) = A·B / (‖A‖·‖B‖)
```

### Key Equations:
```
Transition:     T(A→B) = ρ(A) ⊙ B           [non-commutative!]
Multi-hop:      T(A→B→C) = T(A→B) ⊙ ρ(T(B→C))
Triple encode:  HDV(s,r,o) = C(s) ⊙ C(r) ⊙ ρ(C(o))
Path encode:    HDV(π) = sgn(Σᵢ ρⁱ(HDV(τᵢ)))
Energy:         E(π,q) = λ_r·E_rel + λ_c·E_coh + λ_l·E_len + λ_s·E_simp
Selection:      best_path = argmin_π E(π, query)
```

### Why This Can't Be Done With LLMs:
- LLM: train billions of params → approximate P(next|context)
- AXIOM: encode facts algebraically → traverse KG → compose deterministically
- AXIOM is EXACT (no approximation), INSTANT (no forward pass), INTERPRETABLE (full trace)

---

## IMPACT ASSESSMENT — IF AXIOM SUCCEEDS

### What "Success" Means:
- **Minimum viable:** TriviaQA 40%+, fluent paragraphs, auto-learns from web
- **Game-changer:** TriviaQA 60%+, reasoning competitive with GPT-3.5, 50K tok/s
- **Nobel-level:** Proves algebraic composition = general intelligence without training

### Impact at Each Level:

**Level 1: Useful Tool (TriviaQA 40%+, Week 12)**
- Personal knowledge assistant that runs offline
- Study/education tool
- Domain-specific expert systems
- Privacy-first AI (data never leaves device)
- **Market:** Edge AI, education, enterprise knowledge management
- **Impact:** Millions of users who can't afford GPU/API

**Level 2: Paradigm Shift (TriviaQA 60%+, Month 6)**
- Proves training is NOT required for useful AI
- Every device becomes an AI device (phone, watch, IoT)
- Democratizes AI completely — no GPU monopoly
- Research community takes notice
- **Impact:** Billions of devices gain intelligence
- **Papers:** Multiple top-conference publications

**Level 3: Scientific Revolution (competitive with LLMs, Year 1-2)**
- New mathematical theory of intelligence
- Algebraic computation > statistical approximation (for structured knowledge)
- Spawns new field: "Algebraic AI"
- Every CS/AI curriculum adds VSA/TBA as foundational
- **Impact:** Reshapes the entire AI field
- **Awards:** Turing Award territory (not Nobel — AI isn't a Nobel category yet, but equivalent)

### Comparison: Why AXIOM Could Win

| Dimension | LLM (GPT-4) | AXIOM (Target) | Winner |
|-----------|:---:|:---:|:---:|
| Training cost | $100M+ | **$0** | AXIOM |
| Inference speed | 50 tok/s | **50,000 tok/s** | AXIOM 1000× |
| Hardware | 8× A100 GPU | **Any CPU** | AXIOM |
| Energy per query | ~0.01 kWh | **~0.000001 kWh** | AXIOM 10,000× |
| Deterministic | ❌ | **✅** | AXIOM |
| Interpretable | ❌ | **✅** | AXIOM |
| Instant learning | ❌ (fine-tune = hours) | **✅ (µs)** | AXIOM |
| Privacy | ❌ (cloud) | **✅ (local)** | AXIOM |
| Fluency | **95%** | 70-85% | LLM (for now) |
| General reasoning | **90%** | 40-60% | LLM (for now) |
| Creative writing | **90%** | 20-30% | LLM |

**The bet:** ถ้าปิด fluency gap ได้ (70→85%) — AXIOM ชนะ LLM ใน 8 ใน 10 dimensions

### Why It Could Actually Work:

1. **Moore's Law of Knowledge:** ยิ่ง auto-learn มาก → ยิ่งฉลาด (ไม่มี ceiling เหมือน training)
2. **Composition is powerful:** 100K facts × 4-hop paths = billions of answerable questions
3. **Speed advantage compounds:** 1000× faster = can do 1000× more reasoning per query
4. **Edge deployment:** 8 billion smartphones × AXIOM = 8 billion AI assistants

---

## AGENT HANDOFF PROTOCOL

When switching agents, provide this context:

```
"Continue developing AXIOM at /home/eggchad/eakject/research/Deep_Man/topological-latent-engine/

Read this file first: docs/AGENT_HANDOFF.md (this file)

Current task: [Phase X, Week Y, specific task]
Last commit: [hash + message]
Known blockers: [any]

Build: cargo build --release
Test: cargo test
Run: cargo run --release -p tle-deepman"
```

---

## FILE LOCATIONS

| What | Where |
|------|-------|
| This handoff doc | `docs/AGENT_HANDOFF.md` |
| Workspace root | `Cargo.toml` |
| Main binary | `crates/tle-deepman/src/main.rs` |
| AXIOM-Gen engine | `crates/tle-axiom-gen/src/engine.rs` |
| AFC + IncrementalStore | `crates/tle-afc/src/` |
| Knowledge graph | `crates/tle-axiom-gen/src/graph.rs` |
| Energy function | `crates/tle-axiom-gen/src/energy.rs` |
| Linearizer | `crates/tle-axiom-gen/src/linearize.rs` |
| Beam search | `crates/tle-axiom-gen/src/search.rs` |
| VSA core | `crates/tle-vsa/src/` |
| N-gram hash | `crates/tle-engram/src/` |
| Research docs | `docs/` |
| Python experiments | `experiments/` |
| Data files | `data/` (wiki_train.txt etc — gitignored if large) |
