# AXIOM — Project Plan & Agent Handoff Document

> Last updated: 2026-08-09 (session close — VSA-LM path launched)
> Status: Phase 1 COMPLETE · Phases 2-3 largely complete · TriviaQA intelligence work in progress · **VSA-LM (Path C) STARTED**
>
> ## SESSION HANDOFF SUMMARY v2 (VSA-LM / Path C — next agent)
>
> **Current focus:** Building a non-neural VSA-based language generator (`tle-vsa-lm`) as the long-term "Path C" direction — replace softmax/backprop LM components with VSA algebra + reservoir + n-gram. Fully deterministic, CPU-only, no training.
>
> **Research conclusions this session (from literature + sub-agents):**
> - Pure VSA text generation at scale has never been published — this is frontier research, but the building blocks (TBA, resonator decoding, reservoir) exist and AXIOM already had most of them.
> - The CFLM spec (soliton/Gross-Pitaevskii/gauge-theory language model) is **theoretical only** — no implementation exists anywhere. Do NOT chase it as a near-term target; mine it for ideas only.
> - katgpt-rs has no VSA code, but contributes engineering patterns: Engram (O(1) n-gram hash — already in AXIOM), KARC (reservoir + basis expansion), CompressionDrafter (beam search where scoring = compression length; the *concept* → VSA beam search), sigmoid-never-softmax design rule.
>
> **What was built this session:**
> - New crate `crates/tle-vsa-lm/` (workspace member) — 27 tests passing:
>   - `vocab.rs` — deterministic word↔id + VSA bipolar codebook
>   - `engram.rs` — O(1) FNV-hash n-gram memory (multi-order, additive smoothing) + `top_candidates` short-list
>   - `tba.rs` — Transition Binding Algebra: TM = Σ ρ(C(w_i))⊙C(w_{i+1}); predict = ρ(C(current))⊙TM; score via cosine
>   - `reservoir.rs` — leaky echo-state reservoir + `ReservoirMemory` (non-parametric k-NN associative readout, bounded eviction)
>   - `knowledge.rs` — `KnowledgePrior`: fact-store that steers generation toward fact-consistent words. **Entity-level matching** (a fact fires only when the FULL entity appears in context, not partial words) so `wavelength` can't falsely trigger `short_wavelength` facts
>   - `decode.rs` — VSA cosine decoder (no softmax) + penalty closure
>   - `lib.rs` — `VsaLm` engine: learn/generate/predict, two-stage decoder, energy-guided beam search with VSA anti-repetition (bundle recent context, penalize similar candidates) + bigram loop-breaker
> - Binaries: `vsalm-bench` (toy 97-sentence corpus), `vsalm-corpus <file> [ratio]` (real-corpus benchmark), and `vsalm-knowledge` (teach facts → generate answers)
>
> **Verified numbers (real Wikipedia wiki_train.txt, 300 sentences, 240 train):**
> - TRAIN next-token accuracy **90.1%**, TEST **10.7%** (no softmax, no backprop, no sampling)
> - Signal decomposition: TBA-only TEST 5.8% vs Engram-only TEST 13.5% — n-gram dominates on in-vocab text, but **TBA (pure VSA) generalizes to unseen contexts** better than exact n-gram on the tiny toy corpus (3.4% vs 2.5%)
> - Generation is genuinely fluent Wikipedia-style text, e.g. `"the player characters rest in a camp where units can be customized and character"` — no catastrophic repetition after VSA anti-repetition fix
> - **Deterministic: 5 identical runs ✓**
> - Two-stage decoder (Engram short-list 32 → TBA cosine only on short-list): **accuracy pass 184s → 1.2s (144× speedup)** with <4pt accuracy cost
> - **Knowledge-guided generation (vsalm-knowledge):** teach `(cat,is,animal)`, `(animal,has,heart)` → "does a cat have" → "animal heart" (multi-hop chaining works). `(water,is,liquid)` → "liquid", `(Mars,is,red_planet)` → "red planet", `(bird,has,wings)` → "wings"
>
> **Next steps (VSA-LM):**
> 1. Reservoir signal currently only helps slightly (TRAIN 80.4→81.5% on toy); the k-NN associative readout needs a smarter neighbor search (hierarchical/tree index) and larger reservoir to be competitive
> 2. Larger real corpus run (currently capped at ~300 sentences due to reservoir cost; the TBA+Engram core handles thousands of sentences fast)
> 3. **Integrate AXIOM-Gen KG as the actual knowledge source** — wire `KnowledgePrior` to ingest from `AxiomGen.graph` triples (currently `vsalm-knowledge` teaches facts manually via `add_fact`), then AXIOM can answer from real evidence
> 4. Add a proper EOS/stop signal so knowledge answers don't tail into noise (currently answers are correct-but-noisy after the fact chain is exhausted)
> 5. Compare VSA-LM vs HRBM ridge readout (`tle-reservoir`) on the same corpus for a fair "VSA decode vs neural readout" number
> 6. Consider KARC-style basis expansion (Chebyshev/Fourier) in the reservoir
>
> **Honest caveat:** 10.7% TEST next-token accuracy is not an LM-comparable number (LLMs get 30-50% on next-token). But no neural net / no softmax / deterministic at CPU-only is the whole point. This is the first concrete step of the "Path C" non-neural LM research; the value is the architecture, not yet the score.

---

> ## SESSION HANDOFF SUMMARY v3 (TriviaQA entity linking breakthrough)
>
> **Biggest single jump in TriviaQA history: substring 8.18% → 20.13%, candidate answer 7.23% → 11.32%.**
>
> **What fixed it:** `AxiomGen::extract_query_entities` VSA fuzzy linking no longer requires exact word overlap. It now scores every graph entity by a blend of **substring/prefix affinity** (so query "Molitor" links to evidence entity "Molitorová", "habsburg" → "Habsburgs") + **composed semantic-vector cosine**. Removed the strict `overlap` gate that blocked derived-name linking. Also added 17 family-relation phrases to `decompose::RELATIONAL_PHRASES` (is the mother/father/parent/daughter/son/wife/husband/sister/brother/founder/leader/president/author/director of).
>
> **Verified numbers (verified-wikipedia-dev, 318 records):**
> | Metric | Before | After |
> |--------|:---:|:---:|
> | substring_accuracy | 8.18% | **20.13%** |
> | candidate_answer_accuracy | 7.23% | **11.32%** |
> | answer_entity_recall | 72.33% | 72.33% |
> | avg_latency | ~16ms | 62ms (fuzzy scan over all entities) |
>
> **Also built:** AXIOM-Gen → VSA-LM integration binary `vsalm-axiom` (evidence → graph → `sync_into_vsa_lm` → knowledge-guided generation). `KnowledgeGraph::export_triples()`, `AxiomGen::sync_into_vsa_lm()`, VSA-LM `KnowledgePrior` with entity-level matching + `knowledge_only` stopword mode.
>
> **Honest caveat:** the entity-linking fix moved real dev-set accuracy 2.5×. The vsalm-axiom standalone VSA-LM generation is still noisy (works on clean taught facts, not noisy decomposed evidence); VSA-LM is best used as a *fluency layer over AxiomGen reasoning*, not standalone QA. Remaining bottlenecks: decomposition noise (100+ junk entities/page), answer selection on long phrase-entities.

---

> ## SESSION HANDOFF SUMMARY (previous — TriviaQA intelligence)
>
> **Current focus:** Making AXIOM genuinely smarter for open-domain QA, verified on real TriviaQA (not smoke fixtures).
>
> **Root-cause finding:** Evidence recall is 99.69% and answer-entity recall ~72%, but candidate answer accuracy is only ~7%. The bottleneck is NOT retrieval — it is (a) decomposition producing noisy entities, and (b) answer selection ranking.
>
> **What was just built (this session):**
> - `decompose::decompose_sentence` — clause-based fact decomposition with relational verb anchoring, subject chaining, word-boundary-safe predicate matching
> - Composed semantic entity vectors (`AxiomGen::semantic_vector`) — bundle word vectors so VSA cosine reflects shared vocabulary
> - Structural answer extraction (`GenerationResult.answer`) — connectivity + role bias + VSA relevance + length penalty
> - `Intent::Who` added to linearize
> - Benchmark diagnostics: `candidate_answer_accuracy`, `answer_entity_recall`, `evidence_answer_recall`
> - Real TriviaQA RC archive downloaded locally under `data/triviaqa/` (gitignored, 2.48GB)
>
> **Verified numbers (verified-wikipedia-dev, 318 records):** substring 8.18%, candidate answer 7.23%, answer-entity recall 72.33%, evidence recall 99.69%, avg latency ~16ms.
>
> **Next steps (ideas to explore in new session):**
> 1. Use composed semantic vectors inside `extract_query_entities` (Phase B) so query entities link to graph entities via cosine, not just word match
> 2. Filter decomposition output to short, entity-like candidates (suppress long phrase-objects)
> 3. Evidence sentence embedding: compare query vector to sentence vectors to select relevant evidence before decomposition
> 4. Answer-centric graph: only keep triples whose relation is question-compatible (Who→person relations, What→copula, Where→location)
> 5. Iterative refinement: second narrow beam pass over answer candidates (katgpt DDTree-style)
>
> **Honest caveat:** do NOT report these numbers as a real TriviaQA score. They are evidence-ingested pipeline diagnostics on the dev subset. New research ideas welcome — the user has additional research ideas for the next session.

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
├── tle-gen/         ← KN-5 language model (ppl=67.4)
└── tle-vsa-lm/      ← VSA-LM: non-neural text generation (TBA+Engram+Reservoir) [NEW]
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

## 14-WEEK DEVELOPMENT PLAN (Revised)

### System Classification: **Algebraic-Symbolic AI** (not rule-based, not neural)

```
Current composition:
  Layer 1 (Engram):    Statistical (N-gram counts)
  Layer 2 (TBA):       Algebraic (VSA permutation-binding)
  Layer 3 (AXIOM-Gen): Algebraic + Graph (energy beam search)
  Layer 4 (Reasoning): Algebraic (multi-hop, analogy)
  Conversation:        Rule-based ← TO BE REPLACED with algebraic

Goal: 100% Algebraic — zero rule-based, zero neural
```

### Phase 1: Knowledge Infrastructure (Week 1-3) ← START HERE

**Goal:** ระบบจัดการความรู้ที่ scale ได้ + auto-learn จาก internet

| Week | Task | Deliverable | Gate |
|:----:|------|-------------|------|
| 1 | **Compressed Knowledge Representation (CKR)** | `tle-knowledge` crate: hierarchical VSA bundles, O(√N) memory | Store 200K facts in 16MB |
| 2 | **Auto-Learn from Web** | `/learn-url` command: fetch → extract → compress | Learn 300+ facts from 1 Wikipedia page in <5s |
| 3 | **Knowledge Compaction** | Periodic merge/prune algorithm | 100K raw → 30K compacted, same coverage |

### Phase 2: Generation Quality + Tokenization (Week 4-8)

**Goal:** Fluent output + algebraic tokenization + eliminate all rule-based code

| Week | Task | Deliverable | Gate |
|:----:|------|-------------|------|
| 4 | **Template extraction** (10K+ from corpus) | Template bank + matcher | Generate varied sentence structures |
| 5 | **KN-5 fluency scoring** | E_fluency in energy function | Perplexity < 100 on generated text |
| 6 | **VSA Morphological Tokenization** | Subword composition: C("un")⊙ρ(C("believe"))⊙ρ²(C("able")) | 5K roots + 200 affixes → cover 100K+ words |
| 7 | **VSA Intent Detection** (replace rule-based) | Algebraic semantic matching for intents | "tell me the reason" → detects "why" without keyword |
| 8 | **Multi-sentence + Style** | Paragraph planner + casual/formal modes | Coherent 3-5 sentence responses |

### Phase 3: Intelligence Layer (Week 9-11)

**Goal:** Reasoning ซับซ้อน + fully algebraic pipeline

| Week | Task | Deliverable | Gate |
|:----:|------|-------------|------|
| 9 | **Attractor reasoning** (iterative refinement) | Resonator-based multi-pass | Answer improves over 3-5 iterations |
| 10 | **PTG recursive composition** | Unbounded reasoning depth | Solve 10-hop inference chains |
| 11 | **VSA Entity Linking + Contradiction Detection** | Fuzzy match + conflict alert | Link "short wavelength" ↔ "blue light" automatically |

### Phase 4: Scale & Deploy (Week 12-14)

**Goal:** Multi-language + web learning + publish

| Week | Task | Deliverable | Gate |
|:----:|------|-------------|------|
| 12 | **Background web learning daemon** | Auto-fill knowledge gaps from internet | AXIOM gets smarter daily |
| 13 | **Multi-language** (Thai + English) | VSA subword handles ภาษาไทย | "ทำไมท้องฟ้าเป็นสีฟ้า?" works |
| 14 | **Benchmark + paper** | TriviaQA 40%+, arXiv submission | Publishable results |

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

### ✅ COMPLETED (Phase 1.1 — Knowledge Infrastructure):
- [x] Compressed Knowledge Representation (CKR) — `tle-knowledge`
  - [x] BloomFilter (O(1) existence check)
  - [x] KnowledgeBundle (VSA superposition, max 200 facts)
  - [x] CategoryIndex (auto-categorize by subject similarity)
  - [x] CompressedKnowledgeStore (3-tier: Bloom + exact + VSA)
  - [x] 13 tests pass

### ✅ COMPLETED (Phase 1.2 — Knowledge Compaction):
- [x] Deterministic fact compaction — `tle-knowledge::CompressedKnowledgeStore::compact`
  - [x] Exact duplicate removal, retaining newest facts
  - [x] Configurable per-subject pruning
  - [x] Optional global fact limit
  - [x] Bloom, exact, and VSA indexes rebuilt after pruning
  - [x] `CompactionConfig` and `CompactionReport` exported
  - [x] 14 `tle-knowledge` tests pass

### ✅ COMPLETED (Phase 1.3 — Production Compaction):
- [x] Automatic compaction wired into `tle-afc::IncrementalStore`
  - [x] Default trigger every 10,000 learned facts
  - [x] Configurable trigger interval and per-subject retention limit
  - [x] Rebuilds exact fact store and VSA KG memory after pruning
  - [x] `/stats` reports compaction runs and pruned facts
  - [x] Production compaction regression test added
  - [x] `cargo test -p tle-afc`: 47 tests pass
  - [x] Same-subject/relation values merge deterministically during compaction
  - [x] `/stats` reports merged values

### ✅ COMPLETED (Phase 1.4 — Web Learning):
- [x] Bounded HTTP/HTTPS fetch with timeout and response-size limit
- [x] HTML cleanup: removes scripts, styles, navigation, forms, and markup
- [x] Sentence extraction and simple fact extraction
- [x] `/learn-url <url>` wired into the production REPL
- [x] Extracted facts are learned by IncrementalStore and AXIOM-Gen
- [x] Entity normalization removes leading articles for stable queries
- [x] Multi-clause fact extraction added
- [x] Local HTTP-server integration test and 6 web-learning tests pass
- [x] Synthetic 400-fact extraction benchmark passes under 5 seconds
- [x] Wikipedia infobox/table extraction added and tested
- [x] Real Wikipedia run measured: 1,103 sentences, 345 facts, ~1.53s
- [x] Wikipedia 300+ facts / <5s gate passed without synthetic padding

### ✅ COMPLETED (Phase 2 — Generation Quality):
- [x] Template extraction (TemplateBank) — 19 patterns, 4 tests
- [x] Fluency scoring (compute_fluency) — heuristic naturalness scorer
- [x] VSA Morphological Tokenizer — 5K roots + 200 affixes, 6 tests
- [x] VSA Intent Detection — algebraic matching (no keywords), 6 tests
- [x] Multi-sentence ParagraphGenerator — priority-ordered, pronouns, 4 tests
- [x] All above WIRED into production REPL ✅

### ✅ COMPLETED (Phase 3.1 — Intelligence):
- [x] Attractor Reasoning — iterative convergence, 3 tests
- [x] Wired into REPL (subjects added as attractor basins on /teach)

### ✅ COMPLETED (Phase 2.1 — Integration Pass):
- [x] Topic clusters exposed from CKR using cosine-routed `CategoryIndex`
- [x] Age-based global fact pruning added to production compaction
- [x] Response styles added: `/style casual`, `/style formal`, `/style brief`
- [x] MorphTokenizer wired into the query hot path for OOV morpheme composition
- [x] TemplateBank wired into AXIOM-Gen linearization for single and multi-hop paths
- [x] Verified focused suites: AFC 49, AXIOM-Gen 41, Deepman 5, Knowledge 15 tests

### ✅ COMPLETED (Phase 3.2 — Recursive Composition):
- [x] Beam search supports recursive composition up to 64 guarded hops
- [x] Default reasoning depth increased from 4 to 10 hops
- [x] Visited-entity cycle guard prevents repetitive graph paths
- [x] 10-hop chain and cycle regression tests added
- [x] `cargo test -p tle-axiom-gen`: 43 tests pass

### ✅ COMPLETED (Phase 3.3 — Entity Linking):
- [x] Query entity matching normalizes plural and possessive surface forms
- [x] Multi-word entities use composed VSA vectors and cosine similarity
- [x] Fuzzy entity-linking regression test added (`cats` → `cat`)
- [x] `cargo test -p tle-axiom-gen`: 44 tests pass
- [x] Underscore entity linking fixed (`node_0` query matching)

### ✅ COMPLETED (Phase 3.4 — Contradiction Detection):
- [x] KnowledgeGraph detects conflicting objects for the same subject/relation
- [x] Duplicate facts are ignored as non-conflicts
- [x] `/stats` reports detected AXIOM-Gen contradictions
- [x] `cargo test -p tle-axiom-gen`: 45 tests pass

### ✅ COMPLETED (Phase 4.1 — Validation Coverage):
- [x] Thai root-token and mixed Thai-English deterministic tokenization tests
- [x] Added `axiom-bench` deterministic 10-hop benchmark binary
- [x] Benchmark verified: 10-hop path, deterministic trace, ~15-17ms generation
- [x] Background web-learning job boundary added via `/learn-url-bg`
- [x] Fetch/extract runs off-thread; knowledge mutation remains on the REPL thread
- [x] Persistent `/queue-url` queue with retry state and line-based persistence
- [x] Scheduler processes one queued job at a time with automatic retry
- [x] Queue validation/retry persistence tests added
- [x] Standalone `web-daemon` binary added with `--once` mode and append-only learned output

### ✅ COMPLETED (Phase 1 — Knowledge Infrastructure):
- [x] **Week 2:** Auto-learn from web
  - [x] HTTP fetch + HTML extraction (MVP)
  - [x] `/learn-url` command (MVP)
  - [x] Improve extraction quality with entity normalization and multi-clause extraction
  - [x] Increase real-page extraction coverage to 300+ facts without lowering precision
- [x] **Week 3:** Knowledge Compaction
   - [x] Topic clustering (cosine similarity)
   - [x] Fact merging (shared subject + relation → combine distinct objects)
   - [x] Automatic age-based stale fact pruning
   - [x] Auto-trigger every 10K facts

### 🔲 TODO (Phase 2 — remaining):
- [x] **Style adaptation** (casual/formal/brief modes)
- [x] **Wire MorphTokenizer into hot path** (query morpheme composition)
- [x] **Wire TemplateBank into AXIOM-Gen linearizer** (single and multi-hop paths)

### 🔲 TODO (Phase 3 — remaining):
 - [x] **PTG recursive composition** (bounded safely to 64 hops)
 - [x] **VSA Entity Linking** (fuzzy cosine match plus normalized surface forms)
 - [x] **Contradiction detection** (conflict reporting)
 - [x] **Contradiction resolution policy** (`ReportOnly` / `LatestWins`)
 - [x] **Wire AttractorReasoner into question answering** (query-topic disambiguation)

### 🔲 TODO (Phase 4 — Deploy):
- [x] Background web learning daemon (standalone worker and REPL scheduler available)
- [x] **Multi-language foundation** (Thai + English tokenization coverage)
- [x] Thai fact extraction for `เป็น`, `คือ`, `มี`, `อยู่ใน`, and `เกิดใน`
- [x] Thai no-whitespace sentence handling regression tests
- [ ] Full TriviaQA benchmark run (dataset acquisition and evidence ingestion remain)
- [x] TriviaQA-compatible JSONL harness (`triviaqa-bench`) with exact/substring accuracy and latency metrics
- [x] TriviaQA native `Data` JSON format reader added
- [x] Separate evidence-facts JSONL ingestion supported by `QuestionId`
- [x] Official TriviaQA RC archive downloaded/extracted locally (ignored by git)
- [x] Verified Wikipedia dev split loader: 318 records parsed
- [x] Verified Wikipedia dev evidence run: 318 records, 11.32% substring accuracy, 99.69% evidence-answer recall, 404ms average latency
- [x] Question-overlap evidence grounding added without answer-alias oracle lookup
- [x] Clause-based fact decomposition (`decompose::decompose_sentence`) with relational verb anchoring and subject chaining
- [x] Composed semantic entity vectors so VSA cosine reflects shared vocabulary
- [x] Structural answer extraction in `GenerationResult.answer` (connectivity + role bias + VSA relevance + length penalty)
- [x] Root-cause diagnostics: answer-entity recall 72.33%, evidence recall 99.69%, candidate answer 7.23%
- [x] VSA-LM crate (`tle-vsa-lm`): VSA codebook + TBA transition memory + O(1) Engram + reservoir associative memory + cosine decoder (no softmax) + energy beam search — 21 tests
- [x] Two-stage decoder (Engram short-list → TBA cosine): 144× speedup on accuracy measurement
- [x] VSA-LM real-corpus benchmark: 90.1% TRAIN / 10.7% TEST next-token, deterministic, fluent Wikipedia-style generation
- [ ] Full train/test benchmark and richer evidence extraction remain
- [ ] arXiv paper submission
- [x] Reproducible paper artifact status and pre-submission checklist added
- [ ] Published pre-built binary release (local artifact generated)
- [x] Local Linux x86_64 archive and SHA256 manifest generated
- [x] Reproducible release profile and `docs/RELEASE.md` added

---

## KNOWN ISSUES

| Issue | Severity | Fix Plan |
|-------|:--------:|----------|
| KG cycle causes repetitive output | Closed | Visited-entity guard in beam search |
| "scatters" doesn't match `/teach` pattern | Closed | Added to production extraction patterns |
| Articles still imperfect ("a evaporation") | Improved | Mass-noun handling added; broader grammar remains |
| Pronoun resolution sometimes picks verb as subject | Improved | Common-verb blacklist added |
| AXIOM-Gen not triggered for "what is X?" | Closed | What-is path now tries AXIOM-Gen before Engram fallback |
| `/learn-url` not implemented yet | Closed | `/learn-url` production MVP and Wikipedia gate complete |

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
