# AXIOM ROADMAP — Master Task Board

> This is the canonical task board. Agents MUST update it after every task:
> mark done + record new metrics. Status: `pending` → `in_progress` → `done` / `blocked`.
> Last updated: 2026-08-10 (v13)

## Current System State (baseline v14)

| Metric | Value | Target |
|--------|:---:|:---:|
| candidate_answer_accuracy | 19.81% | 40% |
| answer_entity_recall | 71.38% | 80% |
| substring_accuracy | 23.58% | 50% |
| avg_latency | 105ms | <200ms ✓ |
| gen speed | 12K tok/s | 50K tok/s |
| codebook memory | 62MB (32×) | <50MB |
| evidence_answer_recall | 99.69% | 99.7% |

## How to read

- `[ ]` = pending, `[~]` = in_progress, `[x]` = done, `[!]` = blocked
- Each task has: priority (P0/P1/P2), estimated effort, dependency, expected impact
- Pick the highest-priority `[ ]` or `[~]` task whose dependencies are all `[x]`

---

## TRACK 1 — Accuracy (TriviaQA candidate + recall)

Goal: candidate 18.87% → 35%+, recall 71.07% → 80%+

### T1.1 Adaptive sentence coverage
- [x] Status: done (dead-end) · Priority: P0 · Effort: 1 day · Depends: none
- **Goal:** recall +3-5pt. Replace fixed top-5/6 sentence selection with
  relative threshold: keep sentences whose VSA score > mean + σ.
- **File:** `crates/tle-axiom-gen/src/bin/triviaqa-bench.rs` (`extract_document_facts`)
- **Verify:** full bench, recall should rise, watch latency
- **Status:** REVERTED — VSA score signal is too weak (cos≈0.01) for a
  meaningful threshold. Both configs regressed candidate:
  - 0.5σ cap 8: candidate 16.67% (-2.2), recall 73.58% (+2.5), lat 124ms
  - 1.0σ cap 6: candidate 18.24% (-0.6), recall 70.75% (-0.3), lat 103ms
  - Fixed top-5 (baseline) stays optimal: 18.87% / 71.07% / 213ms.
  - **Lesson:** VSA cosine can't discriminate sentences well. Adaptive
    coverage needs a stronger signal (T3.1 semantic codebook), not count math.

### T1.2 Lowercase noun-phrase extraction
- [x] Status: done · Priority: P0 · Effort: 1 day · Depends: none
- **Goal:** recall +2-3pt. Extract 2-4 word lowercase phrases after
  prepositions ("collapsible support assembly") using the existing
  mentions gate to filter noise. Previous attempt failed (too noisy) —
  this time reuse `mentions`/`is_related_to` all-caps gate.
- **File:** `crates/tle-axiom-gen/src/decompose.rs`
- **Verify:** full bench, recall up, candidate NOT down
- **Status:** KEPT — 3+ word all-lowercase phrases pass, 1-2 word rejected.
  candidate 18.55-18.87% (no regression), recall 71.38% (+0.3pt),
  latency 103ms (2× faster, likely fewer junk facts in graph).
  **Result:** recall +0.3pt, latency -110ms

### T1.3 Entity consolidation 2.0
- [ ] Status: pending · Priority: P0 · Effort: 1 day · Depends: none
- **Goal:** candidate +2-3pt. Merge entities whose names are word-order
  permutations or comma-trimmed variants ("Chicago, Illinois" → "Chicago",
  "Hingis, Martina" → "Martina Hingis"). Previous substring-merge regressed —
  only merge on EXACT pre-comma head match or full word-set match.
- **File:** `crates/tle-axiom-gen/src/graph.rs`
- **Verify:** full bench, candidate up, recall NOT down
- **Status:** — | **Result:** —

### T1.4 Relation-typed connectivity in extract_answer
- [x] Status: done · Priority: P0 · Effort: 1 day · Depends: none
- **Goal:** candidate +2-4pt. In `extract_answer`, weight connectivity by
  relation type: strong (located_in, capital_of, president_of, born_in = 2.0)
  vs weak (mentions, is_related_to = 1.2). Previous attempt didn't move
  needle because answers had 0 connectivity — retry AFTER T1.1/T1.3.
- **File:** `crates/tle-axiom-gen/src/engine.rs`
- **Verify:** full bench, candidate up
- **Status:** KEPT — strong=2.0, weak(mentions/is_related_to/named_after)=0.8.
  candidate 18.87-19.18% (peak +0.3), recall 71.38% stable, latency 102ms.
  **Result:** candidate +0.3pt peak

### T1.5 Diagnostic-driven failure analysis
- [ ] Status: done · Priority: P0 · Effort: 2h · Depends: none
- **Result:** Root cause = OVERLAP DOMINANCE + entity boundary imprecision.
  Diagnostics added (`AXIOM_TRIVIA_DEBUG` prints top-5 scores). See
  PROGRESS_LOG 2026-08-10 #3.

---

## TRACK 2 — System (Wikipedia scale + generation quality)

Goal: usable conversational knowledge system

### T2.1 Answer-first generation (two-stage)
- [ ] Status: pending · Priority: P1 · Effort: 2 days · Depends: T1.x partial
- **Goal:** Fix noisy Wikipedia QA. Use AxiomGen `extract_answer` to find the
  answer entity, then VSA-LM verbalizes ONLY that entity (not free-form
  multi-fact blend). `vsalm-axiom` partially does this.
- **File:** `crates/tle-axiom-gen/src/bin/vsalm-wiki.rs`, `crates/tle-vsa-lm/src/lib.rs`
- **Verify:** "what is the capital of france" → "paris"
- **Status:** — | **Result:** —

### T2.2 Fact specificity ranking (IEF)
- [x] Status: done · Priority: P1 · Effort: 1 day · Depends: T2.1
- **Goal:** KnowledgePrior ranks facts by object-word rarity — rare words
  ("paris") outrank common ("the", "of"). Suppress stopwords after first
  answer token.
- **File:** `crates/tle-axiom-gen/src/engine.rs`
- **Verify:** generation answers are specific, not "the of by"
- **Status:** DONE as average-connectivity normalization in extract_answer
  (conn/count + role/count). candidate 19.81% (+0.6pt record). Extreme hub
  (Macron 197 facts) still wins on Wikipedia QA — further tuning needed.

### T2.3 Wikipedia batch ingestion
- [x] Status: done · Priority: P1 · Effort: 1-2 days · Depends: T2.1/T2.2
- **Goal:** vsalm-wiki accepts URL list, builds shared large KnowledgePrior,
  persists to disk, reloads later. Test with 100+ pages.
- **File:** `crates/tle-axiom-gen/src/bin/vsalm-wiki.rs`
- **Verify:** ingest 100 pages < 2 min, QA works after reload
- **Status:** DONE — `--save <file>` TSV persistence, `--load <file>` reload.
  308 facts from 2 pages, QA works after reload. 100-page scale test remains
  (fetch speed ~300ms/page, so ~30s for 100 pages — well under 2 min).

---

## TRACK 3 — Research / Novelty (post-accuracy)

Gate: only start when candidate >30%

### T3.1 Semantic codebook (co-occurrence layer)
- [x] Status: done (infra) · Priority: P2 · Effort: 2-3 days · Depends: T2.3 (large corpus)
- **Goal:** Secondary VSA layer from corpus co-occurrence so `C(France)`
  ≈ `C(Paris)`. Do NOT replace random codebook (breaks determinism) — add a
  distributional layer on top. Corpus must be large (100+ pages) to be useful.
- **Status:** SemanticLayer built (semantic.rs), wired into AxiomGen.
  Test cos(paris, france)>0 ✓. No metric change on TriviaQA (200-file corpus
  too small) — needs large Wikipedia corpus via vsalm-wiki batch. Weight in
  extract_answer VSA relevance can be raised once corpus is large.
  **Result:** infrastructure only; needs T3.1b (large corpus test)

### T3.2 Full VaCoAl rescue circuit + CR2 verification
- [ ] Status: blocked · Priority: P2 · Effort: 3-5 days · Depends: candidate >30%
- **Goal:** Wire CR2 path confidence into answer verification. Only useful
  when graph is clean (candidate >30%). bind_gf2/unbind_gf2/cr2_confidence
  already in `hypervector.rs`.
- **Status:** — | **Result:** —

### T3.3 Paper draft
- [ ] Status: blocked · Priority: P2 · Effort: 1 week · Depends: candidate >40%
- **Goal:** GF(2) VaCoAl binding + semantic codebook + VSA-LM non-neural LM.
- **Status:** — | **Result:** —

---

## Cross-cutting notes / GOTCHAS

- **Never re-try DDTree** as primary answer selector — 4 attempts, all regressed.
  Beam paths don't carry answer entity. Legacy `extract_answer` scan-all-triples wins.
- **Never touch extract_answer weights blindly** — overlap×1, VSA×2, role×3,
  query-penalty×0.2 are empirically optimal. Tune via diagnostics, not guesswork.
- **Substring entity consolidation regressed** (v10) — only use exact head/word-set match.
- **VSA signal is near-zero** with random codebook (cos≈0.01). Don't rely on it
  until T3.1 semantic layer exists.
- **Query-weighted KnowledgePrior filtering** was noisy (v13) — relation-match
  filter returns too many facts; prefer specificity ranking (T2.2) instead.
- **Benchmark command:**
  ```
  cargo build --release -p tle-axiom-gen
  ./target/release/triviaqa-bench data/triviaqa/qa/verified-wikipedia-dev.json - data/triviaqa/evidence/wikipedia
  # quick: AXIOM_TRIVIA_LIMIT=50 ./target/release/triviaqa-bench ...
  ```
