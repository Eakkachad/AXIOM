# AXIOM PROGRESS LOG — Chronological Development Journal

> Append-only journal. Newest entry at the TOP. Each entry: date · what changed ·
> measured results. Agents MUST append after every working session.

---

## 2026-08-12 — v18 (T1.18e PathHD relation-schema retrieval — FIRST NEW-SIGNAL WIN)

**Commits:** new crate `tle-ghrr` + engine integration + workspace + docs

### What was built
- **New crate `crates/tle-ghrr/`** (14 tests): GHRR block-unitary binding
  (real O(4) blocks, D=128, d=2048; deterministic Householder-product blocks;
  `bind_path` order-sensitive; blockwise cosine) + `RelationSchemaIndex`
  (training-free IDF) + `calibrated_score` (α=0.2/β=0.1/λ=0.8, Table-11).
- **Engine integration** (`ghrr_pathhd_signal`): question relation INTENT
  (content words → graph relation names, substring/prefix match; fallback =
  most frequent relation at query entities), per-candidate MAX calibrated
  blockwise cosine over its 1-hop/2-hop relation paths from query entities vs
  the intent. Added as signal `AXIOM_W_PATHHD` (default **2.0**).

### Bench (full 318, STRICT metrics, stable 3 runs)
| metric | before (v17) | after (PathHD on) | Δ |
|---|---|---|---|
| candidate_exact | 15.09% | **15.72%** | +0.63 |
| candidate_f1 | 16.98% | **17.61%** | +0.63 |
| strict_recall | 54.72% | 54.72% | 0 |
| substring candidate | 24.21% | **25.16%** (best ever) | +0.95 |
| avg latency | ~135ms | ~153ms | ok |

Weight sweep: peak at 2.0 (exact 15.72/f1 17.61); 2.5→15.41, 3.0→14.78,
5.0→13.21 (over-boost regresses). Keep-gate passed (strict candidate up,
strict recall NOT down).

### Why this one worked (banked)
PathHD is a **genuinely different signal**: it scores the RELATION SEQUENCE of
the candidate's path against the question's relation intent using
order-sensitive GHRR binding — NOT query-connectivity/count. The deep-rank
buried golds (which D1/D2/VSA-boost all failed on) are exactly the ones that
connect to the question via the RIGHT relation, and this signal rewards that.
First new-signal win since the deep review; confirms the research conclusion
that the gap needs NEW orthogonal signals, not re-combination.

---

## 2026-08-12 — v17 (strict metrics + QNP default ON; D1/D2/VSA-boost negative)

**Commits:** code (bench metrics, QNP, answer_type.rs, typed expansion, D2,
VSA-boost) + docs

### Kept (measured, stable 3 runs, STRICT metrics)
- **Strict metrics live** (`triviaqa-bench.rs`): `candidate_f1_accuracy`
  (EM-or-token-F1≥0.7 over aliases — SQuAD/TriviaQA protocol),
  `candidate_exact_accuracy`, `strict_recall` (F1≥0.7 vs graph nodes). Honest
  baselines: **f1 16.98% · exact 15.09% · strict_recall 54.72%** (substring
  recall 76.10% has ~21pt phantom).
- **QNP default ON** (`AXIOM_V1_QNP`): full penalty for query-named entities
  with conn=0 AND hop2=0 (the reference/anchor, never the answer). STRICT:
  exact 14.78→15.09 (+0.31), f1 16.67→16.98 (+0.31), strict_recall unchanged.
  Substring candidate 24.84→24.21 is the metric artifact (suppressing reference
  entities that substring rewarded).

### Negative (env-gated OFF, kept as tested infra)
- **D1 typed final-hop expansion** (answer_type.rs + AXIOM_W_TYPED): fires but
  typed_cands=0 for discriminative predictions — Who/Where/Number answers are
  1-2 hop (already candidates); no distance-3 type-compatible golds. Additive
  signal inert (w≤3) / regresses (w≥5). Mode-C golds are 2-hop ranked-low, not
  distance-3.
- **D2 conditional+saturated count** (AXIOM_W_RATIO): exact 15.09→7.55% at
  ratio 0.5. count_cond of buried golds is small BY DEFINITION (that's why
  they're buried) → destroys their signal.
- **Conditional VSA boost** (AXIOM_VSA_NOSTRUCT): inert at 2-6, regresses at 8.
  Buried golds are query-named (QNP penalty ×0.6) so even vsa=0.97/0.99 can't
  beat a connected winner; boost raises noise equally.

### Conclusion (banked)
The buried-gold class (Mode C, ~26%: gold conn=0) is NOT liftable by any
query-derived signal — golds are query-named+penalized+structurally weak, and
every fix that helps them helps the structurally-connected+count-rich winners
equally (net≈0). Remaining levers: genuinely different signal (PathHD
relation-schema, T1.18e) or decomposition so golds connect properly (T1.7
family).

---

## 2026-08-12 — v16c (T1.13 MDL tiebreak — negative → env-gated off, documented)

**Commits:** code (engine.rs shingle_cover + tiebreak) + docs

### What changed
- `engine.rs`: `shingle_cover(a, b, l)` — greedy length-l substring coverage
  (deterministic LZ77 proxy, no deps, katgpt MatchLengthScorer-style) +
  MDL differenced tiebreak Δ(e) = cover(q,name) − cover(q,facts), reordering
  only candidates within `AXIOM_MDL_BAND` (0.02) of the top. Env
  `AXIOM_V1_MDL` (default **off**). 3 unit tests (coverage, determinism,
  gold-vs-junk Δ direction).

### Bench (full 318, stable 3+ runs)
| Config | candidate | recall | substring | latency |
|--------|:---:|:---:|:---:|:---:|
| baseline (MDL off) | 24.84% | 76.10% | 23.27% | ~146ms |
| naive MDL | **24.53%** | 76.10% | 23.27% | ~214ms |
| MDL query-named-excluded | **24.84%** | 76.10% | 23.27% | ~237ms |

### Honest finding (banked in LESSONS_LEARNED §1.5)
Query-named entities' facts trivially match the query (they ARE the reference),
so any "facts explain the query" signal promotes them, undoing the query
penalty. Excluding query-named removes the harm but leaves zero gain → M2
near-tie stays irreducible on this dataset. Kept env-gated (off) as tested
infra. **Result: neutral, no metric change.**

---

## 2026-08-12 — v16b (T1.12 F2 random-linear-code infra + honest wiring finding)

**Commits:** code (tle-vsa/src/gf2.rs) + docs

### What changed
- **`crates/tle-vsa/src/gf2.rs`** (new, exported via lib.rs): deterministic F2
  linear algebra + random linear codes per arXiv 2403.03278:
  - `Gf2Mat` — packed u64 bit-matrix; `rref`/`rank`/`solve` (Gaussian
    elimination over F2), `mul_vec`/`mul_vec_transposed`, `from_columns`.
  - `LinearCode` — systematic [I_k | A] code; `encode`/`decode`/`factorize`
    (unique c = key⊕value, K∩V={0})/`syndrome`/`is_codeword`.
  - `factorize_bundle` — recover the exact subset summing to a bundle by
    solving the linear system (iteration-free counterpart to resonator).
  - 8 unit tests + 1 HyperVector-layer integration test (exact deterministic
    bundle recovery of {apple, cherry}).

### Verification
- `cargo test -p tle-vsa`: 30 passed (incl. 9 new).
- Full workspace compiles; all 15 tle-vsa-dependent crates' tests pass
  (incl. tle-pipeline slow 100-run determinism, 551.72s — pre-existing slow).
- Full 318 bench: **neutral** — candidate 24.84%, recall 76.10%, substring
  23.27% (additive module, default scoring untouched).

### Honest finding (banked — saves future hours)
Direct wiring of F2 codes into the CURRENT random-bipolar d=2048 scoring path
is degenerate: random vectors are full-rank (rank ≈ min(dim, n)), so Gaussian
elimination gives no search-space reduction and syndrome/subspace signals
reduce to the already-rejected S1 overlap family. The F2 machinery needs a
**structured codebook** (codewords in a low-dim subspace C = K×V) — tracked as
T1.12b (deferred, high regression risk). Prioritize T1.13 (MDL tiebreak, a real
new signal) and T1.15 (PathHD) over T1.12b.

---

## 2026-08-12 — v16 (T1.11 M1 conditional overlap-veto + T1.11+ research-gated roadmap)

**Commits:** code (engine.rs) + docs (ROADMAP/PROGRESS_LOG/AGENT_HANDOFF)

### Research session (sub-agents, 2026-08-12)
- **Codebase audit:** GF(2) primitives exist (`bind_gf2`/`unbind_gf2`/`cr2_confidence`,
  `hypervector.rs:237-286`) but NO Gaussian elimination/rank over F2 anywhere;
  no `faer`/`nalgebra`/`rustfft` deps; Ascent (Datalog) declared but never wired;
  commutator is ~5 lines from existing Cl(3,0) geometric product.
- **7-signal proposal validated:** S1-S6 are re-labelings of existing signals
  (overlap/PPR/degree/cosine²/cosine) — fail the Spearman-orthogonality bar,
  predicted to regress to the 12.58-19.18% fusion floor. S6 (commutator) is
  mathematically dangerous (zero for identical AND orthogonal). Only **S7
  (Allen interval)** is genuinely orthogonal but doesn't hit M1-M5. Category
  theory (B) and projective-measurement fusion (C) = framing only.
- **katgpt-rs (github.com/katopz/katgpt-rs):** confirmed no VSA code; actionable
  patterns = CLR `(mean)^M` reliability gate, `MatchLengthScorer` (suffix-match
  compression proxy), multi-head prime-modulus Engram, SplitMix64 seeds.
- **Prior art:** PathHD (arXiv:2512.09369) — calibrated blockwise cosine +
  Top-K prune, closest published architecture to the 52pt gap. VaCoAl
  (2607.16573) warning: perfect cleanup destroys discrimination signal.

### What changed (T1.11 M1)
- **`extract_answer` conditional overlap-veto** (`engine.rs`): overlap counts
  ONLY when candidate is structurally connected to query entities (conn>0 OR
  hop2>0 OR PPR support>τ). Env `AXIOM_V1_M1` (default **1** now best-known),
  `AXIOM_V1_M1_TAU` (default 0.0). Kills M1 overlap-dominance — the conditional
  the linear sum provably cannot express. Magnitude-preserving (immune to the
  fusion failure class).

### Bench results (full 318-record, verified-wikipedia-dev)
| Metric | v15 baseline | T1.11 M1 ON | Δ |
|--------|:---:|:---:|:---:|
| candidate_answer_accuracy | 24.53% | **24.84%** | +0.31 |
| answer_entity_recall | 76.10% | 76.10% | 0 |
| substring_accuracy | 23.27% | 23.27% | 0 |
| evidence_answer_recall | 99.69% | 99.69% | 0 |
| avg_latency | ~167ms | ~146ms | faster |

Stable 3+ runs (24.84% / 76.10% ×3). Keep-gate passed (candidate AND recall no
regress). Quick bench on the 50-record subset regressed (38→36%) — confirmed the
subset is non-representative; full bench is the only trusted gate.

### Lessons banked
- Quick-bench subsets can disagree with the full bench in **direction** (M1 was
  −2pt on n=50, +0.31pt on n=318) — never revert from a 50-record regression.
- The research session's core finding is now roadmapped: **the fix for the 52pt
  gap is new signals + hard filters (M1-veto, F2-codes, MDL tiebreak, PathHD),
  NOT re-fusing the existing 6.** S1-S6 are documented dead-ends to avoid.

---

## 2026-08-11 — v15 (LESSONS_LEARNED registry — anti-pattern consolidation)

**Commits:** docs only

### What changed
- **`docs/LESSONS_LEARNED.md`** created: permanent anti-pattern registry
  consolidating every failure from history (DDTree ×4, percentile, IEF,
  semantic-in-scoring, substring-consolidation, weight-tuning) + this session's
  T1.8-T1.10 results. 9 "ห้ามทำเด็ดขาด" entries with measured numbers +
  mechanism + the "way out" for each.
- **AGENT_WORKFLOW** now mandates reading it before every experiment.
- **AGENT_HANDOFF** lists it as file #5 in the source-of-truth system.

### Key content (all bench-verified)
- **Fusion redesigns all fail** (RRF 11.95-15.41%, percentile 12.58%, conformal
  log-odds 12.58-19.18%) — p-value/rank normalization destroys magnitude gaps.
  Linear sum with raw magnitudes keeps winning.
- **IEF/log-frequency kills evidence mass** (5-10%); count term IS evidence,
  not pure hub inflation (lowering it → 17.61%).
- **Answer-type needs POS/NER-lite**, not relation heuristics (19.81%).
- **Decomposition truncation breaks transitivity chains** — inference rules
  need intermediate nodes that only L1 (POS/clause) can provide.
- What WORKS: bug-fixing gates (query-penalty +2.2), new structural signals
  (PPR +0.63), decomposition quality (T1.7 recall +4.72).

### Measured results
No metric change (docs only). Current: candidate 24.21% · recall 76.10%.

---

## 2026-08-11 — v15 (T1.9b exploration + T1.9c hub-corrected PPR)

**Commits:** T1.9b, T1.9c

### T1.9b (exploration, neutral) what changed
- **Intent-aware query penalty** (`AXIOM_QP_WHAT`): What/Who/Why query-named
  entities get milder penalty (0.6) because "What is the Milky Way?" — X IS the
  answer. Removing penalty entirely → 22.01%; uniform 0.2 vs mild 0.6 is
  net-neutral but mild fixes What-is-X cases with no regression.
- **Count-weight split** (`AXIOM_W_COUNT`): 0.2 is the plateau (0.15→23.27,
  0.1→21.38, 0.05→17.61). The frequency term is EVIDENCE MASS, not pure hub
  inflation — RCA's "hub domination via count" is not the main driver.
- **decompose location relations**: "is a village/town/city in"→located_in,
  "is from"→from; tail proper nouns inherit the head fact's strong location
  relation ("village in X, Scotland" → located_in Scotland). Restores location
  hierarchy but no aggregate change on this dataset.

### T1.9c (KEPT) hub-corrected personalized PageRank
- `graph.personalized_pagerank(seeds, iters)`: π_q=(1-c)v+cPᵀπ_q, c=0.85, 60
  fixed iterations (deterministic), degree-normalized transition, relative-PPR
  hub debias score(e) = log π_q(e) − log π(e) (Milne-Witten). Wired as 7th
  signal in extract_answer, AXIOM_W_PPR=0.3 default.

### Measured results (full 318 bench)
| Metric | v15 T1.9a | T1.9b | T1.9c | Δ (9a→9c) |
|--------|:---:|:---:|:---:|:---:|
| candidate_answer_accuracy | 23.58% | 23.58% | **24.21%** | +0.63pt |
| answer_entity_recall | 76.10% | 76.10% | 76.10% | 0 |
| substring_accuracy | 22.33% | 22.33% | 22.33% | 0 |

### Key findings
- PPR weight search: 0.3-0.35 optimum (24.21%), 0.05-0.2 (23.90%), regresses
  above 0.5 (0.5→23.58, 1.0→22.64, 3.0→17.30). Small weights act as a
  structural tiebreak; large weights let graph topology dominate.
- Failure-mode analysis (165): 23/57 top-5 gold losses won primarily via heur
  (count+len+cap), 11 via conn, 9 via role; 8/25 near-ties decided purely by
  VSA cosine noise (identical conn/role/heur); deep-rank golds (rank 6-114) are
  mostly CLEAN entities connected but outranked.

---

## 2026-08-11 — v15 (T1.9a query-entity punctuation fix — candidate 21.38→23.58%)

**Commits:** T1.9a (this entry)

### What changed
- **T1.9a** `engine.rs`:
  1. RRF rank fusion added as env-gated experiment (`AXIOM_RANK=rrf`, k via
     `AXIOM_RRF_K`, per-signal weights via `AXIOM_RRF_W_*`, VSA default 0,
     query-penalty re-applied). **ALONE it regresses** (equal-weight 11.95%,
     tuned-weight 15.41%) — confirms the documented "rank/equal-weight fusion
     fails" (T1.6: 12.58%). Kept gated-off for future hard-filter testing.
  2. **Query-entity punctuation-stripped matching** (the real win): debug
     showed "O'Hare" winning with `fin=4.43` — if the ×0.2 query penalty had
     fired it would be 0.88. Root cause: `extract_query_entities` split the
     query on non-alphanumerics, so "O'Hare"→["o","hare"] never matched the
     entity "O'Hare"; question-named entities dodged the penalty and won via
     overlap (M1). Fix: strip non-alphanumerics WITHIN each raw whitespace
     token and compare to punctuation-stripped entity tokens ("ohare"≡"ohare").

### Measured results (full 318 bench, stable across 3 runs)
| Metric | v15 (T1.8a) | T1.9a | Δ |
|--------|:---:|:---:|:---:|
| candidate_answer_accuracy | 21.38% | **23.58%** | **+2.2pt** |
| answer_entity_recall | 76.10% | 76.10% | 0 |
| substring_accuracy | 22.64% | 22.33% | -0.31pt |
| evidence_answer_recall | 99.69% | 99.69% | 0 |

### Key findings
- The linear-sum scorer is NOT the only bottleneck — **query-entity detection
  precision gates the query penalty** (×0.2). Question-named entities that
  dodge detection (punctuation: O'Hare, Jaws (film), Milky Way) win via
  overlap even at weight 0.05. Fixing detection = +2.2pt candidate.
- RRF alone fails exactly as predicted by prior docs (equal-weight fusion is
  dead). RRF needs the hard-filter veto (T1.9b) to matter.
- 165 failures remain (was 171). O'Hare case verified fixed in debug.

---

## 2026-08-11 — v15 (T1.8c IEF/distinctness — dead-end, reverted)

**Commits:** (docs only — code reverted to T1.8a)

### What changed
- **T1.8c** `engine.rs`: env-gated IEF experiment (AXIOM_W_IEF) replaced the
  raw-count term in `heur` with distinctness = -log(count/graph_size) per RCA
  §4.2 step 2 (hub-invariance). Swept scales 0.1-2.5 on the full bench.
- **REVERTED** — no scale helped; the log-frequency bonus removes the
  discriminative raw-count signal and the score collapses.

### Measured results (full 318 bench)
| AXIOM_W_IEF | candidate |
|:---:|:---:|
| 0 (legacy) | **21.38%** |
| 0.1 | 10.38% |
| 0.3 | 8.81% |
| 0.6 | 7.23% |
| 1.0 | 6.29% |
| 1.5 | 5.35% |

### Key findings
- Matches the documented failure "freq-bonus log-scale regressed (17.30%)" —
  raw count (0.2×) is already a useful, weakly-preferential signal; replacing
  it with a hub-invariant log term destroys ranking.
- T1.8b (percentile + calibrate) NOT attempted: T1.6 equal-weight percentile
  already failed (12.58%), and T1.8a single-weight sweeps show the tuned
  linear sum is flat in every direction except overlap → no calibrated-
  aggregation headroom. T1.8 ranking work CLOSED with ov 0.15→0.05 as the
  only win.

---

## 2026-08-11 — v15 (T1.8a overlap weight calibration — candidate 20.75→21.38%)

**Commits:** T1.8a (this entry)

### What changed
- **T1.8a** `engine.rs` `extract_answer`: overlap weight 0.15→0.05 (default).
  Coordinate-ascent weight search infrastructure added: `weight_env()` reads
  AXIOM_W_CONN/ROLE/HOP2/OV/VSA/HEUR env overrides so the search sweeps the
  full 318 bench without recompiling. 6 new weights, defaults identical to the
  tuned linear sum.
- Search procedure (RCA §4.1, avoiding the T1.6 equal-weight trap): sweep each
  of the 6 signals individually, hold winner, re-sweep others.

### Measured results (full 318 bench)
| Metric | v15 baseline | ov=0.05 | Δ |
|--------|:---:|:---:|:---:|
| candidate_answer_accuracy | 20.44-20.75% | **21.07-21.38%** | +0.6-0.9pt |
| answer_entity_recall | 76.10% | 76.10% | 0 |
| substring_accuracy | 22.64% | 22.64% | 0 |
| avg_latency | ~100-235ms | ~100-235ms | 0 |

### Key findings
- **Only OVERLAP moved the needle.** CONN/ROLE/HOP2/VSA/HEUR single-weight
  changes are flat around the tuned baseline (as documented — strong local
  optimum). Overlap dominance was the actionable signal: question-named
  entities get many overlap points and were suppressing correct connected
  answers. Cutting 0.15→0.05 (still nonzero, keeps tie-breaking) is a clean win.
- Weight search infra (env overrides) is deterministic and cheap (~83s/bench,
  8-way parallel) — keep for T1.8b/c and future recalibration.
- Recall 76.10% / substring 22.64% untouched — pure ranking change.

---

## 2026-08-11 — v15 (T1.7 decomposition quality: proper-noun boundary precision)

**Commits:** T1.7 (this entry)

### What changed
- **T1.7** `decompose.rs` `extract_proper_nouns`: proper-noun phrases now stop
  at comma/semicolon and numeric tokens, and at connectors/prepositions that
  were previously swallowed (`by`, `or`, `as`, `alongside`, `named`, `such`,
  `like`, `including`, `between`, `after`, `before`, `through`, `under`,
  `over`, `during`, `within`, `about`, `around`). Trailing `, ; .` trimmed.
- Single capitalized proper nouns admitted when (a) comma/semicolon-terminated
  (apposition: "Chicago, Illinois"), (b) preceded by a lowercase token
  ("present-day Switzerland", "of Alaska"), and NOT article-headed ("the Loop"
  rejected) or sentence-initial common words ("Located" rejected) or
  discardable function words.
- **Root cause targeted (RCA row 5):** gold answers previously entered the
  graph only as polluted surfaces, e.g. `(O'Hare International Airport,
  mentions, Chicago, Illinois, 17 mi northwest)` → connectivity never fires.
  Now `Chicago` and `Illinois` are clean entities. O'Hare evidence verified in
  debug before benching.

### Measured results (full 318 bench, stable across 6 runs)
| Metric | v14 baseline | v15 T1.7 | Δ |
|--------|:---:|:---:|:---:|
| candidate_answer_accuracy | 19.50-19.81% | **20.44-20.75%** | +0.6-0.9pt |
| answer_entity_recall | 71.38% | **76.10%** | **+4.72pt** |
| substring_accuracy | 23.90% | 22.64% | -1.26pt |
| evidence_answer_recall | 99.69% | 99.69% | 0 |
| avg_latency | ~100-265ms | ~100-235ms | ≈ |

### Key findings
- Recall +4.72pt is the largest single-session jump in AXIOM history — the
  boundary fix surfaces answers (Chicago, Switzerland, LH, Baby Buggy) as
  clean graph nodes that connectivity can now reach.
- Candidate rose only +0.6-0.9pt despite the huge recall gain → confirms RCA:
  once entities are clean, the remaining bottleneck is the ranking
  aggregation (hub domination), not decomposition. Next step should be
  rank-normalized scoring WITH weight calibration (not blind equal-weight).
- substring -1.26pt is a sentence-linearization artifact (beam path → spoken
  sentence), NOT answer selection. Keep-gate (candidate primary + recall
  secondary) both improved → KEPT.
- Per-record diagnostics are NOT stable across identical runs (147/318 flip)
  due to HashMap iteration order — only aggregate metrics are trustworthy.
  Use aggregates, not per-record diffs, for A/B decisions.

---

## 2026-08-10 — v14 (RCA + T1.6 retrieve-then-rank experiments)

**Commits:** RCA doc (`docs/ROOT_CAUSE_ANALYSIS.md`), experiments reverted

### What changed
- **docs/ROOT_CAUSE_ANALYSIS.md**: cross-layer RCA — root cause = hub
  domination + non-normalized linear score aggregation (Macron 197 facts beats
  Paris 1 strong capital_of link). 4-layer analysis + redesign proposal.
- **T1.6 experiments** (reverted): retrieve-then-rank, percentile-normalized
  equal-weight signals → candidate 12.58% (worse). +0.5 VSA → 13.84%.
  Tuned linear-sum baseline (19.81%) still wins.

### Measured results
| Approach | candidate |
|---|---|
| tuned linear sum (baseline) | **19.81%** |
| percentile equal-weight | 12.58% |
| percentile +0.5 VSA | 13.84% |

### Key findings
- RCA theory correct (hub-invariance, signal parity needed) but percentile
  equal-weight is NOT the right implementation — needs per-signal weight
  calibration. VSA with random codebook is noise → must stay a weak tiebreaker.
- Tuned linear sum survived 5+ redesign attempts — strong local optimum.
  Structural gains require decomposition quality (cleaner entities → better
  conn signal), not re-weighting ranking.

---

## 2026-08-10 — v14 (T3.1b semantic scoring attempt — reverted)

**Commits:** `81271f8`

### What changed
- Semantic layer verified: cos(capital,paris)=0.56 vs cos(capital,macron)=0.27
  on 4-page Wikipedia corpus. Proof the co-occurrence concept works.
- Wired into extract_answer scoring → candidate HURT. Root cause: query
  vector bundles ALL words incl stopwords → semantic noise from co-occurring
  words inflated vsa for wrong entities.
- Freq-bonus log-scale experiment also regressed (17.30%).
- Reverted all scoring changes; baseline restored 19.81%.

### Key findings
- Co-occurrence semantic layer is real but needs 3 fixes before scoring:
  1. Larger corpus (100+ pages), 2. content-word-only query vector,
  3. calibrated vsa weight. Rushing it was premature.

---

## 2026-08-10 — v14 (T3.1 semantic layer)

**Commits:** `726be9b`

### What changed
- **SemanticLayer** (semantic.rs): window co-occurrence → distributional
  semantic vectors. cos(paris, france) > 0 verified (was ~0 with random
  codebook).
- Wired into AxiomGen semantic_vector() + bench shared corpus.

### Measured results
- No metric change on TriviaQA (candidate 19.50-19.81%, within noise).
- Root cause: 200-file TriviaQA corpus too small for cross-record
  co-occurrence. Needs large Wikipedia corpus (vsalm-wiki batch).

---

## 2026-08-10 — v14 (2-hop connectivity + workspace cleanup)

**Commits:** `e5e69e4`

### What changed
- **2-hop connectivity** in extract_answer: entities reachable from a query
  entity through one intermediate node get a 0.5× bonus (relation-typed).
  Captures answers like "LH" connecting via "Ovulation" when no direct link.
- **Workspace cleanup**: Phase 2 refactor made HyperVector.data private, but
  tle-resonator/tle-clifford/tle-tda-router/tle-deepman still accessed it.
  All switched to `as_slice()`. Full workspace builds + 135 tests pass.

### Measured results (full 318 bench)
| Metric | v14 Track2 | v14 2-hop |
|--------|:---:|:---:|
| candidate_answer_accuracy | 19.81% | 19.81% |
| answer_entity_recall | 71.38% | 71.38% |
| avg_latency | 100ms | 105ms |

### Key findings
- 2-hop connectivity is correct infrastructure but no metric change on
  TriviaQA — most answers are 1-hop from query entities already.
- Workspace was silently broken in non-tested crates (resonator, clifford,
  tda-router) since Phase 2 — now compiles + tests pass.

---

## 2026-08-10 — v14 Track 2 (T2.3)

**Commits:** `8b11ad4`

### What changed
- **T2.3 Wikipedia batch ingestion** — vsalm-wiki `--save <file>` (TSV
  persistence of facts), `--load <file>` (reload without re-fetching).
  Tested: 2 pages → 308 facts → save → load → QA "where is paris located" → France.

### Measured results
- Fetch speed ~300ms/page → 100 pages ≈ 30s (well under 2 min target).
- Persistence round-trip verified.

---

## 2026-08-10 — v14 Track 2 (T2.1-T2.2)

**Commits:** `e42442b`, `cf6b295`

### What changed
- **T2.1 Answer-first generation** — vsalm-wiki wired AxiomGen; answers
  entities directly via extract_answer, VSA-LM as fluency fallback.
  "who is the president of france" → Emmanuel_Macron ✓, "where is paris
  located" → France ✓.
- **T2.2 Average connectivity** — extract_answer normalizes conn/role by
  link count. Fixes hub problem (Macron 197 facts vs Paris 1 strong link).

### Measured results (full 318 bench)
| Metric | v14 Track1 | v14 Track2 |
|--------|:---:|:---:|
| candidate_answer_accuracy | 18.87-19.18% | **19.81%** (+0.6) |
| answer_entity_recall | 71.38% | 71.38% |
| avg_latency | 102ms | 100ms |

### Key findings
- Average-connectivity normalization is correct direction (candidate +0.6pt).
- Extreme hubs (Macron 197 facts) still dominate on Wikipedia QA — link-count
  normalization not enough; needs degree-relative or IEF weighting per fact.
- Answer-first generation produces clean entity answers (not noisy blends).

---

## 2026-08-10 — v14 Track 1 (T1.1-T1.4)

**Commits:** `a66c92f`, `22a800e`, `b489d17`, `155cb09`, `e8a870a`

### What changed
- **System**: continuous-dev system (ROADMAP/PROGRESS_LOG/WORKFLOW)
- **T1.1 Adaptive sentence coverage** — REVERTED (VSA signal too weak)
- **T1.2 Lowercase NP extraction** — KEPT (recall +0.3pt, latency 2×)
- **T1.3 Permutation entity consolidation** — KEPT (infra, no regression)
- **T1.4 Relation-typed connectivity** — KEPT (candidate peak 19.18%)

### Measured results (full 318 bench, v13 → v14)
| Metric | v13 | v14 |
|--------|:---:|:---:|
| candidate_answer_accuracy | 18.87% | **18.87-19.18%** |
| answer_entity_recall | 71.07% | **71.38%** (+0.3) |
| avg_latency | 213ms | **102ms** (2×) |

### Key findings
- T1.1 lesson: VSA cosine can't discriminate sentences — needs semantic
  codebook (T3.1), not threshold math.
- Relation typing on connectivity works (+0.3pt candidate) once graph is
  clean (T1.2 reduced junk).
- Latency halved from 213→102ms via cleaner graph (fewer junk facts).

---

## 2026-08-10 — v13 (continuous-dev system bootstrap)

**Commits:** `a66c92f`

### What changed
- **v11**: Diagnostic mode (`AXIOM_TRIVIA_DEBUG` prints top-5 entity score
  breakdown). Root cause found: OVERLAP DOMINANCE — entities named in question
  get 20-54 overlap points, drowning answer entities (0 overlap).
- **v11 fix**: connectivity-first scoring — overlap weight 1.0→0.15,
  query-named penalty ×0.2. candidate 16.35%→18.55% (+2.2pt).
- **v12**: subject preposition truncation, comma entity consolidation,
  MorphTokenizer wired into extract_query_entities, wider sentence coverage
  (6 overlap sentences). candidate→18.87%, recall→71.07%.
- **v13**: vsalm-wiki binary (Wikipedia fetch→clean→decompose→KnowledgePrior→QA).
  Query-weighted fact filtering (noisy, reverted).

### Measured results (full 318 bench)
| Metric | before v11 | v13 |
|--------|:---:|:---:|
| candidate_answer_accuracy | 16.35% | **18.87%** |
| answer_entity_recall | 70.44% | **71.07%** |
| substring_accuracy | 23.58% | 23.58% |
| avg_latency | 198ms | 213ms |

### Infrastructure completed
- 32× codebook compression (Phase 2), GF(2) encoding (Phase 3)
- TBA TopK cache 12K tok/s (Phase 4), KnowledgePrior O(1) hash index
- vsalm-wiki Wikipedia ingestion pipeline (300 facts per 2 pages)

### Key findings / gotchas
- **DDTree dead** — 4 attempts, all regressed. Beam paths don't carry answer.
- **extract_answer weights** (overlap×1, VSA×2, role×3, query×0.2) are optimal.
- **Substring entity consolidation regressed** — only exact-match merge works.
- **Query-weighted KnowledgePrior** noisy — prefer specificity ranking.
- **Semantic codebook** (co-occurrence) deferred — corpus too small (<200 words/page).

---

## 2026-08-10 — v10 (cleaner decomposition)

**Commits:** `6112bdc`, `2d4e44c`

### What changed
- mentions/is_related_to gate: require ≥2 words + ≥1 capital non-article word
- preposition truncation for long objects (≥5 words, excludes "of")
- tail entity extraction, bonus overlap 3→4

### Measured results
- candidate 17.30%, recall 69.18% (both records at the time)

---

## 2026-08-10 — v9 (infrastructure)

**Commits:** `d244b02`, `0d98714`, `6a439d9`, `5943739`, `505913b`

### What changed
- Phase 2: BipolarVector lazy decompression (32× memory)
- Phase 3: GF(2) triple encoding (XOR packed bits, 3× faster)
- Phase 4: TBA TopK cache (2-3× gen speedup)
- KnowledgePrior O(1) hash index
- CR2 path-confidence energy term

### Measured results
- codebook: 2GB → 62MB
- gen speed: 4.4K → 12K tok/s
- TriviaQA candidate 16.98%, recall 68.55%
