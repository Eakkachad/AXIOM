# AXIOM ROADMAP — Master Task Board

> This is the canonical task board. Agents MUST update it after every task:
> mark done + record new metrics. Status: `pending` → `in_progress` → `done` / `blocked`.
> Last updated: 2026-08-12 (v16)

## Current System State (baseline v16)

| Metric | Value | Target |
|--------|:---:|:---:|
| candidate_answer_accuracy | **24.84%** | 40% |
| answer_entity_recall | 76.10% | 80% |
| substring_accuracy | 23.27% | 50% |
| avg_latency | ~146ms (idle) | <200ms ✓ |
| gen speed | 12K tok/s | 50K tok/s |
| codebook memory | 62MB (32×) | <50MB |
| evidence_answer_recall | 99.69% | 99.7% |

## RCA deliverable

See `docs/ROOT_CAUSE_ANALYSIS.md` — cross-layer analysis of the answer-selection
gap (hub domination + non-normalized linear score aggregation). T1.6 experiments
below recorded but reverted; the tuned linear-sum baseline remains optimal.

## T1.6 retrieve-then-rank (RCA-driven) — EXPERIMENTS, reverted
- [x] Status: done (reverted) · Priority: P0 · Effort: 1 day
- Percentile-normalized + equal-weight signals: candidate **12.58%** (worse)
- +0.5 VSA weight: candidate **13.84%** (worse)
- Conclusion: tuned linear sum (19.81%) beats percentile-equal-weight on this
  dataset. RCA theory correct (hub-invariance, signal parity) but needs
  per-signal weight calibration, not blind equal weighting. VSA stays 2.0×
  (noise tiebreaker). Do NOT re-apply without weight tuning.

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
- [x] Status: done (infra) · Priority: P0 · Effort: 1 day · Depends: none
- **Goal:** candidate +2-3pt. Merge entities whose names are word-order
  permutations or comma-trimmed variants. No metric change on TriviaQA
  (few permutation cases), but correct infra for real-world names.
  **Result:** infra only, no metric change

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
- [x] Status: done · Priority: P0 · Effort: 2h · Depends: none
- **Result:** Root cause = OVERLAP DOMINANCE + entity boundary imprecision.
  Diagnostics added (`AXIOM_TRIVIA_DEBUG` prints top-5 scores). See
  PROGRESS_LOG 2026-08-10 #3.

### T1.7 Proper-noun entity boundary precision (decomposition quality)
- [x] Status: done · Priority: P0 · Effort: 1 day · Depends: none
- **Goal:** candidate +3-5pt via cleaner entity boundaries. RCA: gold answer
  often enters the graph ONLY as a polluted entity surface ("Chicago, Illinois,
  17 mi northwest" instead of "Chicago"), so connectivity never fires for it.
- **File:** `crates/tle-axiom-gen/src/decompose.rs` (`extract_proper_nouns`)
- **Verify:** full bench, candidate up, recall NOT down
- **Status:** KEPT — phrases now stop at commas/numbers/`by`/`alongside`/`or`/
  `as`/etc; trailing punctuation trimmed; single proper nouns admitted when
  comma-terminated, preceded by lowercase, or mid-sentence (NOT article-headed
  or sentence-initial). candidate 19.50→20.44-20.75% (+0.6-0.9pt), recall
  71.38→76.10% (+4.72pt — record single-jump), substring 23.90→22.64%
  (-1.26pt, sentence-linearization metric; keep-gate = candidate+recall both up).
  6 unit tests added. **Result:** recall +4.72pt, candidate +0.6-0.9pt

### T1.8 Rank calibration (RCA-driven, weight-search based)
- [x] Status: done · Priority: P0 · Effort: 1-2 days · Depends: none
- **Goal:** candidate 20.44-20.75% → 25-30%. RCA conclusion: with clean
  entities (T1.7), the remaining gap is extract_answer's non-normalized linear
  sum (hub domination + signal-scale mismatch). T1.6 equal-weight percentile
  failed (12.58%); blind re-weighting failed 5+ times — this task uses
  **coordinate-ascent weight search** with the full 318 bench as objective.
- **Sub-steps (in order, stop when plateau):**
  - T1.8a: coordinate-ascent sweep of the 6 extract_answer weights
    (conn_avg, role_avg, hop2_avg, overlap, vsa, heur) around current
    (1.0/0.8/0.5/0.15/2.0/0.2). ~30-60 full-bench runs.
  - T1.8b: percentile-normalize each signal within candidate set THEN
    calibrate weights (fixes hub-invariance per RCA) — NOT equal weight.
  - T1.8c: distinctness/IEF (-log(freq/graph_size)) replaces raw count in
    `heur` (RCA §4.2 step 2).
- **Guardrails:** VSA stays a weak tiebreaker; never DDTree; aggregate bench
  metrics only (per-record is HashMap-noise); run quick bench before full.
- **Verify:** full 318 bench, candidate up, recall NOT down
- **Status:** T1.8a DONE — overlap weight 0.15→0.05. Coordinate-ascent weight
  search (env-driven, no recompile): swept CONN/ROLE/HOP2/OV/VSA/HEUR
  individually around current defaults. Only OV moved the needle:
  ov=0.05 → candidate **21.07-21.38%** (baseline 20.44-20.75, +0.6-0.9pt),
  stable across 8+ runs; recall 76.10% unchanged; substring 22.64% unchanged.
  All other weights flat (no single-weight gain). Overlap dominance (question
  -named entities scoring high) was suppressing correct connected answers.
  T1.8c DONE (dead-end, REVERTED) — IEF/distinctness replaced raw count in
  heur (env-gated AXIOM_W_IEF): candidate collapsed to 5-10% at any scale
  (log-frequency bonus removes the discriminative raw-count signal, matches
  documented "freq-bonus log-scale regressed 17.30%"). T1.8b (percentile +
  calibrate) NOT tried — T1.6 already proved equal-weight percentile fails;
  single-weight sweeps around the tuned linear sum are flat, so calibrated
  aggregation offers no headroom. Weight search infra (env overrides) kept.
  **Result:** candidate +0.63pt (T1.8a only)

### T1.9 Rank redesign: hard filter + rank fusion (RRF) — research-backed
- [~] Status: in_progress · Priority: P0 · Effort: 2-3 days · Depends: none
- **Goal:** candidate 21.38% → 25-30%. Replaces linear-sum+argmax (proven flat
  local optimum) with the research synthesis in
  `docs/RANKING_RESEARCH_SYNTHESIS.md` (3 deep-research memos converged):
  hard structural veto BEFORE ranking + rank-position fusion instead of score
  fusion.
- **Empirical anchor (171 failures):** M1 overlap dominance (21+), M2 near-tie
  noise (18), M3 hub/degree (5), M4 structural conn=0 (6+), M5 junk surfaces.
  Math: linear sums of differently-scaled signals (overlap ~50 vs conn ~2)
  cannot express "name-match only counts when connectivity present" — no
  single weight fixes both regimes.
- **Sub-steps (each independent, bench after each):**
  - T1.9a: **RRF rank fusion** — `score(e) = Σᵢ wᵢ/(k+rankᵢ(e))`, k=60, over
    the 6 existing per-signal lists (conn/role/hop2/overlap/vsa/heur); wᵢ=AUCᵢ
    data-derived once on the bench. Env-gated A/B vs linear sum first.
  - T1.9b: **hard structural filter** before ranking — F1 answer-type
    (intent→τ: Who→PERSON, Where→LOCATION, When→TIME), F2 question-relation
    reachability, F3 distance≤3 (fall back to widening when empty). Veto
    cannot be outvoted by magnitude → kills M1/M2/M5.
  - T1.9c: **hub-corrected PPR** — `π_q=(1-c)v+cPᵀπ_q` (c≈0.85, 60 iters,
    v=teleport to query entities); `ppq(e)=log π_q(e)−log π(e)` (Milne-Witten
    hub debias) as 7th RRF list + candidate expansion → fixes M4 (the only
    thing that does) + M3.
- **Guardrails:** sigmoid-never-softmax (katgpt rule — no softmax competition);
  VSA demoted to verification gate only (it's N(0,1/√2048) noise with random
  codebook); never DDTree; aggregate bench only; quick bench before full.
- **File:** `crates/tle-axiom-gen/src/engine.rs` (`extract_answer`)
- **Verify:** full 318 bench, candidate up, recall NOT down
- **Status:** T1.9a FOUND THE FIRST REAL WIN (unexpected direction): RRF rank
  fusion ALONE regresses (11.95-15.41%, equal-weight and tuned-weight — matches
  documented "equal/rank-weight fusion fails", T1.6: 12.58%). BUT the debug
  analysis revealed a **query-entity matching bug**: "O'Hare"/"Jaws (film)"/
  "Milky Way" were NOT detected as query entities (punctuation split
  "O'Hare"→["o","hare"]), so the ×0.2 query penalty never fired and
  question-named entities won via overlap. Fix: punctuation-stripped whole-token
  matching in `extract_query_entities`. candidate 21.38→**23.58%** (+2.2pt,
  stable 3+ runs), recall 76.10% unchanged, substring 22.64→22.33%.
  RRF kept env-gated (AXIOM_RANK=rrf, off by default) for future filter tests.
  **Result:** candidate +2.2pt (query-penalty fix)
- **Status:** T1.9b exploration (neutral-or-correct, committed): intent-aware
  query penalty (What/Who milder — "What is X?" X IS often the answer;
  removing penalty hurts 22.01%), count-weight split (AXIOM_W_COUNT — 0.2 is
  plateau, count = evidence mass not pure hub), location relational phrases +
  tail-relation inheritance in decompose (restores "village in X, Scotland"→
  located_in). Candidate stays 23.58%. **Result:** infra, no metric change.
- **Status:** T1.9c DONE — hub-corrected personalized PageRank as 7th signal
  (AXIOM_W_PPR=0.3 default). `π_q=(1-c)v+cPᵀπ_q` (60 iters) with relative-PPR
  hub debias `log π_q(e) − log π(e)` (Milne-Witten). Weight search: 0.3-0.35
   optimum. candidate 23.58→**24.21%** (+0.63pt, stable 4+ runs), recall 76.10%
   unchanged, substring 22.33% unchanged. Higher weights regress (1.0→22.64,
   3.0→17.30). 1 new PPR unit test. **Result:** candidate +0.63pt

### T1.10 Great Fusion Framework (research-backed 4-layer architecture)
- [~] Status: in_progress · Priority: P0 · Effort: 3-5 days · Depends: none
- **Goal:** candidate 24.21% → 30%+. Replaces the linear weighted sum with the
  4-layer architecture from `docs/RANKING_RESEARCH_SYNTHESIS.md` + deep
  research (conformal prediction, Datalog/Ascent, resonator, POS/NP-chunking).
- **Layers (build in dependency order, bench after each):**
  - T1.10a: **L4 conformal + calibrated log-odds** (primary ranking fix):
    per-signal empirical p-value `p_i(e)=(#cand with s_i≥s_i(e))/|cand|`,
    fuse as log-odds PoE `Σ wᵢ·logit(ĉᵢ)` with per-bin calibration from the
    bench; sigmoid-never-softmax; temperature sharpening T≈0.3 for near-ties.
    Fixes M2 (25 near-ties) + scale mismatch (overlap ~50 vs conn ~2).
    **STATUS: NEGATIVE RESULT (reverted to env-gated).** Conformal p-value
    fusion regresses at every config: equal-weight log-odds 12.58% (exactly
    the documented T1.6 percentile failure), tuned-weight 19.18%, all below
    tuned linear 24.21%. Root cause: converting raw scores to within-candidate
    p-values flattens real magnitude gaps (conn 2.0 vs 0.5 → just "rank 1 vs 2").
    The linear sum's raw magnitudes carry information normalization destroys.
    Lesson: the ~52pt gap is NOT fixable by re-fusing the same 6 signals —
    it needs NEW signals (PPR worked +0.63) or HARD FILTERS that remove wrong
    candidates entirely (answer-type veto). Prioritize T1.10b (Datalog hard
    filter) over T1.10a fusion.
  - T1.10b: **L2 embedded Datalog** (Ascent/Datafrog): deductive inference
    rules over the KG — transitivity, inversion ("mother of"⟺"has mother"),
    class hierarchy, comparator semantics (largest/smallest → sort). Hard
    answer-type filter (F1) before ranking. Fixes M1/M3 + logic questions.
    **STATUS: INFRA BUILT, no metric change (env-gated).** New
    `inference.rs`: `derive_facts()` (inversion → *_inv relations, located_in/
    part_of transitivity closure), `relation_family()`/`entity_families()`,
    `passes_answer_type()`. Ascent added as dep. 3 unit tests. Bench hook
    AXIOM_INFER={INV,TRANS}. Results: inv-only 24.21%, trans-only 24.21%,
    both 24.21% (recall 76.10% — derived facts don't fire on this dataset:
    family relations + location chains are rare in TriviaQA evidence).
    Answer-type veto (AXIOM_TYPE_VETO) TESTED, REVERTED: relation-heuristic
    families misfire ("won"→Person for "which team", capital_of object/subject
    ambiguity) → 19.81%. Confirms research report caveat: answer-type needs
    POS/NER-lite (L1), NOT raw relation heuristics. Transitivity needs L1 to
    recover intermediate steps (decomposition truncates "located_in Dumfries
    and Galloway" at comma, breaking the →Scotland chain).
  - T1.10c: **L1 deterministic POS/NP-chunking** (DFA + small lexicon):
    clause typing, entity boundary precision → kills M5 junk surfaces,
    feeds answer-type into L2.
  - T1.10d: **L3 resonator networks** as VSA confidence tiebreak (NOT primary
    — VSA cosine is noise; only useful after graph is clean).
- **Guardrails:** keep all env-gated A/B; VSA never primary; no softmax
  (sigmoid-never-softmax, katgpt rule); no DDTree; deterministic.
- **Verify:** full 318 bench, candidate up, recall NOT down
- **Status:** T1.10a NEGATIVE (conformal fusion 12.58-19.18%, reverted).
  T1.10b INFRA (Datalog rules built, no metric change; type-veto reverted).
  T1.10c NEGATIVE (surface filter at graph regresses — see LESSONS_LEARNED §2.4).
  T1.10d resonator not attempted (VSA noise).
  **T1.10e DONE — subject resolution** (deep-rank fix): trailing-copula strip
  ("Zadok the Priest were" → "Zadok the Priest"), leading-copula inherit with
  proper-noun guard ("is a ballet composed by" → inherit "Swan Lake"), passive
  `*_by` relation patterns + strong weights. candidate 24.21→**24.53%**
  (+0.32, stable 3+ runs), recall 76.10% unchanged, substring 22.33→23.27
  (+0.94). Recovered odql_15009 (Steve Miller Band). 2 new unit tests.
   **Result:** candidate +0.32pt (subject resolution)

### T1.11+ Rank redesign v2 — NEW-SIGNAL/HARD-FILTER track (research-gated)

> Source of this track: sub-agent research session (2026-08-12) — codebase audit +
> katgpt-rs prior-art analysis + arXiv verification of the "7 orthogonal signals"
> proposal (Linear Codes for HDC 2403.03278, VSA category theory 2501.05368,
> sheaf laplacian 2309.03773, PathHD 2512.09369, VaCoAl 2607.16573, etc.).
>
> **Verdict on the 7-signal proposal:** S1 (F2 subspace) / S2 (e^A communicability)
> / S3 (sheaf Dirichlet) / S4 (quantum fidelity) / S5 (FFT phase) / S6 (commutator
> gate) are all **functional re-labelings of existing signals** (overlap, PPR,
> degree, cosine², cosine) — held to the project's own Spearman-orthogonality bar
> they fail, and are predicted (from the T1.6/T1.10a evidence base) to regress to
> the 12.58-19.18% fusion floor. S6 is actively dangerous (rank-1 commutator is
> zero for BOTH identical and orthogonal). **Do NOT implement S1-S6 as ranking.**
> Only **S7 (Allen's interval algebra)** is genuinely orthogonal, but it has no
> metadata precondition today and does not touch M1-M5.
> **Category-theory framing (B) and projective-measurement fusion (C) are framing
> only** — C is a re-labeled percentile-style fusion (renormalizes magnitude away).
>
> **What IS worth building (each env-gated, bench after each):**
>
> - **M1 conditional overlap-veto** (the real lever, 21/165 failures): linear sum
>   provably cannot express "overlap counts only when connectivity present".
> - **F2 linear-code deterministic unbinding** (math verified vs 2403.03278):
>   exact Gaussian elimination over F2 replaces iterative recovery → cleaner
>   candidate sets (M2/M5). ~300-500 LOC new bit-matrix module in `tle-vsa`.
> - **Compression/MDL differenced tiebreak** (from synthesis §L3 + katgpt
>   MatchLengthScorer): `[C(q⊕fact)−C(q)] − [C(q⊕name)−C(q)]` — only genuinely
>   orthogonal + magnitude-preserving + no-dep signal available. Breaks near-ties.
> - **PathHD-style calibrated blockwise cosine + Top-K prune** (2512.09369):
>   best-matching published architecture for the 52pt answer-selection gap.
> - **CLR `(mean)^M` sigmoid reliability gate** (katgpt): widen near-tie margins
>   deterministically instead of re-weighting.
> - **S7** as a when-question hard filter (optional; needs date-interval
>   extraction precondition first).

### T1.11 M1 conditional overlap-veto (hard filter, not fusion)
- [x] Status: done · Priority: P0 · Effort: 0.5 day · Depends: none
- **Goal:** candidate 24.53% → ~28-30%. Kill overlap-dominance (M1, 21/165
  failures): question-named / query-word entities that have ZERO structural
  connectivity (conn=0 AND hop2=0 AND PPR support below τ) must not win via
  overlap. The linear sum cannot express this conditional — a filter can.
  Magnitude-preserving (does NOT normalize signals), so immune to the
  percentile/fusion failure class.
- **File:** `crates/tle-axiom-gen/src/engine.rs` (`extract_answer`)
- **Verify:** full 318 bench, candidate up, recall NOT down
- **Status:** KEPT — env `AXIOM_V1_M1` (default 1, best-known), `AXIOM_V1_M1_TAU`
  (default 0.0). Overlap zeroed for ov>0 candidates with conn=0 AND hop2=0 AND
  ppr≤τ. Full 318 bench: candidate 24.53→**24.84%** (+0.31pt, stable 3+ runs),
  recall 76.10% unchanged, substring 23.27% unchanged, latency ~167→~146ms.
  Note: quick 50-record subset showed −2pt (38→36) — subset is non-representative;
  full bench is the only trusted gate. **Result:** candidate +0.31pt

### T1.12 F2 random-linear-code deterministic unbinding (decomposition cleanup)
- [x] Status: done (infra) · Priority: P1 · Effort: 2-3 days · Depends: T1.11
- **Goal:** replace iterative/approximate recovery in the decomposition/cleanup
  path with exact Gaussian elimination over GF(2) (verified against arXiv
  2403.03278: C = K×V direct-sum subcode, K∩V={0}, unique factorization, no
  iteration, deterministic by construction). Note: `bind_gf2`/`unbind_gf2`/
  `cr2_confidence` already exist in `tle-vsa/src/hypervector.rs:237-286` but
  there is NO rank/elimination over F2 anywhere — build a bit-matrix module
  (u64-word packed rows, ~300-500 LOC, no deps). Caveat from 2607.16573: do NOT
  repair every collision (perfect cleanup ⇒ candidates become indistinguishable).
- **File:** `crates/tle-vsa/src/` (new `gf2_linalg.rs` or similar) + wiring in
  `tle-axiom-gen/src/decompose.rs`
- **Verify:** full bench, candidate up / recall NOT down
- **Status:** INFRA BUILT — new `crates/tle-vsa/src/gf2.rs` (exported via lib.rs):
  `Gf2Mat` (packed bit-matrix, deterministic rref/rank/solve/mul_vec[ᵀ]),
  `LinearCode` (systematic [I_k|A] code, encode/decode/factorize/syndrome —
  unique c = key⊕value split, K∩V={0}), `factorize_bundle` (bundle→subset via
  Gaussian elimination, the exact counterpart to resonator iteration). 8 unit
  tests + 1 HyperVector-layer integration test (bundle recovery exact &
  deterministic). All tle-vsa dependents pass (`cargo test` on 15 crates,
  incl. slow tle-pipeline 100-run determinism, 0 failures). Full bench neutral:
  candidate 24.84%, recall 76.10% (additive module, default scoring untouched).
  **Honest finding:** direct scoring wiring on the RANDOM-bipolar d=2048
  codebook is degenerate — random vectors are full-rank (rank ≈ min(dim, n)),
  so Gaussian elimination offers no search-space reduction and the syndrome/
  subspace signals reduce to the already-rejected S1 overlap family. The F2
  benefit requires a **structured codebook** (codewords in a low-dim subspace
  C = K×V) which breaks the existing codebook contract → T1.12b.
  **Result:** infra + tests, bench-neutral, no metric change

### T1.12b Structured F2 codebook (unblocks F2 scoring) — DEFERRED
- [ ] Status: blocked · Priority: P1 · Effort: 3-5 days · Depends: decision on
  breaking the random-bipolar codebook contract
- **Goal:** make codewords live in a low-dim subspace C = K×V so the T1.12
  `factorize_bundle`/syndrome machinery becomes non-degenerate in scoring
  (query-entity verification, near-tie tiebreak). Requires re-encoding the
  codebook (deterministic, seeded) — must re-verify 100-run determinism +
  full bench. High risk of regression; only pursue after T1.13/T1.15.
- **Status:** — | **Result:** —

### T1.13 Compression/MDL differenced tiebreak (8th signal)
- [x] Status: done (negative → env-gated, off) · Priority: P1 · Effort: 1 day · Depends: T1.11
- **Goal:** break M2 near-ties (25/165) deterministically with a genuinely
  orthogonal, magnitude-preserving signal: `Δ = [C(q⊕fact(e)) − C(q)] −
  [C(q⊕name(e)) − C(q)]` (match-length proxy per katgpt MatchLengthScorer —
  inverted byte-index + suffix-match, NOT full LZ). Tiebreak-only by design
  (katgpt's own CompressionDrafter failed its GOAT gate — never primary).
- **File:** `crates/tle-axiom-gen/src/engine.rs`
- **Verify:** full bench, candidate up / recall NOT down
- **Status:** IMPLEMENTED, tested, REVERTED TO ENV-GATED OFF.
  `shingle_cover` (greedy length-l substring coverage = LZ proxy, no deps) +
  Δ = cover(q,name) − cover(q,facts) as a near-tie reorder within
  `AXIOM_MDL_BAND` (0.02) of the top. Full 318 bench (3+ runs):
  - naive (all band members): candidate 24.84→**24.53%** (−0.31, mirrors M1)
  - **query-named-excluded** (AXIOM_V1_MDL=1, current code): candidate
    **24.84%**, recall 76.10%, substring 23.27% — **exactly neutral**, +90ms
    latency (146→237ms).
  Root cause of both: a query-named entity's facts TRIVIALLY match the query
  (it IS the query), so any "facts explain the query" signal promotes the
  reference and undoes the query penalty. Excluding query-named removes the
  harm but leaves zero gain — M2 near-tie is not addressable by this signal
  class on this dataset (consistent with LESSONS_LEARNED §2.6). Env-gated
  (default off) for future experiments. 3 unit tests kept.
  **Result:** neutral, no metric change (kept as tested infra, off)

### T1.14 S7 Allen's interval algebra — when-question hard filter (optional)
- [ ] Status: pending · Priority: P2 · Effort: 1-2 days · Depends: date/interval
  extraction in `decompose.rs` (only `happened_in` year triples exist today)
- **Goal:** only genuinely orthogonal signal of the proposed 7. Veto candidates
  whose event ordering contradicts the query's temporal constraint (13 Allen
  relations + composition table + difference logic x−y≤k). Does NOT touch M1-M5.
- **File:** new `crates/tle-axiom-gen/src/interval.rs`
- **Status:** — | **Result:** —

### T1.15 PathHD-style calibrated blockwise cosine + Top-K prune
- [ ] Status: pending · Priority: P1 · Effort: 2 days · Depends: T1.11
- **Goal:** port the published answer-selection pipeline closest to this gap
  (arXiv 2512.09369): GHRR-style non-commutative path hypervectors per candidate
  (order-sensitive, aligns with existing `HDV(π)=Σρⁱ(HDV(τᵢ))`), calibrated
  blockwise cosine to comparable scale, hard Top-K prune before final argmax.
  Veto-first (katgpt screening-007 lesson), env-gated.
- **File:** `crates/tle-axiom-gen/src/engine.rs`
- **Status:** — | **Result:** —

### T1.16 CLR (mean)^M sigmoid reliability gate for near-ties (M2)
- [ ] Status: pending · Priority: P2 · Effort: 0.5 day · Depends: T1.11
- **Goal:** instead of re-weighting (all flat/regress), widen the near-tie margin
  deterministically: `score_k = (mean_m σ(s_k,m − b))^M` with M≈4 (katgpt CLR
  reliability gate). Only applies when top candidates are within a noise band;
  never primary scoring.
- **File:** `crates/tle-axiom-gen/src/engine.rs`
- **Status:** — | **Result:** —

### T1.17 katgpt engineering adoption (determinism/robustness, no metric)
- [ ] Status: pending · Priority: P2 · Effort: 1 day · Depends: none
- **Goal:** SplitMix64 seed-mixing for bipolar codebook (fixes small-seed
  degeneration — katgpt Issue 296 bug class); multi-head prime-modulus hashing
  for `tle-engram` (collision dilution); coll-count confidence stat. Determinism
  hardening, no accuracy expectation.
- **Status:** — | **Result:** —

---

## TRACK 2 — System (Wikipedia scale + generation quality)

Goal: usable conversational knowledge system

### T2.1 Answer-first generation (two-stage)
- [x] Status: done · Priority: P1 · Effort: 2 days · Depends: T1.x partial
- **Goal:** Fix noisy Wikipedia QA. AxiomGen extract_answer finds entity,
  VSA-LM verbalizes only that entity. vsalm-wiki wired; answers entities
  directly. "who is the president of france" → Emmanuel_Macron.
  **Result:** clean entity answers, no free-form noise

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
- [x] Status: done (infra, NOT scoring) · Priority: P2 · Effort: 2-3 days · Depends: T2.3 (large corpus)
- **Goal:** Secondary VSA layer from corpus co-occurrence so `C(France)`
  ≈ `C(Paris)`. Do NOT replace random codebook (breaks determinism) — add a
  distributional layer on top. Corpus must be large (100+ pages) to be useful.
- **Status:** SemanticLayer infra built + verified (cos capital-paris 0.56 on
  4-page corpus). Wired into extract_answer → HURT accuracy (vsa noise from
  stopword-heavy query vector). REVERTED from scoring. Keep infra for future;
  needs (a) bigger corpus, (b) content-word-only query vector, (c) vsa weight
  calibration before re-enabling.
  **Result:** infrastructure only — DO NOT re-enable without fixes

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
