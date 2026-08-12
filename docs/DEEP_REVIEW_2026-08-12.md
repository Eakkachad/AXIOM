# AXIOM Deep Review & Solution Research — 2026-08-12

> **Purpose:** Records the deep-review findings (the real problem) and the
> solution research that followed (what the literature + original analysis says
> to do). Read this before the next optimization sprint.
> **Status:** v16c (candidate 24.84% reported / 13.84% exact · recall 76.10%).

---

## PART 1 — THE PROBLEM (deep review, measured on the live bench)

### 1.1 The reported metric is inflated ~2× (H5 CONFIRMED)

The `candidate_answer_accuracy` metric uses **bidirectional substring** matching.
Measured on the full 318-record bench (new stricter metrics added to the bench):

| metric | value | note |
|---|---|---|
| candidate_answer (bidirectional substring) | 24.84% | reported baseline |
| candidate_token (token containment, any alias) | 24.53% | still inflated (roosevelt ⊂ eleanor roosevelt passes) |
| **candidate_exact (token-set equality, any alias)** | **13.84%** | honest number |
| answer_entity_recall (bidirectional substring) | 76.10% | also lenient |

- **52% (41/79) of "correct" are substring-only artifacts**: gold `roosevelt`
  (maiden name) ↔ picked `eleanor roosevelt` (the person — objectively wrong),
  gold `e` ↔ `vehicle registration plate`, gold `wales` ↔ `holidays in wales`.
- Official benchmarks (SQuAD/TriviaQA/HotpotQA/NQ/WebQSP) never use bidirectional
  substring — they use normalized EM or token-F1 **max over the full alias list**
  (TriviaQA's `NormalizedAliases`, which the bench doesn't even load).
- Agent-measured: official-style EM = 18.2%, **EM-or-token-F1≥0.7 = 18.6%**
  (recommended primary), token-exact = 16.0%.
- **True selectable ceiling ≈ 54.7%** (EM-or-F1≥0.7 against graph nodes, 174/318),
  NOT 76.1% — ~21pt of claimed recall is phantom (gold a token-subset of a
  longer node).
- Implication: the real gap is ~40pt (55% ceiling → ~19% honest selection), and
  the substring metric **actively rewards picking the long reference entity** —
  it has been misdirecting optimization.

### 1.2 Failure decomposition (measured, current baseline, 163 failures)

Per-record dump (new `AXIOM_DUMP` diagnostic) on the current baseline:

| mechanism | % | winner profile |
|---|---|---|
| Query-topic/reference entity | 20% | winner conn=0 but overlap+count high; entity name appears in question ("Buddy Holly", "House of Habsburg") |
| Heur(count) dominance | 15% | winner = most-documented entity ("Natty Bumppo" beats "hawkeye") |
| Structural blind spot (gold conn=0) | 20% | gold connected to nothing → rank ~21, gap ~5 |
| Near-tie noise (gap ≤ 0.5) | 18% | irreducible per LESSONS_LEARNED §2.6 |
| Mixed/small | 27% | — |

Deep-rank (gold rank ≥ 6) = 62% of failures; gold rank 0 (not in candidate set)
= 0 — **every recalled gold is in the candidate set; selection is purely a
ranking failure.**

### 1.3 Empirical tests this session (env-only, full 318 bench)

| config | substring | exact | verdict |
|---|---|---|---|
| baseline | 24.84% | 13.84% | — |
| `AXIOM_W_OV=0` (no overlap) | 23.58% | — | overlap is double-edged, needed for near-tie golds |
| `AXIOM_QP_WHAT` ∈ {0.4,0.3,0.1} | 24.21% flat | — | universal penalty strengthening nets −2 |
| `AXIOM_W_HEUR` ∈ {0.7,0.5,0.3,0.1} | 22.33→11.01% | — | count IS evidence mass; cannot be reduced globally |
| **`AXIOM_V1_QNP=1`** (full penalty for conn=0 query-named) | 24.21% (−0.63) | **14.47% (+0.63)** | **real improvement hidden by substring metric** |

`AXIOM_V1_QNP` (code committed, default off) suppresses the reference entity
(conn=0 query-named) — exact-match improves +0.63pt (stable 3 runs) while the
substring metric reports a regression. **Keep-gate must switch to the honest
metric or it will reject real improvements.**

### 1.4 Historical reconciliation (agent review)

- **RCA (08-10) vs LESSONS_LEARNED (08-11) contradiction resolved**: RCA
  prescribed percentile/IEF hub-invariance; its own experiments refuted it
  (percentile 12.58% twice, IEF 5-10%). The gap is a *signal-scarcity* problem,
  not an aggregation problem. Every re-combination of the existing signals fails;
  every real win was an information-gate fix, decomposition-quality fix, a new
  structural signal (PPR), or a hard conditional constraint.
- Wins pattern: T1.7 (entity boundaries +4.72 recall), T1.9a (query-penalty
  gate +2.2), T1.9c (relative PPR +0.63), T1.10e (subject resolution +0.32),
  T1.11 (M1 overlap veto +0.31) — all "signal-scalpel", never "mixer".

---

## PART 2 — SOLUTION RESEARCH SYNTHESIS (4 deep-research tracks)

### Track A — Fix the metric first (prerequisite)

Recommendation (ANSWER_METRIC_RESEARCH.md): gate on **EM-or-token-F1≥0.7 over
the FULL official alias list** (`NormalizedAliases` + `Value` + `Aliases` +
`HumanAnswers` + `MatchedWikiEntityName`). Token-F1's precision term fixes the
`roosevelt ⊂ eleanor roosevelt` hole that containment misses. Keep substring
metrics as print-only diagnostics. Env-gate as `AXIOM_METRIC=strict` (default
keeps legacy for continuity) so old-vs-new can be diffed. Honest caveat (Bulian
et al. EMNLP 2022): no token metric separates `ely cathedral` (correct) from
`eleanor roosevelt` (wrong); strictness is the defensible default.

### Track B — PathHD: relation-schema path retrieval (the big structural lever)

arXiv:2512.09369 full algorithm extracted → full Rust spec in
`docs/research/PATHHD_ENGINEERING_SPEC.md`. Core ideas that transfer to
d=2048 CPU-only deterministic:

1. **GHRR block-unitary binding** — real O(4) blocks, D=128 blocks, d=2048
   (D·m² = 128·16 = 2048 exactly). Products of two Householder reflections per
   block (the paper's `diag(e^{iφ})` family is commuting — a paper bug; use
   Householder). Order- and direction-sensitive: `r1→r2 ≠ r2→r1`. Cost ~16K
   flops/hop. Existing `tle-vsa` Hadamard bind is commutative → insufficient.
2. **Plan-based query encoding** (Table 12, +1.7-2.8pt): the query becomes a
   relation-sequence plan `z_q`, derived deterministically by
   phrase→relation mapping (reuse the ~180-phrase RELATIONAL_PHRASES map),
   BFS over a relation-schema graph, argmax by content-word overlap. This is the
   key precision lever — NOT text projection (needs SBERT).
3. **Calibrated score** `s(z) = blockwise_cos(v_q, v_z) + α·IDF(z) − β·λ^|z|`,
   α/β/λ from Table 11 (0.2/0.1/0.8). IDF over evidence-document schema
   frequency (training-free).
4. **Top-K = 3 hard prune** (Table 5: no-prune blurs decisions; K=3 best).
5. **Deterministic adjudicator** replacing the LLM (+0.7-0.8pt in paper):
   intent-consistency relation veto + entity-evidence second pass over the
   top-K end entities (reuses extract_answer signals).
6. **Distractor bound (Thm 1, Rademacher = AXIOM's exact setting)**: d=2048 is
   sufficient for M≤200 candidates at ε≈0.10, δ=0.01.

Verdict: highest structural leverage. New crate `tle-ghrr` (do NOT modify
`tle-vsa`; AGENTS.md rule 6). Env `AXIOM_PATHHD=1`, default off.

### Track C — Reference/topic-entity suppression (20% of failures)

(REFERENCE_ENTITY_RESEARCH.md — full report saved.) Key literature: exclusion =
set subtraction (von Fintel 1993); NegEx cue-lexicon detector (Chapman 2001);
VSA negation = −X (Kanerva 2009) but weak at d=2048 (tiebreak only); STAGG query
graph (P15-1128): named entities are ANCHORS, answer = λ-variable node NOT in
question → validates the conn=0 discriminator; topic-signature/residual-IDF
(Lin&Hovy C00-1072, Church&Gale 1995) for frequency.

Recommended mechanisms (env-gated, default off — no v16 risk):
- **C1 exclusion-cue detection + full penalty on named anchors** (NegEx-style:
  "other than", "besides", "apart from", "the other one") → highest precision.
- **C2 query-focus classifier**: default = Identity (today's behavior, keeps
  Milky-Way correct); flip to Anchor only on high-precision surface triggers
  (possessive `X's`, `of`-PP, "for X and Y"). Winner must then have
  `has_struct` and not be a named anchor. Medium risk.
- **C3 query-conditional count**: zero/soft the `heur` count ONLY for anchors;
  untouched for everyone else. NOT the failed global count reduction.
- Test order C1 → C2 → C2+C3; VSA-NOT last (within-band tiebreak).

### Track D — Structural blind spot (gold conn=0, 20%) + count dominance (15%)

(CONNECTIVITY_COUNT_RESEARCH.md — full report saved.) Key literature: OPI
(arXiv:2606.28076, +4.6/+8.9 Hit@1, type-constrained final-hop expansion),
QASA (2606.30133, query-aware spreading-activation gate, +3.6-7.4 F1),
CatRAG (2602.01965, Static Graph Fallacy + query-aware edge weighting),
BM25/RSJ (Robertson-Zaragoza 2009; Robertson-Spärck-Jones 1976), Milne-Witten
relative PPR (already proven +0.63 in AXIOM).

Recommended mechanisms:
- **D1 (primary) Answer-type + typed final-hop expansion** (OPI-style):
  `predict_answer_type(intent, query)` → Entity/Person/Place/Temporal/Number;
  `RelationKind{head_type, tail_type}` table for ~40 relations; expand to
  candidates whose final-hop relation's tail-type matches predicted (and
  Number/Temporal candidates must parse numeric); new additive signal
  `w_typed·typed_avg` (env `AXIOM_W_TYPED`). Gives conn=0 attribute/value golds
  a dedicated strong signal. Expected biggest gain. Must ship BEFORE D2 (D2's
  conditional count would zero conn=0 golds).
- **D2 Conditional + saturated count** (BM25-grounded): `count_cond(e)` =
  query-connected triples (raw_conn_count + raw_2hop_count, already computed);
  `count_ratio(e)` = count_cond/count (Milne-Witten style, hub-invariant);
  `heur = w_count·BM25_sat(count_cond) + w_ratio·count_ratio − ...` with
  `BM25_sat(c) = c(k1+1)/(c+k1)`, k1≈2-3. Fixes Mode B without removing
  evidence mass. Env `AXIOM_W_RATIO`.
- **D3 (tertiary) QASA-style query-aware PPR gate**: gate σ(v) = lexical
  overlap between question content words and `fact_texts[v]` (NOT VSA cosine —
  noise), applied to the PPR mass-share; + CatRAG symbolic-anchor teleport
  (ε=0.2). Deterministic ~20 LOC. Makes D1/D2 reach further.

Recommended sequence: **A (metric) → B (PathHD) or D1 (typed expansion) → D2 →
D3 → C1/C2** — each env-gated, full-318 keep-gate on the STRICT metric.

---

## PART 3 — ACTION PLAN (prioritized)

| # | Task | Fixes | Mechanism | Env | Expected |
|---|---|---|---|---|---|
| 1 | **Adopt strict metric** (EM-or-F1≥0.7 over aliases) | metric inflation | Track A | `AXIOM_METRIC=strict` | honest 18.6% baseline |
| 2 | **QNP → default on** (conditional full penalty for conn=0 query-named) | 20% reference entities | already coded | `AXIOM_V1_QNP=1` | +0.63 exact (stable) |
| 3 | **D1 typed final-hop expansion** | 20% conn=0 + attribute answers | OPI-style | `AXIOM_W_TYPED` | largest |
| 4 | **B PathHD relation-schema retrieval** | deep-rank discrimination | GHRR plan encoding | `AXIOM_PATHHD=1` | structural |
| 5 | **D2 conditional+saturated count** | 15% count dominance | BM25/RSJ | `AXIOM_W_RATIO` | +1.5-3pp |
| 6 | **C1/C2 reference suppression** | 20% (with QNP) | NegEx/exclusion + query-focus | `AXIOM_V2_*` | — |
| 7 | **D3 gated PPR** | makes 3-6 reach further | QASA lexical gate | — | +0.5-1.5pp |

**Keep-gate rule update:** decisions on the STRICT metric (candidate exact /
EM-or-F1≥0.7) + recall (strict). Legacy substring metrics are print-only.

---

## Files produced this session
- `docs/research/ANSWER_METRIC_RESEARCH.md` — metric audit + recommendation
- `docs/research/REFERENCE_ENTITY_RESEARCH.md` — reference/topic suppression
- `docs/research/PATHHD_ENGINEERING_SPEC.md` — full PathHD Rust spec
- `docs/research/CONNECTIVITY_COUNT_RESEARCH.md` — typed expansion + count
- `crates/tle-axiom-gen/src/bin/triviaqa-bench.rs` — `AXIOM_DUMP` diagnostic +
  `candidate_token_accuracy` / `candidate_exact_accuracy` metrics
- `crates/tle-axiom-gen/src/engine.rs` — `AXIOM_V1_QNP` conditional penalty
  (default off, pending metric adoption)
