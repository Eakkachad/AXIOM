# AXIOM — Session Research Summary: 6 Negative Rounds + What Actually Works

> **Date:** 2026-08-11 · **Scope:** T1.8-T1.10 (ranking redesign deep-dive)
> **Baseline in / out:** candidate 19.81% → 24.53% · recall 71.38% → 76.10%
> **Purpose:** consolidate the negative results + the working patterns so the
> next research round does not repeat them. Complements LESSONS_LEARNED.md.

---

## 1. Executive Summary

Six experimental rounds aimed at fixing answer selection over a knowledge graph
all returned **negative or neutral**, yet the session still gained +4.72pt
(19.81→24.53%). The gains came from **bug-fixing gates, decomposition quality,
and one new structural signal** — never from redesigning the score fusion.

**The central finding:** the ~52pt gap between recall (76%) and candidate (24%)
is **NOT fixable by re-combining the same 6 signals**. Every fusion/normalization
redesign (weight tuning, percentile, RRF, conformal, IEF, sigmoid) destroys
information or hits a flat local optimum. The gap needs **new signals** and
**cleaner graph input** — which is where the real wins came from.

---

## 2. The 6 Negative Rounds (all bench-verified, all reverted)

| # | Round | Technique | Result | Why it failed |
|---|-------|-----------|--------|---------------|
| 1 | T1.8a/c | coordinate-ascent weight search + IEF | flat; IEF 5-10% | linear sum is a flat local optimum; log-freq kills evidence mass |
| 2 | T1.9a | RRF rank fusion | 11.95-15.41% | rank/equal-weight fusion destroys magnitude gaps |
| 3 | T1.10a | conformal p-value + log-odds PoE | 12.58-19.18% | p-value normalization = same failure as #2 (12.58% exact match) |
| 4 | T1.10b | Datalog type-veto (relation heuristics) | 19.81% | "won"→Person misfires; answer-type needs POS, not relations |
| 5 | T1.10c | NP-surface filter at graph | 23.58-23.90% | valid entities legitimately contain those chars; M5 mostly already filtered |
| 6 | T1.10f | VSA clamp + count re-weight (near-tie) | all regress | VSA noise ±0.08 flips ties randomly; count is genuine evidence mass |

**Pattern:** every attempt to make the signals "fair" (normalize, equalize,
calibrate) lost to the raw linear sum. The raw magnitudes carry information
that normalization discards.

---

## 3. What Actually Worked (the 4 real gains)

| Gain | Technique | Type | Result |
|------|-----------|------|--------|
| **+2.2pt** | query-entity punctuation fix | **bug-fix at a gate** | "O'Hare"/"Jaws (film)" were escaping the ×0.2 query penalty (punctuation split) |
| **+4.72pt recall** | proper-noun boundary precision | **decomposition quality** | clean entities ("Chicago" not "Chicago, Illinois, 17 mi") → connectivity fires |
| **+0.63pt** | hub-corrected PPR (relative PPR) | **new structural signal** | `log π_q(e) − log π(e)` (Milne-Witten) debiases hubs |
| **+0.32pt** | subject resolution | **decomposition quality** | copula-tail strip + leading-copula inherit + passive `*_by` patterns |

**The recipe that works (repeatedly):**
1. **Fix bugs at the gates** (query matching, subject resolution)
2. **Improve decomposition quality** (entity boundaries, subject anchoring)
3. **Add new structural signals** (PPR — orthogonal to the 6 existing)

---

## 4. The Math Why (for the paper / next research)

### 4.1 Why linear-sum tuning is a flat local optimum
If signals have noise σᵢ, the min-variance linear combiner needs wᵢ ∝ 1/σᵢ².
Overlap (range ~50) has σ≈10+, so it wants a tiny weight — but overlap is also
the MOST informative when it truly fires. **No single linear weight is correct
for both "overlap=gold" and "overlap=trap" regimes.** This is a structural
limit of `Σ wᵢfᵢ`, not a calibration problem (5+ variants confirmed).

### 4.2 Why normalization/fusion fails (12.58% exact match, twice)
Converting raw scores (conn 2.0 vs 0.5) to percentile/p-value (rank 1 vs 2)
discards the magnitude gap. Two independent implementations (T1.6 percentile,
T1.10a conformal) hit the SAME 12.58% — strong evidence this is a real
information-theoretic loss, not coincidence.

### 4.3 Why VSA cosine is noise with a random codebook
Random bipolar vectors in d-dim give cos ~ N(0, 1/√d). At d=2048, σ≈0.022.
Any claim that raw VSA cosine carries signal is false by construction unless
the vectors share structured components. VSA can only be a **verification gate**
or a bounded factor, never a deciding term.

### 4.4 What a fixable gap looks like (data-driven)
- deep-rank golds (149): mostly fixed via **subject resolution** (WHO+creator 12)
  + decomposition; the rest need stronger connectivity (L1 clause typing)
- near-ties (22): irreducible under linear sum; need **structural entity-type
  filter** (POS/NER-lite) to break — NOT scoring weights

---

## 5. Recommendations for Next Research Round

1. **Decomposition (highest ROI):** clause/subject typing (L1) — completes
   subject resolution, enables location transitivity, feeds answer-type.
   Directly attacks the largest remaining buckets (deep-rank 149, WHERE 35).
2. **New signals over fusion:** anything orthogonal to the 6 existing signals
   (PPR proved the pattern). Candidates: clause-role encoding, evidence-path
   count, co-occurrence (with proper content-word query, big corpus).
3. **POS/NER-lite lexicon** — the ONLY way to break M2 near-ties (entity-type
   veto) without relation-heuristic misfires (T1.10b failure).
4. **Do NOT re-attempt:** any fusion/normalization/weight redesign of the 6
   signals; IEF; relation-heuristic type veto; graph-surface filters; VSA as
   primary. All documented with numbers in LESSONS_LEARNED.md.

---

## 6. Final Bench State

| Metric | v14 | v15 end | Δ |
|--------|:---:|:---:|:---:|
| candidate_answer_accuracy | 19.81% | **24.53%** | +4.72pt |
| answer_entity_recall | 71.38% | **76.10%** | +4.72pt |
| substring_accuracy | 23.90% | 23.27% | -0.63pt |
| evidence_answer_recall | 99.69% | 99.69% | 0 |

*Commits: T1.7, T1.8a, T1.9a, T1.9c, T1.10e + neutral infra (T1.9b, T1.10b/d).*
