# AXIOM — Answer-Selection Redesign: Problem Analysis + Research Synthesis

> **Date:** 2026-08-11
> **Status:** Research complete — proposals ready for A/B bench
> **Inputs:** 171-failure debug analysis (this session), 3 deep-research memos
> (math/literature, katgpt-rs prior art, local research docs)

---

## 1. Problem Analysis (empirical, 318-record bench)

Baseline v15: candidate 21.38% · recall 76.10% · substring 22.64%. Gold answer
is in the graph 76% of the time but picked only 21% → **answer selection is the
bottleneck, not retrieval.**

171 failures categorized from `AXIOM_TRIVIA_DEBUG` top-5 + gold-rank traces:

| Mode | Cases | Mechanism |
|---|---|---|
| **M1 Overlap dominance** | 21 top-5 + most of 102 deep | Entity whose NAME matches question words wins: "Jaws (film)">Bruce, "O'Hare">Chicago (which city), "John F. Kennedy" ov=54. Question-named ≠ answer. |
| **M2 Near-tie noise** | 18 | Junk beats gold by <0.6pt ("in 2007">SMERSH). One signal's noise band decides. |
| **M3 Hub/degree** | 5 | High-frequency entities (FA Cup, Cricket, United States) win via `0.2·count`. |
| **M4 Structural conn=0** | 6 (+) | Gold IS a node but has zero typed connectivity to query (recall is substring, over-counts). |
| **M5 Junk entity surfaces** | several | `Cast *Gregory Peck`, `inaugurated on 30 April 1890`, `It's Cold Outside"` still enter graph. |

Root cause (confirmed by this + 5 prior sessions): **linear weighted sum +
argmax over 6 differently-scaled signals** cannot express conditional facts:
"name-match is informative ONLY when connectivity is also present." Overlap
lives on scale ~50, conn on ~2 — no single linear weight is right for both the
"overlap=gold" and "overlap=trap" regimes. This is the documented flat local
optimum.

---

## 2. Research Findings (3 memos, converged)

Three independent research streams converge on the SAME structural prescription:

1. **Hard filter BEFORE ranking** (veto, not vote) — math memo F1/F2/F3,
   katgpt's Screening/007 (`Score = Σ[ln P + ln R]`, hard trim at R=0),
   local docs' query-construction.
2. **Rank fusion (RRF) instead of score fusion** — Cormack 2009; bounded
   contribution `1/(k+rank)` kills scale mismatch by construction.
3. **Nonlinear aggregation** — sigmoid-then-power reliability gate `(mean)^M`
   (katgpt CLR), SmoothMin (katgpt similarity.rs, +12pp), softmax-temperature
   sharpening, calibrated log-odds / product-of-experts.
4. **Hub correction** — Milne-Witten PPR debias `log π_q(e) − log π(e)`
   (math memo), anti-frequency penalty `γ·ln(1+count)` (TBA EBM, local docs).
5. **Orthogonal evidence signal** — compression-length / MDL, but ONLY as
   differenced verification (compress query⊕fact, subtract query⊕name) so it
   cannot re-instantiate M1.

### Why the linear-sum approach failed 5+ times (the math)

If signals are independent with noise σᵢ, the min-variance linear combiner
weights wᵢ ∝ 1/σᵢ². Overlap (range ~50) ⇒ σ≈10+ ⇒ gets tiny weight — but
overlap is ALSO the most informative when it truly fires. Minimax-linear
combination cannot express "highly informative but mostly false-positive."
That is M1 exactly, and why weight tuning hits a flat optimum: **no single
linear weight is correct for both regimes.** The fix is conditional
weighting / hard gates / rank positions — not more weight search.

---

## 3. The Proposal (champion pipeline)

```
[HARD FILTER — vetos, cannot be outvoted by magnitude]
  F1 type:      intent(q)→τ(q) (Who→PERSON, Where→LOCATION, When→TIME, ...)
                candidate survives iff type(e)∈τ(q); widen if empty.
  F2 relation:  e reachable from a query entity via a relation whose surface
                form appears in question content words.
  F3 distance:  dist(e, query) ≤ 3  (PPR expansion for M4).

[RANK — among survivors]
  RRF(e) = Σᵢ wᵢ / (k + rankᵢ(e)),   k=60
  lists:  conn, role, hop2, overlap, vsa(→verification gate only), heur,
          + ppq (hub-corrected PPR) as 7th list
  wᵢ = AUCᵢ/ΣAUCᵢ  (calibrated once on the bench — data-derived, not tuned)
  answer = argmax RRF(e)
```

Optional layers (A/B separately):
- **L2 sigmoid-then-power:** `score(e) = Σ wᵢ·σ(aᵢ(sᵢ−bᵢ))^γ` — reliability-gated, super-linear separation for M2.
- **L3 compression verification:** `S(e) = [C(q⊕fact(e)) − C(q)] − [C(q⊕name(e)) − C(q)]` as a final tiebreak (M5 / orthogonal-to-VSA).

### Implementation order (each independent, cheap, benchmarkable)

1. **Scheme 3 first** — hub-corrected PPR as 7th list + M4 candidate expansion.
   Fixes the 6 structural cases nothing else touches. Deterministic power
   iteration, ~60 iters, O(E).
2. **Scheme 1** — hard filter F1/F2/F3 + RRF. Biggest lever (M1 21 + M2 18 + M3 5).
3. **Scheme 2 / conformal p-value fusion** (Rüschendorf `2·mean`) as the scoring
   layer if RRF leaves intra-list noise.

---

## 4. References (verified)

- Cormack, Clarke, Buettcher (2009). *Reciprocal Rank Fusion.* SIGIR. https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
- Sun, Bedrax-Weiss, Cohen (2019). *PullNet.* ACL. https://arxiv.org/abs/1904.09537
- Penzkofer et al. (2024). *VSA4VQA.* CogSci. https://arxiv.org/abs/2405.03852
- Kleyko et al. *HDC/VSA Survey I & II.* ACM Computing Surveys. https://arxiv.org/abs/2111.06077, https://arxiv.org/abs/2112.15424
- Vovk & Wang (2012). *Combining p-values via averaging.* https://arxiv.org/abs/1212.4966
- Hinton et al. (2015). *Distilling Knowledge* (temperature). https://arxiv.org/abs/1503.02531
- Milne & Witten (2008). *Learning to Link with Wikipedia* (PPR hub debias). CIKM.
- Haveliwala (2002). *Topic-sensitive PageRank.* WWW.
- Jeh & Widom (2003). *Scaling Personalized Web Search.* WWW.
- Hinton (2002). *Products of Experts.* Neural Computation.
- katgpt-rs (local): `compression_drafter.rs`, `clr/vote.rs` (reliability gate),
  `similarity.rs` (SmoothMin), `screening/complexity_prior.rs` (sigmoid-never-softmax).
- Local docs: `AXIOM_MATH_FRAMEWORKS_RESEARCH.md` §7.2 (RRF), `RESEARCH_PAPER_DRAFT.md` §3.3 (TBA anti-frequency).

---

## 5. Failure-mode × proposal matrix

| Mode (cases) | Linear+argmax now | Hard filter | RRF | PPR hub-correct | est. recovered |
|---|---|---|---|---|---|
| M1 Overlap (21) | 21 lost | F1 veto | bounded | — | 15-19 |
| M2 Near-tie (18) | 18 lost | F2 | cross-list | diffusion margin | 9-16 |
| M3 Hub (5) | 5 lost | — | capped list | π/π_q cancels | 4-5 |
| M4 conn=0 (6) | 6 lost | F3 | — | **only fix** | 3-5 |
| **Combined** | — | — | — | — | **~31-40 / 50** |

Projection: candidate 21.38% → est. 25-30% if the recovered cases hold.
**These are hypotheses to verify on the bench, not measurements.**

---

*Ground truth for the next task: ROADMAP T1.9. Do NOT re-propose manual weight
tuning / percentile equal-weight / IEF / semantic-in-scoring / DDTree.*
