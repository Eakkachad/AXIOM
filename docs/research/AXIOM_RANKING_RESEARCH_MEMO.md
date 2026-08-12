# AXIOM — Ranking & Answer-Selection Research Memo

**Scope:** Non-neural, deterministic, CPU-only, VSA (d=2048, random bipolar codebook) + KG QA system in Rust.
**Empirical anchor:** 318-record TriviaQA bench. `extract_answer` = linear-weighted score + argmax over 6 signals.

**Documented failure modes (50 of 318):**
1. **Overlap dominance (21)** — entity whose *name* matches question words beats gold. ("Jaws (film)" > "Bruce"; "O'Hare" > "Chicago".)
2. **Near-tie noise (18)** — junk beats gold by <0.6 composite points. ("in 2007" > "SMERSH".)
3. **Hub/degree (5)** — high-frequency entities (FA Cup, Cricket, United States) win via `0.2·count`.
4. **Structural conn=0 (6)** — gold IS a node but has zero typed connectivity to the query entities (substring recall over-counts).

**Failed (do not re-propose):** manual weight tuning; equal-weight percentile normalization (12.58%, worse); IEF in place of count; DDTree beam scoring (4 tries); semantic co-occurrence in scoring; VSA-cosine-as-primary (noise).

---

## TL;DR — top 3 candidate schemes (ranked by expected failure-mode impact)

| # | Scheme | Formula (exact) | Fixes | Expected |
|---|--------|-----------------|-------|----------|
| **1** | **RRF + hard structural filter** (rank fusion over per-signal lists, k=60) | `score(e) = Σ_i w_i·1/(k + rank_i(e))`, `w_i = AUC_i/ΣAUC`; pre-filter by answer-type + question-relation match | #1 (bounded 1/(k+rank) kills overlap's ~50-scale dominance), #2 (cross-list agreement), #3 (frequency becomes one bounded list) | 44/50 non-structural cases, est. 33–40 recovered |
| **2** | **Empirically calibrated log-odds product** (reliability-gated PoE) | `score(e) = Σ_i w_i·logit(ĉ_i(e))`, `ĉ_i` = Laplace-smoothed P(gold\|bin of s_i) calibrated once on the 318 bench | #1, #2 (margin→probability), #3 (count's calibrated weight collapses) | est. 32–38 recovered |
| **3** | **Personalized PageRank, hub-corrected (π_q/π), as a fused structural signal** | `π_q = (1−c)v + cPᵀπ_q`, 40–100 power iterations; `ppq(e) = log π_q(e) − log π(e)` | #4 (multi-hop reachability, the only thing that fixes it), #3 (stationary/π debias), supports #2 | est. 3–5 of the 6 conn=0 cases |

**Champion pipeline:** `HardFilter(intent, relation, type)` → candidate set expanded to 3-hop via PPR → `RRF` over 6 per-signal rankings with AUC weights, freq-list capped → argmax; VSA cosine retained only as a post-hoc sanity gate (drop to a single bounded term or remove).

Sections below: (1) rank aggregation/RRF, (2) graph centrality/PPR, (3) HDC/VSA for QA, (4) non-neural query construction, (5) score-combination theory, then full formulas for the three schemes and a failure-mode matrix. Sources marked **[verified]** were fetched during this session; **[memory]** are from training knowledge (titles/metadata reliable, exact IDs to double-check).

---

## 1. Learning-to-rank without ML: rank aggregation and RRF

### 1.1 The core idea

Replace "fuse raw scores" with "fuse **ranks**". Every signal `i` defines an ordering; aggregate orderings instead of numbers. The family: **Borda** (points = #items ranked below + 1, summed over lists), **Copeland** (pairwise win counts), **Kemeny** (permutation minimizing sum of Kendall-tau distances; NP-hard [memory: Kemeny 1959; Bartholdi et al. 1989]), **RRF** (weighted sum of reciprocal ranks).

**Reciprocal Rank Fusion** — Cormack, Clarke & Buettcher, SIGIR 2009 [verified, PDF: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf]:

```
RRF(e) = Σ_i  1 / (k + rank_i(e))
```
where `rank_i(e)` is the rank of `e` in list `i` (1-based; absent → not summed), and `k` is a smoothing constant (Cormack reports `k=60` robust; tune `k∈[10,200]` on the bench).

**Bounded contribution is the whole point.** Each list contributes at most `1/(k+1) ≈ 0.0164` (k=60). Six lists → total score ≤ `0.098`. Compare the current scheme where `overlap` lives on a ~0–50 scale and `conn_avg` on ~0–2: no single signal can dominate, because *rank* — not magnitude — is what carries information.

### 1.2 Does RRF fix the scale mismatch (overlap ~50 vs conn ~2)?

**Yes, by construction.** Rank is invariant under monotone rescaling of each signal (scale-free, position-of-item-based). "Jaws (film)" gets `1/61` from the overlap list but contributes nothing where overlap doesn't rank it (conn/role/hop2 all rank it low or absent); "Bruce" collects contributions from role (subject/object bias), conn and hop2. RRF is exactly the fusion rule Cormack designed for **heterogeneous lists from unrelated systems** — the textbook setup you have.

### 1.3 RRF vs weighted score fusion — when is rank-fusion preferred?

- **Rank fusion wins** when signals are (a) differently scaled, (b) noisy in magnitude but order-stable, (c) sparse/partial (item present in only some lists). RRF needs no per-signal calibration and no normalization, and it degrades gracefully: a list missing a candidate just doesn't contribute.
- **Score fusion wins** when magnitudes carry meaning (e.g., a calibrated probability). Once you calibrate signals into probabilities (Scheme 2), score fusion is theoretically nicer.
- The formal statement: for top-1 selection under **independent** monotone-noised scores, rank fusion (Borda/RRF) is minimax-robust to unknown scale; for calibrated probabilities, log-product is optimal. So: **calibrate → fuse scores; don't calibrate → fuse ranks.**

### 1.4 RRF vs failure modes

- **#1 overlap dominance (21):** **Fixes.** Overlap's contribution is capped at `1/(k+1)`; gold wins whenever it ranks in ≥2–3 of the remaining 5 lists. This is the largest single win available.
- **#2 near-tie noise (18):** **Partially fixes.** RRF kills *inter*-list scale noise (the 0.6pt junk-beats-gold margins are largely scale artifacts) but not *intra*-list ordering noise (if junk is ranked just above gold inside one list, ranks still prefer junk). Pair with Scheme 2's calibration or temperature sharpening (Sec 5.4) for intra-list noise.
- **#3 hub/degree (5):** **Fixes mostly.** The frequency term becomes one bounded list; worst case a hub adds `1/(k+1)` ≈ 0.016. Optionally **drop the count term entirely** or cap its list so hubs never appear in its top ranks when count is the *only* signal (add a "≥2 lists" sparsity guard, see Scheme 1).
- **#4 structural conn=0 (6):** **Does not fix.** No rank operation creates a signal that is zero. Requires PPR / candidate expansion (Scheme 3).

### 1.5 Rust feasibility
Trivial. Per signal: one sort of the candidate vector → rank map. Fusion: O(candidates × signals). Deterministic, no floats beyond f64. Kemeny is the only infeasible member (NP-hard); Borda/Copeland/RRF are all O(n·log n).

---

## 2. Graph-based entity scoring: PPR, RWR, HITS, Katz, SimRank

### 2.1 Formulas

**Personalized PageRank / random-walk-with-restart** [memory, canonical: Jeh & Widom, "Scaling Personalized Web Search," WWW 2003; Haveliwala, "Topic-sensitive PageRank," WWW 2002]:

```
π_q = (1 − c)·v  +  c·Pᵀ·π_q
```
- `c` restart discount (e.g., 0.85; RWR literature: 0.1–0.15 restart *prob*),
- `P` row-stochastic transition matrix, `P[u,v] = w(u,v)/Σ_w(u,·)` — **this is itself the degree normalization** (each hop divides by out-weight),
- `v` = teleport distribution = uniform over query entities (or weighted by question-word overlap).

Solved by power iteration: `π^{(t+1)} = (1−c)v + cPᵀπ^{(t)}`, 40–100 iterations, O(edges) each. Deterministic; converges since `(1−c)v` is a positive mass-1 distribution and P is stochastic (spectral radius 1, contraction factor c).

**Katz centrality** [memory: Katz 1953]: `score(e) = Σ_paths α^len` — closed form `(I − αA)^{-1}`, weights *all* path lengths with geometric decay. PPR is the probability-normalized cousin of Katz with a teleport term; on CPU for <1e5 nodes either is fine (Katz needs a linear solve or series truncation).

**HITS** [memory: Kleinberg 1999]: alternating hub/authority iteration on the adjacency matrix. **SimRank** [memory: Jeh & Widom 2002]: recursive "similar if neighbors are similar" — expensive, not needed here.

### 2.2 Does PPR fix hub domination? — yes, but only with the right normalization

**The hub problem is real and has a known, standard fix.** PPR without personalization concentrates mass on hubs (stationary distribution favors high in-degree). Two orthogonal corrections:

1. **Degree normalization is already in `P`** (divide row by out-weight) — this handles *out*-degree hubs but not *in*-degree popularity.
2. **Background-stationary debias (the standard one):** divide the personalized score by the global stationary score,

```
ppq(e) = log π_q(e) − log π(e),   π = PageRank with uniform teleport
```
This is a PMI-style normalization ("relative PPR"). It is the basis of the Wikipedia Link-based Measure / Milne-Witten entity-linking prior [memory: Milne & Witten, "Learning to Link with Wikipedia," CIKM 2008], where PPR was normalized by the stationary distribution precisely to stop common-entity inflation. **This is the documented hub fix.** A hub that is 1–2 hops from every query gets a large `π_q(e)` *and* a large `π(e)`; the ratio cancels. FA Cup / Cricket / United States would stop winning.

### 2.3 Does PPR fix the structural conn=0 failure (#4)?

**Yes — this is the only scheme in this memo that does.** `hop2_avg` and `conn_avg` are truncated at 2 hops / typed edges only; PPR sums over *all* path lengths with decay `c`, so an answer 3–5 hops out, or one reached through untyped/weak edges that `conn` ignores, accumulates mass. Note the honest caveat: if gold is *topologically disconnected* from the query entities (its own connected component), every diffusion method scores it 0. Expect to recover the majority of the 6, not all of them.

### 2.4 PPR in KGQA (evidence)

- **PullNet** [verified, arXiv:1904.09537, ACL 2019 — Sun, Bedrax-Weiss, Cohen] constructs a question-specific subgraph via iterative retrieval and reasons over it — PPR/RWR-style subgraph construction is the dominant Freebase/DBpedia QA pattern.
- **GQA** "Graph Based Question Answering" [memory: Kag, Salim, Gavish, NAACL 2018] extracts a small weighted graph from the question words, runs a walk/PPR, then *uses the extracted answer type and relations to re-score candidates*. This is precisely Scheme 1+3: structural scoring on a query-localized subgraph, then a type/relation-constrained re-rank.

### 2.5 Rust feasibility
Power iteration on a HashMap<edge> graph is straightforward and deterministic (fixed iteration count — no tolerance-based nondeterminism). Cost: O(iter × E). For the reported graph scale (sub-100k edges) this is microseconds-to-ms. Precompute global `π` once offline.

---

## 3. Hyperdimensional computing (HDC/VSA) for QA

### 3.1 Why vsa_cosine is noise — quantified

With a random bipolar codebook in d=2048, the cosine between an *independent* question bundle and an entity name vector is approximately N(0, 1/√d), i.e., std ≈ 0.022. (Dot product of two independent random ±1 vectors ~ Binomial(d,½)-centered, variance d ⇒ cosine ~ N(0,1/d) ⇒ std 1/√d.) **This matches your measurement exactly, and it is a theorem about random codebooks, not a tuning problem.** Any claim that raw VSA cosine carries signal on this setup is false by construction *unless* the two vectors share structured components.

### 3.2 What published HDC QA actually does

- **VSA4VQA** [verified, arXiv:2405.03852, CogSci 2024 — Penzkofer, Shi, Bulling] is the closest "HDC for QA" system: it encodes natural images in a 4D VSA (Semantic Pointer Architecture) and answers spatial questions with **learned query masks** and a **pre-trained VLM**. Both crutches violate your no-training constraint. Its lesson: state-of-the-art HDC QA needs *learned* query encoders — the codebook analog of "the query mask is tuned."
- **HDC/VSA Survey I & II** [verified: arXiv:2111.06077, arXiv:2112.15424, ACM Computing Surveys — Kleyko et al.] — the definitive taxonomy. Relevant section: NLP applications use VSA for *classification/retrieval by bundling*, where the encoder (how you turn tokens into hypervectors) is the entire game. With a random codebook and no learning, the only usable signal is **shared-subcomponent count**, not cosine magnitude.

### 3.3 What is salvageable for AXIOM (deterministic, no training)

**(a) Neighborhood-encoded entity hypervectors.** Instead of `H(e) = code(entity_name)`, encode an entity by its *neighbors*:

```
H(e) = ⊕_r  bind(code(relation_r), code(neighbor_name_r))      (bundle over 1-hop neighbors)
Q    = ⊕_q  code(query_entity_name_q)
```
Then `cos(Q, H(e))` has expected value ≈ (fraction of shared terms) > 0 whenever `e` shares named neighbors with the query entities — i.e., it becomes a soft *semantic-neighbor* signal rather than white noise. Caveat: this is literally "semantic co-occurrence in scoring," which you report failed — but it failed as a *raw lexical* co-occurrence; the VSA version is the same computation through a linear codebook and will behave equivalently. **Recommendation: skip.** The signal you want is structural (PPR), not distributed-lookalike-lexical.

**(b) VSA as an answer *verification gate*, not a ranker.** Standard, well-attested VSA-NLP use: encode the passage as a bundle of window n-grams; verify that the candidate answer token co-occurs with a query entity in the evidence text. This is deterministic, uses the codebook as intended (bundled co-occurrence detection), and can act as a `+ε` consistency bonus or a `0/1` gate inside Scheme 1. This is *not* what you tried (you tried cosine-to-rank).

**Bottom line for #RQ3:** no published non-neural, untrained HDC system does better answer selection than a rank/structural fusion of the kind in Schemes 1–3. Drop vsa_cosine from the primary sum; keep it only as the verification gate (b) or a capped `+0.02·I(verified)` term. The VSA budget is better spent on the hypervector side of *encoding* (role-binding), not on the ranking side.

---

## 4. Non-neural KGQA: query construction / hard structural filtering

### 4.1 On "Grecx"
I could not locate a KGQA system named "Grecx." The only indexed item is *GRecX: An Efficient and Unified Benchmark for GNN-based Recommendation* [verified, arXiv:2111.10342] — unrelated. **Likely a misattribution; treat the "Grecx = parse-to-SPARQL" claim as unverified.** The underlying idea it gestures at is real and well-documented, though:

- **Dong et al., "Question Answering Over Freebase with Multi-Column Convolutional Neural Networks"** [memory: ACL 2015] — relation-column matching: score = word-match of question against (subject, relation, answer-type) columns, with a hard answer-type prior.
- **PullNet** [verified, arXiv:1904.09537] — constructs a question subgraph, *then* scores.
- **GQA** [memory: NAACL 2018] — phase 1 extracts candidate set + answer type + relations, phase 2 re-ranks. **The two-phase structure (filter-then-rank) is the robust pattern**, and it's free (no neural nets).

### 4.2 Deterministic hard filters that fix #1 (21 cases) directly

**F1 — Answer-type filter (kills "Jaws (film)" beats "Bruce"):**
```
intent(q) → expected types τ(q)   (Who→PERSON; What/Which→CONCEPT/thing; Where→LOCATION; When→TIME; How many→NUMBER)
candidate survives  ⇔  type(e) ∈ τ(q)   (soft fallback: if no candidate survives, widen τ to all)
```
The question-named entity in the 21 overlap cases is virtually always the *wrong type* for the intent (a film is not the answer to "what was the shark's name"). This is the single most targeted fix for failure mode #1.

**F2 — Relation/predicate filter (kills near-ties and wrong-type noise):**
```
candidate survives ⇔  e reachable from a query entity via a relation whose surface form (or a frozen synonym set)
                      appears among the question's content words
```
Cheap, deterministic (string + a small hand-built synonym table). This removes "in 2007"-style fragments that pass substring recall but are not attached to a question-matching relation.

**F3 — Connectivity requirement (attacks #4):**
```
survives ⇔  dist(e, query entities) ≤ 3   (via PPR expansion, Scheme 3)
```
Expands the candidate set from the substring-recalled set (which over/under-counts) to a structural neighborhood, so gold nodes with zero lexical recall still enter the ranking.

**Why filtering before ranking is more robust than ranking alone:** a linear ranker can be outvoted by a strong-but-wrong signal (overlap at scale 50). A *filter* is a veto — it cannot be outvoted by magnitude. Failures become countable, diagnosable cases instead of soft score losses. This is the mathematically important asymmetry: **hard constraints fix mode #1 deterministically; ranking fixes modes #2/#3 probabilistically.**

### 4.3 Rust feasibility
All string/type set operations; O(candidates × relations). Trivial.

---

## 5. Score-combination theory

### 5.1 Why linear sums of differently-scaled signals fail

Two clean statements:

1. **Scale mismatch ≠ calibration error.** If signals are independent with noise σ_i, the minimum-variance linear combiner weights `w_i ∝ 1/σ_i²`. Overlap (range ~50) has σ≈10+ ⇒ gets a *tiny* weight; but overlap is also the *most informative* signal when it truly fires. Minimax-linear combination cannot express "highly informative but mostly false-positive." That is exactly the 21-case mode-#1 failure, and it's why manual weight tuning hit a flat local optimum: **no single linear weight is correct for both the "overlap is gold" and "overlap is a trap" regimes.** The fix is *conditional* weighting: calibrate P(correct | overlap=k).

2. **Linear sums let a strong noisy signal win on margin, and margins are not comparable across signals.** The 18 near-tie cases (junk > gold by <0.6) are decisions made inside one signal's noise band.

### 5.2 Product of experts / geometric mean / log-sum-exp

**Product-of-experts** [memory: Hinton, "Training Products of Experts by Minimizing Contrastive Divergence," Neural Computation 2002]:
```
PoE:      score(e) = Π_i  g_i(s_i(e))^{w_i}         (requires g_i > 0 everywhere)
log-form: score(e) = Σ_i w_i·ln g_i(s_i(e))
```
Properties: requires **agreement across all experts** (any one zero kills the candidate) — excellent for noise (#2) but fatal for structural zeros (#4), where `conn=0` is legitimate. Therefore: **use a gated/calibrated variant** — map each raw signal to a probability in (0,1) first (Scheme 2), so a zero raw value maps to `p≈small`, not `p=0`. Then the log-product is finite, and it behaves like a **soft-AND**: candidates must be plausible on every signal, which is precisely the anti-noise property you want.

**Smooth-max / smooth-min (β-parameterized)** — the differentiable min/max relaxations:
```
SoftMaxβ(s_1..s_m) = β⁻¹ · ln( Σ_i exp(β·s_i) )      → max as β→∞
SmoothMinβ(s_1..s_m) = −β⁻¹ · ln( Σ_i exp(−β·s_i) )   → min as β→∞
```
- **SmoothMin = soft-AND / "all filters" gate** — good for *hard* structural requirements (mode #4, #1) where you want the weakest-link signal to dominate.
- **SoftMax = soft-OR** — good when "any strong signal suffices."
- Both are log-sum-exp family; `β` is a single scalar tuned on the bench (coarse grid, 3–5 values — NOT the manual-weight tuning that failed, because β only sets how "hard" the gate is, and the geometry of the operation — not a per-signal weight — does the work).

### 5.3 Sigmoid-then-power (reliability gates)

```
score(e) = Σ_i w_i · σ( a_i·(s_i(e) − b_i) )^γ      (a_i = steepness, b_i = threshold, γ = exponent)
```
Equivalent structure to the calibrated variant: each signal is bounded to (0,1) (fixes scale mismatch), and raising to power γ sharpens or flattens the "region of influence." The clean deterministic version of "reliability gating" is Scheme 2's empirical calibration (below) — gates are then *data-derived* instead of hand-picked, which is the lesson from your failed weight tuning.

### 5.4 Softmax temperature sharpening

```
p(e) = exp(s(e)/T) / Σ_j exp(s(j)/T)      (T < 1 sharpens:  p_top ↑, second-place ↓)
```
Temperature sharpening **amplifies margins before argmax**, directly attacking mode #2's <0.6pt decisions — *but only if the scores feeding it are trustworthy* (calibrated). Sharpening raw uncalibrated scores amplifies noise too. Reference for the temperature mechanism: Hinton et al., "Distilling the Knowledge in a Neural Network" [verified, arXiv:1503.02531]. Use `T ≈ 0.2–0.5` *after* Scheme 2's calibration, or drop it.

### 5.5 Conformal p-value combination (distribution-free fusion)

The most theoretically satisfying deterministic answer to "combine 6 differently-scaled signals": convert each signal to a **p-value against the per-question candidate distribution**, then combine p-values under the Rüschendorf averaging rule, which is **valid under arbitrary dependence** [verified: Vovk & Wang, "Combining p-values via averaging," arXiv:1212.4966]:

```
p_i(e) = ( # candidates with s_i(·) ≥ s_i(e) ) / ( # candidates )        # empirical CDF
p_comb(e) = min( 1,  2 · mean_i p_i(e) )                                 # Rüschendorf 2·mean
answer = argmin_e p_comb(e)
```
- Scale mismatch disappears **by construction** — every p-value lives on [0,1].
- The `2·mean` rule is the maximally robust combiner (factor 2 is optimal under worst-case dependence); alternatives with different trade-offs: Fisher (`−2Σ ln p_i` → χ²_{2m}), Stouffer (`Σ Φ⁻¹(p_i)`), harmonic mean p (HMP) and the Lévy combination test [verified: Wilson, arXiv:2105.01501], and exchangeable improvements [verified: Gasparin, Wang, Ramdas, arXiv:2404.03484].
- Mode #2: near-ties in raw score space often *are* resolved because p-values are relative — a junk entity whose raw signal is only slightly above gold still has a comparable p, so 2·mean punishes candidates that are mediocre on *multiple* signals.
- Rust feasibility: trivial, deterministic, one sort + running counts. **This is the lowest-engineering, highest-rigor fusion available** — recommend it as the fallback fusion if AUC-calibration (Scheme 2) feels over-engineered.
- Conformal framing (as a guardrail, not required): distribution-free coverage guarantees for the top-k answer set — Angelopoulos & Bates [verified, arXiv:2107.07511]. Use for *abstention*: if `p_comb(argmin)` ≥ α, return "don't know" instead of argmax.

---

## 6. The three concrete schemes (full formulas)

**Notation.** Candidates `E` per question. Raw signals `s_i(e)`, `i ∈ {conn, role, hop2, overlap, vsa, heur}`. Offline artifacts (built once, deterministically, from the 318-record bench — this is *calibration data*, not training): AUC_i per signal, and bin tables.

### Scheme 1 — RRF + hard structural filter  (recommended primary)

```
[OFFLINE]  for each signal i: compute AUC_i = P(rank(gold) < rank(junk)) over the bench.
           w_i = AUC_i / Σ_j AUC_j

[PER QUESTION]
  E' = { e : F1 type(e)∈τ(intent)  AND  F2 reachable-via-question-relation  AND  F3 dist≤3 }
  if |E'| == 0: widen to {e : F1 only}; if still 0: fall back to unfiltered candidate set.

  for each i: L_i = sort E' by s_i descending;  rank_i(e) = position (absent → +∞)
  RRF(e) = Σ_i  w_i / (k + rank_i(e)),   k = 60 (grid-search k ∈ {10,30,60,100} on bench)
  answer = argmax_{e∈E'} RRF(e)
```
Sparsity guard: drop candidates appearing in ≤1 list (they are single-signal flukes — the anti-#2 step). Frequency list: include `s_freq = count` as a 6th list **but** cap its AUC weight to ≤0.1·Σ (or drop it; the bench decides).

**Why it's ranked #1:** it attacks all three biggest failure classes at once with a single mechanism (bounded rank contribution), is standard IR practice, and needs only two scalar tunables (k, and the freq weight cap) which — unlike per-signal weights — do not sit on a flat optimum.

### Scheme 2 — Calibrated log-odds product (reliability-gated PoE)

```
[OFFLINE]  for each signal i, bin s_i over the bench into B=10 quantile bins over its positive support.
           c_i[b] = ( #gold in bin b  + 1 ) / ( #candidates in bin b + 2 )     # Laplace α=1
           w_i = AUC_i / Σ_j AUC_j

[PER QUESTION]
  score(e) = Σ_i  w_i · ln( ( c_i[bin(s_i(e))] ) / ( 1 − c_i[bin(s_i(e))] ) )
  answer = argmax_e score(e)
```
- Handles `s_i=0` gracefully (maps to a low-but-nonzero bin, so log is finite — the PoE zero-problem is avoided).
- Fixes #1: an overlap=1 name-match maps to whatever the bench says overlap=1 is worth as *probability*, not a ~50 raw points.
- Fixes #2: margins become probability gaps; add temperature sharpening (Sec 5.4, `T=0.3`) if you want more decisiveness.
- Fixes #3: count's calibrated `c` collapses to ≈prior for all but true-rare-hubs.
- Deterministic-Rust: two precomputed `[usize]→f64` lookup tables + an ln. Ten minutes of work.

### Scheme 3 — Hub-corrected personalized PageRank, fused

```
[OFFLINE]  π = global PageRank (uniform teleport, c=0.85), converged, stored per node.

[PER QUESTION]
  v = 1/|Q| over query entities (or ∝ word-overlap weights)
  iterate  π_q ← (1−c)·v + c·Pᵀ·π_q   for 60 iterations           # P = degree-normalized
  ppq(e) = ln π_q(e) − ln π(e)                                     # hub-corrected
  fuse:   treat ppq as a 7th signal in Scheme 1, or as a hard expansion:
          E'' = E' ∪ { top-30 nodes by ppq }  (fixes #4: structural recall)
```
**Use as the fix for #4 (the 6 conn=0 cases) and the deep anchor for #2**, not as a standalone ranker — it has no lexical/type knowledge, so it must be fused with Scheme 1's lists.

---

## 7. Failure-mode × scheme matrix

| Failure mode (count) | Linear+argmax (now) | Scheme 1 (RRF+filter) | Scheme 2 (calibrated log-odds) | Scheme 3 (PPR) |
|---|---|---|---|---|
| #1 Overlap dominance (21) | loses 21 | **filter F1+F2 veto + bounded overlap** — recovers est. 15–19 | **recalibrated overlap probability** — recovers est. 14–18 | no (no lexical signal) |
| #2 Near-tie noise (18) | loses 18 | partial (inter-list) — est. 9–13 | **strong** (probability gaps + T-sharpen) — est. 12–16 | supports (diffusion margin) |
| #3 Hub/degree (5) | loses 5 | **freq list bounded/capped** — est. 4 | **count calibration collapses** — est. 4 | **π/π_q ratio cancels hubs** — est. 4 |
| #4 Structural conn=0 (6) | loses 6 | no | no | **only fix** — est. 3–5 |
| **Combined expected recovery** | — | **33–40 / 50** | **32–38 / 50** | **3–5 / 6 (of #4)** |

Recommended order of experimentation (each is independent and cheap):
1. **Scheme 3 first** (fixes the 6 structural cases that nothing else touches) → fuse as extra list.
2. **Scheme 1** (biggest lever: 21+18+5 cases).
3. **Scheme 2 or conformal p-value fusion** (Sec 5.5) as the scoring layer under the RRF (best of both: bounded + calibrated).

---

## 8. References

**[verified — fetched this session]**
- Cormack, Clarke, Buettcher (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR. PDF: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
- Sun, Bedrax-Weiss, Cohen (2019). *PullNet: Open Domain QA with Iterative Retrieval on KBs and Text.* ACL. arXiv:1904.09537 — https://arxiv.org/abs/1904.09537
- Penzkofer, Shi, Bulling (2024). *VSA4VQA: Scaling a VSA to Visual QA on Natural Images.* CogSci. arXiv:2405.03852 — https://arxiv.org/abs/2405.03852
- Kleyko et al. *A Survey on HDC aka VSA, Part I: Models and Data Transformations.* ACM Computing Surveys. arXiv:2111.06077 — https://arxiv.org/abs/2111.06077
- Kleyko et al. *…Part II: Applications, Cognitive Models, Challenges.* arXiv:2112.15424 — https://arxiv.org/abs/2112.15424
- Vovk & Wang (2012/2019). *Combining p-values via averaging.* arXiv:1212.4966 — https://arxiv.org/abs/1212.4966
- Wilson (2021). *The Lévy combination test* (HMP/CCT family). arXiv:2105.01501 — https://arxiv.org/abs/2105.01501
- Gasparin, Wang, Ramdas (2024). *Combining exchangeable p-values.* PNAS. arXiv:2404.03484 — https://arxiv.org/abs/2404.03484
- Hinton, Vinyals, Dean (2015). *Distilling the Knowledge in a Neural Network* (temperature sharpening). arXiv:1503.02531 — https://arxiv.org/abs/1503.02531
- Angelopoulos & Bates (2021). *A Gentle Introduction to Conformal Prediction.* arXiv:2107.07511 — https://arxiv.org/abs/2107.07511
- Cai et al. (2021). *GRecX: GNN-based Recommendation benchmark.* arXiv:2111.10342 — https://arxiv.org/abs/2111.10342 (the only indexed "Grecx"; likely not the intended source)

**[memory — titles/metadata reliable, exact IDs not re-verified]**
- Jeh & Widom (2003). *Scaling Personalized Web Search.* WWW. (PPR formulation; also SimRank, KDD 2002.)
- Haveliwala (2002). *Topic-sensitive PageRank.* WWW.
- Tong, Faloutsos, Pan (2006). *Fast Random Walk with Restart and its Applications.* KDD.
- Kleinberg (1999). *Authoritative Sources in a Hyperlinked Environment.* JACM (HITS).
- Katz (1953). *A new status index derived from sociometric analysis.* Psychometrika.
- Milne & Witten (2008). *Learning to Link with Wikipedia.* CIKM. (PPR/stationary-normalized entity prior — the hub-debias precedent.)
- Dwork, Kumar, Naor, Sivakumar (2001). *Rank Aggregation Methods for the Web.* WWW. (Borda/Copeland/Kemeny survey.)
- Fagin, Kumar, Sivakumar (2003). *Comparing Top k Lists.* SODA.
- Hinton (2002). *Training Products of Experts by Minimizing Contrastive Divergence.* Neural Computation.
- Fox & Shaw (1994). *Combination of Multiple Searches.* NIST TREC-2. (CombSUM/CombMNZ.)
- Kemeny (1959). *Mathematics without numbers.* Daedalus.
- Vovk, Gammerman, Shafer (2005). *Algorithmic Learning in a Random World.* Springer. (Conformal prediction.)
- Dong et al. (2015). *Question Answering Over Freebase with Multi-Column CNNs.* ACL.
- Kag, Salim, Gavish (2018). *Graph Based Question Answering.* NAACL. (GQA — two-phase PPR + type/relation re-rank; anthology ID not confirmed this session.)

---

## 9. Methodological note on this memo

Web access worked for the arXiv search UI, OpenAlex/Crossref APIs, the ACL Anthology, and the RRF PDF; Google/Bing/Semantic Scholar were blocked or rate-limited. All **[verified]** items above were confirmed against live pages during this session (arXiv abstract pages, PDF, API JSON). Everything else is explicitly **[memory]**. The failure-mode counts, the N(0,1/√2048) noise claim, and the 12.58% percentile-normalization baseline are taken from your brief and treated as ground truth; the recovery estimates in §7 are my projections from the mechanism analysis, not measured results — treat them as hypotheses to confirm on the bench.
