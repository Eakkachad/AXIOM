# Connectivity Expansion + Conditional Count — Research Report

> 2026-08-12. Fixes for AXIOM Mode C (gold conn=0, ~20% of failures) and
> Mode B (count dominance, ~15%). Sources verified against arXiv full texts
> where marked [V]; [C] = canonical citation; [I] = AXIOM-specific inference.

## Anchor papers (verified)

- **CatRAG** arXiv:2602.01965 [V] — "Static Graph Fallacy" (fixed transition
  matrix): 3 mechanisms — Symbolic Anchoring (weak teleport seeds, ε=0.2,
  weighted by inverse passage count; ablation −3.2% HoVer), Query-Aware Dynamic
  Edge Weighting (prune seed edges to top-K by query↔fact sim; LLM tier × static
  weight, forward-only), Key-Fact Passage Enhancement (β=2.5 boost to passages
  with verified seed triples; −1.4 on structured 2Wiki). Headline = Full Chain
  Retrieval 34.6 vs 30.5, not raw recall.
- **QASA** arXiv:2606.30133 [V] — per-iteration gate: `A(v)={u:(u,v)∈E, r(u)>τ,
  r(u)>r(v)}` (monotonicity kills cycles), `Δr(v)=α·σ(v)·Σr(u)` with
  `σ(v)=max(cos(e_v,e_q),0)` — gate = query-relevance of TARGET node. Params
  τ=0.01, α=0.7, T=3. Deterministic, one Cypher query. Gate ablation +7.4 F1
  MuSiQue, +3.6 2Wiki, latency ÷1.5-4.9. **Depth sweep: T=1→2 is the biggest
  lever (+18.8 2Wiki); T=4 degrades** (pulls irrelevant entities).
- **QAFD-RAG** arXiv:2605.18775 [V] — edge weight `w_uv = H_sim(u,v)·(a + b·
  (H_sim(u,q)+H_sim(v,q)))`, a=1, b=¼, flow-diffusion solver. Formula = the
  "gold standard" query-conditioned edge weight; solver not practical for AXIOM.
- **OPI** arXiv:2606.28076 [V] — **most on-point**: relation-centric ontology
  (head/tail TYPE constraints per relation, e.g. born_in: Person→Place),
  predict answer type → map to compatible FINAL-HOP relations → bidirectional
  retrieval (topic-side prefix + answer-side final-hop matching). "Suppresses
  noisy mixed-type expansion." **+4.6 Hit@1/+5.0 F1 WebQSP, +8.9/+3.3 CWQ.**

## Mode C — structural blind spot (gold conn=0)

Verified answer: **don't widen blindly — widen only along answer-type-compatible
final hops** (OPI), from restricted bridge anchors, with monotonicity/flow guard
(QASA τ + r(u)>r(v); PathRAG flow-pruning). AXIOM's 2-hop pass is blind (all
neighbors of all 1-hop nodes) — that's the noise source, not depth.

**Attribute/literal answers**: WebQSP (Yih ACL 2016) answers are entities OR
literals (years, numbers); MCCNN (Dong ACL 2015) — answer-type prediction is a
separate head that filters candidates (type-consistent reranked up). "How many"
= cardinality operator (COUNT), a separate code path.

### D1 (primary) — Answer-type + typed final-hop expansion (OPI-style)

1. `predict_answer_type(intent, query) -> {Entity, Person, Place, Temporal,
   Number}` — word rules ("in what year/which year/when"→Temporal, "how
   many/how much"→Number, "where"→Place, "who"→Person). Pure function.
2. `RelationKind{head_type, tail_type}` table for ~40 relation labels the
   decomposer emits (released_in/born_in/…→Place; released/created/won/…→
   Temporal for year objects). Deterministic match table.
3. Typed final-hop expansion after the 2-hop pass: for each bridge b in {query
   entities ∪ 1-hop nodes}, for each out-triple (b,r,v) with tail_type(r) ==
   predicted (Number/Temporal: v parses numeric), add v to candidates with
   `raw_typed[k] += conf·kind_weight`. QASA-style monotonicity/visited guards.
4. New signal `+ w_typed · typed_avg` (env `AXIOM_W_TYPED`, start 1.5-2.0).
   Type-shape veto optional for Number/Temporal.
~150 LOC, no new deps. Recall monotone (expansion only adds). Expected: biggest
single gain; upper bound +3-5pp [I].

## Mode B — count dominance

BM25/RSJ (Robertson-Zaragoza 2009; Robertson-Spärck-Jones 1976) gives the
structural diagnosis: AXIOM's `heur = 0.2·raw_count` sums over ALL triples
regardless of query = a **document-length statistic, not a relevance
statistic**. Three structural fixes:
1. **Query-conditioning**: count only triples reachable from query entities
   (raw_conn_count + raw_2hop_count — already computed).
2. **TF saturation** k1: `BM25_sat(c) = c(k1+1)/(c+k1)`, k1≈2-3 — keeps
   evidence monotonic, caps the hub's multiplicative advantage.
3. **Milne-Witten relative ratio**: `count_ratio(e) = count_cond(e)/count(e)`
   — the count analog of the relative PPR that already worked (+0.63). A
   topic/alias entity has huge count but small relevant FRACTION; a
   genuinely-frequent answer has a high fraction.

IEF failed before because it discarded absolute evidence; these keep it inside
the connected component. **Ship AFTER D1** (D2's conditional count zeroes conn=0
golds; D1 gives them typed connectivity first).

### D2 — Conditional + saturated count
`heur = w_count·BM25_sat(count_cond) + w_ratio·count_ratio − len_pen +
cap_bonus + det_pen`. Env `AXIOM_W_RATIO`. Expected +1.5-3pp [I].

### D3 (tertiary) — QASA-style query-aware PPR gate
Gate `σ[v] = max(0, Σ_{w∈q} idf(w)·𝟙[w ∈ tokens(fact_texts[v])])` (lexical, NOT
VSA cosine — noise), multiply PPR mass-share by `σ[v]^γ` (γ≈0.5-0.8); + CatRAG
symbolic-anchor teleport ε=0.2. ~20 LOC, deterministic. Makes D1/D2 reach
further. Expected +0.5-1.5pp [I].

## Recommended sequence
D1 (typed expansion) → bench → D2 (conditional count) → bench → D3 (gated PPR)
only if plateau. All additive env-gated signals (the documented recipe that
works: new signals + gate fixes, never re-fusion).

## References
CatRAG 2602.01965; QASA 2606.30133; QAFD-RAG 2605.18775; OPI 2606.28076;
PathRAG 2502.14902; HippoRAG 2405.14831; HippoRAG 2 2502.14802; SMART
2304.12395; MuSiQue 2108.00573; 2Wiki 2011.01060; topic-sensitive PPR
(Haveliwala WWW 2002 [C]); Milne-Witten CIKM 2008 [C]; PRA (Lao EMNLP 2011 [C]);
spreading activation (Crestani CSUR 1997 [C]); WebQSP (Yih ACL 2016 [C]);
MCCNN (Dong ACL 2015 [C]); BM25/RSJ (Robertson & Zaragoza 2009 [C]);
BM25F (Robertson-Zaragoza-Taylor CIKM 2004 [C]).
