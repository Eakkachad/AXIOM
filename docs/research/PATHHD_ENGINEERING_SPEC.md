# PathHD → AXIOM Engineering Spec (arXiv:2512.09369)

> Extracted 2026-08-12 from the full paper (HTML v2). Complete algorithm spec
> for a CPU-only, deterministic, zero-training Rust port WITHOUT any LLM.

## Verdict: what transfers to d=2048 random-bipolar

| Ingredient | Carries over? | Verdict |
|---|---|---|
| GHRR block-unitary binding | needs real O(m) blocks | **build faithfully** (real 4×4, D=128, d=2048) |
| Hadamard bind (tle-vsa::ops::bind) | commutative | **insufficient** (Table 3: 2.3-2.7pt gap) |
| Circular convolution | commutative + norm drift | **not recommended** |
| Positional-permutation bind | order-sensitive, bipolar | cheap fallback, untested by paper |
| Blockwise cosine (Eq 4) | ≡ flat cosine when normalized | implement as block loop (=flat cosine) |
| Calibrated score α·IDF − β·λ^|z| | exact formula | adopt Table-11 hyperparams |
| Plan-based query encoding | yes (+1.7-2.8pt) | **adopt**; text-projection needs SBERT |
| Top-K=3 pruning | yes | adopt (Table 5) |
| Distractor bound (Thm 1) | Rademacher = AXIOM's setting | d=2048 sufficient for M≤10³, ε≈0.1 |

**Paper bug flagged:** the suggested `diag(e^{iφ})` block family is commuting
(contradicts their non-commutativity claim). Use products of TWO Householder
reflections (real O(4)) — the paper's own alternative.

## 1. GHRR binding

- Symbol x → block vector `v_x = [A_1(x); …; A_D(x)]`, `A_j ∈ U(m)`, flat dim
  `d = D·m²`. m=4 fixed; D = 2048/16 = 128 blocks.
- **Real adaptation**: O(4) blocks (equations identical with `tr(AᵀB)`, halves
  memory/compute). Optional d=4096 ⇒ D=256 if needed.
- Deterministic construction: seed = FNV1a(name) ^ base_seed ^ (j·0x9E3779B9…),
  ChaCha20 → gaussians → `A = H(v)·H(w)` (two Householder reflections,
  `H(u) = I − 2uuᵀ/‖u‖²`). Orthogonal ⇒ ‖A‖_F = √m exactly; products of
  orthogonal blocks stay orthogonal ⇒ **no variance blow-up under binding depth**
  (the reason GHRR beats circular convolution).
- Binding `X ⊛ Y = [X_1Y_1; …; X_DY_D]` (blockwise matrix product, left-to-right)
  + blockwise Frobenius normalization. Non-commutative: `X_jY_j ≠ Y_jX_j`.
- Unbind: `X ≈ Z Y*`, `Y ≈ X* Z` (unitary inverse).

## 2. Path encoding (Eq 3)

`v_z = v_{r1} ⊛ v_{r2} ⊛ … ⊛ v_{rℓ}` — relations only (not entities); entity
anchoring happens at enumeration/adjudication. Order & direction sensitive.

## 3. Query encoding: the relation plan (Table 12: +1.7-2.8pt)

Plan = a relation-sequence schema `z_q`, NOT text. Deterministic derivation:
1. Topic entity + intent from extract_query_entities / classify_intent.
2. Phrase→relation mapping: tokenize relation names; `R_q = {r : W(r) ∩ C(q) ≠ ∅}`
   (exact + prefix), fallback = relations adjacent to topic entity.
3. Relation-schema graph: relation-id adjacency where type-consistent.
4. BFS enumerate plans from each r ∈ R_q, depth ≤ L_max (=3), beam B.
5. Select plan: argmax `Σ_{r∈z} |W(r) ∩ C(q)|`, tiebreak shorter.
6. `v_q ← BindPath(z_q)`.

## 4. Blockwise cosine (Eq 4)

`sim(X,Y) = (1/D) Σ_j tr(X_jᵀY_j)/(‖X_j‖_F‖Y_j‖_F)`. When blocks are unit-Frobenius
this ≡ flat cosine of the 2048-vector. Blockwise framing = variance-consistency
device (keeps capacity bound d = D·m² valid), not a separate metric.

## 5. Calibrated score (Eq 5-6)

`s(z) = sim(v_q, v_z) + α·IDF(z) − β·λ^|z|`
`IDF(z) = log(1 + N_train/(1 + freq(schema(z))))`
Table 11 hyperparams: WebQSP (0.2,0.1,0.8), CWQ (0.3,0.1,0.8), GrailQA
(0.2,0.2,0.8). Note λ<1 ⇒ longer paths penalized LESS (counteracts accumulated
binding noise). AXIOM IDF: corpus = evidence docs; freq(schema) = number of
distinct topic entities instantiating it.

## 6. Top-K pruning (Table 5)

Score ALL candidates, keep K=3, adjudicate only those. Evidence: no-prune
blurs decisions (near-duplicate/noisy paths). K=3 best (86.2/71.5), K=5 86.1.

## 7. Distractor bound (Prop 1 / Cor 1 / Thm 1)

Rademacher/bipolar case (AXIOM's exact setting, c=1/2): `d ≥ (2/ε²)ln(2M/δ)`.
At d=2048, δ=0.01: M=200 ⇒ ε≈0.102; matched path at cos≈1, gold path well above
0.1. **d=2048 comfortably sufficient.**

## 8. Deterministic adjudicator (replaces LLM, worth +0.7-0.8pt in paper)

1. Intent-consistency veto: terminal relation of top path must produce the
   predicted answer type (Who→person-producing, Where→location-producing,
   When→date-producing); swap if a lower top-K path passes.
2. Entity-evidence second pass: run extract_answer signals restricted to the
   top-K end entities; `final = w_p·s_calibrated + w_e·entity_score`.
3. Near-tie direction rule: prefer schema direction agreeing with intent.

## 9. Rust module plan — new crate `tle-ghrr` (do NOT modify tle-vsa)

```
crates/tle-ghrr/
  src/block.rs      # [[f32;4];4]: mat_mul, frob_inner/norm, householder, random_orthogonal_block(seed)
  src/vector.rs     # GhrrVector{blocks: Vec<[[f32;4];4]>} (8KB); bind_path, blockwise_cosine
  src/codebook.rs   # GhrrCodebook HashMap<String,GhrrVector>; FNV1a^base^(j·φ) seeds
  src/schema.rs     # RelationSchemaGraph, enumerate_plans(BFS,L_max,B), phrase_to_relations, select_plan
  src/retrieval.rs  # Idftable, calibrated_score, rank_candidates, top_k(K=3)
  src/adjudicate.rs # intent-veto + entity-evidence second pass + direction rule
```
Complexity at AXIOM scale (graph ~100-1000 nodes, N≈50-200, ℓ≤4): ~10 Mflops
+ 0.4 M MACs ≈ well under 1 ms release. Compose at graph level: reuse
tle-axiom-gen beam_search path enumeration; re-rank by relation-schema.

Implementation order (each gated): 1 block.rs+vector.rs primitives + tests
(orthogonality, determinism, order-sensitivity cos(r1⊛r2, r2⊛r1)≈0) → 2 codebook
→ 3 schema/plan → 4 retrieval/calibration/top-k → 5 adjudicate → 6 integrate
in generate() behind `AXIOM_PATHHD=1` → 7 bench (quick then full 318, keep-gate
candidate+recall on STRICT metric).

## Flagged non-transfers
Complex U(4)→real O(4); paper's diag(e^{iφ}) family is commuting → Householder;
blockwise-vs-flat cosine gain → none when normalized; text-projection query →
SBERT, use plan-based; LLM adjudication → deterministic §8; d peak 3k-6k →
d=2048 ok (one-line knob to 4096); IDF "training questions" → evidence-frequency.
