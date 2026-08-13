# Chat-like-an-LLM Research Strategy — 2026-08-13

> User direction (2026-08-13): deep-research katgpt + literature, then build an
> ORDERED tree of falsifiable hypotheses and prove them step-by-step. Be
> willing to think outside the box. Sub-agent research (3 tracks) + G0 probe.

## The honest verdict (research + evidence)

**A deterministic, zero-training, CPU-only system cannot reach GPT-3.5+ chat
quality.** Verified literature: the best non-neural next-token predictor is the
∞-gram (5T tokens) at **47%** — and its own paper says it is "harmful for
generation" (digresses). Kneser-Ney 5-gram = 67.6 PPL vs Transformer-XL 21.8.
gzip-as-generator is miscalibrated. Retrieval chatbots (Cleverbot-class) recycle
human utterances. **The realistic target is a "fluent domain-expert":**
deterministic, graph-grounded, attributable, corpus-bounded. The best backbone
mix is **retrieval + template/graph + statistical filler, always
candidate-restricted (never full-vocab VSA decode).**

## katgpt-rs findings (deep forensic, 30 crates)

- **NO train-free generative mechanism exists in katgpt.** Its generative power
  is wholly neural. But it provides train-free ASSISTIVE cores + engineering:
  - **Engram** O(1) associative memory + sigmoid gate → usable as a
    content-addressed LM (repurposable for AXIOM).
  - **MatchLengthScorer** (compression_drafter.rs) → train-free corpus-plausible
    reranker (compression/MDL prior); GOAT failure was only at an extreme
    ≥3×-compression threshold, irrelevant to reranking.
  - **ConstraintPruner + DDTree chain_seed** → force grammar-valid generation
    (prune-before-add); the cheapest "sounds like an LLM" surface win.
  - **KARC** closed-form ridge reservoir — periodic-blind, auxiliary only.
- Implication: katgpt confirms the ceiling is structural (n-gram/statistical +
  grammar-constrained), not a missing trick.

## Ordered hypothesis tree (from strategy agent)

| # | Hypothesis | Capability | Cost | Pass bar | Status |
|---|---|---|---|---|---|
| **G0** | Backbone ceiling probe (KN-5 vs VSA-LM same split) | C4 fluency | 4-8h | ≥30% / recall | **DONE** |
| H1 | Grammar finisher post-processor | C8 | 4-8h | err<10% | pending |
| H2 | Turn-memory wiring (δ-Mem) into axiom-chat | C7 | 6-10h | ref≥80% | pending |
| H3 | KN-5 shortlist + VSA rerank beats VSA-LM | C4 | 6-10h | TEST≥20% | **next** |
| H4 | Retrieval-augmented generation (RAG) | C1+C3 | 8-16h | R@1≥50% | pending |
| H5 | Corpus scaling 30-50MB for filler | C4 | 6-12h | ≥30% | pending |
| H6 | Template extraction + variation | C6 | 8-12h | ≥60% distinct | pending |
| H7 | Hybrid planner (graph+templates+KN-5) beats pure VSA | C3-C5 | 12-20h | ≥70% H2H | pending |
| H8 | Offline-LLM distillation into index | C4 | 12-24h | ≥35% ≤100MB | long-shot |

## G0 result (DONE, measured 2026-08-13)

`kn5-acc` (new bin): Kneser-Ney 5-gram on wiki_train.txt, SAME split as
vsalm-scale (2400 train / 600 test, vocab 6432):

| metric | KN-5 | VSA-LM |
|---|---|---|
| TEST full-vocab argmax | **16.7%** | 11.0% |
| top-32 shortlist recall | **48.0%** | 29.3% |
| top-128 shortlist recall | **62.0%** | — |

**Verdict:** KN-5 statistically dominates the VSA-LM for local coherence on this
corpus. The shortlist recall (+18.7pt) is the load-bearing number: the
candidate-restricted ceiling rises from ~15% to ~24%+. The VSA-LM's engram
shortlist should be REPLACED by a KN-5 shortlist. However 16.7% argmax is still
far from 30% — corpus is only 11MB (∞-gram needed 5T tokens for 47%), so
statistics alone caps here; retrieval+template+graph (H4/H7) carries the actual
"chat" quality.

## H3 result (DONE, 2026-08-13) — KN-5 as the filler, NOT fused

Integrated a KN-5 shortlist + KN-5 probability signal into `tle-vsa-lm`
(`kn5.rs` module; env `AXIOM_LM_KN5`, `AXIOM_LM_W_KN5`). Measured:
- KN-5 shortlist + fusion in the VSA-LM pipeline: **TEST 10.7%** (no gain —
  the fusion destroys KN-5's signal, the recurring lesson: weighted sums of
  different-scale signals lose).
- KN-5 standalone (kn5-acc probe, full-vocab argmax): **16.7%** — clearly
  better.
- TRAIN with KN-5 shortlist: 92.8% (memorizes) — confirms the model works.

**Conclusion:** do NOT fuse KN-5 into the VSA signal soup. Use KN-5 as a
STANDALONE filler / generator (argmax by KN-5 distribution), or as a hard
candidate filter whose own argmax decides. The hybrid chat (H7) should use
KN-5 directly for free-form generation.

## Next steps (in order)
1. **H3**: add a KN-5 shortlist to `tle-vsa-lm` predict (env `AXIOM_LM_KN5`),
   measure TEST — expect ~16-24%.
2. **H7**: hybrid planner in `axiom-chat` (intent → graph reasoning → template
   realization + KN-5 filler) — the "fluent domain-expert".
3. **H4**: retrieval-augmented generation (best-matching corpus sentence + graph
   adaptation).
4. **H1/H2/H6**: grammar finisher, turn memory, template variation (cheap polish).

## Files
- `crates/tle-gen/src/bin/kn5-acc.rs` — G0 probe (KN-5 next-token + recall).
- (katgpt analysis, feasibility research — from sub-agents, 2026-08-13)
