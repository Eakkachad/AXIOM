# AXIOM — Algebraic neXt-token Inference On Memory

> Solve for X. No training required.

**AXIOM** is a deterministic, CPU-only, zero-training question-answering and
reasoning system built on hyperdimensional vectors (VSA) + a knowledge graph.
Every number below is measured on a real benchmark — see **Honest Status** for
what the numbers do and do NOT mean.

## Status (2026-08-11 · v15)

| Metric | Value | What it measures |
|--------|:---:|------|
| TriviaQA candidate answer | **24.53%** | AXIOM picks the correct entity |
| Answer entity recall | **76.10%** | Correct answer is a graph node |
| Substring accuracy | **23.27%** | Answer appears in generated sentence |
| Evidence answer recall | **99.69%** | Answer exists in ingested evidence |
| Latency | ~100ms | Full evidence ingest + answer |

**The honest reading:** with evidence pre-ingested (recall 99.69%), AXIOM can
*find* the right answer 76% of the time but *select* it only 24.53%. The
52pt gap between finding and selecting is the open research problem. This is a
**pipeline diagnostic on evidence-ingested data — NOT an open-domain benchmark
score.** Standard TriviaQA EM/F1 equivalents would be lower.

## What Works (verified)

- **1-hop & some multi-hop reasoning** on taught facts:
```
AXIOM> /teach sky is blue
AXIOM> /teach blue has short wavelength
AXIOM> why is the sky blue?
  A sky is blue, because the blue has short wavelength. [633µs]
```
- **Deterministic**: same input → same output, always. No sampling.
- **Instant learning**: `/teach` a fact → answerable in µs.
- **Zero training**: no gradients, no backprop, CPU-only, ~18MB.

## What Does NOT Work Yet (honest)

- **"does cat have a heart?" fails** when taught `animals have hearts` —
  the system does not reliably chain `cat is an animal` → `animals have hearts`.
  It is key-value lookup + shallow traversal, **not** real compositional reasoning.
- **Open-domain QA is weak** (24.53% candidate). The 52pt gap is the bottleneck.
- **No LLM-style conversation.** Output is formulaic; it cannot free-form chat.
- **VSA cosine with a random codebook is near-noise** (N(0,1/√2048)) — it cannot
  be a primary signal; it only works as a weak tiebreaker.

## Recent Progress (v14 → v15, all bench-verified)

| Change | Type | Result |
|--------|------|--------|
| Proper-noun boundary precision | decomposition | recall +4.72pt |
| Query-entity punctuation fix | bug-fix at a gate | candidate +2.2pt |
| Hub-corrected PPR (relative PageRank) | new structural signal | candidate +0.63pt |
| Subject resolution (copula handling) | decomposition | candidate +0.32pt |
| Overlap weight calibration | tuning | candidate +0.63pt |

**What did NOT work (6 documented rounds):** weight tuning (flat local optimum),
percentile/rank fusion, conformal p-value fusion, IEF, relation-heuristic type
veto, graph-surface filters. All reverted and recorded in
`docs/LESSONS_LEARNED.md` so they are never re-tried. Full analysis in
`docs/SESSION_RESEARCH_SUMMARY.md`.

## Try It

```bash
cargo build --release

# Interactive REPL
cargo run --release -p tle-deepman

# TriviaQA diagnostic (evidence-ingested)
./target/release/triviaqa-bench data/triviaqa/qa/verified-wikipedia-dev.json \
  - data/triviaqa/evidence/wikipedia

# Tests (75+ pass)
cargo test -p tle-axiom-gen
```

## Architecture (high level)

```
Input → [Intent classify] → [Query entities] → [extract_answer: rank entities]
      → [Knowledge Graph: triples + relations + PPR] → [Linearize] → Output
Signals used to rank: connectivity, role bias, 2-hop, overlap,
VSA cosine (weak), heuristics, hub-corrected PPR.
```

## Honest Limitations

- **Quality gap vs LLMs** — correct but formulaic; no free-form conversation.
- **TriviaQA numbers are internal diagnostics** on evidence-ingested data, not
  standard open-domain EM/F1. Treat them as pipeline health, not a leaderboard.
- **Knowledge must be taught or ingested** — knows nothing by default.
- **Decomposition still noisy** (~30% junk) — the main lever for recall.
- **Answer selection is the open gap** (76% find → 24% select) — the active
  research focus. See `docs/RANKING_RESEARCH_SYNTHESIS.md`.
- **English only** (Thai planned).
- **No creative writing** — composes facts, cannot tell stories.

## Research Docs

- `docs/AGENT_HANDOFF.md` — current state + next steps (read first)
- `docs/LESSONS_LEARNED.md` — anti-pattern registry (what NOT to re-try)
- `docs/ROOT_CAUSE_ANALYSIS.md` — cross-layer analysis of the selection gap
- `docs/SESSION_RESEARCH_SUMMARY.md` — this session's 6 negatives + 4 gains
- `docs/RANKING_RESEARCH_SYNTHESIS.md` — deep research on ranking math
- `docs/RESEARCH_PAPER_DRAFT.md` — TBA + AXIOM-Gen paper draft

## License

MIT
