# AGENTS.md — AXIOM Session Onboarding

> **Purpose:** This file is auto-loaded by opencode at session start so ANY new
> agent can understand this project in ~2 minutes and start helping immediately.
> It is the ONLY file guaranteed to be read; everything else must be reachable
> from here.

> **⚠️ READ THIS FIRST:** `docs/STATUS_VISION_ASSESSMENT.md` is the ground-truth
> reality check (2026-08-13). It records the honest verdict: **no scientific
> breakthrough** — the project empirically confirmed the limits of pure
> deterministic VSA for language. Older docs may still contain aspirational/
> overclaimed language ("AI เปลี่ยนโลก", "Nobel-level"); treat those as
> superseded by the assessment doc.

## What AXIOM Is (30-second version)

A **deterministic, zero-training, CPU-only** question-answering + reasoning system
in Rust (18 crates) built on hyperdimensional vectors (VSA, d=2048 random bipolar),
cellular sheaf routing, and continuous Hopfield memory over a knowledge graph.
It ingests Wikipedia/evidence → decomposes to triples → ranks entities → answers.
**Not** an LLM. No gradients, no sampling.

**Current state (v18c):** TriviaQA candidate **24.84%** · **candidate_exact 16.35%**
· **candidate_f1 18.24%** · answer_entity_recall **76.73%** · **strict_recall 55.35%**
· avg_latency **66.59ms** (⚡ 17.5% faster on CPU).
**Transmuted Algebraic CPU Engine:** 10,000 Vocab · **100.0% (26/26)** Factual Recall ·
**694.7 tok/s** on 1 CPU Thread · **2.76 MB** L3-Cache Footprint.

## MUST READ — in this order (source of truth)

| # | File | Why |
|---|------|-----|
| 1 | `docs/AGENT_HANDOFF.md` | Current state, quick-start for new environment, next steps |
| 2 | `docs/TRANSMUTED_WEIGHT_ARCHITECTURE.md` | **Master spec:** Two-Tier Transmuted Architecture, ZCA Torus Phasor, Gated Sheaf, Hopfield, HiPPO |
| 3 | `docs/ROADMAP.md` | Task board — pick highest-priority `[ ]` whose deps are `[x]` |
| 4 | `docs/PROGRESS_LOG.md` | Chronological journal — what was tried, results |
| 5 | `docs/LESSONS_LEARNED.md` | **Anti-pattern registry — READ BEFORE ANY EXPERIMENT** |
| 6 | `docs/ROOT_CAUSE_ANALYSIS.md` | Why the selection gap exists (read before touching scoring) |
| 7 | `docs/AGENT_WORKFLOW.md` | Operating procedure (bench → keep/revert → update docs → commit) |

## Startup Procedure (do this every session)

```bash
# 1. Verify tests pass (all 18 crates)
cargo test -p tle-axiom-gen -p tle-vsa-lm -p tle-vsa

# 2. Verify baseline (full 318-record bench, ~70s)
cargo build --release -p tle-axiom-gen
./target/release/triviaqa-bench data/triviaqa/qa/verified-wikipedia-dev.json \
  - data/triviaqa/evidence/wikipedia
#   expect: candidate 24.5-24.8%, exact 16.35%, strict_recall 55.35%

# 3. Verify Transmuted CPU Model (10k vocab, ~690 tok/s)
python3 scripts/build_real_scale_model.py data/models/real_transmuted_10k.twotier
./target/release/vsalm-transmute data/models/real_transmuted_10k.twotier

# 4. Check working tree is clean
git status
```

## HARD RULES (violating these wastes hours)

1. **NEVER claim improvement without running the full 318-record bench.**
   Quick bench: `AXIOM_TRIVIA_LIMIT=50 ./target/release/triviaqa-bench ...`
2. **NEVER re-try documented failures** (see LESSONS_LEARNED.md §TL;DR):
   weight-tuning linear sum, percentile/rank/conformal fusion, IEF, DDTree,
   semantic-in-scoring, relation-heuristic type-veto, graph-surface filters.
3. **Keep-gate:** KEEP a change only if candidate (primary) AND recall
   (secondary) both don't regress. Per-record diffs are HashMap-noise —
   decide on aggregates, stable across 3+ runs.
4. **Env-gate every experiment** (`AXIOM_W_*`, `AXIOM_RANK=...`) so you can
   A/B and revert without code churn. Default = best-known (24.53%).
5. **After every task:** update ROADMAP + PROGRESS_LOG + AGENT_HANDOFF, then
   commit (code + docs together).
6. After touching `tle-vsa`: run FULL workspace `cargo test` (other crates
   broke silently once).

## What WORKS vs what DOESN'T (so you don't rediscover)

**Works (bench-verified):** query-entity punctuation fix (+2.2), hub-corrected
PPR relative-PageRank signal (+0.63), proper-noun boundary precision
(+4.72 recall), subject resolution / copula handling (+0.32), overlap weight
0.05 (+0.63), **M1 conditional overlap-veto** (+0.31, env AXIOM_V1_M1 — overlap
only counts when the candidate is structurally connected to the query).

**Does NOT work:** any fusion redesign of the 6 signals (all normalize away
magnitude gaps — 12.58% hit twice). The gap needs **new signals** + **cleaner
decomposition**, not re-combining existing ones.

## Quick commands

```bash
cargo run --release -p tle-deepman                    # interactive REPL
./target/release/triviaqa-bench data/triviaqa/qa/verified-wikipedia-dev.json - data/triviaqa/evidence/wikipedia  # full bench
AXIOM_TRIVIA_LIMIT=50 ./target/release/triviaqa-bench ...   # quick bench
AXIOM_TRIVIA_DEBUG=1 ...   # top-5 per-record diagnostics (debug gate)
cargo test -p tle-axiom-gen -p tle-vsa-lm -p tle-vsa   # core tests
```
