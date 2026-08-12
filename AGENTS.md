# AGENTS.md — AXIOM Session Onboarding

> **Purpose:** This file is auto-loaded by opencode at session start so ANY new
> agent can understand this project in ~2 minutes and start helping immediately.
> It is the ONLY file guaranteed to be read; everything else must be reachable
> from here.

## What AXIOM Is (30-second version)

A **deterministic, zero-training, CPU-only** question-answering + reasoning system
in Rust (18 crates) built on hyperdimensional vectors (VSA, d=2048 random bipolar)
over a knowledge graph. It ingests Wikipedia/evidence → decomposes to triples →
ranks entities → answers. **Not** an LLM. No gradients, no sampling.

**Current state (v16):** TriviaQA candidate 24.84% · entity recall 76.10% ·
substring 23.27% · evidence recall 99.69%. These are **evidence-ingested pipeline
diagnostics, NOT open-domain benchmark scores.** The active research gap: system
*finds* the answer 76% of the time but *selects* it only 24.8% (~51pt gap).

## MUST READ — in this order (source of truth)

| # | File | Why |
|---|------|-----|
| 1 | `docs/AGENT_HANDOFF.md` | Current state, what was built, next steps, gotchas |
| 2 | `docs/ROADMAP.md` | Task board — pick highest-priority `[ ]` whose deps are `[x]` |
| 3 | `docs/PROGRESS_LOG.md` | Chronological journal — what was tried, results |
| 4 | `docs/LESSONS_LEARNED.md` | **Anti-pattern registry — READ BEFORE ANY EXPERIMENT** |
| 5 | `docs/ROOT_CAUSE_ANALYSIS.md` | Why the 52pt selection gap exists (read before touching scoring) |
| 6 | `docs/AGENT_WORKFLOW.md` | Operating procedure (bench → keep/revert → update docs → commit) |

## For RESEARCH work specifically

- `docs/RESEARCH_REQUEST.md` — the 6 open research questions (what to search for)
- `docs/SESSION_RESEARCH_SUMMARY.md` — this session's 6 negative rounds + 4 real gains
- `docs/RANKING_RESEARCH_SYNTHESIS.md` — deep research on ranking math (RRF, PPR, conformal, PoE)
- `docs/research/` — paper draft, algorithm specs, prior-art analysis (katgpt-rs), ranking memo

## Startup Procedure (do this every session)

```bash
# 1. Verify tests pass
cargo test -p tle-axiom-gen -p tle-vsa-lm -p tle-vsa

# 2. Verify baseline (full 318-record bench, ~90s)
cargo build --release -p tle-axiom-gen
./target/release/triviaqa-bench data/triviaqa/qa/verified-wikipedia-dev.json \
  - data/triviaqa/evidence/wikipedia
#   expect: candidate 24.53%, recall 76.10%

# 3. Check working tree is clean (docs + code committed)
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
