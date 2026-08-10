# AXIOM AGENT WORKFLOW — Continuous Development Operating Procedure

> How an agent (or the user) continues AXIOM development across sessions without
> losing context. Follow this EXACT order every time.

## 1. Session Start (2 min)

1. Read `docs/AGENT_HANDOFF.md` → current state + next steps
2. Read `docs/ROADMAP.md` → pick the highest-priority `[ ]` task whose deps are `[x]`
3. Read `docs/PROGRESS_LOG.md` → confirm what was already tried (avoid re-trying failures)
4. Run baseline: `cargo test -p tle-axiom-gen -p tle-vsa-lm -p tle-vsa`

## 2. Task Execution Loop

For each task:

```
1. Mark task [~] in ROADMAP.md
2. Implement the change
3. cargo test -p tle-axiom-gen -p tle-vsa-lm -p tle-vsa   (must pass)
4. cargo build --release -p tle-axiom-gen
5. Quick bench: AXIOM_TRIVIA_LIMIT=50 ./target/release/triviaqa-bench ...
6. If promising, FULL bench: ./target/release/triviaqa-bench ...
7. Compare metrics vs baseline in ROADMAP.md. KEEP only if no regression
   on the primary metric (candidate) AND secondary (recall).
8. If regressed → revert (git checkout) and document WHY in ROADMAP "Result"
9. If improved → mark [x], update metrics in ROADMAP.md
```

## 3. Progress Update (after each task — mandatory)

After EVERY task, update ALL THREE:

| File | What to update |
|------|----------------|
| `docs/ROADMAP.md` | mark task `[x]`/`[~]`/`[!]`, record Result line |
| `docs/PROGRESS_LOG.md` | append new entry at TOP (date, what, metrics) |
| `docs/AGENT_HANDOFF.md` | refresh "Current State" + "Next steps" sections |

## 4. Commit Convention

- One commit per task (or per coherent change)
- Message: `vNN: <task> — <result summary>`
- Include before/after metrics in commit body
- `git add docs/` alongside code — the roadmap/log ARE part of the deliverable

## 5. Session End (3 min)

1. Ensure working tree is clean (all committed)
2. Verify ROADMAP reflects true status (no `[~]` left dangling → mark `[x]` or `[!]`)
3. Update AGENT_HANDOFF "Last updated" + state table
4. Write final PROGRESS_LOG entry

## 6. Anti-Patterns (NEVER do these)

- ❌ Re-try a task marked `[!]` or documented as failed, without a NEW approach
- ❌ Change `extract_answer` weights blindly (documented optimal)
- ❌ Use DDTree as primary answer selector (4 documented failures)
- ❌ Substring entity consolidation (documented regression)
- ❌ Modify code without running the bench (no hallucinated improvements)
- ❌ Leave ROADMAP/PROGRESS_LOG stale — they are the source of truth

## 7. Benchmark Reference

```bash
# Quick iteration (fast)
AXIOM_TRIVIA_LIMIT=50 ./target/release/triviaqa-bench data/triviaqa/qa/verified-wikipedia-dev.json - data/triviaqa/evidence/wikipedia

# Full 318 (definitive, ~3-5 min)
./target/release/triviaqa-bench data/triviaqa/qa/verified-wikipedia-dev.json - data/triviaqa/evidence/wikipedia

# Wikipedia QA
./target/release/vsalm-wiki https://en.wikipedia.org/wiki/Paris https://en.wikipedia.org/wiki/France

# VSA-LM scale
./target/release/vsalm-scale data/wiki_train.txt 5000 0.8
```
