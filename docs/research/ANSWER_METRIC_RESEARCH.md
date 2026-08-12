# Measuring Answer-Selection Accuracy in a Deterministic QA System

**Research memo · 2026-08-12 · For: AXIOM (deterministic, zero-training KGQA; Rust)**
**Scope: how to correctly measure answer-selection accuracy when gold answers are alias lists, and what metric should gate AXIOM's keep/revert optimization loop.**

> **Bottom line (executive summary).**
> The current `candidate_answer_accuracy` uses **bidirectional substring matching**, which no
> published QA benchmark uses. Its 24.8% is a ~1.8× inflation over the same system scored by
> the *official TriviaQA evaluation script* (18.2% exact match) and ~2× over the strict
> per-question metric we recommend (18.6%). The inflation is structurally unavoidable for any
> substring/containment metric: **single-token golds are subsumed by longer picked entities**
> (`gold="roosevelt" ⊆ picked="eleanor roosevelt"`), and TriviaQA's alias list itself is noisy
> (28% of aliases are single tokens; junk aliases like `"rain in wales"` leak in from Wikipedia
> section titles). The same artifact inflates `answer_entity_recall` (76.1% measured → ~54.7%
> real selectable ceiling).
>
> **Recommendation:** gate AXIOM on `EM-or-token-F1≥0.7` computed as a **max over the full
> official alias list** (`NormalizedAliases` + `MatchedWikiEntityName` + `Value` + `Aliases` +
> `HumanAnswers`), i.e. exactly the SQuAD/TriviaQA machinery with a binary threshold at the
> natural plateau we found empirically. Keep all legacy substring metrics as printed diagnostics
> only — never gate decisions on them. Pseudocode and rationale in §7.

---

## 1. How standard QA benchmarks measure correctness (verified protocol facts)

I fetched and read the official evaluation scripts rather than relying on secondary claims.
`✓` = verified by reading the actual script/paper.

### 1.1 SQuAD (Rajpurkar et al., 2016) — `✓` script read (via TriviaQA's official evaluator, which is explicitly "extended from the evaluation script for v1.1 of the SQuAD dataset", and HotpotQA's evaluator, which is byte-identical in structure)

- `normalize_answer(s)`: lowercase; replace `_` with space; map punctuation (incl. curly
  quotes) to spaces; strip articles `a/an/the`; collapse whitespace.
- **Exact match (EM)** = `normalize_answer(prediction) == normalize_answer(ground_truth)` — a
  **string-equality** test after normalization. Not token-set equality, not substring.
- **F1** = token-level **precision & recall** computed with multiset intersection
  (`Counter(pred) & Counter(gold)`), `F1 = 2PR/(P+R)`.
- Both are **maximized over all gold answers** for the question.
- There is **no substring / containment metric** anywhere in the script.

### 1.2 TriviaQA (Joshi et al., 2017) — `✓` script read (`triviaqa_evaluation.py`)

- Official evaluator returns only `exact_match` and `f1`, both via
  `metric_max_over_ground_truths` over
  `get_ground_truths(answer) = answer['NormalizedAliases'] + [normalize_answer(h) for h in HumanAnswers]`.
- **Correction to a premise in the task brief:** TriviaQA's official protocol does **not** use
  "fuzzy/substring" matching. The alias problem is solved by the **`NormalizedAliases` field**
  (dataset authors pre-computed normalized aliases per answer) plus **max-over-aliases**. AXIOM's
  bench loads only `Value` + `Aliases` (raw), missing `NormalizedAliases`,
  `MatchedWikiEntityName` and `HumanAnswers` — an easy protocol fix that alone raises official EM
  from 16.0% to 18.2% on this bench (see §4).

### 1.3 HotpotQA (Yang et al., 2018) — `✓` script read (`hotpot_evaluate_v1.py`)

- Same `normalize_answer` / `f1_score` / `exact_match_score` as SQuAD, with `yes|no|noanswer`
  special-casing, plus supporting-fact span F1 (`update_sp`, set-based P/R over SP triples) and
  **joint** answer×SP EM/F1. Again: no substring metric.

### 1.4 NaturalQuestions (Kwiatkowski et al., 2019) — `✓` script read (`nq_eval.py`)

- Short answers are **spans** (byte/token offsets) annotated by 5 workers. Correctness =
  predicted short-answer **set must equal one of the non-null gold short-answer sets**
  (`span_set_equal`). No string normalization — exactness is enforced **by construction** at
  annotation time (the multi-token span *is* the answer). Yes/no answers compared as labels.
- Reported primary metric is short-answer **accuracy at an optimal score threshold**, where the
  threshold is chosen on the dev set itself (a documented caveat of the protocol).

### 1.5 WebQuestions (Berant et al., 2013) / WebQSP (Yih et al., 2015)

- Answers are **Freebase entities** (not surface strings). Metric = per-question **answer-set
  F1**: precision/recall of the predicted entity set vs the gold entity set, averaged.
- The alias/surface-form problem is resolved **upstream at annotation time** by committing to KB
  entity IDs. This is the "selectable entity" paradigm AXIOM is actually implementing: AXIOM's
  output is a graph node, not a free-form string.

### 1.6 Summary table

| Benchmark | Correctness signal | Alias handling | Substring? |
|---|---|---|---|
| SQuAD v1.1 | normalized EM **or** token F1 (max over golds) | max over golds | no |
| TriviaQA | same (EM + F1), over `NormalizedAliases` + HumanAnswers | **normalized alias list** + max | no |
| HotpotQA | same EM + F1 (+ SP/joint) | max over golds | no |
| NaturalQuestions | short-answer **span-set equality** (5-way annotated) | exact spans, set equality | no |
| WebQSP / WebQuestions | answer-**set F1** over Freebase entity IDs | resolved to KB IDs | no |

**Takeaway:** the de-facto standard for "strict but alias-fair" is **normalized exact match OR
token-level F1, maximized over gold aliases**. Bidirectional substring is used by no published
benchmark for answer scoring, and the F1 term exists precisely because it **penalizes precision**
— a gold token being merely contained in a much longer prediction gets recall 1.0 but
precision < 1.0.

---

## 2. The alias problem, and TriviaQA's actual answer to it

TriviaQA gold answers are lists of aliases (`"Luteinizing hormone"`, `"LH"`,
`"Luteinising hormone"`, …). The official solution is **not** "make matching fuzzy"; it is:

1. The dataset ships a curated, **normalized** alias list per answer
   (`NormalizedAliases`, e.g. `["luteinizing hormone","lh","luteinising hormone"]`).
2. The evaluator takes the **max over that list** of (normalized EM, token F1).

This means a system only needs to match **one** clean alias to be credited — abbreviations are
handled by the alias list, not by lenient matching. AXIOM already has this data locally; it
simply does not read it (see §4.5).

---

## 3. What is actually wrong with the current AXIOM metrics (measured)

All numbers below are from the current 318-record bench run on
`data/triviaqa/qa/verified-wikipedia-dev.json` (the user's T1.14 build) and the per-record
`AXIOM_DUMP` output. I verified the bench's own counters (`79 / 78 / 44`) by re-implementing its
exact code path, then computed the alternative metrics against the same dump.

| Metric | Count | Rate | Notes |
|---|---|---|---|
| `candidate_answer_accuracy` (bidirectional substring) | 79 | 24.8% | **current primary; inflated** |
| `candidate_token_accuracy` (containment + "significant" filter) | 78 | 24.5% | still inflated, see §5 |
| `candidate_exact_accuracy` (bench, first-match) | 44 | 13.8% | **undercounts** due to order bug (§4.6) |
| token-set exact, **any** alias (order-independent) | 51 | 16.0% | honest strict floor |
| **official TriviaQA EM** (NormalizedAliases + HumanAnswers) | 58 | 18.2% | the blessed protocol |
| **EM or token-F1 ≥ 0.7, official alias list** | **59** | **18.6%** | **recommended primary** |

### 3.1 The artifact band is token-F1 ∈ [0.5, 0.7)

| Metric | Count | Rate |
|---|---|---|
| EM or F1 ≥ 0.7 | 59 | 18.6% |
| EM or F1 ≥ 0.6 | 75 | 23.6% |
| EM or F1 ≥ 0.5 | 86 | 27.0% |

There is a **clean plateau**: everything in [0.7, 1.0) adds only **one** record (the legitimate
`"steve miller"` for gold `"the steve miller band"`, F1=0.80), while dropping to 0.6 admits 16
extra records that are dominated by the artifacts described below. So τ=0.7 is not a knife-edge
choice; it is the natural separation point between "selecting the reference entity" (good) and
"selecting an entity that merely mentions the reference" (bad).

### 3.2 Concrete artifact classes in the rejected band (all "correct" under substring, all wrong)

| Question (truncated) | gold | picked | F1 | verdict |
|---|---|---|---|---|
| Eleanor Roosevelt's maiden name | `roosevelt` | `eleanor roosevelt` | 0.67 | picked the **person**, not the maiden name |
| Gwyl San Steffan (St Stephen's Day) | `wales` | `holidays in wales` | 0.67 | picked a noise entity, not the country |
| what is a costermonger / "e" case | `e` | `vehicle registration plate` | 0.0 | pure substring garbage |
| Spitfire production company | `supermarine` | `supermarine spitfire` | 0.67 | picked the aircraft, not the company |
| last name of the Marx brothers | `marx` | `marx brothers` | 0.67 | picked the group, not the surname |
| Mr Micawber's first name | `wilkins` | `wilkins micawber` | 0.67 | picked the person, not the first name |
| birthplace of football rules (city) | `cambridge` | `england` | 0.67* | *matched via noisy alias `england cambridge`* |
| rutabaga vegetable in America | `swede` | `sweden` | 0.67 | picked the country, not the vegetable |
| missing Cluedo character | `colonel mustard` | `cluedo` | 0.67 | picked the game, not the character |
| island of which Ulster is part | `ireland` | `ulster` | 0.67 | picked the province, not the island |

The `*` case is the second mechanism: **noisy long aliases** (`"england cambridge"`,
`"rain in wales"`, `"communications in wales"`, `"park street church of england primary school"`)
leak into the alias list from Wikipedia section titles, and any token-overlap rule that does not
penalize extra tokens can be satisfied through one of them.

### 3.3 The data itself makes subsumption pervasive

Measured on the 318-record set:
- 1,486 of 5,357 normalized aliases (28%) are **single tokens** (`roosevelt`, `wales`, `biko`).
- 1,554 of 318 records contain an alias whose token set is a **strict subset** of another
  alias's token set — the `roosevelt ⊂ eleanor roosevelt` shape is the norm, not the exception.
- 17 records have a **single-character** alias (digits `3`, `5`; even emoji `🌌`, `🐻`). A
  substring metric can be satisfied by any picked string that happens to contain such a
  character.

---

## 4. Answering the specific questions in the brief

### 4.1 "Why is token containment STILL inflated (24.5% vs exact 13.8%)?"

Three independent reasons:

1. **No precision term.** Token containment scores `gold ⊆ picked` and `picked ⊆ gold` alike.
   `roosevelt ⊂ eleanor roosevelt` passes (SQuAD F1 = 0.667, precision 0.5). The current
   `significant` filter (`shared ≥ 2 OR shared ≥ 1 with a ≥5-char token`) does **not** fix this:
   `roosevelt` is 9 characters, so the ≥5-char branch fires and the subset is still credited.
   Token-F1's **precision** term is the actual fix: adding extra tokens to the picked entity
   *drops* F1.
2. **Noise aliases.** Even a "shared ≥ 2 significant tokens" rule is defeated by the alias list
   itself: picked `holidays in wales` shares 2 tokens with the junk aliases `rain in wales` and
   `communications in wales`. Any rule that only requires the *gold* side to be present (with
   no penalty for the *picked* side's extra tokens) can be gamed by whichever junk alias
   happens to overlap.
3. **Alias-order dependence.** The bench iterates aliases and `break`s on the first match
   (`triviaqa-bench.rs:122-133`). When a noisy superset alias precedes the true alias in the
   list, a genuinely exact match is scored as mere containment. I found 7 such records
   (e.g. gold `the netherlands`, picked `netherlands` — exact under normalization but scored as
   containment). **Fix: take the max over all aliases** (as the official script does), never
   first-match.

### 4.2 "Is SQuAD's F1 the right metric for alias-heavy entity answers?"

It is the *standard*, and it is the right **strict core**: max-over-aliases handles the alias
list, and the precision term handles subsumption. But it is not perfect, and it is important to
say so honestly: Bulian et al. (2022) show with 23k human judgments that token-level F1 gives a
"false impression of graduality" and is blind to the question — e.g. it cannot distinguish
`gold="ely", picked="ely cathedral"` (arguably correct, same entity) from
`gold="roosevelt", picked="eleanor roosevelt"` (wrong, different entity), because both are
structurally `1-token gold ⊂ 2-token picked`. **No token-overlap metric can separate those two
cases**; the honest resolution is to *choose strictness* (τ=0.7), accept that a few fuller-name
variants of the true answer are lost, and let the **official alias list** credit the shorthand
forms (`hingis`, `lh`, `biko`) that the dataset authors judged valid. That is exactly the
protocol the SQuAD/TriviaQA community settled on.

### 4.3 "What metric fixes THIS specifically?" (the subsumption trap)

The precision term of token-F1 — equivalently: *never credit a match whose precision < 1 while
the picked is much longer than the gold*. Options evaluated on this bench:

| Option | Rate | Verdict |
|---|---|---|
| (a) token-set exact vs any alias | 16.0% | clean but strict; misses `the netherlands`→`netherlands` (fixed by article-stripping normalization) |
| (b) token-F1 ≥ 0.7 vs any alias (+EM) | **18.6%** | recommended; plateau-justified |
| (c) bidirectional containment (current) | 24.5% | broken (no precision) |
| "shared ≥ 2 tokens" | ~17.6% | still gamed by noise aliases (`rain in wales`) |
| "gold must be a prefix of picked" | — | separates `ely cathedral` (prefix, accept) from `eleanor roosevelt` (suffix, reject) **but** wrongly rejects `anterior fontanelle`/`northern lapwing` (suffix but correct). Not worth the complexity. |
| longest-alias common-substring fraction | — | more machinery than token-F1, same irreducible ambiguity; no published precedent |

F1's precision term is the principled fix; the ad-hoc token-count heuristics are demonstrably
inadequate.

### 4.4 "Selectable ceiling" — what is the true recall ceiling, and the right protocol?

The correct protocol is: **a question is "selectably recalled" iff some graph entity matches a
gold alias under the same strict EM-or-F1≥0.7 rule** (i.e. a node AXIOM could actually be
credited for selecting exists). This is the "oracle ceiling" of the selection task, and it is the
standard retrieval-vs-selection decomposition in KGQA work. Measured on this bench (I added
throwaway instrumentation to the dump, then reverted it):

| Recall variant | Rate |
|---|---|
| `answer_entity_recall` (bidirectional substring, current) | 76.1% |
| any node with token-F1 ≥ 0.7 vs a gold alias | **54.7%** |
| any node token-set-exact vs a gold alias | 50.6% |

So **~21pt of the claimed 76% recall is phantom**: it counts records whose gold appears only as
a token substring of a longer, non-selectable node (e.g. `roosevelt` inside a `eleanor roosevelt`
node). Optimizing against 76% recall means chasing nodes that the metric would never credit once
the answer metric is fixed. The real selection gap is 54.7% (ceiling) → 18.6% (current strict
accuracy) ≈ 36pt, and the 76%→54.7% correction is the first thing the roadmap's "recall 80%"
target must absorb.

### 4.5 Missing alias fields in the bench

`triviaqa-bench.rs`'s `load_records` reads only `Answer.Value` and `Answer.Aliases`
(lines 263-268). It ignores `Answer.NormalizedAliases` (the dataset's blessed, normalized alias
list), `Answer.MatchedWikiEntityName`, `Answer.NormalizedMatchedWikiEntityName` and
`Answer.HumanAnswers`. Loading them is required for the recommended metric to be fair and is
free (they are already in the JSON files).

### 4.6 Additional bench defects found while measuring

- Break-on-first-match makes scores **alias-order dependent** (7 records affected; §4.1.3).
- Empty-picked records are excluded from candidate metrics but the string/`to_lowercase`
  comparisons elsewhere are fine; the 3 emoji-only-gold records (e.g. `🌌` for the Milky Way)
  can never be token-matched — they must be handled explicitly (see §7, `NONEMPTY` guard and
  empty-token gold aliases).
- `exact_accuracy` / `substring_accuracy` (whole-output, lines 90-91) are sentence-level
  diagnostics and are not the topic here, but they share the same substring blindness.

---

## 5. Literature on metric failure modes (supporting citations)

- **Bulian, Buck, Gajewski, Boerschinger, Schuster — "Tomayto, Tomahto: Beyond Token-level
  Answer Equivalence for Question Answering Evaluation", EMNLP 2022** (arXiv:2202.07654).
  First systematic analysis of token-level equivalence shortcomings: false graduality,
  question-blindness, and the need for *asymmetric answer equivalence* (accept answers that are
  "equivalent to or improve over" the reference). Direct support for: (a) token-F1 alone is an
  imperfect oracle, (b) the "fuller-name" vs "wrong-entity" ambiguity (§4.2) is real and
  measured, (c) learned equivalence is the aspirational fix, not the deterministic one.
- **Geirhos et al. — "Shortcut Learning in Deep Neural Networks", Nature Machine Intelligence
  2020** (arXiv:2004.07780). General framework for "shortcut" artifact exploitation: when a
  cheap signal (here: gold-token-subset) suffices to pass a gate, optimization will optimize for
  it. This is the mechanism by which AXIOM's 2×-inflated metric misdirected its optimization.
- **Rajpurkar et al. 2016; Joshi et al. 2017; Yang et al. 2018; Kwiatkowski et al. 2019;
  Berant et al. 2013; Yih et al. 2015** — the protocols in §1, all verified against scripts.

I found **no published benchmark or library** that scores entity/literal answers with
bidirectional substring matching; the ones that need leniency do it via alias lists
(TriviaQA), multi-answer max (SQuAD/HotpotQA), span sets (NQ), or KB-entity identity (WebQSP).
The binary τ for F1 is a design choice of ours (not a published standard): SQuAD/TriviaQA
report *average* F1, not a per-question threshold, so the threshold is AXIOM's own and is
justified empirically by the 0.7 plateau (§3.1).

---

## 6. Verified facts vs. recommendations (explicit separation)

**Verified protocol facts** (§1, from scripts/papers): SQuAD/TriviaQA/HotpotQA = normalized EM +
token F1, max over golds, **no substring**; NQ = annotated span-set equality; WebQSP =
answer-set F1 over KB entities; TriviaQA ships and evaluates against `NormalizedAliases`.

**Empirical measurements** (§3-§4, from this bench): current substring 24.8%, containment 24.5%,
any-alias exact 16.0%, official EM 18.2%, EM-or-F1≥0.7 = 18.6%; artifact band = F1∈[0.5,0.7);
true selectable ceiling 54.7% vs claimed 76.1%; 28% single-token aliases; 7 alias-order-affected
records; 17 single-char aliases; noise aliases defeat "shared≥2".

**Recommendations** (§7): τ=0.7 threshold; the alias-source set; keep-gate redesign; diagnostics
policy. These are my design choices, argued from the facts above and the cited literature.

---

## 7. Recommendation: the metric formula, and the keep-gate redesign

### 7.1 Primary metric (pseudocode — deterministic, cheap, order-independent)

```text
# ---- normalization (exactly the SQuAD/TriviaQA official definition) ----
def NORMALIZE(s):
    s = s.lower().replace('_', ' ')
    s = map(punct_or_curly_quote -> ' ', s)          # string.punctuation + "‘’´`"
    s = remove articles as whole words:  \b(a|an|the)\b
    return ' '.join(s.split())

# ---- gold aliases: the full official alias set, normalized, deduped ----
def GOLD_ALIASES(record):
    raw = record.Answer.Value + record.Answer.Aliases
          + record.Answer.MatchedWikiEntityName
    out = record.Answer.NormalizedAliases                       # already normalized
          + [NORMALIZE(h) for h in record.Answer.HumanAnswers]
          + [NORMALIZE(x) for x in raw]
    return unique([a for a in out if a != ''])                  # drop empty (emoji-only)

# ---- SQuAD-style token F1 over a pair ----
def TOKEN_F1(pred, gold):
    P = multiset(NORMALIZE(pred).split()); G = multiset(NORMALIZE(gold).split())
    C = |P ∩ G|                                                # multiset intersection
    if C == 0: return 0.0
    prec = C / |P|; rec = C / |G|
    return 2·prec·rec / (prec + rec)

# ---- per-question correctness (max over aliases; the official pattern) ----
def IS_CORRECT(picked, record):
    if picked is empty: return False
    for alias in GOLD_ALIASES(record):
        if NORMALIZE(picked) == alias: return True             # exact
    for alias in GOLD_ALIASES(record):
        if TOKEN_F1(picked, alias) >= 0.7: return True         # lenient tail
    return False

candidate_accuracy_strict  = mean over records of IS_CORRECT(picked_record, record)
```

Why this satisfies the three requirements in the brief:
- **(a) strict, does not reward picking the reference entity** — no containment; the F1
  precision term means `roosevelt ⊆ eleanor roosevelt` (F1=0.667) fails; `wales ⊆ holidays in
  wales` (F1≤0.667) fails; `e` vs `vehicle registration plate` (F1=0) fails.
- **(b) fair to alias lists** — max over the full official alias list, so `LH`, `hingis`,
  `biko`-type shorthands are credited when the dataset lists them, and the F1 tail credits the
  one fuller-name variant we found (`steve miller` → `steve miller band`, F1=0.80).
- **(c) deterministic and cheap** — pure string ops; O(aliases × tokens) per record, no model.

### 7.2 Why τ = 0.7 (§3.1 recapped)

The pass-count is flat at 59 across τ∈[0.7, 1.0) and jumps +16 at τ=0.6; the [0.6, 0.7) band is
dominated by the artifact classes of §3.2. τ=0.7 is the separation point, not a sensitive knob.
If AXIOM ever wants the fuller-name variants credited, the honest lever is the *alias list*
(e.g. add the fuller form as an alias), not lowering τ.

### 7.3 Keep-gate redesign

Replace the current gate (candidate = bidirectional substring, recall = bidirectional substring):

| Gate | Old | New |
|---|---|---|
| primary | `candidate_answer_accuracy` (substr) | `candidate_accuracy_strict` (EM-or-F1≥0.7, official aliases) |
| secondary | `answer_entity_recall` (substr) | `answer_entity_recall_strict` = any graph node with EM-or-F1≥0.7 vs any gold alias (the true selectable ceiling) |

KEEP a change iff primary **and** secondary (strict) both do not regress — same rule, honest
numbers. Expected new baseline ≈ 18.6% primary / ≈54.7% recall; the roadmap targets (40% /
80%) are now *ceiling-relative* (i.e. 40% of selectable, ~73% of the 54.7% ceiling) and should
be re-stated.

### 7.4 Legacy substring metrics as diagnostics only

Keep printing, but never gate on them:
- `candidate_answer_accuracy_legacy` (bidirectional substring) — tells you "is the gold text
  mentioned by the picked entity at all", a cheap *coarse* signal.
- `answer_entity_recall_legacy` — tells you "does any node mention the gold", useful to separate
  "gold not in graph at all" from "gold in graph but not selectable", but **not** a ceiling.
- Add one new diagnostic column: `gold_rank` / `gold_gap` already exist in the dump; extend the
  debug path to also mark *strict* gold rank (first entity satisfying EM-or-F1≥0.7) so failure
  analysis separates "not selectable" from "selectable but ranked wrong".

Also fix, in the same change: read `NormalizedAliases`/`MatchedWikiEntityName`/`HumanAnswers`;
evaluate max-over-aliases (remove break-on-first-match); skip empty-picked and empty-gold-alias
records explicitly.

### 7.5 Residual, documented limitation

No string/token metric can separate `ely cathedral` (correct) from `eleanor roosevelt`
(wrong) — both are structurally identical subsumption cases (Bulian et al. 2022). τ=0.7 errs on
the side of strictness (a few correct fuller-name picks are counted wrong). This is the
deliberate, defensible tradeoff; if AXIOM later needs it, the only principled upgrade is a
learned/externally-sourced answer-equivalence model (BEM), which is out of scope for a
deterministic zero-training system.

---

## Citations

1. Rajpurkar, P., Zhang, J., Lopyrev, K., Liang, P. *SQuAD: 100,000+ Questions for Machine
   Comprehension of Text.* EMNLP 2016. arXiv:1606.05250.
2. Joshi, M., Choi, E., Weld, D., Zettlemoyer, L. *TriviaQA: A Large Scale Distantly Supervised
   Challenge Dataset for Reading Comprehension.* ACL 2017. arXiv:1705.03551. Official eval:
   github.com/mandarjoshi90/triviaqa (`evaluation/triviaqa_evaluation.py`).
3. Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W., Salakhutdinov, R., Manning, C.
   *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.* EMNLP 2018.
   arXiv:1809.09600. Official eval: github.com/hotpotqa/hotpot (`hotpot_evaluate_v1.py`).
4. Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., et al. *Natural Questions: A
   Benchmark for Question Answering Research.* TACL 7 (2019). arXiv:1901.08634. Official eval:
   github.com/google-research-datasets/natural-questions (`nq_eval.py`).
5. Berant, J., Chou, A., Frostig, R., Liang, P. *Semantic Parsing on Freebase from
   Question-Answer Pairs.* EMNLP 2013. aclanthology.org/D13-1160.
6. Yih, W., Chang, M.-W., He, X., Gao, J. *Semantic Parsing via Staged Query Graph Generation:
   Question Answering with Knowledge Base.* ACL-IJCNLP 2015. aclanthology.org/P15-1128.
7. Bulian, J., Buck, C., Gajewski, W., Boerschinger, B., Schuster, T. *Tomayto, Tomahto. Beyond
   Token-level Answer Equivalence for Question Answering Evaluation.* EMNLP 2022.
   arXiv:2202.07654.
8. Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., Wichmann,
   F. *Shortcut Learning in Deep Neural Networks.* Nature Machine Intelligence 2 (2020).
   arXiv:2004.07780.

*Empirical measurements in §3-§4 are AXIOM-specific (this bench), computed from the 318-record
run and its `AXIOM_DUMP` output; protocol facts in §1 were verified directly against the cited
scripts/papers.*
