# Research Report — The Reference-Entity Failure Mode in AXIOM

> **Scope:** The ~20% of AXIOM answer-selection failures where the winning (wrong)
> candidate is an entity whose **name appears literally in the question** — the
> reference/topic entity (e.g., "Buddy Holly … Who was the other one?" → AXIOM
> selects "Buddy Holly"; gold = "Richie Valens").
> **Date:** 2026-08-12 · **Status:** literature review + mechanism proposals,
> none bench-tested yet.
> **Grounding:** `crates/tle-axiom-gen/src/engine.rs` (`extract_answer`, v16),
> `docs/PROJECT_SUMMARY.md`, `docs/ROOT_CAUSE_ANALYSIS.md`, `AGENTS.md`.

Every citation below is tagged **\[V\]** = fetched/verified this session (arXiv /
ACL Anthology / DBLP / journal page), or **\[K\]** = canonical reference cited
from knowledge and NOT re-verified in this session. The concrete mechanism
proposals in §5 are analysis/inference grounded in the verified literature.

---

## 0. Why the reference entity wins, in AXIOM's own scoring terms

From `engine.rs:extract_answer` (v16), for the reference/topic entity `R`
(e.g., "Buddy Holly"):

| Signal | Value for `R` | Why |
|---|---|---|
| `conn_avg` | **0** | In every triple `R` participates in, `R` is the query side → `connected_id` is the *other* node (engine.rs:510). `R` is never "the node connected to the query". |
| `hop2_avg` | **0** | 2-hop pass explicitly skips query entities (engine.rs:537). |
| `overlap` | high | `R`'s name words are in the question. |
| `heur = 0.2·count(R)` | **huge** | `count` = # triples mentioning `R`. The evidence doc is *about* `R`, so `R` is the subject of the majority of decomposed triples. |
| `ppr` | **maximal** | `ppr = log π_q(R) − log π(R)` and `R` *is* the PageRank seed → `π_q(R)` is the largest value in the personalized distribution. |
| `vsa_cosine` | ~noise | `N(0, 1/√D)`, D=2048 → σ≈0.02 (RCA). |
| `query_penalty` | 0.6 (mild) | `What`/`Who` intent → `qp_mild` (engine.rs:649), because "What is X?" legitimately has `X` as the answer. |

So `R` wins via `heur + overlap + ppr` surviving the mild penalty; the gold answer
`A` (Richie Valens) has `conn_avg > 0` but the magnitude gap is too large.

**Why the three already-tried global fixes fail (and why this report does not
re-propose them):**

1. *Universal query-penalty strengthening*: crashes identity questions
   ("What is X?" / "Who is X?" — the Milky Way case) where the named entity *is*
   the answer. The penalty must be **conditional**, not universal.
2. *Overlap removal*: kills a genuine (weak) tiebreak for connected candidates
   whose names share words; `R` still survives via `heur + ppr`.
3. *Global count-weight reduction* (24.8% → 11%): `count` *is* genuine
   evidence-mass for `A` (Valens appears in the Buddy Holly doc a handful of
   times). Cutting `w_count` cuts `A`'s signal with `R`'s.

The three mechanisms in §5 are all **conditional on question structure**, which
is the axis the failed fixes ignored. This conditional-vs-global distinction is
itself supported by the negation literature: every effective negation fix found in
§1 is a *targeted* fix applied after negation detection, not a global model change.

---

## 1. Exclusion / contrastive semantics in QA

### 1.1 What the failure semantically is

"The other one / other than X / besides X / apart from X" is **exceptive
semantics**: the answer is required to lie in a set **minus** the named
entities. This is set subtraction, not mere negation of a relation.

- **von Fintel (1993), "Exceptive constructions", Natural Language Semantics 1(2):123–148** \[V, DOI 10.1007/BF00372560\] — the semantics of exceptives is "primarily one of subtraction from the domain of a quantifier". The named set is subtracted from the answer domain. This is the formal backbone of mechanism (a): once an exceptive cue is present, the named entities are *by definition* not in the answer set.
- **Moltmann (1995), "Exception sentences and polyadic quantification", Linguistics & Philosophy 18(3):223–280** \[K\] — exception sentences; complement-set semantics.
- **Hoeksema (1995), "The semantics of exception phrases"** \[K\] — the *except* / *besides* / *apart from* family.
- **Rooth (1992), "A theory of focus interpretation", NLS 1:75–116** \[K\] — alternative-set semantics; "besides X" and "other than X" evoke an alternative set from which X is removed.

The anaphoric "the other one" resolves against a salient set (the three musicians
who died in the crash). Deterministic resolution of "other"-anaphora is a
set-complement operation: `answer ∈ S \ {named}` where `S` is the set of entities
sharing the relation named in the question.

### 1.2 Neural systems handle this badly (evidence)

- **CONDAQA** (Ravichander et al., 2022), arXiv:2211.00295 \[V\] — "A Contrastive Reading Comprehension Dataset for Reasoning about Negation". Questions whose answer depends on negated statements in the passage; three edit types (paraphrase / scope-change / reversal) form clusters models solve spuriously. Directly models the implication-of-negation reasoning AXIOM's reference-questions require.
- **NeQA** (Yanaka et al. / Wang et al. 2023), arXiv:2305.17311 \[V\] — "Beyond Positive Scaling: How Negation Impacts Scaling Trends of Language Models". Negation is *compositional*: solving = task1 (answer) + task2 (negation), and task2 has a separate sigmoid scaling curve — i.e., negation is a *learnable but distinct sub-capability*, which is exactly why a deterministic gate (rather than a scoring tweak) is the right shape of fix.
- **Kassner & Schütze (2020)**, "Negated and Misprimed Probes for Pretrained Language Models: Birds Can Talk, But Cannot Fly", ACL 2020, DOI 10.18653/V1/2020.ACL-MAIN.698 \[V\] — probing shows pretrained LMs fail negated LAMA probes. Frequency-based priors (exactly `R`'s advantage) dominate when negation is present.
- **Paraphrasing in Affirmative Terms Improves Negation Understanding**, arXiv:2406.07492 \[V\] — the strongest fixes *detect* the negation and re-formulate the query. Same shape as mechanism (a): detect, then transform the query.
- **Making Language Models Robust Against Negation**, arXiv:2502.07717 \[V\] — dedicated pre-training objectives; +1.8–9.1% on CONDAQA. Again a targeted capability, not a global prior change.
- **Negation in Cognitive Reasoning**, arXiv:2012.12641 \[V\] — negation must be *treated as an operator* over an incomplete logical store; forward reasoning alone can't derive answers. Supports building exclusion into the query (operator) rather than re-weighting evidence.

### 1.3 Deterministic / rule-based exclusion detection — NegEx

The canonical deterministic negation-cue detector is **NegEx**:

- **Chapman, Bridewell, Hanbury, Cooper & Buchanan (2001)**, "A Simple Algorithm for Identifying Negated Findings and Diseases in Discharge Summaries", Journal of Biomedical Informatics 34(5):301–310, DOI 10.1006/jbin.2001.1029 \[V\] — trigger-word lists (pseudo-negation vs. true negation) plus a **scope window** (k tokens around the cue). Precisely the machinery AXIOM needs, and it is surface-based (no POS/NER required), so it fits AXIOM's deterministic, lexicon-free-of-training constraint. Mechanism (a) is a NegEx-style cue lexicon + scope window specialised to trivia exceptives ("other than", "besides", "apart from", "except", "excluding", "the other one", "another", "who else", "rather than", "instead of").

- **NEST-KGQA**, "Which bird does not have wings: Negative-constrained KGQA with Schema-guided Semantic Matching and Self-directed Refinement", arXiv:2604.14749 \[V\] — a 2026 task definition: KBQA questions with at least one **negative constraint**, and an explicit **Python-formatted logical form** (PyLF) because "existing logical forms are hardly suitable to express negation clearly". This is the KBQA-side confirmation that exclusion must become an explicit logical-form operator, not an implicit scoring preference. In SPARQL terms: `FILTER(?a NOT IN {anchors})`.

Bottom line for §1: the semantics is **set subtraction** (von Fintel); the detection machinery is **cue-lexicon + scope window** (NegEx); the system-side lesson is **make exclusion an explicit operator on the query** (NEST). AXIOM's linear-sum scorer cannot express an exclusion operator — that requires a gate (hard) or a query-space operation (VSA-NOT, §2).

---

## 2. VSA/HDC representation of negation and exclusion

### 2.1 The canonical mechanism: negation = complement / −1

- **Kanerva (2009)**, "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors", Cognitive Computation 1(2):139–159, DOI 10.1007/s12559-009-9009-8 \[V\] — primary source. The paper's algebra section (verified from the PDF, §6.3) states: *"Subtracting one vector from another is accomplished by adding the vector's complement. The complement of a real vector is gotten by multiplying each component by −1."* It also fixes the bipolar model: binding = XOR (equivalently componentwise multiply in the ±1 encoding), bundling = majority/sign, negation = multiply by −1 (i.e., XOR with the all-ones vector in binary; bitwise complement).
- **Gayler (1998)**, "Multiplicative Binding, Representation Operators & Analogy" \[K\] — MAP architecture; negation of an atomic vector by multiplication with the all-(−1) vector; the standard reference for bipolar negation.
- **Plate (1995/2003)**, "Holographic Reduced Representations" \[K\] — the HRR family; negation/zero-vector conventions differ but subtraction is defined.
- **Kleyko, Rachkovskij, Osipov & Rahimi**, "A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part I: Models and Data Transformations", arXiv:2111.06077 \[V\] — the field's canonical survey. **Negative finding:** the survey indexes *no* dedicated negation treatment (I grepped the full text: 0 hits for "negat"); negation is implicit in the bipolar/majority primitives. So "NOT X in VSA" is standard but **under-theorised** — there is no published "exclusion operator for question answering in HDC".
- **Kleyko et al.**, "Vector Symbolic Architectures as a Computing Framework for Emerging Hardware", arXiv:2106.05268 \[V\] — same implicit treatment; field-algebra view where −1 multiplication is the negation endomorphism.
- **Rachkovskij & Kussul (2001)**, "Binding and Normalization of Binary Sparse Distributed Representations" \[K\] — for **sparse binary** representations, set operations (union=OR, intersection=AND, complement=NOT, symmetric difference=(A∨B)∧¬(A∧B)) are *exact*. AXIOM uses dense bipolar, so it gets NOT for free but AND/OR only approximately.

### 2.2 Encoding "NOT X" / exclusion for AXIOM's query

In AXIOM's dense bipolar VSA (d=2048, ±1):

- **NOT X = −X** (componentwise), equivalently XOR with the all-(−1) vector. Deterministic, zero-cost, exactly the Kanerva complement.
- **Bundle with negation**: `Q' = bundle(Q, −R)` — superimposing the negated reference into the query hypervector used for the `rel` (cosine) term suppresses `R` and is quasi-orthogonal-noise for everything else. Effect size: `cos(Q', R) ≈ cos(Q, R) − 2/√D ≈ −0.044`; for any other `e`, `Δcos ≈ ±2/√D ≈ ±0.022` — i.e., **weak but exactly targeted**. Because AXIOM's RCA already showed `vsa_cosine` is ~noise (σ≈0.02), this shove is *insufficient alone* (consistent with AXIOM's negative result on VSA clamp + count re-weight), but it is the only mechanism that puts the exclusion **inside the query representation**, which §1.3 says is where the operator belongs.
- **Logical XOR** ("exclusive") in HDC: `A XOR B = (A ∧ ¬B) ∨ (¬A ∧ B)` using NOT (−1), AND (binding ⊙), OR (bundle ⊕) — textbook composition of the primitives in §2.1. There is no published dedicated "exclusive bundle" operator; the published exclusion machinery lives instead in **geometry-based KG-query embeddings**:

### 2.3 Published negation in embedding-space KG reasoning (the transferable prior)

These are *trained* models (AXIOM cannot train), but they establish the **correct abstraction for exclusion**: the answer set is a *region*; NOT is the *set complement*; exclusion = complement of the anchor region.

- **BetaE** (Ren & Leskovec, 2020), arXiv:2010.11465 \[V\] — first to support ¬ in FOL query embeddings; negation = complementing the Beta distribution (1−β).
- **ConE** (Zhang et al., 2021), arXiv:2110.13715 \[V\] — cone embeddings; ¬ = cone complement ("the first geometry-based model to handle conjunction, disjunction and negation").
- **GammaE** (Yang et al., 2022), arXiv:2210.15578 \[V\] — Gamma embeddings with complement via linearity/boundary properties.
- **Logic Embeddings for Complex Query Answering**, arXiv:2103.00418 \[V].
- **HDReason**, arXiv:2403.05763 \[V\] — HDC applied to KG reasoning (KG completion), but *not* negation; confirms the field has not wired exclusion into HDC-based KG reasoning.
- **PathHD**, arXiv:2512.09369 \[V\] — HDC path retrieval for KG-QA; again no negation. (Path-like retrieval is however conceptually what AXIOM's `conn/hop2` approximate.)

**Verdict for §2:** "Multiply the reference hypervector by −1 and add it to the query bundle" is the published, canonical encoding of NOT in bipolar VSA (Kanerva 2009), with the "exclusive" reading backed by the geometry-based complement literature (BetaE/ConE). It is **not a complete fix** in AXIOM's noise regime — treat it as a supplementary shove on `rel`, never the primary signal.

---

## 3. Intent-aware answer-type vs reference — the query-graph answer to "which node is the answer"

### 3.1 The standard formalism: named entities are anchors, the answer is the variable

The KB-QA semantic-parsing literature is unambiguous: **the entities named in the
question are anchors (grounded constraints); the answer is the single free
variable node** of the query graph.

- **Yih, Chang, He & Gao (2015), STAGG** — "Semantic Parsing via Staged Query Graph Generation: Question Answering with Knowledge Base", ACL-IJCNLP 2015, P15-1128, DOI 10.3115/v1/P15-1128 \[V\] — "We define a query graph that resembles subgraphs of the knowledge base and can be directly mapped to a logical form." The query graph = anchor entities + relations + **one λ-variable (the answer node)**. The "core inferential chain" connects the anchors to the answer node. **The answer position is chosen by construction: it is the node not grounded to any mention.** This is the exact discriminator AXIOM's data shows: reference entities have `conn=0` *because they are the anchors*, and the gold answer is the node connected *to* them.
- **Sun, Ma, Yih, Tsai, Liu & Chang (2015)**, "Open Domain Question Answering via Semantic Enrichment", WWW 2015, DOI 10.1145/2736277.2741651 \[V\] — the **"topic entity"** is the anchor extracted from the question; the answer is derived by expansion (enrichment) around it.
- **Berant, Chou, Frostig & Liang (2013)**, "Semantic Parsing on Freebase from Question-Answer Pairs", EMNLP 2013, D13-1160 \[V\] — answer is an argument position of the query (lambda-DCS variable).
- **Berant & Liang (2014)**, "Semantic Parsing via Paraphrasing", ACL 2014, P14-1133 \[V\].
- **Bao, Duan, Yan, Zhou & Zhao (2016)**, "Constraint-Based Question Answering with Knowledge Graph", COLING 2016, C16-1236 \[V\] — multi-constraint questions; each named entity + relation pair is a *constraint*; the answer satisfies all constraints. "Other than X" is a **negative constraint** — the direct ancestor of mechanism (a)'s operator.
- **Talmor & Berant (2018)**, "The Web as a Knowledge-base for Answering Complex Questions" (ComplexWebQuestions), NAACL 2018, N18-1059 \[V\] — complex constraints incl. **negation and comparison** on the variable; these are the question families AXIOM's reference-questions belong to.
- **Wolfson et al. (2020)**, "Break It Down: A Question Understanding Benchmark" (QDMR), TACL 8:183–198, 2020.tacl-1.13, DOI 10.1162/tacl_a_00309 \[V\] — questions decompose into an ordered list of steps; "the other one" decomposes to [select the set; remove the named members] — a set-difference step. QDMR is deterministic-convertible to pseudo-SQL, confirming the operator-shaped treatment.
- **Zheng & Zhang**, "Question Answering over Knowledge Graphs via Structural Query Patterns", arXiv:1910.09760 \[V\]; **Wu et al.**, "Modeling Global Semantics for QA over KBs" (gRGCN), arXiv:2101.01510 \[V\]; **"Better Query Graph Selection for KBQA"**, arXiv:2204.12662 \[V\]; **"Semantic Parsing for QA over Knowledge Graphs"**, arXiv:2401.06772 \[V\] — query-graph selection/construction line; all keep the anchor/variable split.
- **Li & Roth (2002)**, "Learning Question Classifiers", COLING 2002, C02-1150 \[V\] — the classic answer-type taxonomy (HUM, ENT, LOC, NUM, …). The answer-type is predicted *independently of the named entities*; a named PERSON in a "What song…" question has the wrong type for the answer slot. Type mismatch is a discriminative signal AXIOM does not currently use.

### 3.2 Heuristics in the query-graph literature for "which position is the answer"

Consolidating the above (this sub-section is **analysis**, not a single citation):
the answer node is the one that is (i) **not** mentioned in the question,
(ii) **reachable from every anchor** through typed relations, (iii) of the
**type** indicated by the interrogative head. Each maps onto an existing AXIOM
signal: (i) `is_query_named`, (ii) `conn_avg`/`hop2_avg`/`ppr`, (iii) intent +
head-noun (partially). **AXIOM already computes every ingredient; what is missing
is the conditional that binds them:** *"if the question structure says X is an
anchor, then the winner must satisfy (ii) and must not be (i)."* That conditional
is mechanism (b).

Note on scope of the literature: these systems *build* the query graph, so the
answer position never has to be *recovered from a ranking* — AXIOM's exact problem
(selecting the answer from a scored entity list without a query graph) is not
solved by any single paper in this line. The invariant transfers; the scoring
mechanism does not. This is also why AXIOM's earlier graph-surface filters and
relation-heuristic type-veto failed: they were applied as universal filters, not
as anchor-conditional gates.

---

## 4. Document-topic-frequency as a confound

### 4.1 The confound, stated precisely

`count(R)` is not "evidence that R is the answer"; it is **evidence that R is the
topic of the evidence document** — and, because the question quotes R, `count(R)`
is *explained by the query itself*. `count(A)` (Valens) is not explained by the
query in the same way: A's mentions in the Buddy Holly doc are genuine
evidence-mass for A as a co-participant. The published name for this separation
problem is the **topic-signature / residual-frequency** family.

### 4.2 Published conditional / relative frequency measures

- **Sparck Jones (1972)**, "A Statistical Interpretation of Term Specificity and Its Application in Retrieval", Journal of Documentation 28(1):11–21 \[K\] — IDF: `log(N/df)`. Collection-level only; it does **not** separate "topic of this doc" from "genuine co-evidence", so collection IDF is the wrong tool for this failure (consistent with AXIOM's IEF negative result).
- **Robertson & Zaragoza (2009)**, "The Probabilistic Relevance Framework: BM25 and Beyond", Foundations and Trends in IR 3(4):333–389 \[K\] — BM25 IDF `log((N−df+0.5)/(df+0.5))`; again collection-level.
- **Salton & Buckley (1988)**, "Term-weighting approaches in automatic text retrieval", Information Processing & Management 24(5):513–523 \[K\] — the term-weighting taxonomy (TF/IDF/…); establishes that *component* weighting (not the aggregate) is where specificity lives.
- **Lin & Hovy (2000)**, "The Automated Acquisition of Topic Signatures for Text Summarization", COLING 2000, C00-1072 \[V\] — **topic signatures**: score a term by the **ratio of its frequency in topic-conditioned text to its marginal frequency** (`log P(w|T)/P(w)`). This is the published "frequency conditioned on the topic/question" the failure needs: the reference's frequency is *fully explained* by the topic (`P(R|T)≈1`), so its marginal-vs-conditional ratio carries no answer discrimination; the co-participant's ratio does.
- **Church & Gale (1995)**, "Poisson Mixtures", Natural Language Engineering 1(2):163–190 \[V\] — **residual IDF**: model the expected document-frequency of a term under a Poisson-mixture null; the *residual* `log(df_observed / df_expected)` is the genuine specificity signal. This is the canonical way to "discount the frequency that is explainable by the document topic" **without discarding the count as a whole** — precisely the 24.8→11 crash the user described, avoided by keeping the *residual*, not the raw count.
- **Lavrenko & Croft (2001)**, "Relevance-Based Language Models", SIGIR 2001 \[K\] — relevance models condition directly on the query: `P(w | q)` via pseudo-relevance. The theoretical home of "frequency conditioned on the query".
- **IDF information in BERT** (Merrill et al. / authors not captured), arXiv:2202.12191 \[V\] — IDF-like specificity is recoverable from a trained model; not needed here, but shows the field treats relative-vs-collection frequency as orthogonal axes.
- **The hypergeometric test performs comparably to TF-IDF on standard text analysis tasks**, arXiv:2002.11844 \[V\] — an alternative significance-style specificity measure; supports the "use a *significance* of co-occurrence, not a raw count" family.

### 4.3 Entity-salience line (the reference IS the salient topic)

The reference entity of the evidence document is precisely the *salient topic
entity* that the salience literature identifies:

- **Ganea & Hofmann (2017)**, "SWAT: A System for Detecting Salient Wikipedia Entities in Texts", arXiv:1804.03580 \[V\] — identifies the salient/central entities of a document; **the reference entity of a TriviaQA evidence doc is its SWAT entity**. Salience of `R` is a *document property*, not an answer property.
- **Xiong et al.**, "Towards Better Text Understanding and Retrieval through Kernel Entity Salience Modeling" (KESM), arXiv:1805.01334 \[V\] — salience beats frequency-based features; "modeling the salience of query entities in candidate documents" is exactly the query-conditional mass AXIOM should discount.
- **GUM-SAGE**, arXiv:2504.10792 \[V\] — graded entity salience; centrality is graded, so a *soft* conditional discount (not a hard zero) is defensible.
- **"How Knowledge Popularity Influences and Enhances LLM Knowledge Boundary Perception"**, arXiv:2505.17537 \[V\] — entity-popularity in the question vs answer is a real, measurable confound in entity-centric QA; supports conditioning answer scoring on question-side popularity.
- **"Quantifying the Impact of Cognitive Biases in QA Systems"**, arXiv:1909.09633 \[V\] — perceived popularity amplifies a candidate's selection; the QA analogue of AXIOM's count-driven `R` wins.

**Verdict for §4:** the published, correct conditional measures are **topic
signatures** (Lin & Hovy), **residual IDF** (Church & Gale), and **query
conditioning** (Lavrenko & Croft). All are *per-candidate relative* measures that
subtract the query/topic-explainable component from the raw count. AXIOM's global
`w_count` reduction failed because it was *global*; the conditional versions keep
the residual evidence-mass of `A` intact. Mechanism (c) instantiates the
query-conditional residual deterministically.

---

## 5. Synthesis — concrete mechanisms (ordered by expected precision/risk)

> Design constraints honoured: deterministic, zero-training, env-gated A/B
> (per AGENTS.md HARD RULE 4), **conditional not universal** (the axis on which
> the three failed fixes all fell). All three mechanisms only fire when the
> question structure says the named entity is an anchor; identity questions
> ("What/Who is X?") keep today's behavior.

### Mechanism (a) — Exclusion-cue detection + anchor-set penalty (highest precision, narrowest coverage)

**Mechanism.** A NegEx-style surface cue lexicon + scope window (§1.3) that
detects exceptive/exclusion constructions: `the other one`, `the other`, `other
than`, `besides`, `apart from`, `except`, `excepting`, `excluding`, `rather
than`, `instead of`, `who else`, `another` (as "was another X who…"), `but not`,
`not … or`. When a cue is present **and** the question names ≥1 entity, for
**every** named entity apply `qp_full` (0.2), and require the winner to satisfy
`has_struct` (`conn_avg>0 ∨ hop2_avg>0 ∨ ppr>τ`) — the `has_struct` predicate M1
already computes at engine.rs:678. Optionally also subtract `Σ(−anchor_i)` from
the query hypervector feeding `rel` (VSA-NOT, §2.2) as a supplementary shove.

**Evidence.** von Fintel 1993 (exceptive = set subtraction) \[V\]; NegEx 2001
(deterministic cue+scope detection) \[V\]; NEST-KGQA 2026 (negative constraints
need an explicit logical-form operator) \[V\]; CONDAQA/NeQA/Kassner-Schütze
(negation is a distinct, targeted capability — targeted fixes, not global ones)
\[V\].

**Implementation.** New function `detect_exclusion(query) -> bool` (regex/word-set
on the already-lowercased token stream; scope window ±3 tokens, as in NegEx),
consumed in `extract_answer` before `query_penalty` selection. Env gate
`AXIOM_V2_EXCL` (default 0). ~50–80 LOC, no new dependencies.

**Expected risk.** LOW–MEDIUM. Precision risk is the additive "besides" reading
("besides being a singer, …" — clause-initial, followed by a verb form) and "not"
scope; both are handled by restricting cues to the exceptive forms and requiring a
named entity within the scope window. Coverage is limited to the exclusion
question family (one slice of the ~20% class); it does not touch the
attribute/co-anchor families (those go to (b)). This is a **gate on the argmax**,
not a re-weighting — immune to the fusion-normalisation failure class.

### Mechanism (b) — Answer-position determination: anchors vs answer variable (keystone)

**Mechanism.** A deterministic query-focus classifier (no POS/NER) that labels the
answer position:

| Trigger (surface) | Named entities are… | Answer position |
|---|---|---|
| `What/Who is X?` with no possessive / `of`-PP / co-anchor | possibly the answer | `Identity` (today's mild 0.6 penalty; **unchanged**) |
| `X's <N>`, `<N> of X`, `What <N> did/was X …`, `for X and Y`, `by X and Y`, `between X and Y`, `with X and Y` | **anchors** | `ObjectNode`/`SharedValue` — a node **connected** to the anchors |
| exclusion cue (from (a)) | **negative anchors** (excluded set) | `Excluded` — same-relation completion, `∉ anchors` |

For `ObjectNode`/`SharedValue`/`Excluded`: named anchors get `qp_full`;
**non-anchor candidates with `has_struct=false` cannot win** (hard gate on the
winner, preserving score magnitudes and thus avoiding the fusion failure class).
This is the query-graph invariant (§3): the answer is the node *not* mentioned,
*reachable* from every anchor, and of the head-noun type.

**Evidence.** STAGG (answer = λ-variable; anchors grounded) \[V\]; Semantic
Enrichment (topic entity) \[V\]; Bao 2016 (constraints) \[V\]; CWQ (complex
constraints) \[V\]; Li & Roth (answer-type orthogonal to named entities) \[V\];
QDMR/Break ("the other one" = set-difference step) \[V\].

**Implementation.** `classify_answer_position(query, named_ids) -> {Identity |
ObjectNode | SharedValue | Excluded}` — regex rules on the same token stream as
(a): possessive apostrophe-s, `of`-PP, coordinator `and` with two named entities,
`for/by/between/with` + named entities. Consumed in `extract_answer` to select
penalty tier and to gate the winner. Env gate `AXIOM_V2_POS` (default 0).

**Expected risk.** MEDIUM — the load-bearing decision is Identity vs anchor, and
misclassification of an identity question (Milky-Way case) into anchor mode would
zero the correct answer. **Mitigation: default to Identity**; flip to anchor mode
*only* on the high-precision surface triggers above (possessive, `of`-PP,
co-anchor `and/for/by/between/with`, exclusion cues). These triggers are
near-unambiguous in trivia questions and cover exactly the intent families the
user enumerated. Additionally, `has_struct` uses the same thresholds M1 already
validated, so no new hyper-parameters enter the linear sum.

### Mechanism (c) — Query-conditional count: residual evidence-mass for anchors (the scoring fix)

**Mechanism.** When (b) labels the question as non-Identity, for each **anchor**
`a` set the `heur` count contribution to **zero** (the anchor's mentions are
topic-context, not answer-evidence), while leaving `w_count` untouched for all
non-anchor candidates. Formally: `heur(e) = w_count · count(e) · [e ∉ anchors] −
len_pen + …`. Optionally a *soft* variant: keep `count(a)` but add a residual
term `count(a) − expected_count(a | topic)` using the anchor's marginal frequency
across the corpus (a deterministic, offline-computable analogue of topic
signatures / residual IDF, §4.2). The soft variant is safer for borderline
identity questions.

**Evidence.** Lin & Hovy topic signatures \[V\]; Church & Gale residual IDF \[V\];
Lavrenko & Croft query-conditioning \[K\]; SWAT/KESM/GUM-SAGE salience \[V\].

**Implementation.** A `Set<usize>` of anchor ids threaded from (b); one line
guarding `heur`. Env gate `AXIOM_V2_RESID` (default 0), with a
`AXIOM_V2_RESID_SOFT` sub-switch. ~10 LOC once (b) exists.

**Expected risk.** MEDIUM, **conditional only** — it is strictly gated on (b), so
a (b) misclassification propagates. It does **not** re-propose global count-weight
reduction (24.8→11): the discount is per-candidate and only for anchors, so
Valens's `w_count·count(Valens)` survives. This is the difference between the
failed global fix and the published conditional measures.

### The VSA-NOT supplement (do not run alone)

Add `Q' = bundle(Q, −R_1, −R_2, …)` to the `rel` cosine when (a) fires. Rationale:
it is the literal published encoding of NOT in bipolar VSA (Kanerva 2009) and the
only mechanism that places the exclusion inside the query representation (NEST's
logical-form lesson). **Do not expect it to win alone**: at D=2048 the effect is
`Δcos ≈ ±0.02–0.04`, i.e. below AXIOM's measured noise floor (RCA: σ≈0.02), and
it leaves `overlap`/`heur` untouched. Treat it as a tie-break shove inside the
already-gated pipeline, not as a ranking signal.

### Recommended order of attack and A/B plan

1. **(a) alone** — cheapest, highest precision, env `AXIOM_V2_EXCL`; measures the
   exclusion subset of the ~20%.
2. **(b) alone** — env `AXIOM_V2_POS`; measures the attribute/co-anchor families.
3. **(b)+(c)** — env `AXIOM_V2_POS` + `AXIOM_V2_RESID`; the full conditional-count
   stack. Run (c) soft first, then hard.
4. Add the VSA-NOT shove only as a within-band tiebreak, and only after (1)–(3)
   show a stable delta on the full 318-record bench.

All gates default off → zero regression risk to the v16 baseline; every change is
revertible per AGENTS.md HARD RULE 4.

---

## Appendix — Citation registry

**Verified this session \[V\]:**
- Kanerva 2009, *Cognitive Computation* 1(2):139–159, DOI 10.1007/s12559-009-9009-8
- Kleyko et al., *A Survey on HDC aka VSA, Part I*, arXiv:2111.06077
- Kleyko et al., *VSA as a Computing Framework for Emerging Hardware*, arXiv:2106.05268
- Ren & Leskovec, *BetaE*, arXiv:2010.11465
- Zhang et al., *ConE*, arXiv:2110.13715
- Yang et al., *GammaE*, arXiv:2210.15578
- *Logic Embeddings for Complex Query Answering*, arXiv:2103.00418
- *NEST-KGQA: Which bird does not have wings…*, arXiv:2604.14749
- Ravichander et al., *CONDAQA*, arXiv:2211.00295
- *NeQA: Beyond Positive Scaling*, arXiv:2305.17311
- Kassner & Schütze 2020, ACL, DOI 10.18653/V1/2020.ACL-MAIN.698
- *Paraphrasing in Affirmative Terms Improves Negation Understanding*, arXiv:2406.07492
- *Making Language Models Robust Against Negation*, arXiv:2502.07717
- *Negation in Cognitive Reasoning*, arXiv:2012.12641
- Joshi et al., *TriviaQA*, arXiv:1705.03551
- *What Question Answering can Learn from Trivia Nerds*, arXiv:1910.14464
- *Quizbowl: The Case for Incremental QA*, arXiv:1904.04792
- Yih et al., *STAGG*, ACL-IJCNLP 2015, P15-1128, DOI 10.3115/v1/P15-1128
- Berant et al., *Semantic Parsing on Freebase*, EMNLP 2013, D13-1160
- Berant & Liang, *Semantic Parsing via Paraphrasing*, ACL 2014, P14-1133
- Sun et al., *Open Domain QA via Semantic Enrichment*, WWW 2015, DOI 10.1145/2736277.2741651
- Li & Roth, *Learning Question Classifiers*, COLING 2002, C02-1150
- Talmor & Berant, *CWQ*, NAACL 2018, N18-1059
- Wolfson et al., *Break It Down (QDMR)*, TACL 8:183–198, DOI 10.1162/tacl_a_00309
- Gardner et al., *Contrast Sets*, Findings EMNLP 2020, DOI 10.18653/V1/2020.FINDINGS-EMNLP.117
- Bao et al., *Constraint-Based QA with KG*, COLING 2016, C16-1236
- Ganea & Hofmann, *SWAT*, arXiv:1804.03580
- Xiong et al., *KESM*, arXiv:1805.01334
- *GUM-SAGE*, arXiv:2504.10792
- *Finding IDF Information in BERT*, arXiv:2202.12191
- Lin & Hovy, *Topic Signatures*, COLING 2000, C00-1072
- Church & Gale, *Poisson Mixtures*, Natural Language Engineering 1(2), 1995
- von Fintel, *Exceptive Constructions*, NLS 1(2):123–148, 1993, DOI 10.1007/BF00372560
- Chapman et al., *NegEx*, J. Biomed. Inform. 34(5):301–310, 2001, DOI 10.1006/jbin.2001.1029
- *HDReason*, arXiv:2403.05763; *PathHD*, arXiv:2512.09369
- Zheng & Zhang, *Structural Query Patterns*, arXiv:1910.09760
- Wu et al., *gRGCN*, arXiv:2101.01510
- *Better Query Graph Selection for KBQA*, arXiv:2204.12662
- *Semantic Parsing for QA over KGs*, arXiv:2401.06772
- *How Knowledge Popularity Influences…*, arXiv:2505.17537
- *Quantifying the Impact of Cognitive Biases in QA*, arXiv:1909.09633
- *Efficient Hyperdimensional Computing*, arXiv:2301.10902
- *The hypergeometric test performs comparably to TF-IDF…*, arXiv:2002.11844

**Canonical, cited from knowledge (not re-verified this session) \[K\]:**
- Sparck Jones 1972, *J. Documentation* 28(1):11–21 (IDF)
- Robertson & Zaragoza 2009, *FnTIR* 3(4):333–389 (BM25)
- Salton & Buckley 1988, *IP&M* 24(5):513–523 (term weighting)
- Lavrenko & Croft 2001, *SIGIR* (relevance-based language models)
- Gayler 1998/2003, MAP / multiplicative binding
- Plate 1995/2003, Holographic Reduced Representations
- Moltmann 1995, *Ling. & Phil.* 18(3):223–280 (exception sentences)
- Hoeksema 1995, *The semantics of exception phrases* (CSLI)
- Rooth 1992, *NLS* 1:75–116 (association with focus)
- Rachkovskij & Kussul 2001, *Binary sparse distributed representations*
