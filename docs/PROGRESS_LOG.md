# AXIOM PROGRESS LOG — Chronological Development Journal

> Append-only journal. Newest entry at the TOP. Each entry: date · what changed ·
> measured results. Agents MUST append after every working session.

---

## 2026-08-10 — v14 Track 2 (T2.1-T2.2)

**Commits:** `e42442b`, `cf6b295`

### What changed
- **T2.1 Answer-first generation** — vsalm-wiki wired AxiomGen; answers
  entities directly via extract_answer, VSA-LM as fluency fallback.
  "who is the president of france" → Emmanuel_Macron ✓, "where is paris
  located" → France ✓.
- **T2.2 Average connectivity** — extract_answer normalizes conn/role by
  link count. Fixes hub problem (Macron 197 facts vs Paris 1 strong link).

### Measured results (full 318 bench)
| Metric | v14 Track1 | v14 Track2 |
|--------|:---:|:---:|
| candidate_answer_accuracy | 18.87-19.18% | **19.81%** (+0.6) |
| answer_entity_recall | 71.38% | 71.38% |
| avg_latency | 102ms | 100ms |

### Key findings
- Average-connectivity normalization is correct direction (candidate +0.6pt).
- Extreme hubs (Macron 197 facts) still dominate on Wikipedia QA — link-count
  normalization not enough; needs degree-relative or IEF weighting per fact.
- Answer-first generation produces clean entity answers (not noisy blends).

---

## 2026-08-10 — v14 Track 1 (T1.1-T1.4)

**Commits:** `a66c92f`, `22a800e`, `b489d17`, `155cb09`, `e8a870a`

### What changed
- **System**: continuous-dev system (ROADMAP/PROGRESS_LOG/WORKFLOW)
- **T1.1 Adaptive sentence coverage** — REVERTED (VSA signal too weak)
- **T1.2 Lowercase NP extraction** — KEPT (recall +0.3pt, latency 2×)
- **T1.3 Permutation entity consolidation** — KEPT (infra, no regression)
- **T1.4 Relation-typed connectivity** — KEPT (candidate peak 19.18%)

### Measured results (full 318 bench, v13 → v14)
| Metric | v13 | v14 |
|--------|:---:|:---:|
| candidate_answer_accuracy | 18.87% | **18.87-19.18%** |
| answer_entity_recall | 71.07% | **71.38%** (+0.3) |
| avg_latency | 213ms | **102ms** (2×) |

### Key findings
- T1.1 lesson: VSA cosine can't discriminate sentences — needs semantic
  codebook (T3.1), not threshold math.
- Relation typing on connectivity works (+0.3pt candidate) once graph is
  clean (T1.2 reduced junk).
- Latency halved from 213→102ms via cleaner graph (fewer junk facts).

---

## 2026-08-10 — v13 (continuous-dev system bootstrap)

**Commits:** `a66c92f`

### What changed
- **v11**: Diagnostic mode (`AXIOM_TRIVIA_DEBUG` prints top-5 entity score
  breakdown). Root cause found: OVERLAP DOMINANCE — entities named in question
  get 20-54 overlap points, drowning answer entities (0 overlap).
- **v11 fix**: connectivity-first scoring — overlap weight 1.0→0.15,
  query-named penalty ×0.2. candidate 16.35%→18.55% (+2.2pt).
- **v12**: subject preposition truncation, comma entity consolidation,
  MorphTokenizer wired into extract_query_entities, wider sentence coverage
  (6 overlap sentences). candidate→18.87%, recall→71.07%.
- **v13**: vsalm-wiki binary (Wikipedia fetch→clean→decompose→KnowledgePrior→QA).
  Query-weighted fact filtering (noisy, reverted).

### Measured results (full 318 bench)
| Metric | before v11 | v13 |
|--------|:---:|:---:|
| candidate_answer_accuracy | 16.35% | **18.87%** |
| answer_entity_recall | 70.44% | **71.07%** |
| substring_accuracy | 23.58% | 23.58% |
| avg_latency | 198ms | 213ms |

### Infrastructure completed
- 32× codebook compression (Phase 2), GF(2) encoding (Phase 3)
- TBA TopK cache 12K tok/s (Phase 4), KnowledgePrior O(1) hash index
- vsalm-wiki Wikipedia ingestion pipeline (300 facts per 2 pages)

### Key findings / gotchas
- **DDTree dead** — 4 attempts, all regressed. Beam paths don't carry answer.
- **extract_answer weights** (overlap×1, VSA×2, role×3, query×0.2) are optimal.
- **Substring entity consolidation regressed** — only exact-match merge works.
- **Query-weighted KnowledgePrior** noisy — prefer specificity ranking.
- **Semantic codebook** (co-occurrence) deferred — corpus too small (<200 words/page).

---

## 2026-08-10 — v10 (cleaner decomposition)

**Commits:** `6112bdc`, `2d4e44c`

### What changed
- mentions/is_related_to gate: require ≥2 words + ≥1 capital non-article word
- preposition truncation for long objects (≥5 words, excludes "of")
- tail entity extraction, bonus overlap 3→4

### Measured results
- candidate 17.30%, recall 69.18% (both records at the time)

---

## 2026-08-10 — v9 (infrastructure)

**Commits:** `d244b02`, `0d98714`, `6a439d9`, `5943739`, `505913b`

### What changed
- Phase 2: BipolarVector lazy decompression (32× memory)
- Phase 3: GF(2) triple encoding (XOR packed bits, 3× faster)
- Phase 4: TBA TopK cache (2-3× gen speedup)
- KnowledgePrior O(1) hash index
- CR2 path-confidence energy term

### Measured results
- codebook: 2GB → 62MB
- gen speed: 4.4K → 12K tok/s
- TriviaQA candidate 16.98%, recall 68.55%
