# AXIOM — โครงการวิจัยแบบครบถ้วน (Self-Contained Summary for Further Research)

> **เอกสารนี้จงใจให้ self-contained** — อ่านจบได้เข้าใจทั้งโปรเจกต์โดยไม่ต้องเปิดไฟล์อื่น
> เหมาะสำหรับนำไปค้นคว้า/คิดงานวิจัยใหม่ในมุมอื่นๆ เพื่อพัฒนาต่อ
> อัปเดต: 2026-08-11 · v15
>
> **⚠️ 2026-08-13 REALITY CHECK:** ตัวเลขและสถานะในเอกสารนี้เป็น v15
> (candidate 24.53% substring) — **metric นั้นโปะ ~2×** (จริง 13.84-16% exact)
> และมีข้อความ aspirational ที่ถูก supersede แล้ว อ่าน
> `docs/STATUS_VISION_ASSESSMENT.md` ก่อน ตัวเลขซื่อตรงล่าสุด: exact 16.04%,
> f1 17.92%, strict_recall 55.03%.

---

## 1. ภาพรวมโปรเจกต์ (30 วินาที)

**AXIOM** (Algebraic neXt-token Inference On Memory) — ระบบถาม-ตอบ + reasoning
แบบ **deterministic 100% · zero-training · CPU-only · ~18MB** สร้างด้วย Rust 18 crates
สถาปัตยกรรม: **VSA (hyperdimensional computing, d=2048 random bipolar) + Knowledge Graph**

**ปรัชญา:** แทนที่ neural network + training ด้วยพีชคณิต (bind/bundle/permute)
+ กราฟความรู้ + energy-based traversal — ตอบได้โดยไม่ต้อง train

| ด้าน | สถานะ |
|---|---|
| TriviaQA candidate (เลือกคำตอบถูก) | **24.53%** |
| Answer entity recall (คำตอบอยู่ใน graph) | **76.10%** |
| Substring accuracy (ประโยคมีคำตอบ) | **23.27%** |
| Evidence answer recall (คำตอบใน evidence) | **99.69%** |
| Latency | ~100ms |
| Multi-hop reasoning (taught facts) | ✅ บางกรณี ("sky is blue → blue has short wavelength") |
| Compositional reasoning ที่แท้จริง | ❌ ยังพัง |
| "คุยเหมือน LLM" | ❌ ยังห่างไกล (เป็น template) |

> **ข้อควรระวัง:** ตัวเลข TriviaQA เป็น **pipeline diagnostic บน evidence-ingested**
> (ระบบได้ evidence มาชี้เป้าแล้ว) ไม่ใช่ open-domain benchmark score จริง

---

## 2. ปัญหาหลักที่ยังไม่ได้แก้ (The Core Gap)

**ช่องว่าง 52pt:** ระบบ *หา* คำตอบได้ 76% (recall) แต่ *เลือก* ถูกแค่ 24.5% (candidate)

```
recall 76.10% ────► candidate 24.53%
   "answer อยู่ใน graph แล้ว"      "เลือก entity ถูกต้อง"
         └──────── 52pt gap ────────┘
```

**กลไกที่พิสูจน์แล้ว (จาก 171 failure cases):**

| โหมด | จำนวน | กลไก |
|---|---|---|
| **M1 Overlap dominance** | ~4 | entity ที่ชื่อตรงคำถามชนะ ("Jaws (film)" ชนะ "Bruce") |
| **M2 Near-tie noise** | ~22 | junk ชนะ gold แค่ <0.6pt; ~5/22 ถูก VSA noise ตัดสินล้วน |
| **M3 Hub/degree** | ~5 | entity ปรากฏบ่อย (FA Cup, Cricket) ชนะด้วยความถี่ |
| **M4 Structural conn=0** | ~6 | gold เป็น node แต่ไม่มี link ถึง query |
| **M5 Junk surfaces** | ~15 | "Cast *Gregory Peck" ยังเข้า graph |
| **Deep-rank** | ~149 | gold เป็น entity ดีแต่ติด rank #6-114 (โดนทับ) |

---

## 3. ระบบปัจจุบันทำงานยังไง (Architecture)

### 3.1 Data flow
```
Input → Intent classify (tle-afc::vsa_intent)
     → extract_query_entities (VSA fuzzy + punctuation-stripped matching)
     → Decompose (evidence → triples, tle-axiom-gen::decompose)
     → is_fact_worthy filter (ingestion gate)
     → KnowledgeGraph (entities + triples + relations + adjacency)
     → Inference rules (inversion/transitivity, tle-axiom-gen::inference)
     → Hub-corrected PPR (graph::personalized_pagerank)
     → extract_answer (rank entities, 7 signals, env-tunable weights)
     → Beam search + Linearize → answer entity + sentence
```

### 3.2 Scoring (extract_answer)
```
score(e) = w_conn·conn_avg + w_role·role_avg + w_hop2·hop2_avg
         + w_ov·overlap + w_vsa·vsa_cosine + w_heur·heur + w_ppr·PPR
         แล้วคูณ query_penalty (intent-aware: Where=0.2, What/Who=0.6)
argmax → answer
```

สัญญาณ 7 ตัว:
- **conn_avg** — avg connectivity ถึง query entity (relation-typed: located_in=2.0, mentions=0.8)
- **role_avg** — intent bias (Who→subject, What/Where→object)
- **hop2_avg** — 2-hop bonus
- **overlap** — คำถาม word ในชื่อ entity (weight 0.05)
- **vsa_cosine** — ≈ noise (N(0,1/√2048) กับ random codebook)
- **heur** — 0.2·count − len_pen + cap_bonus + det_pen (count = evidence mass)
- **PPR** — relative PageRank: log π_q(e) − log π(e) (hub-corrected)

### 3.3 Crate map (18 crates)
- **Core math:** tle-vsa, tle-transition (TBA), tle-resonator, tle-clifford, tle-tda-router
- **Knowledge/reasoning:** tle-engram, tle-knowledge, tle-afc (incremental store/analogy/attractor/intent), tle-axiom-gen (KG/decompose/search/rank)
- **VSA-LM (Path C):** tle-vsa-lm (TBA+Engram+Reservoir+KnowledgePrior+cosine decoder)
- **Apps:** tle-deepman (REPL), tle-chat, tle-pipeline, tle-memory, tle-decoder, tle-bench

---

## 4. อะไรได้ผลจริง (Bench-Verified Wins)

| Gain | เทคนิค | ประเภท |
|------|--------|--------|
| **+4.72pt recall** | proper-noun boundary precision | decomposition quality |
| **+2.2pt candidate** | query-entity punctuation fix ("O'Hare" หลบ penalty) | bug-fix ที่ gate |
| **+0.63pt** | hub-corrected PPR (relative PageRank) | สัญญาณใหม่เชิงโครงสร้าง |
| **+0.63pt** | overlap weight 0.15→0.05 | calibration |
| **+0.32pt** | subject resolution (copula handling, passive *_by relations) | decomposition quality |

**รูปแบบที่ได้ผลซ้ำๆ:** (1) แก้ bug ที่ gate (2) decomposition quality (3) สัญญาณใหม่เชิงโครงสร้าง

---

## 5. อะไรล้มเหลว (6 Negative Rounds — อย่าไปทำซ้ำ)

| # | เทคนิค | ผล | ทำไมล้มเหลว |
|---|--------|-----|------------|
| 1 | coordinate-ascent weight tuning + IEF | flat; IEF 5-10% | linear sum เป็น flat local optimum |
| 2 | RRF rank fusion | 11.95-15.41% | rank fusion ทำลาย magnitude gap |
| 3 | conformal p-value + log-odds PoE | 12.58-19.18% | p-value normalization = เดียวกับ #2 (12.58% เจอ 2 ครั้ง) |
| 4 | Datalog type-veto (relation heuristic) | 19.81% | "won"→Person ผิด; ต้อง POS ไม่ใช่ relation |
| 5 | NP-surface filter ที่ graph | 23.58-23.90% | entity จริง legitimately มี character พวกนั้น |
| 6 | VSA clamp + count re-weight (near-tie) | ทุกแบบ regress | VSA noise ±0.08 พลิก tie; count = evidence mass จริง |

**บทเรียนกลาง:** gap 52pt **แก้ไม่ได้ด้วยการ re-combine สัญญาณเดิม 6 ตัว** —
ทุก fusion/normalization ทำลายข้อมูล magnitude ที่ linear sum เก็บไว้
ต้องใช้ **สัญญาณใหม่** + **graph ที่สะอาดขึ้น**

---

## 6. สถาปัตยกรรมที่เคยถูกเสนอ (จาก research docs) — สถานะจริง

| แนวคิด | แหล่ง | สถานะ |
|--------|-------|--------|
| RRF (rank fusion) | Cormack 2009 | ❌ ทดสอบแล้ว regress |
| Conformal p-value fusion | Vovk & Wang | ❌ ทดสอบแล้ว regress |
| Hub-corrected PPR | Milne-Witten | ✅ ใช้แล้ว (+0.63) |
| Datalog/Ascent inference rules | Gilray et al. | ⚠️ infra สร้างแล้ว ไม่มี metric gain (graph ขาด intermediate node) |
| Resonator networks (factorization) | NSF/arXiv | ⬜ ยังไม่ได้ทดสอบ (VSA เป็น noise → คาดได้ผลน้อย) |
| POS/NP-chunking (DFA) | — | ⬜ surface filter ล้มเหลว; clause/subject typing ยังเป็นงานเปิด |
| Datalog hard filter (F1 type) | — | ❌ relation-heuristic พัง; ต้อง lexicon/POS จริง |

---

## 7. คำถามวิจัยที่ยังเปิด (เอาไปค้นคว้าต่อได้เลย)

1. **Decomposition/clause typing แบบ deterministic** — ปรับ subject resolution
   ให้สมบูรณ์ (แก้ "Zadok the Priest were" → "Zadok the Priest"), เปิด transitivity
   location chain, ป้อน answer-type — งานที่ data ชี้ว่ามีผลสูงสุด
2. **สัญญาณใหม่ที่ orthogonal ต่อ 6 สัญญาณเดิม** — แบบที่ PPR ทำได้ (+0.63)
   ยังมีอะไรอีก? (clause-role encoding, evidence-path count, co-occurrence)
3. **POS/NER-lite lexicon** — ทางเดียวที่จะ break M2 near-tie (entity-type veto)
   โดยไม่ misfire เหมือน relation heuristic
4. **VSA ที่มี signal จริง** — ตอนนี้ cosine เป็น noise กับ random codebook;
   ต้อง learned/structured codebook แต่ขัดกับ deterministic constraint
5. **Compositional reasoning** — "cat is an animal → animals have hearts"
   ยังพัง; ต้อง inference layer ที่ทำงานจริง ไม่ใช่ key-value lookup

---

## 8. ข้อจำกัดที่งานวิจัยใหม่ต้องสอดคล้อง

- ✅ **Deterministic 100%** (same input → same output)
- ✅ **Zero training / zero gradient**
- ✅ **CPU-only, <50MB memory**
- ✅ **VSA d=2048 random bipolar** (เปลี่ยน codebook = เสีย determinism)
- ✅ **Rust**

---

## 9. ไฟล์ที่เกี่ยวข้อง (ถ้าอยากเจาะลึก)

| ไฟล์ | เนื้อหา |
|------|--------|
| `docs/AGENT_HANDOFF.md` | สถานะ + next steps (อ่านก่อน) |
| `docs/ROADMAP.md` | task board |
| `docs/PROGRESS_LOG.md` | journal |
| `docs/LESSONS_LEARNED.md` | anti-pattern registry |
| `docs/ROOT_CAUSE_ANALYSIS.md` | RCA ของ 52pt gap |
| `docs/RESEARCH_REQUEST.md` | 6 คำถามวิจัยที่อยากได้คำตอบ |
| `docs/SESSION_RESEARCH_SUMMARY.md` | 6 negative + 4 gains |
| `docs/RANKING_RESEARCH_SYNTHESIS.md` | ranking math synthesis (3 memos) |
| `docs/research/` | paper draft, algorithm specs, prior-art (katgpt-rs), ranking memo |
| `AGENTS.md` | session onboarding (auto-loaded) |

---

## 10. ตัวเลขสรุป (state v15)

| Metric | v14 | v15 | Δ |
|--------|:---:|:---:|:---:|
| candidate_answer_accuracy | 19.81% | **24.53%** | +4.72pt |
| answer_entity_recall | 71.38% | **76.10%** | +4.72pt |
| substring_accuracy | 23.90% | 23.27% | -0.63pt |
| evidence_answer_recall | 99.69% | 99.69% | 0 |

*git log: v15 = T1.7, T1.8a, T1.9a, T1.9b/c, T1.10a-f + docs/README/AGENTS.md*
