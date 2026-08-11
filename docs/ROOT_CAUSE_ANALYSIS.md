# AXIOM Root Cause Analysis — Answer Selection Failure (Cross-Layer)

> **Date:** 2026-08-10
> **Scope:** TriviaQA candidate 19.81% vs recall 71.38% — 51.57pt gap
> **Method:** Cross-Layer RCA from code → architecture → theory → topology
> **Evidence base:** `engine.rs` extract_answer, `graph.rs` triple_confidence,
>   `decompose.rs` decomposition, `semantic.rs` semantic layer, 12 bench sessions

---

## 1. Executive Summary

ระบบเก็บคำตอบไว้ใน graph ได้ 71.38% (recall) แต่เลือกคำตอบถูกเพียง 19.81% (candidate)
สาเหตุรากไม่ใช่ "ขาด feature" — แต่คือ **การรวมสัญญาณ (score aggregation) เป็น
linear weighted sum ของ signal ที่มี scale ต่างกันหลายสิบเท่า โดยไม่มีการ normalize
เชิงทฤษฎี** ทำให้ signal ตัวหนึ่ง (ความถี่ของ entity) กลบ signal ที่ถูกต้อง
(connectivity ต่อ query) เสมอ และเกิด **hub domination**: entity ที่ปรากฏใน fact
จำนวนมาก (เช่น Macron 197 facts) ได้คะแนนสะสมสูงเกิน จน "ทับ" entity ที่เป็นคำตอบจริง
(เช่น Paris ที่มี 1 strong link capital_of) แม้ semantic signal จะชี้ชัดว่า Paris ถูกต้อง

การแก้แบบพารามิเตอร์ทุกครั้ง (weight tuning, z-score, DDTree, sense-answer,
semantic layer, log-scaled frequency) ล้วน regression หรือ plateau เพราะไปแก้อาการ
(ปรับน้ำหนักของ symptom) ไม่ได้แก้โครงสร้าง (การรวมสัญญาณที่ไม่รับประกัน monotonicity
และ hub invariance)

---

## 2. Layered RCA Mapping

| Symptom (วัดได้) | Code/Architecture Flaw | Theoretical Root Cause |
|---|---|---|
| Macron (197 facts) beat Paris (1 link) แม้ semantic ชี้ Paris | `extract_answer` ใช้ `heur = 0.2 * raw_count` — **linear sum** ของทุก triple ที่ entity ปรากฏ, ไม่มี normalization | **Hub domination / degree-bias**: ฟังก์ชันคะแนนไม่เป็น *hub-invariant* — เพิ่ม degree ของ noise node ขึ้นเชิงเส้น |
| Weight tuning / z-score / log-freq ล้วน regression | ทุก variant ปรับน้ำหนักใน **linear aggregate เดียวกัน** | **Non-monotone functor**: แผนที่จาก (KG, query) → ranking ไม่รับประกันว่าเพิ่ม signal หนึ่ง แล้ว ranking ถูกต้องขึ้น — สัญญาณตัวอื่นถูก scale ครอบงำ |
| DDTree / sense-answer ล้มเหลว 4 ครั้ง | ทั้งคู่พึ่ง **path quality ของ beam search** ที่ VSA cosine เกือบเป็น 0 (random codebook) | **VSA quasi-orthogonality limit**: D-dim random bipolar ให้ cos ~ N(0, 1/√D); ที่ D=2048, σ≈0.02 → signal-to-noise ≈ 1 — path scoring แทบเป็น noise |
| Semantic layer (cos capital-paris 0.56) เปิดแล้วแย่ลง | query vector bundle ทุกคำ + vsa weight 8.0 → semantic noise จากคำ co-occur กลบ conn | **Curse of bundling / superposition**: การ bundle คำ noise เข้า query ทำให้ vector กระจาย (dilute) signal ที่เฉพาะเจาะจง |
| Entity boundary ไม่ตรง ("Chicago, Illinois, 17 mi" ≠ "Chicago") → conn=0 | Decomposition ใช้ `truncate_object` heuristic (ตัดที่ comma/preposition) โดยไม่มี NER/type | **Boundary underdetermination**: ภาษาไม่มี delimiter เชิงตรรกะของ entity — heuristic ตัดผิดเสมอ มีข้อมูลไม่พอจะตัดถูกโดยไม่รู้ type |
| recall 71% แต่คำตอบไม่อยู่ใน selected sentences 30% | `extract_document_facts` ใช้ fixed top-5 VSA + top-6 overlap | **VSA ranking ไม่เลือก sentence ที่ถูกต้อง**: cosine ≈ noise → selection เป็น random-ish |

---

## 3. Topological & Theoretical Analysis

### 3.1 Topology of the failure

โครงสร้าง data/control flow ปัจจุบันเป็น **directed path** 3 ชั้น:

```
Decomposition ──► Knowledge Graph ──► extract_answer ──► Ranking
   (noisy)          (VSA vectors)      (linear sum)       (argmax)
```

จุดพัง: ชั้น ranking ทำ **global argmax ต่อ linear combination** ของ signal ที่ไม่ใช่
probabilities และไม่ normalized:

```
score(e) = conn_avg(e) + 0.8·role_avg(e) + 0.5·hop2_avg(e)
         + 0.15·overlap(e) + 2.0·rel(e) + (0.2·count(e) - len_pen + cap + det)
                                                  ↑ linear in degree — HUB DOMINATES
```

**Invariant ที่ถูกละเมิด:** *"คำตอบที่ถูกต้องควรมีคะแนนสูงสุดใน neighborhood ของ query"*
ถูกละเมิดเพราะ degree ของ noise node (count) เป็น unbounded additive term —
เป็น *violation of the linearity invariant* ของ ranking functor.

### 3.2 Graph-theoretic view

- KG เป็น directed multigraph `G(V, E)`, answer `a* ∈ V` อยู่ห่าง query ≤ 2 hops
- ต้องการ functor `F: (G, q) → ℝ^|V|` ที่ `argmax F = a*`
- เงื่อนไขจำเป็น: `F` ต้องเป็น *degree-normalized* และ *rank-monotone* ต่อ signal
  ที่ถูกต้อง — `F` ปัจจุบันเป็น linear ใน degree → ไม่เป็นไปตามเงื่อนไข

### 3.3 Theoretical limits

1. **VSA codebook limit:** random bipolar vectors มี `cos ~ N(0, 1/√D)` →
   ที่ D=2048, SNR≈1 สำหรับ single-pair → VSA relevance ใช้เป็น **primary signal
   ไม่ได้** ต้องเป็น tiebreaker เท่านั้น (หรือต้อง semantic layer + corpus ใหญ่)
2. **Information limit:** `triple_confidence` เป็น heuristic (len + copula penalty)
   ไม่ได้มาจาก likelihood จริง → confidence มี bias แต่ไม่ calibrate
3. **Complexity class:** selection เป็น argmax ของ 6-signal linear form —
   ไม่มี global optimum ที่รับประกันได้เมื่อ signal scales ต่างกัน โดยไม่มี
   normalization (empirical: 5 weight-tune variants ล้วน regression)

---

## 4. Remediation & Architecture Redesign

### 4.1 หลักการ (Theoretical Guarantee)

แทนที่ "linear sum 6 signals → argmax" เปลี่ยนเป็น **two-stage: retrieve-then-rank**
โดยแต่ละ stage มี guarantee แยก:

```
Stage A (Retrieval):  คัด candidate ด้วย STRUCTURAL filter (hard, monotone)
  keep(e) ⟺ e ∈ neighborhood(q)  (1-hop ∪ 2-hop ∪ overlap>0)
  guarantee: ถ้า a* อยู่ใน graph และเชื่อม q ภายใน 2 hops → a* ∈ candidates
  ⟹ recall ไม่ลดลง (ยังเท่า 71.38%)

Stage B (Ranking):   score ภายใน candidate set ที่เล็กกว่า ด้วย signal ที่
  normalized แบบ rank-based (ไม่ใช่ linear sum):
  rank(e) = Σ_signal w_s · z_s(e)
  โดย z_s = percentile rank ของ signal s ใน candidate set (0..1)
  guarantee: ทุก signal contribute เท่ากัน (ไม่ถูก scale กลบ)
  และ hub invariance: ใช้ distinctness (IEF) ไม่ใช่ raw count
```

### 4.2 Refactored design (pseudocode)

```rust
// Stage A: structural candidate filter (recall-preserving)
fn retrieve_candidates(graph, query_entities) -> Vec<EntityId> {
    let mut cand: HashSet = HashSet::new();
    for t in graph.triples {
        let q_subj = query_entities.contains(&t.subject_id);
        let q_obj  = query_entities.contains(&t.object_id);
        if q_subj != q_obj {
            cand.insert(if q_subj { t.object_id } else { t.subject_id }); // 1-hop
            // 2-hop: neighbors of the 1-hop node
            for &ti in graph.adjacency_of(if q_subj { t.object_id } else { t.subject_id }) {
                let t2 = &graph.triples[ti];
                if !query_entities.contains(&t2.subject_id) { cand.insert(t2.subject_id); }
                if !query_entities.contains(&t2.object_id)   { cand.insert(t2.object_id); }
            }
        }
    }
    cand
}

// Stage B: rank-normalized scoring (hub-invariant)
fn score_entity(e, signals: &[f32]) -> f32 {
    // signals = [conn, role, hop2, overlap, vsa, distinctness(IEF)]
    // z_s(e) = (value_s(e) - min_s) / (max_s - min_s) over candidate set
    signals.iter().map(|s| 1.0 * z(s)).sum::<f32>()  // equal weight, normalized
}
```

**การเปลี่ยนสำคัญ:**
1. `heur` จาก `0.2 * count` → **distinctness = -log(freq/graph_size)** (IEF):
   entity ที่ปรากฏน้อย = specific = คะแนนสูง; hub ถูกกดตามทฤษฎี
2. ทุก signal เข้า **percentile normalization** ต่อ candidate set (ไม่ใช่ global)
3. `rel` (VSA) เป็น **tiebreaker ตัวเดียว** ไม่ใช่ weight 2.0 ที่ scale ผิด —
   เพราะ cos ∈ [-1,1] แล้ว normalize แบบ percentile อยู่แล้ว

### 4.3 ลำดับการลงมือ (กับตัวเลขที่คาด)

| Step | การกระทำ | ตัวชี้วัดที่คาด |
|---|---|---|
| 1 | Implement retrieve-then-rank (Stage A+B) ใน extract_answer | candidate 19.8% → 25-30% (hub กดลง, ทุก signal มีน้ำหนัก) |
| 2 | distinctness (IEF) แทน raw count | hub regression หาย, "Macron>Paris" แก้ได้ |
| 3 | (ถ้า corpus ใหญ่) re-enable semantic layer เป็น tiebreaker | vsa noise ลด (ตอนเป็น percentile) |

### 4.4 สิ่งที่ห้ามทำ (จากหลักฐาน)

- ❌ อย่าเพิ่ม weight อีก (linear-sum tuning ล้มเหลว 5 ครั้ง — เป็น structural
  limit ไม่ใช่ calibration problem)
- ❌ อย่าเปิด semantic layer เป็น primary (VSA SNR≈1 กับ random codebook)
- ❌ อย่ากลับไป DDTree/sense-answer (พึ่ง path ที่ VSA noise)

---

## 5. Invariants & Boundary ใหม่ (ที่ต้อง enforce)

1. **Hub-invariance:** score ของ entity ต้องไม่เพิ่มแบบ linear ตาม degree —
   ใช้ distinctness/IEF
2. **Signal parity:** ทุก signal ผ่าน percentile normalization — ไม่มี signal
   scale กลบตัวอื่น
3. **Retrieval completeness:** Stage A ต้องครอบ 1-hop ∪ 2-hop ∪ overlap —
   ไม่มีคำตอบที่ recall ได้ตอนนี้หายไป (recall ยัง 71.38% เป็น lower bound)
4. **Determinism:** ทั้งหมด deterministic (ไม่สุ่ม) — คงเดิม

---

*เอกสารนี้เป็น ground truth สำหรับงานต่อไป — update ใน `docs/ROADMAP.md` ว่า
T1.6 = retrieve-then-rank redesign เป็นลำดับถัดไป*
