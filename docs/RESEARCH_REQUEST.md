# AXIOM — ปัญหาที่เจอ + ไอเดียความตั้งใจ (สำหรับค้นหางานวิจัย)

> เอกสารนี้เขียนเพื่อให้เอาไปค้นแนวทาง/paper มาช่วยแก้ปัญหา
> อัปเดต: 2026-08-11 · ระบบ: AXIOM (non-neural, deterministic, VSA d=2048 random bipolar + Knowledge Graph, CPU-only, zero-training, Rust)

---

## 1. ตัวเลขปัจจุบัน (TriviaQA 318 records)

| Metric | ค่า | เป้า | สถานะ |
|--------|:---:|:---:|:---:|
| candidate_answer_accuracy | **24.21%** | 40% | เลือกคำตอบถูก |
| answer_entity_recall | **76.10%** | 80% | คำตอบอยู่ใน graph แล้ว |
| substring_accuracy | 22.33% | 50% | ประโยคที่ generate มีคำตอบ |
| evidence_answer_recall | 99.69% | 99.7% | คำตอบอยู่ใน evidence text ✓ |

**Gap หลัก:** recall 76% (answer อยู่ใน KG แล้ว) แต่ candidate 24% (เลือกถูก)
→ ช่องว่าง ~52pt อยู่ที่ **การเลือก/จัดอันดับ entity** ไม่ใช่ retrieval

---

## 2. ปัญหาเชิงกลไก (จากวิเคราะห์ 165 failures)

### 2.1 ระบบปัจจุบัน score entity แบบนี้
```
score(e) = w_conn·conn_avg(e) + w_role·role_avg(e) + w_hop2·hop2_avg(e)
         + w_ov·overlap(e)     + w_vsa·vsa_cosine(e) + heur(e)
         (คูณ query_penalty ถ้า entity ถูกพูดถึงในคำถาม)
```
แล้วเลือก argmax สัญญาณทั้ง 6 มี scale ต่างกันมาก (overlap ~0-50, conn ~0-2)

### 2.2 Failure modes (วัดจาก 165 ตัว)

| Mode | จำนวน | กลไก |
|---|---|---|
| **M1 overlap dominance** | 4 (เคย 21) | entity ที่ชื่อตรงคำถามชนะ ("Jaws (film)" ชนะ "Bruce") — แก้ไปบางส่วนด้วย query penalty |
| **M2 near-tie noise** | 25 | junk ชนะ gold แค่ <0.6pt; **8/25 ตัวตัดสินด้วย VSA noise ล้วน** (conn/role/heur เหมือนกันเป๊ะ) |
| **M3 hub/degree** | 5 | entity ปรากฏบ่อย (FA Cup, Cricket) ชนะด้วยความถี่ |
| **M4 structural conn=0** | 6 | gold เป็น node แต่ไม่มี link ถึง query entity |
| **M5 junk entity surfaces** | 15 | "Cast *Gregory Peck", "in 1893 a Second Division" ยังเข้า graph |
| **deep-rank golds** | 149 | gold เป็น entity ดีแต่ติด rank #6-114 (เชื่อมแล้วแต่โดนทับ) |

### 2.3 ข้อเท็จจริงที่พิสูจน์แล้ว (สำคัญต่อคนออกแบบ)

1. **VSA cosine กับ random codebook ≈ N(0, 1/√2048) ≈ noise** — ใช้เป็น primary signal ไม่ได้
   (ทฤษฎี: random bipolar d-dim ให้ cos ~ N(0,1/√d))
2. **Linear weighted sum ของ signal ต่าง scale = flat local optimum** — ปรับ weight 5+ ครั้งแล้วไม่ขึ้น
3. **Equal-weight/rank fusion ล้มเหลว** (percentile 12.58%, RRF 11.95%) — ต้อง calibration ไม่ใช่ equal
4. **Count term = evidence mass ไม่ใช่ hub inflation** — ลด weight แล้วแย่ลง (17.61%)
5. **คำถาม type สองแบบสู้กัน**: "What is X?" → X เป็นคำตอบเอง (ต้องไม่โดนลงโทษ)
   ต่างจาก "Where is X?" → X ไม่ใช่คำตอบ (ต้องโดนลงโทษ)

---

## 3. สิ่งที่ทำแล้วได้ผล (อย่าทำซ้ำแนวเดิม)

| ทำแล้ว | ผล |
|---|---|
| query-penalty fix (O'Hare/Jaws หลบ penalty เพราะ punctuation) | +2.2pt |
| hub-corrected PPR (relative PPR, Milne-Witten) เป็น signal ที่ 7 | +0.63pt |
| proper-noun entity boundary (T1.7) | recall +4.72pt |
| overlap weight 0.15→0.05 | +0.63pt |
| **ล้มเหลว:** DDTree (4 ครั้ง), percentile equal-weight, IEF, semantic-in-scoring, RRF alone, substring-query-matching | reverted |

---

## 4. ไอเดีย/ทิศทางที่ตั้งใจ (อยากได้งานวิจัยมาสนับสนุน)

### A. การเลือก/จัดอันดับคำตอบ (แก้ gap ~52pt)
- **Hard filter ก่อน rank** — answer-type (Who→คน, Where→สถานที่), relation-reachability
  → "veto ไม่ให้ scale ชนะได้" ต่างจาก weight tuning
- **Conditional weighting** — "name-match มีค่าต่อเมื่อมี connectivity ด้วย"
  (linear weight ตัวเดียวแก้ทั้งสอง regime ไม่ได้)
- **Calibrated log-odds / product-of-experts** — เปลี่ยน raw score เป็น P(answer|bin)
  แล้วรวมแบบ log — แก้ scale mismatch โดยไม่ equal-weight
- **Temperature sharpening** หลัง calibrated (แก้ near-tie ที่ VSA noise ตัดสิน)
- **Sigmoid-never-softmax** (จาก katgpt) — แต่ละ candidate ได้คะแนนอิสระ ไม่แย่ง mass กัน

### B. Decomposition / เข้าใจภาษา (เพิ่ม recall + คุณภาพ graph)
- **POS / NP-chunking** (deterministic, lexicon-based) — ตัด entity boundary ให้แม่น
  แก้ M5 junk surfaces, ช่วย infer answer-type
- **Clause typing** — แยก subject/object/place/time
- **NER-lite** — รู้ว่า Scotland=สถานที่, มี country อยู่ใน lexicon
  → enable F1 answer-type filter

### C. World model / ตรรกะ (ตอบคำถามที่ต้อง reasoning)
- **Typed KG + inference rules** (deterministic, ไม่ train):
  - Transitivity: A>B, B>C ⟹ A>C
  - Inversion: "is mother of" ⟺ "has mother"
  - Class hierarchy: cat ⊂ animal
  - Comparator semantics: largest/smallest → ต้อง sort ไม่ใช่ rank
- คำถาม "How many teams end in United", "who was the OTHER musician" พังเพราะไม่มี layer นี้

### D. Fluency / ตอบเหมือนพูด (ถ้าจะทำ Path C VSA-LM)
- VSA-LM ผ่าน corpus ขนาดใหญ่ → generalization
- ไม่ใช่ gate ของ TriviaQA

---

## 5. คำถามวิจัยที่อยากได้คำตอบ

1. **Non-neural / deterministic KGQA**: มีงานไหนที่เลือก answer entity ได้แม่น (candidate >60% ของ recall)
   โดยไม่ใช้ neural? (Grecx/GQA/PullNet มี neural — อยากได้แนว filter+rank แบบ deterministic)
2. **Entity linking / ranking hub-resistant**: วิธี normalize hub โดยไม่เสีย signal ของ entity จริง
   (Milne-Witten PPR ทำแล้ว +0.63 — มีอะไรดีกว่านี้?)
3. **Score combination**: รวมสัญญาณต่าง scale ให้ถูกต้องโดยไม่ train —
   calibrated log-odds vs conformal p-value (Vovk & Wang) vs product-of-experts
4. **Hyperdimensional computing for QA**: งาน HDC ที่ทำ QA จริงโดยไม่ต้อง learned query mask
   (VSA4VQA ต้อง learned mask — เราอยากได้ deterministic)
5. **Symbolic decomposition ไร้ train**: POS/NP-chunking / clause typing แบบ rule-based
   ที่แม่นใน open-domain English text
6. **Logical reasoning แบบ deterministic**: typed KG + inference rules
   ที่แก้คำถาม comparative/quantitative ได้

---

## 6. ข้อจำกัดที่งานวิจัยต้องสอดคล้อง

- ✅ Deterministic 100% (same input → same output)
- ✅ Zero training / zero gradient
- ✅ CPU-only, <50MB memory
- ✅ VSA dimension 2048, random bipolar codebook (เปลี่ยนไม่ได้เพราะ deterministic)
- ✅ Rust

---

*Reference ในตัวเรา: docs/RANKING_RESEARCH_SYNTHESIS.md (มี 3 research memos + references พร้อม URL)*
