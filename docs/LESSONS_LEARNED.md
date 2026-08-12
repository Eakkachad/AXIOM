# AXIOM — Lessons Learned & Anti-Pattern Registry (ข้อผิดพลาดที่ห้ามทำซ้ำ)

> **โพสต์แรก:** 2026-08-11 · **หลักการ:** ก่อนลองอะไรใหม่ ให้อ่านไฟล์นี้ก่อน
> ถ้าอยู่ใน list นี้ = ต้องมีแนวทางใหม่ที่ต่างโดยสิ้นเชิง ไม่ใช่ variant เดิม
> ทุกข้อมีหลักฐาน bench จริง (ไม่ใช่ conjecture)

---

## TL;DR — ข้อห้ามเด็ดขาด (อ่านก่อนทุก experiment)

| # | ห้ามทำ | ผลที่วัดได้จริง | ทำไมล้มเหลว |
|---|--------|----------------|-------------|
| 1 | **ปรับ weight linear sum ใหม่** | flat 5+ ครั้ง | linear form แสดง conditional ไม่ออก ("ชื่อตรง แต่ conn=0") |
| 2 | **Equal-weight / percentile / rank fusion** | 12.58% (T1.6, T1.10a) | p-value normalization ทำลาย magnitude gap |
| 3 | **Conformal p-value fusion** | 12.58-19.18% | เดียวกับ #2 — แปลง raw→p-value ทำลายข้อมูล |
| 4 | **IEF/distinctness แทน raw count** | 5-10% | log-frequency กำจัด signal ของ count |
| 5 | **DDTree เป็น answer selector** | 4 ครั้ง regress | beam path ไม่ carry answer entity |
| 6 | **Semantic layer ใน scoring** | regress | VSA cosine ≈ noise กับ random codebook |
| 7 | **Substring entity consolidation** | regress | merge ผิด (ต้อง exact match เท่านั้น) |
| 8 | **Relation-heuristic answer-type veto** | 19.81% | "won"→Person ผิดสำหรับ "which team" |
| 9 | **Sigmoid หลัง fusion** | 12-15% | บีบทุกคะแนนใกล้ 0/1 → tie ตกที่ penalty |

---

## 1. บทเรียนระดับกลไก (คะแนน / ranking)

### 1.1 Linear weighted sum = flat local optimum (ข้อ #1)
**หลักฐาน:** 5+ redesign/weight-tuning variants ล้มเหลว:
- v11-v14: ปรับ overlap/conn/role weight → plateau
- T1.8a: coordinate-ascent สแกนทุก weight → มีแค่ overlap ที่ขยับได้
- **ข้อสรุปเชิงคณิตศาสตร์:** ถ้า signal มี noise σᵢ ต่างกัน min-variance combiner
  ต้องการ wᵢ ∝ 1/σᵢ² แต่ overlap (range ~50) เป็นทั้ง "ข้อมูลที่สุด" และ "กับดักที่สุด"
  ในเวลาเดียวกัน → **ไม่มี linear weight ตัวเดียวถูกทั้งสอง regime**
- **ทางรอด:** ไม่ใช่ weight tuning แต่เป็น (ก) สัญญาณใหม่ (PPR) (ข) hard filter
  (ค) แก้คุณภาพ graph

### 1.2 Percentile / p-value / rank fusion = ทำลายข้อมูล (ข้อ #2, #3)
**หลักฐาน:**
- T1.6 equal-weight percentile: 12.58%
- T1.10a conformal equal-weight log-odds: **12.58% (ตัวเลขเดียวกันเป๊ะ)**
- T1.10a conformal tuned-weight: 19.18% (ยังแพ้ linear 24.21%)
- RRF (T1.9a): 11.95-15.41%

**กลไก:** การแปลง raw score (conn 2.0 vs 0.5) → p-value/percentile
(rank 1 vs rank 2) **ทำให้ช่องว่างจริงหายไป** ระบบ linear sum ชนะเพราะ
มันเก็บ magnitude gap ไว้ normalize เอา magnitude ทิ้ง = ทิ้งข้อมูล

**บทเรียน:** scale mismatch แก้ด้วย "อย่าให้ signal หนึ่ง scale ครอบงำ"
ไม่ใช่ "ทำให้ทุก signal scale เท่ากัน" — ต้อง calibrated weight (AUC) ต่อสัญญาณ
แต่ถึง calibrated ก็แพ้ linear (19.18%) → **fusion ไม่ใช่ทางออกของ gap นี้**

### 1.3 IEF / frequency scaling (ข้อ #4)
**หลักฐาน:** AXIOM_W_IEF 0.1→10.38%, 1.5→5.35% (T1.8c)
**กลไก:** raw count 0.2× เป็น "evidence mass" ที่มีประโยชน์ — entity ที่อยู่ใน
triple เยอะ *มัก* เป็นคำตอบจริง (T1.9b พิสูจน์: ลด count weight → 17.61%)
**บทเรียน:** "hub domination" เป็นทฤษฎีที่ดูถูกต้อง แต่ count term วัด
**หลักฐานจริง** ไม่ใช่แค่ความนิยม → อย่าเอา log/distinctness ไปแทน

### 1.4 Sigmoid หลัง fusion = ผิดที่ (ข้อ #9)
**หลักฐาน:** conformal + sigmoid (T1.10a): ทุก temp → 12-15%
**กลไก:** sigmoid บีบทุกคะแนนเข้า (0,1) → ต่างกันน้อยมาก → argmax
กลายเป็นตัวใครมาก่อน; tie ทั้งหมดตกที่ query-penalty
**บทเรียน:** sigmoid-never-softmax ใช้ได้เมื่อต้องการ *probability output*
ไม่ใช่เมื่อต้องการ *ordering* สำหรับ argmax ใช้ raw score ตรงๆ

---

## 2. บทเรียนระดับ graph / decomposition

### 2.1 Answer-type ต้องมี POS/NER-lite ไม่ใช่ relation heuristic (ข้อ #8)
**หลักฐาน:** AXIOM_TYPE_VETO (T1.10b): relation-family classification
→ 19.81% (แย่กว่า 24.21%)
**กลไกที่พัง:**
- "won"→Person ผิดสำหรับ "which team first scored" (คำตอบคือ team/event)
- capital_of: object=สถานที่ แต่ subject=เมือง → ambiguity
- entity ผ่านหลาย relation → family สับสน
**บทเรียน:** answer-type (F1 filter) เป็นไอเดียถูกต้อง แต่ต้องได้ type จาก
**lexicon/POS** (Scotland=country จากชื่อ/type list) ไม่ใช่เดาจาก relation

### 2.2 Decomposition truncate ตัวกลาง ทำลาย transitivity chain
**หลักฐาน:** T1.10b transitivity ไม่ fire: "Wanlockhead located_in
Dumfries and Galloway, Scotland" → truncate ที่ comma → ได้แค่
`located_in Dumfries` → ไม่มี intermediate node `Dumfries and Galloway`
ให้ chain ไป Scotland
**บทเรียน:** inference rule (Datalog) ดีแค่ไหนก็ตาม **input graph ต้องมี
intermediate node** — L2 ต้องพึ่ง L1 (POS/clause) ก่อน ไม่งั้น rule ไม่มีอะไรให้ fire

### 2.3 Junk entity surfaces (M5) = decomposition ไม่ใช่ ranking
**หลักฐาน:** 15/165 failures เป็น junk surface (`Cast *Gregory Peck`,
`in 1893 a Second Division`, `It's Cold Outside"`)
**บทเรียน:** อย่าเอา junk ไปแก้ที่ ranking (มันจะ pollute candidate set)
ต้องแก้ที่ source (POS/NP-chunking ใน decompose)

### 2.4 Surface filter ที่ graph regress ทุกแบบ (T1.10c)
**หลักฐาน (วัดเต็ม 318 bench):**
- NP edge-word gate (reject preposition/verb ที่ head/tail): candidate 24.21→23.90,
  recall 76.10→75.79
- แก้ให้ allow leading "the" ("The Beatles", "The King Shall Rejoice"): ยัง 23.90/75.79
- เหลือแค่ residue-char + quote + em-dash gate: candidate 24.21→23.58, recall 76.10→75.16
**กลไกที่พัง:** entity จริง legitimately มี character พวกนั้น; การ filter ทุกแบบ
remove entity ที่ถูกต้องด้วยเสมอ (เช่น "Zadok the Priest" case หลุดไป 1 record)
**บทเรียน:**
- Junk surface ส่วนใหญ่ (15 ตัว) **ถูกจัดการแล้ว** โดย is_fact_worthy/proper-noun gate เดิม
- ส่วนที่เหลือ rare มาก → filter เพิ่ม = loss > gain บน aggregate
- **M5 ไม่ใช่ lever หลัก** — 15/165 เท่านั้น อย่าลงทุนกับ surface filter
  ไปทำ deep-rank (149) / near-tie (25) แทน

### 2.5 Deep-rank golds ต้องการ subject resolution ไม่ใช่แค่ relation pattern (T1.10d)
**หลักฐาน:** "Swan Lake, Op. 20, is a ballet **composed by** Pyotr Ilyich
Tchaikovsky" → gold Tchaikovsky ติด deep-rank (#17) เพราะ:
- การ์ด RELATIONAL_PHRASES ขาด passive `composed by`/`written by`/`directed by`
  (มีแต่ bare `composed`) → หลังเติม pattern: เกิด fact ถูกต้อง
  `(Swan Lake, created_by, Tchaikovsky)` **เฉพาะเมื่อ subject ถูกต้อง**
- แต่ subject กลายเป็น "is a ballet" (copula fragment) → fact มี subject ขยะ
  → Tchaikovsky ยังเข้า graph แค่ `mentions` (0.8) ไม่ใช่ created_by (2.0)
- **ทดลอง copula-subject fix** (ถ้า subject เริ่มด้วย is/was/are/were → inherit):
  regress (candidate 24.21→23.90, recall 76.10→75.79, เฉพาะ tb_1826 หายไป)
  เพราะ "Zadok the Priest **were** composed by..." → subject "Zadok the Priest were"
  (copula อยู่ท้ายไม่ใช่หน้า) → กลายเป็น junk subject
**บทเรียน:**
- การ์ด relation เดียวไม่พอ — **ต้องมี subject resolution ที่ถูกต้องด้วย**
  (ระบุว่า copula อยู่ท้าย subject, ตัด "X were" → "X")
- copula fix ต้องดูท้าย subject ("Zadok the Priest were" → "Zadok the Priest")
  ไม่ใช่แค่หน้า ("is a ballet" → inherit) — ทั้งสองกรณีต้องรองรับ
- relation additions (`composed by`→created_by, `_by` forms + strong weights)
  เก็บไว้เป็น infra ถูกต้อง (neutral, ไม่ regress) แต่ยัง fire ไม่เต็มที่
  จนกว่า subject resolution จะเสร็จ — นี่คืองาน L1 ที่แท้จริงสำหรับ deep-rank

---

## 3. บทเรียนระดับ process (วิธีทำงาน)

### 3.1 Per-record bench diagnostics NOT stable ระหว่าง run
**หลักฐาน:** 147/318 records เปลี่ยนระหว่าง run เดียวกัน (HashMap iteration
order) — แต่ aggregate (candidate/recall/substring) stable มาก
**กฎ:** ตัดสินใจจาก **aggregate** เท่านั้น ห้ามสรุปจาก per-record diff

### 3.2 Query-entity detection เป็น "gate" ของทุกอย่าง
**หลักฐาน:** T1.9a — "O'Hare"/"Jaws (film)" หลบ query penalty เพราะ
punctuation split (`O'Hare`→["o","hare"]) → +2.2pt เมื่อแก้
**บทเรียน:** ก่อน redesign ranking ตรวจ query matching ก่อน —
ถ้า query entity ตรวจจับผิด ทุก layer ที่ตามมาใช้ข้อมูลผิด

### 3.3 Env-gated A/B = วิธีที่ถูกต้อง
**หลักฐาน:** ทุก experiment ใน T1.8-T1.10 ใช้ `AXIOM_W_*`/`AXIOM_RANK`/
`AXIOM_INFER` env → กลับ baseline ได้ใน 1 คำสั่ง ไม่ต้อง revert code
**กฎ:** ทุก experiment ใหม่ต้อง gated; default = best-known (ปัจจุบัน 24.21%)

### 3.4 ตัดสินใจจาก failure-mode count ไม่ใช่ intuition
**หลักฐาน:** 165 failures ถูก classify (M1-M5 + deep-rank):
- M1 overlap 21→4 หลัง query-penalty fix (พิสูจน์ว่า bug ที่แท้จริง)
- M2 near-tie 25 (8/25 = VSA noise ตัดสิน)
- deep-rank 149 (clean entity แต่ถูกทับ)
**บทเรียน:** ตัวเลขบอกว่า gap อยู่ที่ "เลือกผิดท่ามกลาง candidate ที่เชื่อมแล้ว"
ไม่ใช่ "ไม่มี candidate" → งาน ranking ต้องโฟกัสที่ deep-rank + near-tie

---

## 4. สิ่งที่ "ได้ผล" (เพื่อไม่ให้ไปทำซ้ำทางตัน แล้วลืมทางออก)

| อะไร | ผล | ทำไมได้ผล |
|------|-----|-----------|
| **T1.9a query-penalty punctuation fix** | +2.2pt | แก้ bug ที่ทำให้ penalty ไม่ fire |
| **T1.9c hub-corrected PPR** (log π_q − log π) | +0.63pt | สัญญาณใหม่เชิงโครงสร้าง แก้ M3/M4 |
| **T1.7 proper-noun boundary** | recall +4.72pt | decomposition quality — entity สะอาด |
| **T1.8a overlap 0.15→0.05** | +0.63pt | ลด overlap dominance (แต่ยัง >0 เป็น tiebreak) |
| **T1.9b intent-aware penalty** | neutral | ถูกต้องเชิงแนวคิด (What-is-X คือคำตอบ) แต่ผลรวม 0 |

**รูปแบบที่ทำซ้ำได้:** (1) แก้ bug ที่ gate (2) เพิ่มสัญญาณใหม่เชิงโครงสร้าง
(3) ทำ decomposition สะอาด — **ทั้งสามชนะทุก fusion redesign**

---

## 5. คำถามที่ยังเปิด (ถ้าจะค้น research รอบ 2)

1. POS/NP-chunking แบบ deterministic (DFA + lexicon) ที่แม่นใน open-domain — ยังไม่ได้ทำ (T1.10c)
2. สัญญาณใหม่แบบ PPR ที่แก้ deep-rank (149) — hub-corrected PPR เริ่มแล้ว +0.63
3. วิธีแก้ near-tie ที่ VSA noise ตัดสิน (25) — ยังไม่มีคำตอบ
4. Type lexicon (ประเทศ/เมือง/คน) ขนาดเล็กที่พอจะ enable answer-type filter

---

## 6. Anti-pattern checklist (ก่อน commit ใหม่)

- [ ] มี bench 318 ครบหรือยัง (ห้าม claim ไม่มี bench)
- [ ] aggregate stable 3+ runs หรือยัง
- [ ] experiment env-gated / revert ได้ไหม
- [ ] ไม่ได้ทำอะไรในตาราง "ห้ามทำเด็ดขาด" ด้านบน
- [ ] ไม่ได้เพิ่ม weight ให้ linear sum แบบ blind
- [ ] ไม่ได้ normalize signal → p-value/percentile
- [ ] ไม่ได้ใช้ relation-heuristic ตัดสิน type
- [ ] ไม่อ้าง "hub domination" โดยไม่เช็คว่า count เป็น evidence mass จริงไหม

---

*Maintainers: อัปเดตไฟล์นี้ทุกครั้งที่เจอ anti-pattern ใหม่ พร้อม bench หลักฐาน*
