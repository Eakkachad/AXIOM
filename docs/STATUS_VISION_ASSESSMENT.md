# AXIOM — สถานะจริง กับ วิสัยทัศน์ (Honest Status & Vision Assessment)

> **อัปเดต:** 2026-08-13 · **เอกสารนี้คือ ground truth ฉบับ reality check**
> — อ่านก่อนเชื่อตัวเลข/คำกล่าวอ้างในเอกสารอื่น เอกสารอื่นที่ขัดแย้งกับไฟล์นี้
> ควรถือว่าเป็น aspirational/overclaim จนกว่าจะถูกแก้ให้สอดคล้อง

---

## 1. งานนี้คืออะไร (สรุปตรง)

ระบบ **deterministic · zero-training · CPU-only** (Rust 18+ crates) ที่ทำสองอย่าง:

1. **TriviaQA answer selection** — รับ evidence → graph → rank entity → ตอบคำถาม
2. **axiom-chat** — REPL โต้ตอบ: graph reasoning (multi-hop) + KN-5 free-form +
   TF-IDF RAG (จำ corpus/Wikipedia) + turn memory + grammar + casual

**สิ่งที่มันไม่ใช่:** ไม่ใช่ LLM, ไม่ใช่ neural network, ไม่ใช่ "AI ทั่วไป"

---

## 2. ตัวเลขที่ซื่อตรง (STRICT metrics — นับจากที่แก้ metric โปะแล้ว)

| Metric | ค่า | หมายเหตุ |
|--------|:---:|---------|
| TriviaQA candidate **exact** | **16.04%** | (metric เดิม 24.84% โปะ 2× — เราแก้เอง) |
| TriviaQA candidate f1 (≥0.7) | **17.92%** | |
| answer_entity_recall (substring) | 76.10% | ~21pt เป็น phantom (substring lenient) |
| **strict_recall** (F1≥0.7 ต่อ node) | **55.03%** | ceiling จริงที่ selectable ได้ |
| evidence_answer_recall | 99.69% | ข้อความ raw มีคำตอบ |
| VSA-LM TEST next-token | **~11%** | ขีดจำกัดเชิงโครงสร้าง (KN-5 = 16.7%) |
| axiom-chat RAG self-R@1 | 23.5% (327K ประโยค) / 54% (เล็ก) | |

**Gap ที่แท้จริง:** strict_recall 55% → exact 16% = **~39pt** (ไม่ใช่ 52pt ที่เคยอ้าง —
metric เดิมโปะทั้งสองด้าน)

---

## 3. Breakthrough? — **ไม่มี**

หลัง 20+ experiments + deep research (katgpt forensic, literature, hypothesis
tree G0-H8):

- **KN-5 > VSA-LM** = ยืนยันความรู้เก่า (n-gram มีมา 70 ปี)
- **GHRR noise-floor** = ยืนยันขีดจำกัด VSA ที่รู้กัน (d=4096, M=13K → distractor cos≈0.09)
- **สิ่งที่งานนี้พิสูจน์ได้จริงคือ negative results ที่มีวินัย:** pure VSA ล้วน
  ไม่สามารถถึง "LLM-level generalization" ได้ — นี่คือผลลัพธ์เชิงวิชาการที่มีค่า
  แต่ **ไม่ใช่ breakthrough**

---

## 4. Impact — จริงแต่จำกัด

**มีจริง:** deterministic QA ที่ไม่หลอน · reproducible 100% · ตอบจากความรู้ที่ให้ ·
เร็ว (µs-ms) · ทำงาน offline — ใช้ได้ใน niche (privacy, education, domain expert)

**ยังไม่มี:** ไม่มี algorithm ใหม่ · ไม่ชนะ benchmark SOTA · ไม่มีผลกระทบระดับโลก

---

## 5. ระดับป.ตรี? ระดับโลก?

- **ป.ตรี thesis: ได้ ถ้า frame ถูก** — "Empirical limits of deterministic
  hyperdimensional computing for language" (engineering + negative results +
  metric audit) เป็น thesis ที่ดี
- **ระดับโลก: ไม่** — ต้อง algorithm ใหม่ / SOTA / insight ปฏิวัติ ซึ่งงานนี้ไม่มี

---

## 6. Overclaim registry (ข้อความที่เคยเขียนเกินจริง — ต้องแก้)

| ไฟล์ | ข้อความเดิม | ความจริง | สถานะ |
|------|-----------|---------|:---:|
| AGENT_HANDOFF (vision) | "สร้าง AI แบบใหม่ที่เปลี่ยนโลก" | ไม่สำเร็จ — งานนี้พิสูจน์ว่าไม่ถึง | แก้แล้ว |
| AGENT_HANDOFF (IMPACT) | "Nobel-level: algebraic composition = general intelligence" | พิสูจน์แล้วไม่เป็นจริง | แก้แล้ว |
| AGENT_HANDOFF | "ชนะ LLM 8/10 dimensions" | เป็นไปไม่ได้ (คนละชั้น) | แก้แล้ว |
| PROJECT_SUMMARY / ตัวเลข | candidate 24.53% | โปะ — จริง 13.84-16% | แก้แล้ว |
| handoff (VSA-LM) | "LM แบบใหม่" | TEST 11% = experiment ไม่ใช่ LM | แก้แล้ว |
| "เร็วกว่า LLM 1000×" | 12K tok/s | cherry-pick (ทำสิ่งที่ LLM ทำไม่ได้) | แก้แล้ว |

**กฎตั้งแต่นี้:** ทุก claim ต้องมี (1) full-318 bench บน **strict metric** และ
(2) ตัวเลขซื่อตรง ไม่ใช่ aspirational

---

## 7. ตรงกับวิสัยทัศน์ไหม? — **ไม่ในตอนนี้**

วิสัยทัศน์เดิม: *"AI ใหม่ที่เปลี่ยนโลก — ไม่ train, เร็วกว่า LLM 1000×, algebraic
composition = general intelligence"*

**ความจริง:** งานนี้ได้**พิสูจน์ว่าเส้นทางนั้นไม่สำเร็จ**ในสถาปัตยกรรมนี้
(noise-floor + literature) สิ่งที่ทำได้คือ **subset ที่ทำได้จริง**:
**deterministic domain-expert** (fluent + ไม่หลอน + ตอบจากความรู้ที่ให้)

### วิสัยทัศน์ที่สมจริง (reframed)
> สร้างระบบ reasoning ที่ **ซื่อสัตย์** — deterministic, ไม่หลอน, reproducible,
> ตอบจากความรู้ที่ให้จริง — เป็นเครื่องมือ domain-expert ที่ไว้ใจได้ แทนที่จะ
> อ้างว่าเป็น "AI ทั่วไป"

---

## 8. มูลค่าที่แท้จริงของงาน (3 อย่าง)

1. **วิศวกรรม deterministic reasoning stack** ที่ทำงานได้จริง (Rust, 18+ crates)
2. **Negative results ที่มีวินัย** — 20+ experiments บันทึกครบใน PROGRESS_LOG/
   LESSONS_LEARNED/ROADMAP — จุดขายทางวิชาการจริง
3. **Metric audit** — ค้นพบ metric ตัวเองโปะ 2× แล้วแก้ + สร้าง strict metric —
   บทเรียนที่เขียนเป็น paper ได้

---

## 9. เส้นทางถัดไป (ถ้าจะทำต่อ)

- **ขยาย "domain-expert"** ที่มี — เป็น product ที่ใช้งานได้ (ไม่ใช่ research claim)
- **เปลี่ยนสมมติฐานหลัก** ถ้าอยาก "ระดับโลก" — ต้อง hybrid (retrieval/statistical
  แทน pure VSA) = เริ่มต้นใหม่ ไม่ใช่ต่อยอด
- **ปิดโปรเจกต์อย่างเป็นทางการ** — เอกสารครบ, สถานะซื่อตรง

---

*เอกสารนี้เป็น ground truth — ไฟล์อื่นที่กล่าวอ้างเกินต้องสอดคล้องกับไฟล์นี้*
