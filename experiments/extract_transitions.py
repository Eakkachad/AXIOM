#!/usr/bin/env python3
"""
Hypothesis 3: Extract Transition Patterns from LLM Weights
============================================================
ดึง transition patterns จาก Qwen3-1.7B weights โดยไม่ต้องรัน inference

วิธีรัน:
    python3 extract_transitions.py

ต้องการ:
    pip3 install safetensors numpy huggingface-hub

ผลลัพธ์:
    - transitions.json: top-10 next-token predictions ต่อคำ
    - extraction_report.txt: สรุปผลการ extract
"""

import json
import time
import os
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("/home/eggchad/eakject/research/Deep_Man/topological-latent-engine/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Hypothesis 3: LLM Weight Extraction → VSA Transitions      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ═══ Step 1: Download ═══
    print("Step 1: Downloading Qwen3-1.7B weights (~3.4GB)...")
    print("  (ถ้าเคยโหลดแล้วจะใช้ cache)")
    print()
    
    from huggingface_hub import hf_hub_download
    
    t0 = time.time()
    file1 = hf_hub_download('Qwen/Qwen3-1.7B', 'model-00001-of-00002.safetensors')
    print(f"  ✓ File 1: {time.time()-t0:.0f}s")
    
    t0 = time.time()
    file2 = hf_hub_download('Qwen/Qwen3-1.7B', 'model-00002-of-00002.safetensors')
    print(f"  ✓ File 2: {time.time()-t0:.0f}s")
    
    # Download tokenizer
    tokenizer_path = hf_hub_download('Qwen/Qwen3-1.7B', 'tokenizer.json')
    print(f"  ✓ Tokenizer")
    print()

    # ═══ Step 2: Load tensors ═══
    print("Step 2: Loading embed_tokens + lm_head (ไม่โหลดทั้ง model)...")
    from safetensors import safe_open
    import struct
    
    t0 = time.time()
    
    # Qwen3 uses bfloat16 which numpy doesn't support.
    # Solution: load via safetensors with pytorch framework, or manual conversion.
    
    try:
        import torch
        print("  Using PyTorch for bf16→f32 conversion...")
        with safe_open(file1, framework="pt", device="cpu") as f:
            embed = f.get_tensor("model.embed_tokens.weight").float().numpy()
        with safe_open(file2, framework="pt", device="cpu") as f:
            lm_head = f.get_tensor("lm_head.weight").float().numpy()
    except ImportError:
        print("  No PyTorch — manual bf16→f32 conversion via raw bytes...")
        
        def load_bf16_tensor(filepath, tensor_name):
            """Load a bfloat16 tensor by reading raw bytes and converting to float32."""
            from safetensors import safe_open as sf_open
            # Get tensor metadata
            with open(filepath, 'rb') as raw_f:
                # Read header length (first 8 bytes, little-endian uint64)
                header_len = struct.unpack('<Q', raw_f.read(8))[0]
                header_bytes = raw_f.read(header_len)
                header = json.loads(header_bytes)
                
                tensor_info = header[tensor_name]
                dtype = tensor_info['dtype']
                shape = tensor_info['shape']
                offsets = tensor_info['data_offsets']
                
                # Read raw tensor data
                data_start = 8 + header_len + offsets[0]
                data_end = 8 + header_len + offsets[1]
                raw_f.seek(data_start)
                raw_data = raw_f.read(data_end - data_start)
            
            # Convert bf16 to f32: each bf16 is upper 16 bits of f32
            num_elements = 1
            for s in shape:
                num_elements *= s
            
            u16_array = np.frombuffer(raw_data, dtype=np.uint16)
            # Shift left by 16 bits to get float32 representation
            u32_array = u16_array.astype(np.uint32) << 16
            f32_array = np.frombuffer(u32_array.tobytes(), dtype=np.float32)
            return f32_array.reshape(shape)
        
        embed = load_bf16_tensor(file1, "model.embed_tokens.weight")
        lm_head = load_bf16_tensor(file2, "lm_head.weight")
    
    print(f"  embed_tokens: {embed.shape} (float32)")
    print(f"  lm_head:      {lm_head.shape} (float32)")
    print(f"  RAM ใช้: ~{(embed.nbytes + lm_head.nbytes)/1e9:.1f} GB")
    print(f"  เวลา: {time.time()-t0:.1f}s")
    print()

    # ═══ Step 3: Build tokenizer map ═══
    print("Step 3: Building token map...")
    with open(tokenizer_path) as f:
        tok_data = json.load(f)
    
    # Get vocab from tokenizer
    vocab = tok_data.get('model', {}).get('vocab', {})
    id_to_token = {idx: token for token, idx in vocab.items()}
    token_to_id = vocab
    
    print(f"  Vocabulary size: {len(id_to_token)}")
    
    # Filter: เอาเฉพาะ English words ที่เป็น single token ที่อ่านได้
    english_tokens = {}
    for token, idx in vocab.items():
        # เอาเฉพาะ token ที่เป็น ASCII, ไม่มี special chars, ความยาว 2-15
        clean = token.replace('Ġ', '').replace('▁', '').strip()
        if (clean.isascii() and clean.isalpha() and 2 <= len(clean) <= 15 
            and clean.islower()):
            english_tokens[clean] = idx
    
    print(f"  English word tokens: {len(english_tokens)}")
    print(f"  ตัวอย่าง: {list(english_tokens.keys())[:20]}")
    print()

    # ═══ Step 4: Compute transitions ═══
    print("Step 4: Computing transition matrix...")
    print("  Method: cosine(lm_head[candidate], embed[source])")
    print("  สำหรับทุก English word → top-10 predicted next words")
    print()
    
    # Normalize vectors for cosine similarity
    embed_norms = np.linalg.norm(embed, axis=1, keepdims=True)
    embed_norms[embed_norms < 1e-8] = 1.0
    embed_normed = embed / embed_norms
    
    lm_head_norms = np.linalg.norm(lm_head, axis=1, keepdims=True)
    lm_head_norms[lm_head_norms < 1e-8] = 1.0
    lm_head_normed = lm_head / lm_head_norms
    
    # For each source word, find top-10 next words
    # Only search within english_tokens (not full 151K vocab)
    english_ids = list(english_tokens.values())
    english_words = list(english_tokens.keys())
    
    # Pre-extract english subset of lm_head
    lm_head_english = lm_head_normed[english_ids]  # [num_english, hidden]
    
    transitions = {}
    total = len(english_tokens)
    t0 = time.time()
    
    for i, (word, word_id) in enumerate(english_tokens.items()):
        if i % 500 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / max(i, 1)) * (total - i)
            print(f"  Processing: {i}/{total} ({i*100//total}%) ETA: {eta:.0f}s", end='\r')
        
        # Get source embedding
        source_vec = embed_normed[word_id]  # [hidden]
        
        # Compute similarity with all english lm_head vectors
        scores = lm_head_english @ source_vec  # [num_english]
        
        # Top 10 (excluding self)
        top_indices = np.argsort(scores)[::-1]
        
        preds = []
        for idx in top_indices:
            if english_words[idx] == word:
                continue
            preds.append({
                "word": english_words[idx],
                "score": float(scores[idx])
            })
            if len(preds) >= 10:
                break
        
        transitions[word] = preds
    
    elapsed = time.time() - t0
    print(f"\n  ✓ Done: {len(transitions)} words processed in {elapsed:.1f}s")
    print()

    # ═══ Step 5: Show examples ═══
    print("Step 5: Sample results (source → predicted next words):")
    print()
    
    sample_words = ["the", "cat", "dog", "is", "love", "big", "sun", "run", 
                    "happy", "water", "eat", "think", "because", "very", "and"]
    
    for word in sample_words:
        if word in transitions:
            preds = transitions[word][:5]
            pred_str = ", ".join([f"{p['word']}({p['score']:.3f})" for p in preds])
            print(f"  '{word}' → [{pred_str}]")
    
    print()

    # ═══ Step 6: Evaluate quality ═══
    print("Step 6: Quality evaluation...")
    
    # Check: do the transitions make linguistic sense?
    # Test known patterns
    tests = [
        ("the", ["a", "an", "this", "that", "it", "is", "was"]),  # articles/function words
        ("cat", ["dog", "cats", "pet", "animal", "kitten"]),  # related nouns
        ("is", ["was", "are", "be", "not", "it"]),  # verbs/aux
        ("big", ["large", "small", "huge", "great", "bigger"]),  # adjectives
    ]
    
    hits = 0
    total_tests = 0
    for source, expected_any in tests:
        if source in transitions:
            predicted_words = [p['word'] for p in transitions[source]]
            for exp in expected_any:
                total_tests += 1
                if exp in predicted_words:
                    hits += 1
    
    semantic_accuracy = hits / max(total_tests, 1) * 100
    print(f"  Semantic relevance: {hits}/{total_tests} ({semantic_accuracy:.0f}%)")
    print()

    # ═══ Step 7: Save ═══
    print("Step 7: Saving results...")
    
    output_file = OUTPUT_DIR / "transitions.json"
    with open(output_file, 'w') as f:
        json.dump(transitions, f, indent=2)
    print(f"  ✓ Saved: {output_file} ({os.path.getsize(output_file)/1e6:.1f} MB)")
    
    # Save report
    report_file = OUTPUT_DIR / "extraction_report.txt"
    with open(report_file, 'w') as f:
        f.write("=== Hypothesis 3: LLM Weight Extraction Report ===\n\n")
        f.write(f"Model: Qwen3-1.7B\n")
        f.write(f"Method: cosine(lm_head, embed) - no forward pass\n")
        f.write(f"Vocab extracted: {len(transitions)} English words\n")
        f.write(f"Transitions per word: 10\n")
        f.write(f"Semantic accuracy: {semantic_accuracy:.0f}%\n\n")
        f.write("Sample transitions:\n")
        for word in sample_words[:10]:
            if word in transitions:
                preds = [p['word'] for p in transitions[word][:5]]
                f.write(f"  {word} → {preds}\n")
    print(f"  ✓ Saved: {report_file}")
    
    print()
    print("═══ DONE ═══")
    print(f"  ไฟล์ transitions.json พร้อมใช้กับ Rust VSA encoder แล้ว")
    print(f"  รัน: cargo run --release -p tle-transition")
    print()
    
    if semantic_accuracy > 30:
        print("  ✓ HYPOTHESIS 3 PROMISING: Weight extraction gives meaningful transitions")
    else:
        print("  ⚠ HYPOTHESIS 3 INCONCLUSIVE: May need MLP layers too")


if __name__ == "__main__":
    main()
