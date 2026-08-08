#!/usr/bin/env python3
"""
Hypothesis 3 — Comparison: Weight Extraction vs Text Corpus
=============================================================
Method A: lm_head × MLP_layer14(embed) — ดึง transitions จาก weights + MLP
Method C: Bigram/Trigram counts จาก text corpus — statistical extraction

รัน: .venv/bin/python extract_v2.py

ผลลัพธ์:
  data/transitions_mlp.json   — Method A results
  data/transitions_corpus.json — Method C results  
  data/comparison_report.txt  — เปรียบเทียบ
"""

import json
import time
import os
import struct
import numpy as np
from pathlib import Path
from collections import Counter

OUTPUT_DIR = Path("/home/eggchad/eakject/research/Deep_Man/topological-latent-engine/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════

def load_bf16_tensor(filepath, tensor_name):
    """Load bfloat16 tensor from safetensors via raw bytes."""
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        info = header[tensor_name]
        shape = info['shape']
        offsets = info['data_offsets']
        f.seek(8 + header_len + offsets[0])
        raw = f.read(offsets[1] - offsets[0])
    
    u16 = np.frombuffer(raw, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    f32 = np.frombuffer(u32.tobytes(), dtype=np.float32)
    return f32.reshape(shape)


def silu(x):
    """SiLU activation: x * sigmoid(x)"""
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))


def get_top_k(scores, k, exclude_idx=None):
    """Get top-k indices excluding a specific index."""
    top = np.argsort(scores)[::-1]
    results = []
    for idx in top:
        if exclude_idx is not None and idx == exclude_idx:
            continue
        results.append(idx)
        if len(results) >= k:
            break
    return results


# ═══════════════════════════════════════════════════════════════
# METHOD A: Weight Extraction with MLP
# ═══════════════════════════════════════════════════════════════

def method_a_mlp_extraction():
    """Extract transitions using: lm_head × MLP_layer14(embed)"""
    print("═══ METHOD A: lm_head × MLP(embed) [Weight Extraction] ═══")
    print()
    
    from huggingface_hub import hf_hub_download
    
    # Download model files
    print("  Downloading model weights...")
    file1 = hf_hub_download('Qwen/Qwen3-1.7B', 'model-00001-of-00002.safetensors')
    file2 = hf_hub_download('Qwen/Qwen3-1.7B', 'model-00002-of-00002.safetensors')
    tok_path = hf_hub_download('Qwen/Qwen3-1.7B', 'tokenizer.json')
    print("  ✓ Downloaded")
    
    # Load required tensors
    print("  Loading tensors (embed + MLP layer 14 + lm_head)...")
    t0 = time.time()
    
    embed = load_bf16_tensor(file1, "model.embed_tokens.weight")        # [151936, 2048]
    gate_proj = load_bf16_tensor(file1, "model.layers.14.mlp.gate_proj.weight")  # [6144, 2048]
    up_proj = load_bf16_tensor(file1, "model.layers.14.mlp.up_proj.weight")      # [6144, 2048]
    down_proj = load_bf16_tensor(file1, "model.layers.14.mlp.down_proj.weight")  # [2048, 6144]
    lm_head = load_bf16_tensor(file2, "lm_head.weight")                # [151936, 2048]
    
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  embed: {embed.shape}, gate: {gate_proj.shape}, up: {up_proj.shape}")
    print(f"  down: {down_proj.shape}, lm_head: {lm_head.shape}")
    print()
    
    # Build token map
    with open(tok_path) as f:
        vocab = json.load(f).get('model', {}).get('vocab', {})
    
    # Filter English words
    english_tokens = {}
    for token, idx in vocab.items():
        clean = token.replace('Ġ', '').replace('▁', '').strip()
        if clean.isascii() and clean.isalpha() and 2 <= len(clean) <= 12 and clean.islower():
            english_tokens[clean] = idx
    
    # Subset: top 5000 most common-looking words (shorter = more common)
    sorted_words = sorted(english_tokens.items(), key=lambda x: len(x[0]))[:5000]
    english_tokens = dict(sorted_words)
    
    print(f"  Working vocabulary: {len(english_tokens)} English words")
    print()
    
    # Compute MLP(embed) for each word, then score with lm_head
    # MLP: output = down_proj @ (silu(gate_proj @ x) * (up_proj @ x))
    print("  Computing lm_head × MLP(embed) transitions...")
    t0 = time.time()
    
    word_list = list(english_tokens.keys())
    word_ids = list(english_tokens.values())
    
    # Pre-compute: get embeddings for our subset
    embed_subset = embed[word_ids]  # [5000, 2048]
    lm_head_subset = lm_head[word_ids]  # [5000, 2048]
    
    # Normalize lm_head for cosine scoring
    lm_norms = np.linalg.norm(lm_head_subset, axis=1, keepdims=True)
    lm_norms[lm_norms < 1e-8] = 1.0
    lm_head_normed = lm_head_subset / lm_norms
    
    transitions_mlp = {}
    total = len(word_list)
    
    for i in range(total):
        if i % 200 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / max(i, 1)) * (total - i)
            print(f"    {i}/{total} ({i*100//total}%) ETA: {eta:.0f}s    ", end='\r')
        
        x = embed_subset[i]  # [2048]
        
        # MLP forward: SwiGLU
        gate = gate_proj @ x    # [6144]
        up = up_proj @ x        # [6144]
        hidden = silu(gate) * up  # [6144] — SwiGLU activation
        mlp_out = down_proj @ hidden  # [2048]
        
        # Residual connection: h = embed + mlp_out
        h = x + mlp_out  # [2048]
        
        # Normalize
        h_norm = h / (np.linalg.norm(h) + 1e-8)
        
        # Score against all vocab with lm_head
        scores = lm_head_normed @ h_norm  # [5000]
        
        # Top 10
        top_ids = get_top_k(scores, 10, exclude_idx=i)
        preds = [{"word": word_list[idx], "score": float(scores[idx])} for idx in top_ids]
        transitions_mlp[word_list[i]] = preds
    
    elapsed = time.time() - t0
    print(f"\n  ✓ Done: {len(transitions_mlp)} words in {elapsed:.1f}s")
    print()
    
    # Show samples
    print("  Sample MLP transitions:")
    samples = ["the", "cat", "dog", "is", "big", "run", "happy", "because", "and", "very"]
    for w in samples:
        if w in transitions_mlp:
            preds = transitions_mlp[w][:5]
            parts = [f"{p['word']}({p['score']:.3f})" for p in preds]
            print(f"    '{w}' → [{', '.join(parts)}]")
    print()
    
    # Save
    out_file = OUTPUT_DIR / "transitions_mlp.json"
    with open(out_file, 'w') as f:
        json.dump(transitions_mlp, f)
    print(f"  ✓ Saved: {out_file} ({os.path.getsize(out_file)/1e6:.1f} MB)")
    
    return transitions_mlp, word_list


# ═══════════════════════════════════════════════════════════════
# METHOD C: Text Corpus Bigram Extraction
# ═══════════════════════════════════════════════════════════════

def method_c_corpus_extraction():
    """Extract transitions from text corpus (bigram statistics)."""
    print("═══ METHOD C: Text Corpus Bigram/Trigram Statistics ═══")
    print()
    
    # Use a built-in English corpus (simple sentences we know)
    # In production: would use Brown corpus, Wikipedia, etc.
    corpus_sentences = [
        # Common patterns
        "the cat sat on the mat",
        "the dog ran in the park",
        "the bird flew over the tree",
        "I love my cat very much",
        "she walked to the store",
        "he is a good person",
        "the sun is bright today",
        "we went to the beach",
        "they are happy together",
        "it was a beautiful day",
        # More diverse
        "the water is cold and clear",
        "I think this is very important",
        "she said that he was right",
        "the children played in the garden",
        "we need to find a solution",
        "the book is on the table",
        "he went to the hospital yesterday",
        "the teacher asked a question",
        "they decided to go home early",
        "I want to learn something new",
        "the car stopped at the light",
        "she opened the door and walked in",
        "the food was delicious and fresh",
        "he told her about the plan",
        "we should try a different approach",
        "the music was loud and energetic",
        "she smiled and said hello",
        "the rain started in the afternoon",
        "he finished his work on time",
        "they bought a new house last year",
        "the movie was interesting but long",
        "she studied hard for the exam",
        "the flowers bloom in the spring",
        "he always drinks coffee in the morning",
        "they traveled to many different countries",
        "the baby cried all night long",
        "she cooked dinner for the family",
        "the sky turned dark before the storm",
        "he read the newspaper every day",
        "they played football in the field",
        "the cat chased the mouse around",
        "she wrote a letter to her friend",
        "the old man sat on the bench",
        "he fixed the broken window yesterday",
        "they celebrated the birthday with cake",
        "the river flows to the sea",
        "she planted trees in the backyard",
        "the train arrived at the station",
        "he painted the wall bright blue",
        "they watched the sunset from the hill",
        "the wind blew the leaves away",
        "she found a coin on the street",
        "the mountain is covered with snow",
        "he carried the heavy box upstairs",
        "they danced all night at the party",
        "the fish jumped out of the water",
        "she cleaned the house before dinner",
        "the phone rang in the middle of the night",
        "he drove the car to the airport",
        "they built a sandcastle on the beach",
        "the stars shine bright in the sky",
        "she waited for the bus in the rain",
        "the cat is sleeping on the couch",
        "he promised to come back soon",
        "they shared the food with everyone",
        "the clock struck twelve at midnight",
        "she picked up the phone and called",
        "the road was long and winding",
        "he jumped over the fence quickly",
        "they sang songs around the campfire",
        "the snow covered the ground completely",
        "she threw the ball to the dog",
        "the market was crowded and noisy",
        "he turned off the light and slept",
        "they walked along the river bank",
        "the cat and dog became friends",
        "she told the truth about everything",
        "the building is very tall and modern",
        "he forgot his keys at home",
        "they learned to swim last summer",
        "the cake was sweet and delicious",
        "she ran faster than anyone else",
        "the door opened with a loud creak",
        "he ate breakfast before going to work",
        "they moved to a new city together",
        "the garden is full of beautiful flowers",
        "she lost her wallet at the mall",
        "the professor explained the theory clearly",
        "he saved enough money for a trip",
        "they invited all their friends to dinner",
        "the night was quiet and peaceful",
        "she asked him a difficult question",
        "the boat sailed across the lake",
        "he woke up early every morning",
        "they finished the project on time",
        "the story had a happy ending",
        "she borrowed a book from the library",
        "the bridge connects the two islands",
        "he smiled when he saw her",
        "they argued about the best solution",
        "the winter was long and cold",
        "she wrapped the gift carefully",
    ]
    
    print(f"  Corpus: {len(corpus_sentences)} sentences")
    
    # Count bigrams
    bigram_counts = Counter()
    trigram_counts = Counter()
    word_counts = Counter()
    
    for sentence in corpus_sentences:
        words = sentence.lower().split()
        for w in words:
            word_counts[w] += 1
        for i in range(len(words) - 1):
            bigram_counts[(words[i], words[i+1])] += 1
        for i in range(len(words) - 2):
            trigram_counts[(words[i], words[i+1], words[i+2])] += 1
    
    vocab = sorted(word_counts.keys())
    print(f"  Vocabulary: {len(vocab)} words")
    print(f"  Unique bigrams: {len(bigram_counts)}")
    print(f"  Unique trigrams: {len(trigram_counts)}")
    print()
    
    # Build transition table: for each word → top-10 most likely next words
    transitions_corpus = {}
    for word in vocab:
        # Get all bigrams starting with this word
        nexts = {}
        for (w1, w2), count in bigram_counts.items():
            if w1 == word:
                nexts[w2] = count
        
        if nexts:
            # Sort by count, take top 10
            sorted_nexts = sorted(nexts.items(), key=lambda x: -x[1])[:10]
            total_count = sum(nexts.values())
            transitions_corpus[word] = [
                {"word": w, "score": float(c / total_count)} 
                for w, c in sorted_nexts
            ]
    
    print("  Sample corpus transitions:")
    samples = ["the", "cat", "dog", "is", "big", "run", "happy", "because", "and", "very"]
    for w in samples:
        if w in transitions_corpus:
            preds = transitions_corpus[w][:5]
            parts = [f"{p['word']}({p['score']:.3f})" for p in preds]
            print(f"    '{w}' → [{', '.join(parts)}]")
        else:
            print(f"    '{w}' → [not in corpus]")
    print()
    
    # Save
    out_file = OUTPUT_DIR / "transitions_corpus.json"
    with open(out_file, 'w') as f:
        json.dump(transitions_corpus, f, indent=2)
    print(f"  ✓ Saved: {out_file} ({os.path.getsize(out_file)/1e6:.1f} MB)")
    
    return transitions_corpus, vocab


# ═══════════════════════════════════════════════════════════════
# COMPARISON
# ═══════════════════════════════════════════════════════════════

def compare_methods(trans_mlp, trans_corpus):
    """Compare Method A vs Method C on quality metrics."""
    print()
    print("═══ COMPARISON: Method A (Weights) vs Method C (Corpus) ═══")
    print()
    
    # Test cases: "given word X, what SHOULD come next?"
    # Ground truth based on common English patterns
    test_cases = [
        ("the", ["cat", "dog", "man", "woman", "big", "old", "new", "first"]),
        ("is", ["a", "the", "very", "not", "an", "also", "still", "now"]),
        ("in", ["the", "a", "this", "my", "his", "her", "our", "their"]),
        ("to", ["the", "a", "be", "do", "go", "get", "make", "have"]),
        ("and", ["the", "a", "he", "she", "it", "they", "I", "we"]),
        ("was", ["a", "the", "not", "very", "so", "too", "quite", "still"]),
        ("he", ["was", "is", "had", "said", "went", "told", "did", "saw"]),
        ("she", ["was", "is", "had", "said", "went", "told", "did", "saw"]),
        ("on", ["the", "a", "his", "her", "my", "this", "that", "time"]),
        ("at", ["the", "a", "his", "her", "home", "night", "first", "least"]),
    ]
    
    def score_method(transitions, test_cases):
        """Score: how many ground truth next-words appear in top-10 predictions."""
        hits = 0
        total = 0
        for word, expected in test_cases:
            if word not in transitions:
                continue
            predicted = [p["word"] for p in transitions[word][:10]]
            for exp in expected:
                total += 1
                if exp in predicted:
                    hits += 1
        return hits, total
    
    hits_a, total_a = score_method(trans_mlp, test_cases) if trans_mlp else (0, 1)
    hits_c, total_c = score_method(trans_corpus, test_cases)
    
    print(f"  Method A (LLM Weights + MLP): {hits_a}/{total_a} ({hits_a*100//max(total_a,1)}%)")
    print(f"  Method C (Text Corpus):        {hits_c}/{total_c} ({hits_c*100//max(total_c,1)}%)")
    print()
    
    # Overlap analysis: do they agree?
    if trans_mlp:
        common_words = set(trans_mlp.keys()) & set(trans_corpus.keys())
        agreement = 0
        total_compared = 0
        for word in list(common_words)[:100]:
            pred_a = set(p["word"] for p in trans_mlp[word][:5])
            pred_c = set(p["word"] for p in trans_corpus[word][:5])
            overlap = len(pred_a & pred_c)
            agreement += overlap
            total_compared += 5
        
        overlap_pct = agreement * 100 // max(total_compared, 1)
        print(f"  Overlap (top-5 agreement): {agreement}/{total_compared} ({overlap_pct}%)")
        print()
    
    # Verdict
    print("  ┌─────────────────────────────────────────────────────┐")
    if trans_mlp and hits_a > hits_c:
        print("  │ VERDICT: Method A (Weights) > Method C (Corpus)     │")
        print("  │ → LLM weights give BETTER transitions than corpus!  │")
    elif hits_c > 0:
        print("  │ VERDICT: Method C (Corpus) ≥ Method A (Weights)     │")
        print("  │ → Text corpus gives BETTER sequential transitions   │")
        print("  │ → Weights give semantic similarity, not sequences   │")
    else:
        print("  │ VERDICT: Need more data to conclude                 │")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    
    # Save report
    report = OUTPUT_DIR / "comparison_report.txt"
    with open(report, 'w') as f:
        f.write("=== Hypothesis 3: Method A vs Method C Comparison ===\n\n")
        f.write(f"Method A (LLM Weights + MLP layer 14): {hits_a}/{total_a} ({hits_a*100//max(total_a,1)}%)\n")
        f.write(f"Method C (Text Corpus bigrams): {hits_c}/{total_c} ({hits_c*100//max(total_c,1)}%)\n\n")
        f.write("Conclusion: ")
        if trans_mlp and hits_a > hits_c:
            f.write("Weight extraction with MLP provides superior transition predictions.\n")
            f.write("The MLP layer encodes sequential patterns not available from embedding similarity alone.\n")
        else:
            f.write("Text corpus bigrams provide more direct sequential information.\n")
            f.write("Weight extraction gives semantic neighbors; corpus gives actual next-word patterns.\n")
            f.write("RECOMMENDATION: Use corpus transitions as primary, weight semantics as secondary.\n")
    print(f"  ✓ Report saved: {report}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Hypothesis 3: Weight Extraction vs Corpus — Comparison     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    # Method C first (fast, no download needed)
    trans_corpus, vocab_corpus = method_c_corpus_extraction()
    print()
    
    # Method A (needs model download + computation)
    print("Starting Method A (may take 5-10 minutes)...")
    print()
    try:
        trans_mlp, vocab_mlp = method_a_mlp_extraction()
    except Exception as e:
        print(f"  ⚠ Method A failed: {e}")
        print(f"  Continuing with Method C only...")
        trans_mlp = None
    
    # Compare
    compare_methods(trans_mlp, trans_corpus)
    
    print()
    print("═══ DONE ═══")
    print("  ไฟล์ที่สร้าง:")
    print(f"    data/transitions_corpus.json — Method C (corpus bigrams)")
    if trans_mlp:
        print(f"    data/transitions_mlp.json — Method A (weight extraction)")
    print(f"    data/comparison_report.txt — สรุปเปรียบเทียบ")
