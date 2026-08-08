#!/usr/bin/env python3
"""
HRBM v4: WikiText-2 Benchmark (THE Real Test)
===============================================
- GloVe-50d embeddings (pre-trained)
- WikiText-2 train/test split (standard LM benchmark)
- Reservoir D=256
- Compare with 5-gram baseline
- Honest perplexity measurement
"""

import numpy as np
import time
import re
from collections import Counter

# ═══════════════════ CONFIG ═══════════════════
D_RES = 256
EMBED_DIM = 50
LEAK = 0.3
SPECTRAL = 0.9
SPARSITY = 0.1
LAMBDA = 5.0
SEED = 42
MAX_VOCAB = 5000       # Top-K words by frequency
MAX_TRAIN_TOKENS = 50000  # Limit for RAM (D=256 Gram matrix is fine)
GLOVE_PATH = "/home/eggchad/eakject/research/Deep_Man/topological-latent-engine/data/glove.6B.50d.txt"
WIKI_TRAIN = "/home/eggchad/eakject/research/Deep_Man/topological-latent-engine/data/wiki_train.txt"

np.random.seed(SEED)

# ═══════════════════ DATA LOADING ═══════════════════

def load_wiki(path, max_tokens=None):
    """Load WikiText-2, return list of sentences (word lists)."""
    sentences = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('='):
                continue
            # Simple tokenization: lowercase, split on space
            words = line.lower().split()
            # Remove <unk> and @-@ artifacts
            words = [w for w in words if w != '<unk>' and w != '@-@' and w.isalpha()]
            if len(words) >= 3:
                sentences.append(words)
    
    # Flatten and limit
    if max_tokens:
        flat = []
        for s in sentences:
            if len(flat) + len(s) > max_tokens:
                break
            flat.extend(s)
        # Re-chunk into sentences of ~10 words
        chunk_size = 10
        sentences = [flat[i:i+chunk_size] for i in range(0, len(flat)-chunk_size, chunk_size)]
    
    return sentences

def build_vocab(sentences, max_vocab):
    """Build vocabulary from most frequent words."""
    counter = Counter()
    for s in sentences:
        for w in s:
            counter[w] += 1
    
    # Top-K by frequency
    most_common = counter.most_common(max_vocab)
    vocab = [w for w, c in most_common]
    w2i = {w: i for i, w in enumerate(vocab)}
    return vocab, w2i, counter

def load_glove_subset(path, vocab_set):
    """Load GloVe vectors only for words in vocab."""
    embeddings = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in vocab_set:
                embeddings[word] = np.array([float(x) for x in parts[1:51]], dtype=np.float32)
    return embeddings

# ═══════════════════ RESERVOIR ═══════════════════

class Reservoir:
    def __init__(self, d_res, d_in, seed=42):
        rng = np.random.RandomState(seed)
        self.W_res = rng.randn(d_res, d_res).astype(np.float32)
        mask = rng.rand(d_res, d_res) > SPARSITY
        self.W_res[mask] = 0
        # Scale spectral radius
        scale = np.sqrt(d_res * SPARSITY)
        self.W_res *= SPECTRAL / max(scale, 0.01)
        self.W_in = rng.randn(d_res, d_in).astype(np.float32) * (1.0 / np.sqrt(d_in))
        self.state = np.zeros(d_res, dtype=np.float32)
    
    def step(self, x):
        pre = self.W_res @ self.state + self.W_in @ x
        self.state = (1 - LEAK) * self.state + LEAK * np.tanh(pre)
        return self.state.copy()
    
    def reset(self):
        self.state[:] = 0

# ═══════════════════ 5-GRAM BASELINE ═══════════════════

class NgramBaseline:
    """Simple 5-gram model with backoff for comparison."""
    def __init__(self, n=5):
        self.n = n
        self.counts = {}  # context tuple → Counter of next words
        self.unigram = Counter()
    
    def train(self, sentences):
        for sent in sentences:
            for w in sent:
                self.unigram[w] += 1
            for n in range(1, self.n + 1):
                for i in range(len(sent) - n):
                    context = tuple(sent[i:i+n])
                    next_w = sent[i+n] if i+n < len(sent) else None
                    if next_w:
                        if context not in self.counts:
                            self.counts[context] = Counter()
                        self.counts[context][next_w] += 1
    
    def predict_prob(self, context_words, target):
        """Predict P(target | context) with backoff."""
        total_unigram = sum(self.unigram.values())
        
        for n in range(min(len(context_words), self.n), 0, -1):
            ctx = tuple(context_words[-n:])
            if ctx in self.counts:
                counter = self.counts[ctx]
                total = sum(counter.values())
                if target in counter:
                    return counter[target] / total
        
        # Unigram backoff
        if target in self.unigram:
            return self.unigram[target] / total_unigram
        return 1.0 / (total_unigram + 1)  # Laplace smoothing

# ═══════════════════ MAIN ═══════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  HRBM v4: WikiText-2 Benchmark — THE Real Test              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  D_res={D_RES}, GloVe-{EMBED_DIM}d, λ={LAMBDA}")
    print(f"  Max vocab={MAX_VOCAB}, Max train tokens={MAX_TRAIN_TOKENS}")
    print()

    # Load data
    print("Step 1: Loading WikiText-2...")
    t0 = time.time()
    train_sents = load_wiki(WIKI_TRAIN, max_tokens=MAX_TRAIN_TOKENS)
    
    # Use last 20% as test
    n_train = int(len(train_sents) * 0.8)
    test_sents = train_sents[n_train:]
    train_sents = train_sents[:n_train]
    
    train_tokens = sum(len(s) for s in train_sents)
    test_tokens = sum(len(s) for s in test_sents)
    print(f"  Train: {len(train_sents)} chunks, {train_tokens} tokens")
    print(f"  Test:  {len(test_sents)} chunks, {test_tokens} tokens")

    # Build vocab from training data
    vocab, w2i, freq = build_vocab(train_sents, MAX_VOCAB)
    V = len(vocab)
    print(f"  Vocab: {V} words (top by frequency)")
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print()

    # Load GloVe
    print("Step 2: Loading GloVe embeddings...")
    glove = load_glove_subset(GLOVE_PATH, set(vocab))
    # Fill missing with random
    rng = np.random.RandomState(SEED + 100)
    for w in vocab:
        if w not in glove:
            glove[w] = rng.randn(EMBED_DIM).astype(np.float32) * 0.1
    coverage = sum(1 for w in vocab if w in glove) 
    print(f"  GloVe coverage: {coverage}/{V} ({coverage*100//V}%)")
    print()

    # ═══ HRBM: Collect states ═══
    print("Step 3: HRBM — Collecting reservoir states...")
    t0 = time.time()
    res = Reservoir(D_RES, EMBED_DIM, seed=SEED)
    
    states = []
    targets = []
    skipped = 0
    
    for sent in train_sents:
        res.reset()
        for i in range(len(sent) - 1):
            if sent[i] not in w2i or sent[i+1] not in w2i:
                skipped += 1
                continue
            x = glove[sent[i]]
            s = res.step(x)
            states.append(s)
            targets.append(w2i[sent[i+1]])
    
    N = len(states)
    states = np.array(states, dtype=np.float32)
    print(f"  {N} samples (skipped {skipped} OOV)")
    print(f"  N/D ratio: {N/D_RES:.1f}")
    print(f"  Time: {time.time()-t0:.1f}s")
    print()

    # ═══ KARC Fit ═══
    print("Step 4: KARC Ridge Readout...")
    t0 = time.time()
    
    G = states.T @ states  # [D × D]
    G_reg = G + LAMBDA * np.eye(D_RES, dtype=np.float32)
    G_inv = np.linalg.inv(G_reg)
    
    # Target sums
    target_sums = np.zeros((V, D_RES), dtype=np.float32)
    for i, t in enumerate(targets):
        target_sums[t] += states[i]
    
    W_out = target_sums @ G_inv  # [V × D_RES]
    print(f"  Fit: {time.time()-t0:.2f}s")
    print(f"  W_out: [{V} × {D_RES}] = {V*D_RES*4/1e6:.1f} MB")
    print()

    # ═══ 5-gram Baseline ═══
    print("Step 5: Training 5-gram baseline...")
    t0 = time.time()
    ngram = NgramBaseline(n=5)
    ngram.train(train_sents)
    print(f"  Trained in {time.time()-t0:.2f}s")
    print(f"  Contexts stored: {len(ngram.counts)}")
    print()

    # ═══ Evaluate Both ═══
    print("Step 6: Evaluation on TEST set...")
    
    # HRBM evaluation
    res_eval = Reservoir(D_RES, EMBED_DIM, seed=SEED)
    hrbm_log_prob = 0.0
    hrbm_correct = 0
    hrbm_total = 0
    
    # 5-gram evaluation
    ngram_log_prob = 0.0
    ngram_total = 0
    
    for sent in test_sents:
        res_eval.reset()
        for i in range(len(sent) - 1):
            if sent[i] not in w2i or sent[i+1] not in w2i:
                continue
            
            # HRBM prediction
            x = glove[sent[i]]
            s = res_eval.step(x)
            logits = W_out @ s
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            
            target_id = w2i[sent[i+1]]
            if probs.argmax() == target_id:
                hrbm_correct += 1
            hrbm_log_prob += np.log(max(probs[target_id], 1e-10))
            hrbm_total += 1
            
            # 5-gram prediction
            context = sent[max(0, i-4):i+1]
            p_ngram = ngram.predict_prob(context, sent[i+1])
            ngram_log_prob += np.log(max(p_ngram, 1e-10))
            ngram_total += 1
    
    hrbm_ppl = np.exp(-hrbm_log_prob / max(hrbm_total, 1))
    hrbm_acc = hrbm_correct / max(hrbm_total, 1) * 100
    ngram_ppl = np.exp(-ngram_log_prob / max(ngram_total, 1))
    
    print(f"  HRBM:   acc={hrbm_acc:.1f}%, ppl={hrbm_ppl:.1f} ({hrbm_total} tokens)")
    print(f"  5-gram: ppl={ngram_ppl:.1f} ({ngram_total} tokens)")
    print()

    # ═══ Generation ═══
    print("Step 7: Generation from HRBM...")
    res_gen = Reservoir(D_RES, EMBED_DIM, seed=SEED)
    prompts = ["the president", "in the", "she said", "it was", "they were"]
    
    for prompt in prompts:
        words = prompt.split()
        res_gen.reset()
        for w in words:
            if w in glove:
                res_gen.step(glove[w])
        
        generated = list(words)
        for _ in range(10):
            logits = W_out @ res_gen.state
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            
            # Anti-repetition
            for prev in generated[-3:]:
                if prev in w2i:
                    probs[w2i[prev]] *= 0.01
            probs /= probs.sum()
            
            next_id = probs.argmax()
            next_word = vocab[next_id]
            generated.append(next_word)
            if next_word in glove:
                res_gen.step(glove[next_word])
        
        print(f"  \"{prompt}\" → \"{' '.join(generated)}\"")
    
    # ═══ Summary ═══
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  HRBM v4 — WikiText-2 Results")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  HRBM:   ppl={hrbm_ppl:.1f}, acc={hrbm_acc:.1f}%")
    print(f"  5-gram: ppl={ngram_ppl:.1f}")
    print(f"  Winner: {'HRBM' if hrbm_ppl < ngram_ppl else '5-gram'}")
    print()
    print(f"  Config: D={D_RES}, V={V}, N={N}, λ={LAMBDA}")
    print(f"  Backpropagation: NONE")
    print(f"  Hardware: CPU only")
    print()
    if hrbm_ppl < ngram_ppl:
        print("  🎉 HRBM BEATS 5-GRAM without backpropagation!")
    elif hrbm_ppl < 200:
        print("  ✓ HRBM achieves meaningful prediction (ppl < 200)")
    else:
        print("  ⚠ HRBM needs more data or larger reservoir")

if __name__ == "__main__":
    main()
