#!/usr/bin/env python3
"""
HRBM v5: Reservoir + 1 Trained Hidden Layer (Minimal Backprop)
================================================================
Architecture:
  Frozen Reservoir (D=512) → 1 Hidden Layer (ReLU, trained) → Vocab Output

"Training" = SGD on 1 layer only. ~2-5 minutes on CPU.
Compare: pure KARC (ridge only) vs KARC+1layer vs 5-gram
"""

import numpy as np
import time

# ═══════════════════ CONFIG ═══════════════════
D_RES = 512
HIDDEN = 256         # Hidden layer size
EMBED_DIM = 50
LEAK = 0.3
SPECTRAL = 0.9
SPARSITY = 0.1
SEED = 42
MAX_VOCAB = 1000
MAX_TRAIN_TOKENS = 50000
LR = 0.01            # Learning rate for 1-layer SGD
EPOCHS = 3           # Very few epochs (minimal training)
BATCH_SIZE = 128

GLOVE_PATH = "/home/eggchad/eakject/research/Deep_Man/topological-latent-engine/data/glove.6B.50d.txt"
WIKI_PATH = "/home/eggchad/eakject/research/Deep_Man/topological-latent-engine/data/wiki_train.txt"

np.random.seed(SEED)

# ═══════════════════ UTILS ═══════════════════

def load_wiki(path, max_tokens):
    sentences = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('='):
                continue
            words = [w for w in line.lower().split() if w.isalpha() and w != 'unk']
            if len(words) >= 3:
                sentences.append(words)
    flat = []
    for s in sentences:
        if len(flat) + len(s) > max_tokens:
            break
        flat.extend(s)
    return [flat[i:i+10] for i in range(0, len(flat)-10, 10)]

def load_glove(path, vocab_set):
    emb = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] in vocab_set:
                emb[parts[0]] = np.array([float(x) for x in parts[1:51]], dtype=np.float32)
    rng = np.random.RandomState(99)
    for w in vocab_set:
        if w not in emb:
            emb[w] = rng.randn(EMBED_DIM).astype(np.float32) * 0.1
    return emb

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy(probs, targets):
    """Average cross-entropy loss."""
    n = len(targets)
    log_probs = np.log(probs[np.arange(n), targets] + 1e-10)
    return -log_probs.mean()

# ═══════════════════ RESERVOIR ═══════════════════

class Reservoir:
    def __init__(self):
        rng = np.random.RandomState(SEED)
        self.W_res = rng.randn(D_RES, D_RES).astype(np.float32)
        mask = rng.rand(D_RES, D_RES) > SPARSITY
        self.W_res[mask] = 0
        scale = np.sqrt(D_RES * SPARSITY)
        self.W_res *= SPECTRAL / max(scale, 0.01)
        self.W_in = rng.randn(D_RES, EMBED_DIM).astype(np.float32) / np.sqrt(EMBED_DIM)
        self.state = np.zeros(D_RES, dtype=np.float32)

    def step(self, x):
        pre = self.W_res @ self.state + self.W_in @ x
        self.state = (1 - LEAK) * self.state + LEAK * np.tanh(pre)
        return self.state.copy()

    def reset(self):
        self.state[:] = 0

# ═══════════════════ MODELS ═══════════════════

class PureKARC:
    """Ridge-only readout (no backprop)."""
    def __init__(self, states, targets, V):
        G = states.T @ states + 0.1 * np.eye(D_RES, dtype=np.float32)
        G_inv = np.linalg.inv(G)
        target_sums = np.zeros((V, D_RES), dtype=np.float32)
        for i, t in enumerate(targets):
            target_sums[t] += states[i]
        self.W = target_sums @ G_inv  # [V, D_RES]

    def predict(self, state):
        return softmax(self.W @ state)

class HybridModel:
    """Reservoir → 1 Hidden Layer (ReLU) → Output (trained with SGD)."""
    def __init__(self, V):
        # Xavier init
        self.W1 = np.random.randn(HIDDEN, D_RES).astype(np.float32) * np.sqrt(2.0 / D_RES)
        self.b1 = np.zeros(HIDDEN, dtype=np.float32)
        self.W2 = np.random.randn(V, HIDDEN).astype(np.float32) * np.sqrt(2.0 / HIDDEN)
        self.b2 = np.zeros(V, dtype=np.float32)
        self.V = V

    def forward(self, states):
        """Forward pass: states [N, D_RES] → probs [N, V]"""
        self.h = states @ self.W1.T + self.b1  # [N, HIDDEN]
        self.a = np.maximum(self.h, 0)          # ReLU
        logits = self.a @ self.W2.T + self.b2   # [N, V]
        return softmax(logits)

    def backward(self, states, probs, targets, lr):
        """Backward pass: compute gradients and update."""
        N = len(targets)
        # dL/d_logits = probs - one_hot(targets)
        d_logits = probs.copy()
        d_logits[np.arange(N), targets] -= 1.0
        d_logits /= N

        # Gradients for W2, b2
        dW2 = d_logits.T @ self.a        # [V, HIDDEN]
        db2 = d_logits.sum(axis=0)        # [V]

        # Backprop through ReLU
        d_a = d_logits @ self.W2           # [N, HIDDEN]
        d_h = d_a * (self.h > 0).astype(np.float32)  # ReLU derivative

        # Gradients for W1, b1
        dW1 = d_h.T @ states              # [HIDDEN, D_RES]
        db1 = d_h.sum(axis=0)             # [HIDDEN]

        # SGD update
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train_epoch(self, states, targets, lr):
        """Train one epoch with mini-batches."""
        N = len(targets)
        indices = np.random.permutation(N)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, N, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N)
            idx = indices[start:end]
            batch_states = states[idx]
            batch_targets = targets[idx]

            probs = self.forward(batch_states)
            loss = cross_entropy(probs, batch_targets)
            total_loss += loss
            n_batches += 1

            self.backward(batch_states, probs, batch_targets, lr)

        return total_loss / n_batches

    def predict(self, state):
        """Single prediction."""
        h = state @ self.W1.T + self.b1
        a = np.maximum(h, 0)
        logits = a @ self.W2.T + self.b2
        return softmax(logits.reshape(1, -1)).flatten()

# ═══════════════════ 5-GRAM ═══════════════════

class Ngram:
    def __init__(self):
        self.counts = {}
        self.unigram = {}
        self.total = 0

    def train(self, sentences):
        from collections import Counter
        uni = Counter()
        for sent in sentences:
            for w in sent:
                uni[w] += 1
            for n in range(1, 6):
                for i in range(len(sent) - n):
                    ctx = tuple(sent[i:i+n])
                    nxt = sent[i+n] if i+n < len(sent) else None
                    if nxt:
                        self.counts.setdefault(ctx, Counter())[nxt] += 1
        self.unigram = uni
        self.total = sum(uni.values())

    def prob(self, context, target):
        for n in range(min(len(context), 5), 0, -1):
            ctx = tuple(context[-n:])
            if ctx in self.counts and target in self.counts[ctx]:
                return self.counts[ctx][target] / sum(self.counts[ctx].values())
        return self.unigram.get(target, 1) / (self.total + 1)

# ═══════════════════ EVALUATE ═══════════════════

def evaluate(model, test_sents, vocab, w2i, glove, label):
    res = Reservoir()
    correct = 0
    total = 0
    log_p = 0.0

    for sent in test_sents:
        res.reset()
        for i in range(len(sent) - 1):
            if sent[i] not in w2i or sent[i+1] not in w2i:
                continue
            s = res.step(glove[sent[i]])
            probs = model.predict(s)
            tid = w2i[sent[i+1]]
            if probs.argmax() == tid:
                correct += 1
            log_p += np.log(max(probs[tid], 1e-10))
            total += 1

    acc = correct / max(total, 1) * 100
    ppl = np.exp(-log_p / max(total, 1))
    print(f"  {label}: acc={acc:.1f}%, ppl={ppl:.1f} ({total} tokens)")
    return ppl, acc

# ═══════════════════ MAIN ═══════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  HRBM v5: Reservoir + 1 Trained Layer (Minimal Backprop)    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Reservoir D={D_RES} (frozen), Hidden={HIDDEN} (trained)")
    print(f"  Training: {EPOCHS} epochs SGD, lr={LR}, batch={BATCH_SIZE}")
    print()

    # Load data
    sents = load_wiki(WIKI_PATH, MAX_TRAIN_TOKENS)
    n_train = int(len(sents) * 0.8)
    train_s, test_s = sents[:n_train], sents[n_train:]

    # Build vocab
    from collections import Counter
    freq = Counter()
    for s in sents:
        for w in s:
            freq[w] += 1
    vocab = [w for w, c in freq.most_common(MAX_VOCAB)]
    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    train_tokens = sum(len(s) for s in train_s)
    test_tokens = sum(len(s) for s in test_s)
    print(f"  Data: train={train_tokens} tokens, test={test_tokens} tokens, V={V}")

    # GloVe
    glove = load_glove(GLOVE_PATH, set(vocab))
    print(f"  GloVe loaded: {sum(1 for w in vocab if w in glove)}/{V}")
    print()

    # Collect reservoir states
    print("Step 1: Reservoir states (frozen)...")
    t0 = time.time()
    res = Reservoir()
    all_states = []
    all_targets = []
    for sent in train_s:
        res.reset()
        for i in range(len(sent) - 1):
            if sent[i] in w2i and sent[i+1] in w2i:
                s = res.step(glove[sent[i]])
                all_states.append(s)
                all_targets.append(w2i[sent[i+1]])

    states_np = np.array(all_states, dtype=np.float32)
    targets_np = np.array(all_targets, dtype=np.int32)
    N = len(states_np)
    print(f"  {N} samples, N/D={N/D_RES:.1f}, time={time.time()-t0:.1f}s")
    print()

    # ═══ Model A: Pure KARC (no backprop) ═══
    print("Step 2a: Pure KARC (ridge only, no backprop)...")
    t0 = time.time()
    karc = PureKARC(states_np, targets_np, V)
    print(f"  Fit: {time.time()-t0:.2f}s")

    # ═══ Model B: Hybrid (1 trained layer) ═══
    print(f"\nStep 2b: Hybrid (1 hidden layer, {EPOCHS} epochs SGD)...")
    hybrid = HybridModel(V)
    t0 = time.time()
    for epoch in range(EPOCHS):
        loss = hybrid.train_epoch(states_np, targets_np, LR)
        print(f"  Epoch {epoch+1}/{EPOCHS}: loss={loss:.4f}")
    train_time = time.time() - t0
    print(f"  Train time: {train_time:.1f}s")

    # ═══ Model C: 5-gram ═══
    print("\nStep 2c: 5-gram baseline...")
    ngram = Ngram()
    ngram.train(train_s)

    # ═══ Evaluate all ═══
    print("\n━━━ TEST SET EVALUATION ━━━")
    karc_ppl, karc_acc = evaluate(karc, test_s, vocab, w2i, glove, "Pure KARC (no backprop)")
    hybrid_ppl, hybrid_acc = evaluate(hybrid, test_s, vocab, w2i, glove, "Hybrid (1 layer trained)")

    # 5-gram eval
    ngram_log_p = 0.0
    ngram_total = 0
    for sent in test_s:
        for i in range(len(sent) - 1):
            if sent[i] in w2i and sent[i+1] in w2i:
                ctx = sent[max(0,i-4):i+1]
                p = ngram.prob(ctx, sent[i+1])
                ngram_log_p += np.log(max(p, 1e-10))
                ngram_total += 1
    ngram_ppl = np.exp(-ngram_log_p / max(ngram_total, 1))
    print(f"  5-gram:                    ppl={ngram_ppl:.1f} ({ngram_total} tokens)")

    # ═══ Generation ═══
    print("\n━━━ GENERATION COMPARISON ━━━")
    prompts = ["the president", "in the", "it was", "they were"]
    res_gen = Reservoir()

    for prompt in prompts:
        words = prompt.split()
        res_gen.reset()
        for w in words:
            if w in glove:
                res_gen.step(glove[w])

        gen_karc = list(words)
        gen_hybrid = list(words)
        res_k = Reservoir()
        res_h = Reservoir()
        for w in words:
            if w in glove:
                res_k.step(glove[w])
                res_h.step(glove[w])

        for _ in range(8):
            # KARC
            pk = karc.predict(res_k.state)
            for prev in gen_karc[-2:]:
                if prev in w2i: pk[w2i[prev]] *= 0.01
            pk /= pk.sum()
            nk = vocab[pk.argmax()]
            gen_karc.append(nk)
            if nk in glove: res_k.step(glove[nk])

            # Hybrid
            ph = hybrid.predict(res_h.state)
            for prev in gen_hybrid[-2:]:
                if prev in w2i: ph[w2i[prev]] *= 0.01
            ph /= ph.sum()
            nh = vocab[ph.argmax()]
            gen_hybrid.append(nh)
            if nh in glove: res_h.step(glove[nh])

        print(f"  Prompt: \"{prompt}\"")
        print(f"    KARC:   \"{' '.join(gen_karc)}\"")
        print(f"    Hybrid: \"{' '.join(gen_hybrid)}\"")

    # ═══ Summary ═══
    print("\n━━━ FINAL COMPARISON ━━━")
    print(f"  {'System':<30} {'Perplexity':>12} {'Accuracy':>10} {'Training':>20}")
    print(f"  {'-'*72}")
    print(f"  {'Pure KARC (no backprop)':<30} {karc_ppl:>12.1f} {karc_acc:>9.1f}% {'0s (1 equation)':>20}")
    print(f"  {'Hybrid (1 layer, 3 epochs)':<30} {hybrid_ppl:>12.1f} {hybrid_acc:>9.1f}% {f'{train_time:.0f}s SGD':>20}")
    print(f"  {'5-gram baseline':<30} {ngram_ppl:>12.1f} {'—':>10} {'0s (counting)':>20}")
    print()

    improvement = (karc_ppl - hybrid_ppl) / karc_ppl * 100
    print(f"  Hybrid improvement over KARC: {improvement:.1f}% perplexity reduction")
    if hybrid_ppl < ngram_ppl:
        print(f"  🎉 HYBRID BEATS 5-GRAM! ({hybrid_ppl:.1f} < {ngram_ppl:.1f})")
    elif hybrid_ppl < karc_ppl * 0.5:
        print(f"  ✓ 1 trained layer cuts perplexity by >50%")
    print(f"\n  Total compute: {train_time:.0f}s on CPU (reservoir frozen, only 1 layer trained)")

if __name__ == "__main__":
    main()
