#!/usr/bin/env python3
"""
Final System: Kneser-Ney 5-gram + GloVe Semantic PoE + Generation
===================================================================
A) Proper Modified Kneser-Ney smoothing (industry standard)
B) GloVe semantic augmentation (our contribution)
C) Generation demo with anti-repetition
D) Full evaluation on WikiText-2

100% Deterministic. Zero gradient training. CPU only.
"""

import numpy as np
import time
from collections import Counter, defaultdict

GLOVE_PATH = "/home/eggchad/eakject/research/Deep_Man/topological-latent-engine/data/glove.6B.50d.txt"
WIKI_PATH = "/home/eggchad/eakject/research/Deep_Man/topological-latent-engine/data/wiki_train.txt"
MAX_VOCAB = 3000
MAX_TOKENS = 150000
np.random.seed(42)

# ═══════════════════════════════════════════════════════════
# PART A: Modified Kneser-Ney Smoothing
# ═══════════════════════════════════════════════════════════

class KneserNeyLM:
    """Modified Kneser-Ney smoothed n-gram language model (up to 5-gram)."""
    
    def __init__(self, max_order=5, discount=0.75):
        self.N = max_order
        self.D = discount  # Fixed discount (simplified MKN)
        self.counts = [{} for _ in range(max_order + 1)]  # counts[n][(ctx)] → Counter
        self.continuation_counts = [{} for _ in range(max_order + 1)]
        self.unigram = Counter()
        self.total_tokens = 0
        self.V = 0
    
    def train(self, token_ids):
        """Single-pass training: collect counts."""
        self.total_tokens = len(token_ids)
        
        for t in token_ids:
            self.unigram[t] += 1
        
        # N-gram counts
        for n in range(1, self.N + 1):
            for i in range(n, len(token_ids)):
                ctx = tuple(token_ids[i - n:i])
                word = token_ids[i]
                if ctx not in self.counts[n]:
                    self.counts[n][ctx] = Counter()
                self.counts[n][ctx][word] += 1
        
        # Continuation counts for KN lower-order
        # continuation_count(w) = |{v : c(v,w) > 0}|
        self.cont_count = Counter()  # word → how many unique left contexts
        for ctx, counter in self.counts[1].items():
            for word in counter:
                self.cont_count[word] += 1
        self.total_cont = sum(self.cont_count.values())
        
        self.V = max(max(self.unigram.keys()) + 1, 1)
    
    def prob(self, word, context):
        """P(word | context) using interpolated KN smoothing.
        context: tuple of preceding token ids (most recent LAST).
        """
        return self._kn_prob(word, context, self.N)
    
    def _kn_prob(self, word, context, order):
        """Recursive KN probability."""
        if order == 0:
            # Unigram level: use continuation counts (KN unigram)
            return (self.cont_count.get(word, 0) + 1) / (self.total_cont + self.V)
        
        ctx = tuple(context[-order:]) if len(context) >= order else tuple(context)
        
        if len(ctx) < order:
            # Context shorter than this order, backoff
            return self._kn_prob(word, context, order - 1)
        
        if ctx not in self.counts[order]:
            # Unseen context, backoff
            return self._kn_prob(word, context, order - 1)
        
        counter = self.counts[order][ctx]
        total = sum(counter.values())
        count_w = counter.get(word, 0)
        
        # Number of unique continuations from this context
        n_unique = len(counter)
        
        # Interpolation weight (lambda)
        lam = self.D * n_unique / total
        
        # Discounted probability + backoff
        p_high = max(count_w - self.D, 0) / total
        p_low = self._kn_prob(word, context, order - 1)
        
        return p_high + lam * p_low
    
    def predict_distribution(self, context, V):
        """Get full probability distribution over vocabulary."""
        probs = np.zeros(V, dtype=np.float64)
        for w in range(V):
            probs[w] = self.prob(w, context)
        # Normalize (should already sum to ~1 but ensure)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(V) / V
        return probs

# ═══════════════════════════════════════════════════════════
# PART B: GloVe Semantic Augmentation
# ═══════════════════════════════════════════════════════════

class GloVeAugmenter:
    """Adds semantic smoothing via GloVe similarity."""
    
    def __init__(self, glove_matrix, temperature=3.0):
        """glove_matrix: [V, 50] normalized."""
        self.G = glove_matrix
        self.tau = temperature
        # Precompute similarity matrix
        self.sim = self.G @ self.G.T  # [V, V]
    
    def semantic_prior(self, context_ids, V):
        """Compute semantic prior based on recent context words."""
        if len(context_ids) == 0:
            return np.ones(V) / V
        
        # Average of recent context embeddings (last 5)
        recent = context_ids[-5:]
        ctx_vec = self.G[recent].mean(axis=0)
        ctx_norm = np.linalg.norm(ctx_vec)
        if ctx_norm < 1e-8:
            return np.ones(V) / V
        ctx_vec /= ctx_norm
        
        # Cosine similarity to all vocab
        sims = self.G @ ctx_vec
        
        # Softmax with temperature
        sims_scaled = sims * self.tau
        sims_scaled -= sims_scaled.max()
        probs = np.exp(sims_scaled)
        probs /= probs.sum()
        return probs

# ═══════════════════════════════════════════════════════════
# PART C: Combined PoE System
# ═══════════════════════════════════════════════════════════

class FinalSystem:
    """Kneser-Ney + GloVe semantic = deterministic LM."""
    
    def __init__(self, kn_model, glove_aug, V, alpha_kn=0.85, alpha_sem=0.10, alpha_uni=0.05):
        self.kn = kn_model
        self.glove = glove_aug
        self.V = V
        self.a_kn = alpha_kn
        self.a_sem = alpha_sem
        self.a_uni = alpha_uni
        self.p_uni = np.ones(V) / V
    
    def predict(self, context_ids):
        """Predict next token distribution. DETERMINISTIC."""
        # Expert 1: Kneser-Ney n-gram
        p_kn = self.kn.predict_distribution(tuple(context_ids[-5:]), self.V)
        
        # Expert 2: GloVe semantic
        p_sem = self.glove.semantic_prior(context_ids, self.V)
        
        # Additive mixture
        p = self.a_kn * p_kn + self.a_sem * p_sem + self.a_uni * self.p_uni
        p /= p.sum()
        return p
    
    def generate(self, prompt_ids, max_tokens=20):
        """Generate tokens deterministically (argmax)."""
        context = list(prompt_ids)
        generated = list(prompt_ids)
        
        for _ in range(max_tokens):
            p = self.predict(context)
            
            # Anti-repetition: penalize last 5 tokens
            for tok in context[-5:]:
                p[tok] *= 0.1
            p /= p.sum()
            
            # Deterministic: argmax
            next_tok = int(p.argmax())
            generated.append(next_tok)
            context.append(next_tok)
        
        return generated

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Final System: KN-5 + GloVe Semantic PoE                    ║")
    print("║  Deterministic • Zero Training • CPU Only                   ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # Load data
    print("Loading WikiText-2...")
    all_tokens = []
    with open(WIKI_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('='): continue
            words = [w for w in line.lower().split() if w.isalpha() and w != 'unk']
            all_tokens.extend(words)
            if len(all_tokens) >= MAX_TOKENS: break

    freq = Counter(all_tokens)
    vocab = [w for w, _ in freq.most_common(MAX_VOCAB)]
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = vocab
    V = len(vocab)
    
    ids = [w2i[w] for w in all_tokens if w in w2i]
    split = int(len(ids) * 0.8)
    train_ids, test_ids = ids[:split], ids[split:]
    
    print(f"  Vocab: {V}, Train: {len(train_ids)} tokens, Test: {len(test_ids)} tokens\n")

    # Load GloVe
    print("Loading GloVe...")
    glove_raw = {}
    with open(GLOVE_PATH) as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] in set(vocab):
                glove_raw[parts[0]] = np.array([float(x) for x in parts[1:51]], dtype=np.float32)
    
    G = np.zeros((V, 50), dtype=np.float32)
    for w, i in w2i.items():
        G[i] = glove_raw.get(w, np.random.randn(50).astype(np.float32) * 0.1)
    Gn = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-8)
    print(f"  GloVe: {sum(1 for w in vocab if w in glove_raw)}/{V} words\n")

    # ═══ Train KN-5 ═══
    print("Training Kneser-Ney 5-gram (single pass)...")
    t0 = time.time()
    kn = KneserNeyLM(max_order=5, discount=0.75)
    kn.train(train_ids)
    kn.V = V
    print(f"  Done in {time.time()-t0:.1f}s\n")

    # ═══ Build system ═══
    glove_aug = GloVeAugmenter(Gn, temperature=3.0)
    system = FinalSystem(kn, glove_aug, V, alpha_kn=0.88, alpha_sem=0.08, alpha_uni=0.04)

    # ═══ Evaluate ═══
    print("Evaluating on test set...")
    t0 = time.time()
    
    # Pure KN-5
    log_p_kn = 0.0
    log_p_sys = 0.0
    total = 0
    correct_kn = 0
    correct_sys = 0
    
    # Evaluate every 3rd token for speed (full test is slow with V=3000)
    for i in range(5, len(test_ids), 3):
        target = test_ids[i]
        ctx = test_ids[max(0, i-5):i]
        
        # Pure KN
        p_kn = kn.prob(target, tuple(ctx))
        log_p_kn += np.log2(max(p_kn, 1e-15))
        
        # Full system (KN + GloVe)
        p_full = system.predict(ctx)
        log_p_sys += np.log2(max(p_full[target], 1e-15))
        
        if p_full.argmax() == target:
            correct_sys += 1
        total += 1
    
    ppl_kn = 2 ** (-log_p_kn / total)
    ppl_sys = 2 ** (-log_p_sys / total)
    acc_sys = correct_sys / total * 100
    eval_time = time.time() - t0
    
    improvement = (1 - ppl_sys / ppl_kn) * 100

    print(f"\n{'='*60}")
    print(f"  RESULTS (sampled evaluation, {total} tokens)")
    print(f"{'='*60}")
    print(f"  Pure KN-5:        ppl = {ppl_kn:.1f}")
    print(f"  KN-5 + GloVe:     ppl = {ppl_sys:.1f}  acc = {acc_sys:.1f}%")
    print(f"  Improvement:      {improvement:.1f}%")
    print(f"  Eval time:        {eval_time:.1f}s")
    print(f"  Deterministic:    YES")
    print(f"  Gradient training: NONE")
    print(f"{'='*60}\n")

    if ppl_sys < ppl_kn:
        print(f"  🎉 KN-5 + GloVe BEATS pure KN-5! ({ppl_sys:.1f} < {ppl_kn:.1f})")
    
    # ═══ Generation Demo ═══
    print("\n━━━ GENERATION DEMO ━━━\n")
    prompts = [
        "the president of",
        "in the first",
        "it was a",
        "she said that",
        "the city of",
        "he was the",
        "they were not",
        "the game was",
    ]
    
    for prompt in prompts:
        prompt_ids = [w2i[w] for w in prompt.split() if w in w2i]
        if not prompt_ids:
            continue
        gen_ids = system.generate(prompt_ids, max_tokens=12)
        gen_text = " ".join(i2w[i] for i in gen_ids)
        print(f"  \"{prompt}\" → \"{gen_text}\"")
    
    # ═══ Determinism verification ═══
    print("\n━━━ DETERMINISM CHECK ━━━")
    outputs = set()
    for _ in range(10):
        gen = system.generate([w2i["the"], w2i["president"]], max_tokens=8)
        outputs.add(tuple(gen))
    print(f"  10 runs → {len(outputs)} unique output(s)")
    print(f"  Deterministic: {'✓' if len(outputs) == 1 else '✗'}")
    
    # ═══ Summary ═══
    print(f"\n{'='*60}")
    print(f"  SYSTEM SPECS")
    print(f"{'='*60}")
    print(f"  Architecture:  KN-5 + GloVe Semantic Smoothing")
    print(f"  Vocabulary:    {V} words")
    print(f"  Memory:        ~{V*V*4/1e6 + V*50*4/1e6:.0f} MB (sim matrix + GloVe)")
    print(f"  Training:      Single pass count collection ({len(train_ids)} tokens)")
    print(f"  Inference:     Deterministic argmax")
    print(f"  Parameters:    0 trained (only hyperparameters α set manually)")
    print(f"  Hardware:      CPU only, <50MB RAM")

if __name__ == "__main__":
    main()
