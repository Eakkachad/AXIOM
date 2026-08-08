#!/usr/bin/env python3
"""
HRBM v3: GloVe Embeddings + Proper Train/Test
================================================
- Pre-trained GloVe 50d (no training, just loading)
- 100 sentences, 80/20 split
- Reservoir D=256 (ensure N >> D for generalization)
- Measure REAL test perplexity
"""

import numpy as np
import time
from collections import Counter

# ═══════════════════ CONFIG ═══════════════════
D_RES = 256          # Reservoir dimension (keep N >> D)
LEAK = 0.3
SPECTRAL = 0.9
SPARSITY = 0.1
LAMBDA = 10.0        # Strong regularization for generalization
SEED = 42
GLOVE_PATH = "/home/eggchad/eakject/research/Deep_Man/topological-latent-engine/data/glove.6B.50d.txt"
EMBED_DIM = 50       # GloVe dimension

np.random.seed(SEED)

# ═══════════════════ LOAD GLOVE ═══════════════════

def load_glove(path, vocab_words):
    """Load GloVe vectors for words in our vocabulary."""
    print(f"  Loading GloVe from {path.split('/')[-1]}...")
    embeddings = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in vocab_words:
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                embeddings[word] = vec
    
    # For words not in GloVe, use random
    missing = 0
    for w in vocab_words:
        if w not in embeddings:
            embeddings[w] = np.random.randn(EMBED_DIM).astype(np.float32) * 0.1
            missing += 1
    
    print(f"  Loaded {len(embeddings)-missing} from GloVe, {missing} random fallback")
    return embeddings

# ═══════════════════ CORPUS ═══════════════════

CORPUS = [
    "the cat sat on the mat", "the dog ran in the park",
    "she walked to the store", "he ate a red apple",
    "the bird flew over the tree", "they played in the garden",
    "i love my cat very much", "the sun is bright and warm",
    "the moon shines at night", "we went to the beach",
    "the fish swam in the water", "she read a good book",
    "he built a small house", "the car stopped at the light",
    "the rain fell all day", "she opened the front door",
    "the child ran to school", "he drove to work early",
    "the flower grew very tall", "we watched the stars at night",
    "the river flows to the sea", "she wrote a long letter",
    "the old man walked slowly", "he fixed the broken chair",
    "the wind blew the leaves", "she found a gold ring",
    "the mountain is very high", "he jumped over the wall",
    "they sang songs all night", "the snow covered the road",
    "she threw the ball to him", "the sky turned very dark",
    "he read the news every day", "they walked by the river",
    "the big cat chased the mouse", "a small bird sang a song",
    "the hot food was very good", "she smiled at the child",
    "he ran faster than the dog", "the cold wind was very sharp",
    "we ate dinner at home", "the tree lost all its leaves",
    "she called her best friend", "he painted the wall blue",
    "the baby slept all night", "they moved to a new city",
    "the teacher spoke very clearly", "she danced in the rain",
    "he climbed the tall tree", "the boat sailed on the water",
    "we played cards all night", "the door closed with a bang",
    "she bought some fresh bread", "he told a funny story",
    "the light came on at night", "they arrived very late",
    "the cat and dog played together", "she looked out the window",
    "he put the book on the table", "the music played all night",
    "we sat under the big tree", "the phone rang very loud",
    "she picked up the red pen", "he walked through the door",
    "the clock struck twelve at night", "they shared the big cake",
    "the birds flew south for winter", "she waited for the bus",
    "he opened his old bag", "the children laughed out loud",
    "we drove along the coast road", "the fire burned all night",
    "she planted flowers in the yard", "he caught the red ball",
    "the train arrived on time today", "they built a sand castle",
    "the stars shine very bright tonight", "she lost her house key",
    "he woke up very early today", "the dog barked at the cat",
    "we finished work at five today", "the story had a happy end",
    "she borrowed a book from library", "he swam across the cold lake",
    "the ice cream melted very fast", "they invited all their friends",
    "the night was cold and very dark", "she asked a very hard question",
    "he saved money for the trip", "the bridge crossed the wide river",
    "we learned something new today here", "the summer was long and hot",
    "she wrapped the gift with care", "he smiled when she came in",
    "the garden had many wild flowers", "they talked about the new plan",
    "the cat is a small animal", "the dog is a good friend",
    "the sun rose in the east", "the moon is full tonight",
    "she is very happy today here", "he is a good man",
    "the water is cold and clear", "the food is hot and fresh",
    "they are good friends now here", "we are very happy today",
]

# ═══════════════════ RESERVOIR ═══════════════════

class Reservoir:
    def __init__(self, d_res, d_in, seed=42):
        rng = np.random.RandomState(seed)
        # Sparse reservoir
        self.W_res = rng.randn(d_res, d_res).astype(np.float32) * (1.0 / np.sqrt(d_res * SPARSITY))
        mask = rng.rand(d_res, d_res) > SPARSITY
        self.W_res[mask] = 0
        # Scale spectral radius
        eigvals = np.abs(np.linalg.eigvals(self.W_res))
        max_eig = eigvals.max() if eigvals.max() > 0 else 1.0
        self.W_res *= SPECTRAL / max_eig
        
        self.W_in = rng.randn(d_res, d_in).astype(np.float32) * (1.0 / np.sqrt(d_in))
        self.state = np.zeros(d_res, dtype=np.float32)
    
    def step(self, x):
        pre = self.W_res @ self.state + self.W_in @ x
        self.state = (1 - LEAK) * self.state + LEAK * np.tanh(pre)
        return self.state.copy()
    
    def reset(self):
        self.state = np.zeros_like(self.state)

# ═══════════════════ MAIN ═══════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  HRBM v3: GloVe Embeddings + Proper Generalization Test     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  D_res={D_RES}, Embed=GloVe-{EMBED_DIM}d, λ={LAMBDA}, leak={LEAK}")
    print()

    # Build vocab
    all_words = set()
    for s in CORPUS:
        for w in s.split():
            all_words.add(w)
    vocab = sorted(all_words)
    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    print(f"  Corpus: {len(CORPUS)} sentences, Vocab: {V} words")

    # Load GloVe
    glove = load_glove(GLOVE_PATH, set(vocab))
    
    # Train/Test split
    n_train = int(len(CORPUS) * 0.8)
    train_sents = CORPUS[:n_train]
    test_sents = CORPUS[n_train:]
    print(f"  Split: train={len(train_sents)}, test={len(test_sents)}")
    print()

    # Collect reservoir states from TRAINING data
    print("Step 1: Collect reservoir states...")
    t0 = time.time()
    res = Reservoir(D_RES, EMBED_DIM, seed=SEED)
    
    states = []
    targets = []
    for sent in train_sents:
        words = sent.split()
        res.reset()
        for i in range(len(words) - 1):
            x = glove[words[i]]
            s = res.step(x)
            states.append(s)
            targets.append(w2i[words[i+1]])
    
    states = np.array(states)  # [N, D_RES]
    N = len(states)
    print(f"  {N} samples collected in {time.time()-t0:.2f}s")
    print(f"  N/D ratio: {N/D_RES:.1f} (want > 2 for generalization)")
    print()

    # KARC Ridge Readout
    print("Step 2: KARC Ridge Readout...")
    t0 = time.time()
    
    # Gram: G = H^T·H [D×D] (since N > D, use feature-space)
    G = states.T @ states  # [D_RES × D_RES]
    
    # Target matrix: Y[v, i] = 1 if targets[i]==v
    # H·y_v = states^T where target==v = sum of states with that target
    target_sums = np.zeros((V, D_RES), dtype=np.float32)
    for i, t in enumerate(targets):
        target_sums[t] += states[i]
    
    # Solve: W_out[v] = (G + λI)^{-1} · target_sums[v]
    G_reg = G + LAMBDA * np.eye(D_RES, dtype=np.float32)
    G_inv = np.linalg.inv(G_reg)
    W_out = target_sums @ G_inv  # [V × D_RES]
    
    print(f"  Fit done in {time.time()-t0:.2f}s")
    print(f"  W_out: [{V} × {D_RES}]")
    print()

    # Evaluate
    print("Step 3: Evaluate...")
    
    def evaluate(sentences, label):
        res_eval = Reservoir(D_RES, EMBED_DIM, seed=SEED)
        correct = 0
        total = 0
        log_prob_sum = 0.0
        
        for sent in sentences:
            words = sent.split()
            res_eval.reset()
            for i in range(len(words) - 1):
                x = glove[words[i]]
                s = res_eval.step(x)
                
                # Predict
                logits = W_out @ s
                # Softmax
                logits -= logits.max()
                exp_l = np.exp(logits)
                probs = exp_l / exp_l.sum()
                
                pred = probs.argmax()
                target = w2i[words[i+1]]
                
                if pred == target:
                    correct += 1
                log_prob_sum += np.log(max(probs[target], 1e-10))
                total += 1
        
        acc = correct / total * 100
        ppl = np.exp(-log_prob_sum / total)
        print(f"  {label}: acc={acc:.1f}%, ppl={ppl:.1f} ({correct}/{total})")
        return acc, ppl
    
    train_acc, train_ppl = evaluate(train_sents, "TRAIN")
    test_acc, test_ppl = evaluate(test_sents, "TEST ")
    print()
    
    # Generation
    print("Step 4: Generation...")
    prompts = ["the cat", "she walked", "he ate", "the sun", "the dog", "we played"]
    res_gen = Reservoir(D_RES, EMBED_DIM, seed=SEED)
    
    for prompt in prompts:
        words = prompt.split()
        res_gen.reset()
        for w in words:
            res_gen.step(glove[w])
        
        generated = list(words)
        for _ in range(8):
            logits = W_out @ res_gen.state
            logits -= logits.max()
            exp_l = np.exp(logits)
            probs = exp_l / exp_l.sum()
            
            # Anti-repetition: penalize last 3 words
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
    
    print()
    print("━━━ HRBM v3 RESULTS ━━━")
    print(f"  Architecture: Reservoir D={D_RES} + GloVe-{EMBED_DIM}d + Ridge(λ={LAMBDA})")
    print(f"  TRAIN: acc={train_acc:.1f}%, ppl={train_ppl:.1f}")
    print(f"  TEST:  acc={test_acc:.1f}%, ppl={test_ppl:.1f}")
    print(f"  N/D ratio: {N/D_RES:.1f}")
    print(f"  Backpropagation: NONE")
    print(f"  Training: single matrix equation")
    
    gap = train_acc - test_acc
    print(f"  Generalization gap: {gap:.1f}% (smaller = better)")
    
    if test_acc > 10:
        print(f"  ✓ GloVe HELPS: test accuracy > 10% (generalization!)")
    if test_ppl < 200:
        print(f"  ✓ Test PPL < 200: meaningful prediction ability")
    if gap < 30:
        print(f"  ✓ Gap < 30%: reasonable generalization")

if __name__ == "__main__":
    main()
