# Deep Mathematical Decomposition of the Attention Mechanism
## What Trained Q/K/V Weights Provide & Algebraic Alternatives

---

## Part 1: What Q, K, V Actually Do (Mechanistic Decomposition)

### The Standard Attention Formula

```
Attn(X) = softmax( (XW_Q)(XW_K)^T / √d_k ) × (XW_V)
```

Where:
- X ∈ ℝ^{n×d} is the input matrix (n tokens, d-dimensional embeddings)
- W_Q, W_K ∈ ℝ^{d×d_k} are query/key projections
- W_V ∈ ℝ^{d×d_v} is the value projection
- The output is ∈ ℝ^{n×d_v}

### Decomposition into Sub-Operations

The full attention computation decomposes into **four distinct operations**:

```
1. Q = XW_Q          (query formation)
2. K = XW_K          (key formation)  
3. A = softmax(QK^T/√d)  (attention pattern computation)
4. O = AV = A(XW_V)      (value aggregation)
```

---

### W_Q (Query Projection): "What Am I Looking For?"

**Mathematical role:** W_Q ∈ ℝ^{d×d_k} is a linear map that projects each token x_i into a d_k-dimensional "query space."

**What it computes:**
```
q_i = x_i W_Q
```

This selects and recombines features of x_i to form a *search vector*. The query encodes: "Given my identity (semantic content + position), what kind of token do I need information from?"

**Mechanistic interpretation:**
- W_Q acts as a **feature selector/rotator** on the current token
- It determines which dimensions of the embedding space are relevant for *finding matches*
- Different attention heads learn different W_Q, meaning each head "looks for" different things

**Information-theoretic view:**
- W_Q compresses d dimensions → d_k dimensions
- This compression is a *learned bottleneck* that forces the model to prioritize certain features for matching
- Rank of W_Q determines the richness of the query space

**Without training, natural candidates for "what to look for":**
1. **Position**: f(i) — "look for tokens near me" (no content needed)
2. **Semantic similarity**: embed(w_i) directly — "look for tokens like me"
3. **Syntactic role**: POS(w_i) — "look for syntactically compatible tokens"
4. **Frequency/salience**: IDF(w_i) — "rare tokens look harder"

---

### W_K (Key Projection): "What Do I Offer as a Match?"

**Mathematical role:** W_K ∈ ℝ^{d×d_k} projects each token into the same d_k-dimensional space as queries.

**What it computes:**
```
k_j = x_j W_K
```

**Critical insight — the QK^T bilinear form:**
```
score(i,j) = q_i · k_j = (x_i W_Q)(x_j W_K)^T = x_i (W_Q W_K^T) x_j^T
```

The product W_Q W_K^T ∈ ℝ^{d×d} defines a **bilinear form** over the embedding space. This means:
- Q and K don't independently matter — only their interaction W_Q W_K^T matters
- This bilinear form defines a *compatibility metric* between token pairs
- It is generally **asymmetric**: score(i→j) ≠ score(j→i)

**Asymmetry is key:**
- W_Q ≠ W_K means query and key roles are different
- Token i looking for j ≠ Token j looking for i
- This enables *directional* attention (e.g., nouns attend to their adjectives, not vice versa)

**Without training, natural candidates for "what I offer":**
1. **Same space as queries** (symmetric): W_K = W_Q → attention ∝ cos(x_i, x_j)
2. **Identity**: W_K = I → keys are raw embeddings
3. **Complementary role**: If Q encodes "what POS do I need?", K encodes "what POS am I"

---

### W_V (Value Projection): "What Information Do I Contribute?"

**Mathematical role:** W_V ∈ ℝ^{d×d_v} projects each token into a "value space" — the information that gets passed forward.

**What it computes:**
```
v_j = x_j W_V
output_i = Σ_j α_{ij} v_j    (where α = attention weights)
```

**Critical distinction from Q/K:**
- Q and K determine the **routing** (who talks to whom)
- V determines the **content** (what information flows)
- These are fundamentally independent functions!

**Mechanistic interpretation:**
- W_V selects which aspects of attended tokens are *useful for downstream computation*
- Different heads project into different value subspaces → different information channels
- The output projection W_O then recombines these channels

**Without training, natural candidates for "value":**
1. **Raw embedding**: W_V = I → pass the GloVe vector directly
2. **Identity residual**: v_j = x_j (the input itself is the value)
3. **Positional encoding**: v_j encodes relative position information
4. **Difference vector**: v_j = x_j - x_i (what j adds beyond i)

---

### The QK^T Product: Attention Pattern Matrix

**What it computes:**
```
A = softmax(QK^T / √d_k) ∈ ℝ^{n×n}
A_{ij} = probability that token i attends to token j
```

**This is a learned routing matrix.** The fundamental question is: *How much of this routing is predictable without learning?*

**Decomposition of what QK^T encodes:**
```
(QK^T)_{ij} = x_i (W_Q W_K^T) x_j^T
            = Σ_{a,b} x_i[a] · M[a,b] · x_j[b]
```
where M = W_Q W_K^T is the learned bilinear interaction matrix.

This score captures:
1. **Content-content interaction**: Do these tokens' meanings match?
2. **Position-position interaction**: Are these tokens at compatible positions?
3. **Content-position interaction**: Does token i's content want token j's position?

