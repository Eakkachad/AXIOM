#!/usr/bin/env python3
"""HRBM v6: 2-3 Layer MLP on frozen reservoir. Target: beat 5-gram."""
import numpy as np, time
from collections import Counter

D_RES=512; EMBED_DIM=50; LEAK=0.3; SPECTRAL=0.9; SPARSITY=0.1; SEED=42
MAX_VOCAB=1000; MAX_TOKENS=50000; BATCH=256; LR=0.003
GLOVE="/home/eggchad/eakject/research/Deep_Man/topological-latent-engine/data/glove.6B.50d.txt"
WIKI="/home/eggchad/eakject/research/Deep_Man/topological-latent-engine/data/wiki_train.txt"
np.random.seed(SEED)

def load_wiki(p,mt):
    ss=[]
    with open(p) as f:
        for l in f:
            l=l.strip()
            if not l or l.startswith('='): continue
            ws=[w for w in l.lower().split() if w.isalpha() and w!='unk']
            if len(ws)>=3: ss.append(ws)
    flat=[]
    for s in ss:
        if len(flat)+len(s)>mt: break
        flat.extend(s)
    return [flat[i:i+10] for i in range(0,len(flat)-10,10)]

def load_glove(p,vs):
    e={}
    with open(p) as f:
        for l in f:
            ps=l.strip().split()
            if ps[0] in vs: e[ps[0]]=np.array([float(x) for x in ps[1:51]],dtype=np.float32)
    rng=np.random.RandomState(99)
    for w in vs:
        if w not in e: e[w]=rng.randn(EMBED_DIM).astype(np.float32)*0.1
    return e

class Reservoir:
    def __init__(self):
        rng=np.random.RandomState(SEED)
        self.Wr=rng.randn(D_RES,D_RES).astype(np.float32)
        self.Wr[rng.rand(D_RES,D_RES)>SPARSITY]=0
        sc=np.sqrt(D_RES*SPARSITY)
        self.Wr*=SPECTRAL/max(sc,0.01)
        self.Wi=rng.randn(D_RES,EMBED_DIM).astype(np.float32)/np.sqrt(EMBED_DIM)
        self.s=np.zeros(D_RES,dtype=np.float32)
    def step(self,x):
        pre=self.Wr@self.s+self.Wi@x
        self.s=(1-LEAK)*self.s+LEAK*np.tanh(pre)
        return self.s.copy()
    def reset(self): self.s[:]=0

class DeepMLP:
    """Multi-layer MLP on reservoir features."""
    def __init__(self, layers, V):
        self.W=[]; self.b=[]
        for i in range(len(layers)-1):
            fan_in,fan_out=layers[i],layers[i+1]
            self.W.append(np.random.randn(fan_out,fan_in).astype(np.float32)*np.sqrt(2.0/fan_in))
            self.b.append(np.zeros(fan_out,dtype=np.float32))
        self.n_layers=len(self.W)

    def forward(self, X):
        self.activations=[X]
        h=X
        for i in range(self.n_layers-1):
            h=h@self.W[i].T+self.b[i]
            h=np.maximum(h,0)  # ReLU
            self.activations.append(h)
        # Last layer: no activation (logits)
        logits=h@self.W[-1].T+self.b[-1]
        # Softmax
        logits-=logits.max(axis=-1,keepdims=True)
        e=np.exp(logits)
        probs=e/e.sum(axis=-1,keepdims=True)
        self.activations.append(probs)
        return probs

    def backward(self, targets, lr):
        N=len(targets)
        probs=self.activations[-1]
        d=probs.copy()
        d[np.arange(N),targets]-=1.0
        d/=N
        # Backprop through layers in reverse
        for i in range(self.n_layers-1,-1,-1):
            inp=self.activations[i]
            dW=d.T@inp
            db=d.sum(axis=0)
            self.W[i]-=lr*dW
            self.b[i]-=lr*db
            if i>0:
                d=d@self.W[i]
                d*=(self.activations[i]>0).astype(np.float32)  # ReLU grad

    def train_epoch(self, X, Y, lr):
        N=len(Y)
        idx=np.random.permutation(N)
        loss_sum=0; nb=0
        for s in range(0,N,BATCH):
            e=min(s+BATCH,N)
            bi=idx[s:e]
            probs=self.forward(X[bi])
            loss=-np.log(probs[np.arange(len(bi)),Y[bi]]+1e-10).mean()
            loss_sum+=loss; nb+=1
            self.backward(Y[bi],lr)
        return loss_sum/nb

    def predict(self, x):
        h=x.reshape(1,-1)
        for i in range(self.n_layers-1):
            h=h@self.W[i].T+self.b[i]
            h=np.maximum(h,0)
        logits=h@self.W[-1].T+self.b[-1]
        logits-=logits.max()
        e=np.exp(logits)
        return (e/e.sum()).flatten()

def evaluate(model, test_s, w2i, glove, vocab):
    res=Reservoir(); c=0; t=0; lp=0.0
    for sent in test_s:
        res.reset()
        for i in range(len(sent)-1):
            if sent[i] not in w2i or sent[i+1] not in w2i: continue
            s=res.step(glove[sent[i]])
            p=model.predict(s)
            tid=w2i[sent[i+1]]
            if p.argmax()==tid: c+=1
            lp+=np.log(max(p[tid],1e-10)); t+=1
    return np.exp(-lp/max(t,1)), c/max(t,1)*100, t

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  HRBM v6: Deep MLP (2-3 layers) on Frozen Reservoir         ║")
    print("║  Target: BEAT 5-gram (ppl ~230) with <5 min CPU training     ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    sents=load_wiki(WIKI,MAX_TOKENS)
    nt=int(len(sents)*0.8)
    train_s,test_s=sents[:nt],sents[nt:]
    freq=Counter(); 
    for s in sents:
        for w in s: freq[w]+=1
    vocab=[w for w,_ in freq.most_common(MAX_VOCAB)]
    w2i={w:i for i,w in enumerate(vocab)}; V=len(vocab)
    glove=load_glove(GLOVE,set(vocab))
    
    print(f"  Data: V={V}, train={sum(len(s) for s in train_s)} tok, test={sum(len(s) for s in test_s)} tok")

    # Collect reservoir states
    res=Reservoir(); states=[]; targets=[]
    for sent in train_s:
        res.reset()
        for i in range(len(sent)-1):
            if sent[i] in w2i and sent[i+1] in w2i:
                states.append(res.step(glove[sent[i]]))
                targets.append(w2i[sent[i+1]])
    X=np.array(states,dtype=np.float32)
    Y=np.array(targets,dtype=np.int32)
    N=len(X)
    print(f"  Reservoir: {N} samples, N/D={N/D_RES:.0f}\n")

    # 5-gram baseline
    from collections import Counter as Ctr
    ngram_counts={}; uni=Ctr()
    for sent in train_s:
        for w in sent: uni[w]+=1
        for n in range(1,6):
            for i in range(len(sent)-n):
                ctx=tuple(sent[i:i+n])
                nxt=sent[i+n] if i+n<len(sent) else None
                if nxt: ngram_counts.setdefault(ctx,Ctr())[nxt]+=1
    total_uni=sum(uni.values())
    
    # Eval 5-gram
    ng_lp=0.0; ng_t=0
    for sent in test_s:
        for i in range(len(sent)-1):
            if sent[i] in w2i and sent[i+1] in w2i:
                for n in range(min(i+1,5),0,-1):
                    ctx=tuple(sent[max(0,i+1-n):i+1])
                    if ctx in ngram_counts and sent[i+1] in ngram_counts[ctx]:
                        p=ngram_counts[ctx][sent[i+1]]/sum(ngram_counts[ctx].values())
                        break
                else:
                    p=uni.get(sent[i+1],1)/(total_uni+1)
                ng_lp+=np.log(max(p,1e-10)); ng_t+=1
    ngram_ppl=np.exp(-ng_lp/max(ng_t,1))
    print(f"  5-gram baseline: ppl={ngram_ppl:.1f}\n")

    # Test different depths
    configs = [
        ("1 layer [512]", [D_RES, 512, V], 10),
        ("2 layers [512,256]", [D_RES, 512, 256, V], 15),
        ("3 layers [512,256,128]", [D_RES, 512, 256, 128, V], 20),
    ]

    print(f"  {'Config':<30} {'Epochs':>6} {'Time':>8} {'Test PPL':>10} {'Test Acc':>10} {'vs 5gram':>10}")
    print(f"  {'-'*76}")

    for name, layers, epochs in configs:
        model = DeepMLP(layers, V)
        t0=time.time()
        for ep in range(epochs):
            model.train_epoch(X, Y, LR)
        train_time=time.time()-t0
        
        ppl, acc, _ = evaluate(model, test_s, w2i, glove, vocab)
        ratio = ppl / ngram_ppl
        marker = "🎉 BEATS!" if ppl < ngram_ppl else f"{ratio:.2f}×"
        print(f"  {name:<30} {epochs:>6} {train_time:>7.0f}s {ppl:>10.1f} {acc:>9.1f}% {marker:>10}")

    # Best model generation
    print(f"\n━━━ Generation (best model: 3 layers, 20 epochs) ━━━")
    best = DeepMLP([D_RES, 512, 256, 128, V], V)
    for _ in range(20):
        best.train_epoch(X, Y, LR)
    
    prompts=["the president","in the","it was","they were","she said"]
    for prompt in prompts:
        words=prompt.split()
        rg=Reservoir()
        for w in words:
            if w in glove: rg.step(glove[w])
        gen=list(words)
        for _ in range(10):
            p=best.predict(rg.state)
            for prev in gen[-3:]:
                if prev in w2i: p[w2i[prev]]*=0.01
            p/=p.sum()
            nid=p.argmax()
            nw=vocab[nid]; gen.append(nw)
            if nw in glove: rg.step(glove[nw])
        print(f"  \"{prompt}\" → \"{' '.join(gen)}\"")

    print(f"\n  5-gram: ppl={ngram_ppl:.1f}")
    print(f"  All training on CPU, frozen reservoir (no GPU needed)")

if __name__=="__main__":
    main()
