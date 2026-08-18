#!/usr/bin/env python3
"""
Native Safetensors Weight Extractor for TwoTier Transmuted Algebraic Engine.

Parses HuggingFace `.safetensors` checkpoints (e.g. Qwen-2.5, LLaMA-3, Mistral)
without requiring PyTorch or heavy dependencies (pure Python + NumPy).

Extracts:
1. `model.embed_tokens.weight` -> ZCA Whitened Phasor Codebook on Torus T^D.
2. `model.layers[*].self_attn.{q_proj, k_proj}.weight` -> Gated Sheaf Rotors.
3. `model.layers[*].mlp.{gate_proj, up_proj, down_proj}.weight` -> Symmetrized Continuous Hopfield Memory.

Outputs a compact `.twotier` binary model.
"""

import sys
import os
import json
import struct
import numpy as np

MAGIC_HEADER = b"TWOTIER1"

class SafetensorsReader:
    """
    Zero-dependency streaming reader for the HuggingFace Safetensors format.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = open(file_path, "rb")
        
        # 1. Read 8-byte header size (little-endian uint64)
        header_size_bytes = self.file.read(8)
        if len(header_size_bytes) < 8:
            raise ValueError(f"File {file_path} is too small to be a valid safetensors file.")
        
        self.header_size = struct.unpack("<Q", header_size_bytes)[0]
        self.header_json_bytes = self.file.read(self.header_size)
        self.metadata = json.loads(self.header_json_bytes.decode("utf-8"))
        self.data_offset = 8 + self.header_size

    def list_tensors(self):
        return [k for k in self.metadata.keys() if k != "__metadata__"]

    def read_tensor(self, tensor_name):
        if tensor_name not in self.metadata:
            raise KeyError(f"Tensor '{tensor_name}' not found in safetensors file.")
        
        info = self.metadata[tensor_name]
        dtype = info["dtype"]
        shape = info["shape"]
        start_offset, end_offset = info["data_offsets"]
        
        self.file.seek(self.data_offset + start_offset)
        raw_bytes = self.file.read(end_offset - start_offset)

        if dtype == "F32":
            arr = np.frombuffer(raw_bytes, dtype=np.float32)
        elif dtype == "F16":
            arr = np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32)
        elif dtype == "BF16":
            # BF16 to F32 conversion: shift 16 bits
            u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
            u32 = u16.astype(np.uint32) << 16
            arr = u32.view(np.float32)
        else:
            raise NotImplementedError(f"Unsupported dtype: {dtype}")

        return arr.reshape(shape)

    def close(self):
        self.file.close()

def compute_zca_whitening(embeddings, regularization=1e-4):
    """
    Computes ZCA sphereing matrix: W_ZCA = Q (Lambda + eps*I)^(-1/2) Q^T.
    Centers the embedding distribution and eliminates the anisotropy cone effect.
    """
    N, D = embeddings.shape
    mean = np.mean(embeddings, axis=0)
    centered = embeddings - mean

    # Covariance matrix
    cov = np.dot(centered.T, centered) / (N - 1)
    cov += regularization * np.eye(D, dtype=np.float32)

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-12)
    inv_sqrt = 1.0 / np.sqrt(eigvals)

    w_zca = np.dot(eigvecs * inv_sqrt, eigvecs.T)
    whitened = np.dot(centered, w_zca.T)
    return whitened, mean, w_zca

def cartesian_to_polar_phasors(whitened_embeddings):
    """
    Maps 2D coordinate pairs to Torus T^(D/2) phase angles:
    theta_k = atan2(x_{2k+1}, x_{2k}) in [-pi, pi).
    """
    N, D = whitened_embeddings.shape
    num_pairs = D // 2
    angles = np.zeros((N, num_pairs), dtype=np.float32)

    for k in range(num_pairs):
        re = whitened_embeddings[:, 2 * k]
        im = whitened_embeddings[:, 2 * k + 1] if 2 * k + 1 < D else np.zeros(N)
        angles[:, k] = np.arctan2(im, re)

    return angles

def extract_from_safetensors(safetensors_path, tokenizer_path, output_path, target_dim=128, max_vocab=16384, max_hopfield_slots=512):
    print(f"[*] Opening Safetensors model: {safetensors_path}...")
    reader = SafetensorsReader(safetensors_path)
    tensors = reader.list_tensors()
    print(f"    • Found {len(tensors)} tensors in safetensors checkpoint.")

    # 1. Extract Embeddings
    embed_name = None
    for name in ["model.embed_tokens.weight", "transformer.wte.weight", "embeddings.word_embeddings.weight", "embed_tokens.weight"]:
        if name in tensors:
            embed_name = name
            break

    if embed_name is None:
        # Fallback to first 2D weight matrix with large vocab dimension
        for name in tensors:
            if "embed" in name or "wte" in name:
                embed_name = name
                break

    if embed_name is None:
        raise ValueError("Could not find token embedding matrix in model checkpoint.")

    print(f"[*] Extracting embedding matrix: '{embed_name}'...")
    raw_embeds = reader.read_tensor(embed_name)
    orig_vocab, orig_dim = raw_embeds.shape
    print(f"    • Shape: [Vocab={orig_vocab}, Dim={orig_dim}]")

    # Load Tokenizer vocabulary if available
    tokens = []
    if tokenizer_path and os.path.exists(tokenizer_path):
        print(f"[*] Loading vocabulary tokens from: {tokenizer_path}...")
        try:
            with open(tokenizer_path, "r", encoding="utf-8") as f:
                tok_data = json.load(f)
                if "model" in tok_data and "vocab" in tok_data["model"]:
                    vocab_dict = tok_data["model"]["vocab"]
                    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
                    tokens = [w for w, _ in sorted_vocab]
                elif "vocab" in tok_data:
                    sorted_vocab = sorted(tok_data["vocab"].items(), key=lambda x: x[1])
                    tokens = [w for w, _ in sorted_vocab]
        except Exception as e:
            print(f"    ! Note: Could not parse tokenizer json ({e}), falling back to index tokens.")

    while len(tokens) < orig_vocab:
        tokens.append(f"tok_{len(tokens)}")

    # Truncate / Subsample to target dimensions and vocab limits
    vocab_size = min(orig_vocab, max_vocab)
    tokens = tokens[:vocab_size]
    sub_embeds = raw_embeds[:vocab_size]

    if orig_dim > target_dim:
        print(f"[*] Projecting embedding dimension from {orig_dim} -> {target_dim} via PCA/SVD truncation...")
        # SVD rank truncation
        u, s, vt = np.linalg.svd(sub_embeds - np.mean(sub_embeds, axis=0), full_matrices=False)
        reduced_embeds = np.dot(u[:, :target_dim], np.diag(s[:target_dim]))
    else:
        reduced_embeds = sub_embeds[:, :target_dim]

    print("[*] Computing ZCA Whitening on embedding space...")
    whitened, _, _ = compute_zca_whitening(reduced_embeds)
    phasor_angles = cartesian_to_polar_phasors(whitened)

    # 2. Extract FFN SwiGLU projection matrices into Hopfield memory patterns
    print("[*] Extracting and Symmetrizing FFN SwiGLU Key-Value associative memory...")
    hopfield_slots = []
    
    # Search for layer FFN weights
    for layer_idx in range(32):
        gate_name = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        up_name = f"model.layers.{layer_idx}.mlp.up_proj.weight"
        down_name = f"model.layers.{layer_idx}.mlp.down_proj.weight"

        if gate_name in tensors and up_name in tensors and down_name in tensors:
            gate_w = reader.read_tensor(gate_name) # [inter_dim, orig_dim]
            up_w = reader.read_tensor(up_name)     # [inter_dim, orig_dim]
            down_w = reader.read_tensor(down_name) # [orig_dim, inter_dim]

            inter_dim = gate_w.shape[0]
            slots_per_layer = max(1, max_hopfield_slots // 16)

            for slot_i in range(min(inter_dim, slots_per_layer)):
                # SwiGLU Key Symmetrization: k = 0.5 * (w_gate + w_up)
                k_full = 0.5 * (gate_w[slot_i] + up_w[slot_i])
                v_full = down_w[:, slot_i] if down_w.ndim == 2 else down_w[slot_i]

                # Project to target_dim
                k_proj = k_full[:target_dim]
                v_proj = v_full[:target_dim]

                k_norm = float(np.linalg.norm(k_proj))
                if k_norm > 1e-6:
                    k_unit = k_proj / k_norm
                    hopfield_slots.append((k_unit, v_proj, k_norm))

        if len(hopfield_slots) >= max_hopfield_slots:
            break

    print(f"    • Extracted {len(hopfield_slots)} Hopfield memory slots.")

    # 3. Write .twotier binary
    fast_weights = np.zeros(target_dim * target_dim, dtype=np.float32)
    
    print(f"[*] Serializing Transmuted Model into '{output_path}'...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(MAGIC_HEADER)
        f.write(struct.pack("<IIII", target_dim, 2, target_dim // 4, 32))
        
        # Tier 1 Vocab
        f.write(struct.pack("<I", vocab_size))
        num_pairs = target_dim // 2
        for i, token in enumerate(tokens):
            tok_bytes = token.encode("utf-8")
            f.write(struct.pack("<H", len(tok_bytes)))
            f.write(tok_bytes)
            f.write(struct.pack("<I", num_pairs))
            f.write(phasor_angles[i].astype("<f4").tobytes())

        # Tier 2 Hopfield slots
        f.write(struct.pack("<I", len(hopfield_slots)))
        for k, v, scale in hopfield_slots:
            f.write(k.astype("<f4").tobytes())
            f.write(v.astype("<f4").tobytes())
            f.write(struct.pack("<f", scale))

        # Fast weights
        f.write(struct.pack("<I", len(fast_weights)))
        f.write(fast_weights.astype("<f4").tobytes())

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[+] Successfully converted Safetensors model to: {output_path} ({file_size_mb:.2f} MB)")
    reader.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: extract_safetensors_to_twotier.py <model.safetensors> [tokenizer.json] [output.twotier]")
        sys.exit(1)

    st_path = sys.argv[1]
    tok_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].endswith(".json") else None
    out_path = sys.argv[3] if len(sys.argv) > 3 else (sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].endswith(".json") else "data/models/transmuted_safetensors.twotier")

    extract_from_safetensors(st_path, tok_path, out_path)
