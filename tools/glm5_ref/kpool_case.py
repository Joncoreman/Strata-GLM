"""k-pool indexer check for tests/cuda/glm5_kpool_test.cu on llama.cpp's own layer-3 tensors.
make : raw key = indexer.attn_k @ attn_norm-3 (float64 here), gate = indexer_gate-3, the LN weights and ape; the q,
       weights and pooled keys of llama.cpp for the score kernel.  Python checks first: LayerNorm of the raw key vs
       indexer_k-3, and pooling of llama.cpp's own keys/gates vs indexer_pool_k_new-3.
check: compares the kernel's keys / pooled key / scores with the Python reference and llama.cpp.
Usage: python kpool_case.py make <full_dir> <case.bin> | python kpool_case.py check <full_dir> <case.bin> <out.bin>
"""
import os
import struct
import sys

import numpy as np

from check_router import rd
from gguf_load import Model

f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32).tobytes()


def ln(x, w, b, eps):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def pool(keys, gates, ape):          # keys/gates [4, 128], ape [4, 128]
    lg = gates + ape
    p = np.exp(lg - lg.max(0, keepdims=True)); p /= p.sum(0, keepdims=True)
    return (p * keys).sum(0)


def main(mode, full, path, outp=None):
    L = lambda n: rd(os.path.join(full, n + "__0.bin")).astype(np.float64)
    m = Model()
    eps = float(m.field("glm5-next.attention.layer_norm_epsilon")[0])
    kw = m.get("blk.3.indexer.k_norm.weight").astype(np.float64)
    kb = m.get("blk.3.indexer.k_norm.bias").astype(np.float64)
    ape = m.get("blk.3.indexer_compressor_ape.weight").astype(np.float64).reshape(4, 128)
    wk = m.get("blk.3.indexer.attn_k.weight").astype(np.float64)
    x = L("attn_norm-3")
    raw = x @ wk.T                                     # [T, 128]
    gate = L("indexer_gate-3")
    T = raw.shape[0]
    key_ll = L("indexer_k-3")
    pool_ll = L("indexer_pool_k_new-3")[0]
    q = L("indexer_q-3"); w = L("indexer_weights-3"); pk = L("indexer_pool_k-3").reshape(-1, 128)
    score_ll = L("indexer_score-3")                     # [T, 64]
    if mode == "make":
        print(f"eps={eps}  LayerNorm(raw) vs llama.cpp indexer_k-3: max|diff| {np.abs(ln(raw, kw, kb, eps) - key_ll).max():.3e} (raw from float64 matmul; llama.cpp rounds x to q8)")
        print(f"pool(llama.cpp keys, gates) vs indexer_pool_k_new-3: max|diff| {np.abs(pool(key_ll[:4], gate[:4], ape) - pool_ll).max():.3e}")
        with open(path, "wb") as f:
            f.write(struct.pack("<iif", T, pk.shape[0], eps))
            for a in (raw, gate, kw, kb, ape, q, w, pk):
                f.write(f32(a))
        print("wrote", path)
        return
    with open(outp, "rb") as f:
        keys = np.frombuffer(f.read(T * 128 * 4), dtype=np.float32).reshape(T, 128).astype(np.float64)
        pooled0 = np.frombuffer(f.read(128 * 4), dtype=np.float32).astype(np.float64)
        sc = np.frombuffer(f.read(), dtype=np.float32).reshape(T, -1).astype(np.float64)
    ok = True
    e1 = np.abs(keys - ln(raw, kw, kb, eps)).max(); ok &= e1 < 1e-4
    print(f"kernel key vs LayerNorm reference: max|diff| {e1:.3e}")
    e2 = np.abs(pooled0 - pool(keys[:4], gate[:4], ape)).max(); ok &= e2 < 1e-5
    print(f"kernel pooled[0] vs pooling reference on its own keys: max|diff| {e2:.3e}")
    e3 = np.abs(pooled0 - pool_ll).max()
    print(f"kernel pooled[0] vs llama.cpp indexer_pool_k_new-3: max|diff| {e3:.3e} (includes the q8 rounding of the key projection)")
    for t in range(T):
        nb = (t + 1) // 4
        if nb == 0:
            print(f"token {t}: no completed pool (tail only)")
            continue
        d = abs(sc[t, 0] - score_ll[t, 0]); ok &= d <= 1e-4 * max(1.0, abs(score_ll[t, 0]))
        print(f"token {t}: score[pool 0] {sc[t,0]:.6f} vs llama.cpp {score_ll[t,0]:.6f}  diff {d:.2e}")
    print("ALL OK" if ok else "FAILED")


if __name__ == "__main__":
    main(*sys.argv[1:])
