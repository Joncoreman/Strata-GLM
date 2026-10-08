"""Case for tests/cuda/glm5_mla_test.cu: Strata's mla_decode_attn_batch on llama.cpp's own layer-3 tensors.
q_abs = q_absorbed-3 [T, 64, 512]; the latent cache = kv_cmpr-3 [T, 512] stored as fp16 (llama.cpp's cache_k is f16);
query t sees cells 0..t (5 cells < 2051: the selection keeps everything, the causal mask decides).
The test writes the attention output O [T, 64, 512]; this script's check step applies wv_b (attn_v_b, per head
512 -> 256) and compares with llama.cpp's fattn_mla-3 [T, 64, 256].
Usage: python make_mla_case.py make <full_dir> <case.bin>  |  python make_mla_case.py check <full_dir> <out.bin>
"""
import os
import struct
import sys

import numpy as np

from check_router import rd
from gguf_load import Model


def main(mode, full, path):
    L = lambda n: rd(os.path.join(full, n + "__0.bin")).astype(np.float32)
    q = L("q_absorbed-3")                       # [T, 64, 512]
    T, H, D = q.shape
    if mode == "make":
        kv = L("kv_cmpr-3").reshape(T, D).astype(np.float16)
        with open(path, "wb") as f:
            f.write(struct.pack("<iiif", T, H, D, 1.0 / np.sqrt(256.0)))
            f.write(q.tobytes()); f.write(kv.tobytes())
        print(f"wrote {path}: T={T} H={H} D={D}")
        return
    with open(path, "rb") as f:
        O = np.frombuffer(f.read(), dtype=np.float32).reshape(T, H, D).astype(np.float64)
    m = Model()
    wvb = m.get("blk.3.attn_v_b.weight").astype(np.float64)          # [64, 256, 512] (gguf-py order)
    wvb = wvb.reshape(H, -1, D)
    ours = np.einsum("hod,thd->tho", wvb, O)                            # [T, 64, 256]
    want = L("fattn_mla-3").astype(np.float64)                          # dump ne {256, 5, 64} -> numpy [64, 5, 256]
    want = want.transpose(1, 0, 2)
    d = ours - want
    rel = np.sqrt((d ** 2).sum() / (want ** 2).sum())
    print(f"MLA attention x wv_b vs llama.cpp fattn_mla-3: max|diff| {np.abs(d).max():.3e}  max|ref| {np.abs(want).max():.3e}  rms rel {rel:.3e}  {'OK' if rel < 1e-3 else 'FAIL'}")
    # reference in float64 from the same inputs, for the kernel alone
    kv = L("kv_cmpr-3").reshape(T, D).astype(np.float16).astype(np.float64)
    qd = q.astype(np.float64)
    ref = np.zeros_like(O)
    for t in range(T):
        s = np.einsum("hd,cd->hc", qd[t], kv[: t + 1]) / 16.0
        p = np.exp(s - s.max(axis=1, keepdims=True)); p /= p.sum(axis=1, keepdims=True)
        ref[t] = p @ kv[: t + 1]
    r2 = np.sqrt(((O - ref) ** 2).sum() / (ref ** 2).sum())
    print(f"MLA attention vs float64 reference on the same inputs: rms rel {r2:.3e}  {'OK' if r2 < 1e-5 else 'FAIL'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
