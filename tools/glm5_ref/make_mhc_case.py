"""Test cases for tests/cuda/glm5_mhc_test.cu (Strata's fused_mhc_*), from real GLM-5.3-Flash data, layer 0.

case A  attention-side read, no fold: R = hc_init (the 5 prompt tokens' embeddings x4), hc_attn_fn/scale/base,
        attn_norm.  Expected y equals llama.cpp's attn_norm-0 (checked here against the dump as well).
case B  FFN-side read WITH the fold: R = hc_init, out_prev = a fixed-seed random block output, post_prev/comb_prev =
        case A's; R' = out*post + comb.R; then hc_ffn_fn/scale/base and ffn_norm.  Also the plain write and exit mean.
Expected values come from mhc_ref.py (float64), which matches the llama.cpp dump.

File (little endian): int32 T, n, iters; float32 rms_eps, hc_eps; R[T*4*n]; then per case:
  int32 apply; fn (int32 nbytes + bytes); scale[3]; base[24]; w_norm[n]; out_prev[T*n]; post_prev[T*4]; comb_prev[T*16]
  expected mixes[T*24] pre[T*4] post[T*4] comb[T*16] y[T*n] R'[T*4*n] exit_mean_of_R'[T*n]
"""
import glob
import struct
import sys

import numpy as np

import dumpparse
import mhc_ref as R
from gguf_load import Model

f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32).tobytes()


def rms_w(x, w, eps):
    return x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps) * w


def one(m, x, which, layer, apply, out_prev, post_prev, comb_prev, hc_eps, iters):
    fn_t = m.raw(f"blk.{layer}.hc_{which}_fn.weight")
    fn = m.get(f"blk.{layer}.hc_{which}_fn.weight").astype(np.float64)
    sc = m.get(f"blk.{layer}.hc_{which}_scale.weight").astype(np.float64)
    bs = m.get(f"blk.{layer}.hc_{which}_base.weight").astype(np.float64)
    wn = m.get(f"blk.{layer}.{'attn' if which == 'attn' else 'ffn'}_norm.weight").astype(np.float64)
    Rp = R.hc_post(out_prev, x, post_prev, comb_prev) if apply else x
    inp, mixes, pre, post, comb = R.hc_pre(Rp, fn, sc, bs, hc_eps, iters)
    y = rms_w(inp, wn, R.RMS_EPS)
    T = x.shape[0]
    blob = struct.pack("<i", int(apply)) + struct.pack("<i", fn_t.data.nbytes) + np.ascontiguousarray(fn_t.data).tobytes()
    blob += f32(sc) + f32(bs) + f32(wn) + f32(out_prev) + f32(post_prev) + f32(comb_prev.reshape(T, -1))
    blob += f32(mixes) + f32(pre) + f32(post) + f32(comb.reshape(T, -1)) + f32(y) + f32(Rp) + f32(Rp.mean(axis=1))
    return blob, y, post, comb


def main(out_path):
    m = Model()
    hc_eps = float(m.field("glm5-next.hyper_connection.epsilon")[0])
    iters = int(m.field("glm5-next.hyper_connection.sinkhorn_iterations")[0])
    emb = m.rows("token_embd.weight", R.TOKENS).astype(np.float64)
    T, n = emb.shape
    x = np.repeat(emb[:, None, :], R.HC, axis=1)
    zeros_o, zeros_p, zeros_c = np.zeros((T, n)), np.zeros((T, 4)), np.zeros((T, 4, 4))
    a, yA, postA, combA = one(m, x, "attn", 0, False, zeros_o, zeros_p, zeros_c, hc_eps, iters)

    dump = sorted(glob.glob(r"D:\AI\LLM\llamacpp-glm5\runs\dump_iq4xs_2026-??-??_??????.txt"))[-1]
    d = dumpparse.parse(dump)
    w, s, ds = dumpparse.compare(d["attn_norm-0"][0], yA)
    print(f"case A y vs llama.cpp attn_norm-0: max|diff| {w:.2e}  sum {s:.6f} vs {ds:.6f} (dump prints 4 decimals)")

    out_prev = np.random.default_rng(1).standard_normal((T, n)) * 0.05
    b, _, _, _ = one(m, x, "ffn", 0, True, out_prev, postA, combA, hc_eps, iters)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<iiiff", T, n, iters, R.RMS_EPS, hc_eps))
        f.write(f32(x)); f.write(a); f.write(b)
    print(f"wrote {out_path}: T={T} n={n}")


if __name__ == "__main__":
    main(sys.argv[1])
