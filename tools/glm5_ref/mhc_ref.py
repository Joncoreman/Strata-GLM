"""mHC (hyper-connection) reference for glm5-next, checked against the llama.cpp dump.

Formulas: llama.cpp b11461 src/models/glm5-next.cpp:445-538 and ggml-cuda/dsv4-hc.cu.
  flat   = rms_norm(concat of the 4 streams, 16384 values, eps = rms eps 1e-5)
  mixes  = hc_fn @ flat                       (24 values)
  pre    = sigmoid(mixes[0:4]  * scale[0] + base[0:4]) + hc_eps
  post   = 2 * sigmoid(mixes[4:8] * scale[1] + base[4:8])
  comb   = softmax over dst of (mixes[8:24] * scale[2] + base[8:24]), index = dst + 4*src, then + hc_eps,
           normalize columns (per dst, over src), then (iters-1) x [rows (per src, over dst), columns]
  input  = sum_h pre[h] * x_h                 (block input, 4096)
  new[d] = out * post[d] + sum_s comb[d, s] * res[s]
Run: python mhc_ref.py [dump.txt]  → prints the max difference vs the dump for layer 0.
"""
import glob
import sys

import numpy as np

import dumpparse
from gguf_load import Model

HC = 4
RMS_EPS = 1e-5        # glm5-next.attention.layer_norm_rms_epsilon (checked below)
TOKENS = [785, 6722, 315, 9621, 374]   # "The capital of France is" (dump stderr)


def rms_norm(x, eps):
    return x / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def coeffs(mixes, scale, base, hc_eps, iters):
    """mixes [nt, 24] -> pre [nt,4], post [nt,4], comb [nt, dst 4, src 4]"""
    pre = sigmoid(mixes[:, 0:4] * scale[0] + base[0:4]) + hc_eps
    post = 2.0 * sigmoid(mixes[:, 4:8] * scale[1] + base[4:8])
    c = mixes[:, 8:24] * scale[2] + base[8:24]                 # flat index = dst + 4*src
    c = c.reshape(-1, HC, HC).transpose(0, 2, 1)               # -> [nt, dst, src]
    c = np.exp(c - c.max(axis=1, keepdims=True))
    c = c / c.sum(axis=1, keepdims=True) + hc_eps               # softmax over dst, per src
    def cols(c):  # per dst: sum over src
        return c / (c.sum(axis=2, keepdims=True) + hc_eps)
    def rows(c):  # per src: sum over dst
        return c / (c.sum(axis=1, keepdims=True) + hc_eps)
    c = cols(c)
    for _ in range(1, iters):
        c = cols(rows(c))
    return pre, post, c


def q8_1(x):
    """ggml-cuda quantize_q8_1 per 32 values: d = amax/127, q = round(x/d) (round half away from zero)."""
    xb = x.reshape(-1, 32)
    amax = np.abs(xb).max(axis=1, keepdims=True)
    d = amax / 127.0
    q = np.where(amax > 0, np.sign(xb) * np.floor(np.abs(xb) / np.where(d > 0, d, 1) + 0.5), 0)
    return (q * d).reshape(x.shape)


def hc_pre(x, hc_fn, scale, base, hc_eps, iters, quantize=True):
    """x [nt, 4, 4096] -> (block input [nt, 4096], mixes, pre, post, comb).  quantize: the projection's activation
    goes through q8_1 as in llama.cpp's CUDA mmvq (Strata's fused_mhc does the same)."""
    nt = x.shape[0]
    flat = rms_norm(x.reshape(nt, -1), RMS_EPS)
    if quantize:
        flat = q8_1(flat.astype(np.float32).astype(np.float64))
    mixes = flat @ hc_fn.T
    pre, post, comb = coeffs(mixes, scale, base, hc_eps, iters)
    inp = np.einsum("th,the->te", pre, x)
    return inp, mixes, pre, post, comb


def hc_post(out, res, post, comb):
    """out [nt,4096], res [nt,4,4096] -> [nt,4,4096]"""
    return out[:, None, :] * post[:, :, None] + np.einsum("tds,tse->tde", comb, res)


def main():
    dump_path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(r"D:\AI\LLM\llamacpp-glm5\runs\dump_iq4xs_2026-??-??_??????.txt"))[-1]
    d = dumpparse.parse(dump_path)
    m = Model()
    eps_meta = float(m.field("glm5-next.attention.layer_norm_rms_epsilon")[0])
    hc_eps = float(m.field("glm5-next.hyper_connection.epsilon")[0])
    iters = int(m.field("glm5-next.hyper_connection.sinkhorn_iterations")[0])
    print(f"meta: rms_eps={eps_meta:g} hc_eps={hc_eps:g} sinkhorn_iters={iters}")
    assert abs(eps_meta - RMS_EPS) < 1e-9

    emb = m.rows("token_embd.weight", TOKENS)                  # [5, 4096]
    worst, s, ds = dumpparse.compare(d["inp_embd"][0], emb)
    print(f"inp_embd:      max|diff| {worst:.2e}  sum {s:.6f} vs {ds:.6f}")

    x = np.repeat(emb[:, None, :], HC, axis=1).astype(np.float64)   # hc_init
    fn = m.get("blk.0.hc_attn_fn.weight").astype(np.float64)       # [24, 16384]
    sc = m.get("blk.0.hc_attn_scale.weight").astype(np.float64)
    bs = m.get("blk.0.hc_attn_base.weight").astype(np.float64)
    inp, mixes, pre, post, comb = hc_pre(x, fn, sc, bs, hc_eps, iters)

    def show(label, entry, arr):
        w, s, ds = dumpparse.compare(entry, arr)
        print(f"{label:14s} max|diff| {w:.2e}  sum {s:.6f} vs {ds:.6f}  (dump prints 4 decimals)")

    show("hc_mixes-0", d["hc_mixes-0"][0], mixes)
    show("hc_pre-0", d["hc_pre-0"][0], pre)
    show("hc_post-0", d["hc_post-0"][0], post)
    show("hc_comb-0", d["hc_comb-0"][0], comb.transpose(0, 2, 1).reshape(-1, HC))   # dump rows: ne1=src, ne0=dst
    show("hc_attn_pre-0", d["hc_attn_pre-0"][0], inp)


if __name__ == "__main__":
    main()
