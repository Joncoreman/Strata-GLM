"""Case for tests/cuda/glm5_kda_test.cu: Strata's gdn_step_norm_multi (KDA branch) and kda_gate_beta_multi on llama.cpp's
own layer-0 tensors (whole-tensor dump): q/k/v after conv+SiLU (kda_*_conv-0), L2-normed here as llama.cpp does
(x * rsqrt(sum x^2 + 1e-6), models.h build_gdn_l2_norm), the gate kda_g1-0, beta kda_beta-0, z = kda_g2-0,
gamma = blk.0.ssm_norm; zero initial state (the first prompt).  Expected: kda_scan_out-0 (before the norm) and
y = kda_normed-0 * sigmoid(kda_g2-0).  For the gate kernel: f = ssm_f_b(ssm_f_a(attn_norm-0)) in float64 here, and
the expected gate from the same f (the dump's kda_g1-0 is printed for comparison: llama.cpp rounds f's input to q8).
File: int32 T, H, S; float rms_eps, lower_bound; h[T*3*H*S]; gate[T*H*S]; beta[T*H]; z[T*H*S]; gamma[S];
       want_o[T*H*S]; want_y[T*H*S]; f[T*H*S]; dt_b[H*S]; ssm_a[H]; beta_logit[T*H]; want_gate_from_f[T*H*S]
Usage: python make_kda_case.py <full_dump_dir> <out.bin>
"""
import os
import struct
import sys

import numpy as np

from check_router import rd
from gguf_load import Model


def main(full, out):
    m = Model()
    L = lambda n: rd(os.path.join(full, n + "__0.bin")).astype(np.float64)
    q, k, v = L("kda_q_conv-0"), L("kda_k_conv-0"), L("kda_v_conv-0")       # [T, H, S]
    T, H, S = q.shape
    l2 = lambda x: x / np.sqrt((x ** 2).sum(axis=-1, keepdims=True) + 1e-6)
    h = np.concatenate([l2(q).reshape(T, -1), l2(k).reshape(T, -1), v.reshape(T, -1)], axis=1)
    gate, beta, z = L("kda_g1-0"), L("kda_beta-0").reshape(T, H), L("kda_g2-0")
    gamma = m.get("blk.0.ssm_norm.weight").astype(np.float64)
    want_o = L("kda_scan_out-0")
    want_y = L("kda_normed-0") * (1 / (1 + np.exp(-z)))
    lb = float(np.asarray(m.field("glm5-next.kda.gate_lower_bound")).ravel()[0]) if "glm5-next.kda.gate_lower_bound" in m.meta else -5.0
    rms_eps = float(m.field("glm5-next.attention.layer_norm_rms_epsilon")[0])

    x = L("attn_norm-0")                                                    # [T, 4096]
    fa = m.get("blk.0.ssm_f_a.weight").astype(np.float64)
    fb = m.get("blk.0.ssm_f_b.weight").astype(np.float64)
    f = (x @ fa.T) @ fb.T                                                   # [T, H*S]
    dt_b = m.get("blk.0.ssm_dt.bias").astype(np.float64).ravel()
    A = m.get("blk.0.ssm_a").astype(np.float64).ravel()
    wb = m.get("blk.0.ssm_beta.weight").astype(np.float64)
    beta_logit = x @ wb.T
    g_from_f = lb * (1 / (1 + np.exp(-(-((f.reshape(T, H, S) + dt_b.reshape(H, S)) * A[None, :, None])))))
    print(f"T={T} H={H} S={S} lower_bound={lb} rms_eps={rms_eps}")
    print(f"python gate from f vs llama.cpp kda_g1-0: max|diff| {np.abs(g_from_f - gate).max():.3e} (q8 activation rounding)")
    print(f"python beta vs llama.cpp kda_beta-0: max|diff| {np.abs(1/(1+np.exp(-beta_logit)) - beta).max():.3e}")
    f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32).tobytes()
    with open(out, "wb") as fo:
        fo.write(struct.pack("<iiiff", T, H, S, rms_eps, lb))
        for a in (h, gate, beta, z, gamma, want_o, want_y, f, dt_b, A, beta_logit, g_from_f):
            fo.write(f32(a))
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
