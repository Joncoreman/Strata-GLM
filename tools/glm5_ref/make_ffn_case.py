"""Cases for tests/cuda/glm5_ffn_test.cu (Strata's shared_expert_multi, gate-less and clamped) from llama.cpp's whole
tensors (STRATA_DUMP_DIR): the dense FFN of layer 0 (x = ffn_norm-0, expected ffn_out-0, n_ff 12288) and the shared
expert of layer 3 (x = ffn_norm-3, expected ffn_shexp-3, n_ff 2048).  Weights: the real Q8_0 bytes.
File: int32 n_cases; per case: int32 T, n_embd, n_ff, type(8); float limit; then gate, up, down (int32 nbytes + bytes),
x[T*n_embd], expected[T*n_embd].
Usage: python make_ffn_case.py <full_dump_dir> <out.bin>
"""
import os
import struct
import sys

import numpy as np

from check_router import rd
from gguf_load import Model


def main(full, out):
    m = Model()
    limit = float(np.asarray(m.field("glm5-next.swiglu_clamp_exp")).ravel()[0]) if "glm5-next.swiglu_clamp_exp" in m.meta else 10.0
    print("swiglu limit", limit)
    cases = [("blk.0.ffn_{}.weight", "ffn_norm-0", "ffn_out-0"),
             ("blk.3.ffn_{}_shexp.weight", "ffn_norm-3", "ffn_shexp-3")]
    with open(out, "wb") as f:
        f.write(struct.pack("<i", len(cases)))
        for wname, xin, yout in cases:
            x = rd(os.path.join(full, xin + "__0.bin")).astype(np.float32)
            y = rd(os.path.join(full, yout + "__0.bin")).astype(np.float32)
            g, u, d = (m.raw(wname.format(k)) for k in ("gate", "up", "down"))
            T, n = x.shape
            n_ff = int(g.shape[1])
            f.write(struct.pack("<iiiif", T, n, n_ff, 8, limit))
            for t in (g, u, d):
                b = np.ascontiguousarray(t.data).tobytes()
                f.write(struct.pack("<i", len(b))); f.write(b)
            f.write(x.tobytes()); f.write(y.tobytes())
            print(f"{wname}: T={T} n_embd={n} n_ff={n_ff} types {g.tensor_type.name}/{u.tensor_type.name}/{d.tensor_type.name}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
