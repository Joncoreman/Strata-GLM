"""Bundle one layer's raw GGUF tensors (bytes as stored: Q8_0 blocks, F32) plus the layer's input and llama.cpp's
outputs, for tests/cuda/glm5_layer_test.cu.
File: int32 n_tensors; per tensor: int32 name_len, name, int32 ggml_type, int32 n_dims, int64 ne[n_dims] (ggml order),
int64 n_bytes, bytes.  The input / expected tensors are written as F32 "tensors" named in:* / want:*.
Usage: python make_layer_bundle.py <layer> <full_dump_dir> <out.bin>
"""
import os
import struct
import sys

import numpy as np

from check_router import rd
from gguf_load import Model


def put(f, name, gtype, ne, data):
    b = name.encode()
    f.write(struct.pack("<i", len(b))); f.write(b)
    f.write(struct.pack("<ii", gtype, len(ne)))
    f.write(struct.pack("<" + "q" * len(ne), *ne))
    f.write(struct.pack("<q", len(data))); f.write(data)


def main(layer, full, out):
    layer = int(layer)
    m = Model()
    names = sorted(n for n in m.where if n.startswith(f"blk.{layer}."))
    def L(n):   # a ubatch-1 dump has one file per token (__0 .. __T-1): stack them on a new token axis
        if os.path.exists(os.path.join(full, n + "__1.bin")):
            parts, k = [], 0
            while os.path.exists(os.path.join(full, f"{n}__{k}.bin")):
                parts.append(rd(os.path.join(full, f"{n}__{k}.bin"))); k += 1
            return np.stack(parts).astype(np.float32)
        return rd(os.path.join(full, n + "__0.bin")).astype(np.float32)
    extra = []
    if layer == 0:
        emb = m.rows("token_embd.weight", [785, 6722, 315, 9621, 374]).astype(np.float32)
        extra.append(("in:R", np.repeat(emb[:, None, :], 4, axis=1)))        # hc_init [T, 4, 4096]
    else:
        extra.append(("in:R", L(f"l_out-{layer - 1}")))
    for n in (f"attn_norm-{layer}", f"kda_v_conv-{layer}", f"kda_g1-{layer}", f"kda_beta-{layer}", f"kda_g2-{layer}",
              f"kda_normed-{layer}", f"kda_out-{layer}", f"attn_out-{layer}", f"ffn_norm-{layer}", f"ffn_out-{layer}",
              f"l_out-{layer}"):
        p = os.path.join(full, n + "__0.bin")
        if os.path.exists(p):
            extra.append(("want:" + n.rsplit("-", 1)[0], L(n)))
    with open(out, "wb") as f:
        f.write(struct.pack("<i", len(names) + len(extra)))
        for n in names:
            t = m.raw(n)
            put(f, n, int(t.tensor_type), [int(s) for s in t.shape], np.ascontiguousarray(t.data).tobytes())
        for n, a in extra:
            put(f, n, 0, list(a.shape[::-1]), np.ascontiguousarray(a, dtype=np.float32).tobytes())
    print(f"wrote {out}: {len(names)} weights + {len(extra)} inputs/expected ({', '.join(n for n, _ in extra)})")


if __name__ == "__main__":
    main(*sys.argv[1:])
