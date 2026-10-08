"""Run tests/cuda/glm5_router_test on llama.cpp's real router logits for layers 3 and 4 and compare with llama.cpp's
own chosen weights (ffn_moe_weights_scaled) and probabilities (the chosen experts are found as the probs entries equal
to llama.cpp's gathered weights).  Usage: python check_router.py <full_dump_dir> <glm5_router_test.exe>
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

from gguf_load import Model


def rd(path):
    with open(path, "rb") as f:
        ne = struct.unpack("<4i", f.read(16))
        a = np.frombuffer(f.read(), dtype=np.float32)
    shape = [n for n in ne[::-1] if n > 1] or [1]
    return a.reshape([ne[3], ne[2], ne[1], ne[0]]).squeeze()


def main(full, exe):
    m = Model()
    k = int(m.field("glm5-next.expert_used_count")[0])
    ws = float(m.field("glm5-next.expert_weights_scale")[0])
    print(f"k={k} expert_weights_scale={ws}")
    bad = 0
    for layer in (3, 4):
        logits = os.path.join(full, f"ffn_moe_logits-{layer}__0.bin")
        bias = m.get(f"blk.{layer}.exp_probs_b.bias").astype(np.float32)
        tmp = tempfile.mkdtemp()
        bpath, opath = os.path.join(tmp, "bias.f32"), os.path.join(tmp, "out.bin")
        bias.tofile(bpath)
        subprocess.run([exe, logits, bpath, str(k), str(ws), opath], check=True, capture_output=True)
        with open(opath, "rb") as f:
            T, kk = struct.unpack("<2i", f.read(8))
            ids = np.frombuffer(f.read(T * kk * 4), dtype=np.int32).reshape(T, kk)
            w = np.frombuffer(f.read(T * kk * 4), dtype=np.float32).reshape(T, kk)
        probs = rd(os.path.join(full, f"ffn_moe_probs-{layer}__0.bin")).reshape(T, -1)
        ref_w = rd(os.path.join(full, f"ffn_moe_weights_scaled-{layer}__0.bin")).reshape(T, kk)
        ref_g = rd(os.path.join(full, f"ffn_moe_weights-{layer}__0.bin")).reshape(T, kk)
        for t in range(T):
            ours = set(ids[t].tolist())
            theirs = set(int(np.argmin(np.abs(probs[t] - g))) for g in ref_g[t])   # which expert each weight came from
            same = ours == theirs
            werr = float(np.max(np.abs(np.sort(w[t]) - np.sort(ref_w[t]))))
            bad += (not same) or werr > 1e-5
            print(f"layer {layer} token {t}: experts {'same' if same else 'DIFFERENT'} {sorted(ours)}  max|w diff| {werr:.2e}")
    print("ALL OK" if bad == 0 else f"FAILED {bad}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
