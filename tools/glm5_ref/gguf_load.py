"""Load individual GLM-5.3-Flash tensors (dequantized to float32) from the split GGUF, by name.

Uses llama.cpp b11461's gguf-py (it knows glm5-next and every quant type). Read-only, memory-mapped.
numpy shape follows gguf-py: reversed ggml order, i.e. a ggml {ne0, ne1} weight comes back as [ne1, ne0].
"""
import glob
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# a llama.cpp checkout that knows glm5-next (b11461 or newer): LLAMA_CPP_DIR, else llamacpp-glm5 beside this repository
LLAMA_CPP = os.environ.get("LLAMA_CPP_DIR", os.path.join(os.path.dirname(_REPO), "llamacpp-glm5"))
sys.path.insert(0, os.path.join(LLAMA_CPP, "gguf-py"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                      # noqa: E402
from gguf import GGUFReader             # noqa: E402
from gguf.quants import dequantize      # noqa: E402

try:   # shard 1 of the GGUF: NATIVE, else strata-glm5.json's --native
    import _glm5_paths                  # noqa: E402
    DEFAULT = str(_glm5_paths.native_gguf())
except Exception:
    DEFAULT = os.environ.get("NATIVE", "")


class Model:
    def __init__(self, shard1=DEFAULT):
        stem = shard1.split("-00001-of-")[0]
        self.paths = sorted(glob.glob(stem + "-0000*-of-*.gguf"))
        self.readers = [GGUFReader(p) for p in self.paths]
        self.where = {}
        for r in self.readers:
            for t in r.tensors:
                self.where[t.name] = t
        self.meta = self.readers[0].fields

    def raw(self, name):
        t = self.where[name]
        return t

    def get(self, name):
        t = self.where[name]
        return np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32)

    def rows(self, name, idx):
        """Dequantize only some rows of a big 2-D tensor (e.g. token_embd)."""
        t = self.where[name]
        data = t.data
        return np.asarray(dequantize(data[list(idx)], t.tensor_type), dtype=np.float32)

    def field(self, key):
        f = self.meta[key]
        v = f.parts[f.data[0]] if len(f.data) == 1 else [f.parts[i] for i in f.data]
        return v
