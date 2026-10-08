"""A first expert profile (STRP, tools/make_profile.py's format) for GLM-5.3-Flash: 45 trunk layers x 288 experts,
the dense layers 0-2 left out, experts ranked round-robin across layers (no usage trace exists yet).  Replace it with
a measured one once the engine runs (make_profile.py from a trace).
Usage: python make_glm5_profile.py <out.bin> [n_layers=45] [n_expert=288] [first_expert_layer=3]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import make_profile as MP  # noqa: E402

out = sys.argv[1]
n_layer = int(sys.argv[2]) if len(sys.argv) > 2 else 45
n_expert = int(sys.argv[3]) if len(sys.argv) > 3 else 288
first = int(sys.argv[4]) if len(sys.argv) > 4 else 3
MP.N_LAYER = n_layer
ranked = [(l, e) for e in range(n_expert) for l in range(first, n_layer)]
MP.write_profile(out, ranked, n_expert=n_expert)
print(f"wrote {out}: {len(ranked)} pairs ({n_layer} layers x {n_expert} experts, layers {first}-{n_layer - 1})")
