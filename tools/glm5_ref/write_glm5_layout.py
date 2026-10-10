"""Write only native_experts.txt (the expert table, no payloads) for a GLM GGUF into <out_dir>, with iq_pack's own
expert_layout(); for tests/core/glm5_expert_layout_test.cpp.  Usage: python write_glm5_layout.py <shard1.gguf> <out_dir>
"""
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import iq_pack  # noqa: E402

src = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
model = iq_pack.Model(src, header_only=True)
res = iq_pack.expert_layout(model, src)
if isinstance(res, str):
    raise SystemExit("expert_layout: " + res)
_, text, n_expert, total = res
(out / "native_experts.txt").write_text(text, encoding="utf-8")
print(f"wrote {out / 'native_experts.txt'}: n_expert {n_expert}, total {total} bytes")
print(text.splitlines()[0][:200])
