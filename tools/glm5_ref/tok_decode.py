# Decode token-id files (comma separated) with the GLM-5.3 pack's tokenizer.  Usage: python tok_decode.py FILE...
# The pack: PACK=<folder>, else strata-glm5.json's --pack (_glm5_paths.py).  Set PYTHONIOENCODING=utf-8 on Windows.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _glm5_paths import load_tokenizer  # noqa: E402

tk = load_tokenizer()
for path in sys.argv[1:]:
    ids = [int(x) for x in open(path, encoding="utf-8").read().split(",") if x]
    print(Path(path).name, repr(tk.decode(ids)))
