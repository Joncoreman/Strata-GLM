# Encode raw text (special tokens such as <|user|> parsed) with the GLM-5.3 pack's tokenizer (2026-10-10).
# Usage: python tok_raw.py "<|user|>question<|assistant|>"   or   python tok_raw.py @file
# The pack: PACK=<folder>, else strata-glm5.json's --pack (_glm5_paths.py).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _glm5_paths import load_tokenizer  # noqa: E402

tk = load_tokenizer()
text = sys.argv[1]
if text.startswith("@"):
    text = open(text[1:], encoding="utf-8").read()
print(",".join(str(i) for i in tk.encode(text, parse_special=True)))
