# Token ids of a GLM-5.3 chat prompt ("[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>...<|assistant|>") with the
# pack's tokenizer.  Usage: python tok_chat.py "message"   or   python tok_chat.py @file   (no argument: a 3-part test)
# The pack: PACK=<folder>, else strata-glm5.json's --pack, else Strata-data next to the repository (_glm5_paths.py).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _glm5_paths import load_tokenizer  # noqa: E402

tk = load_tokenizer()
if len(sys.argv) > 1 and sys.argv[1].startswith("@"):   # @file: a long message (Windows caps a command line at 32K)
    sys.argv[1] = open(sys.argv[1][1:], encoding="utf-8").read()
msg = sys.argv[1] if len(sys.argv) > 1 else (
    "次の3つに順番に答えてください。1) 日本の四季の行事と食べ物を季節ごとに短く説明。"
    "2) Write a Python function that groups CSV rows by a column and returns the mean of another column, "
    "and explain its time complexity. 3) Summarize the causes of the French Revolution in three sentences.")
text = "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>" + msg + "<|assistant|>"
ids = tk.encode(text, parse_special=True)
print(len(ids), file=sys.stderr)
print(",".join(str(i) for i in ids))
