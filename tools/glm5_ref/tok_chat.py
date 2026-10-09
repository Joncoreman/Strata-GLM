import json
import sys

import regex

sys.path.insert(0, r"D:\AI\LLM\strata-glm-gpu\tools")
import strata_tokenizer as ST  # noqa: E402

D = r"D:\AI\LLM\Strata-data\packs\glm53_iq4xs\tokenizer"
meta = json.load(open(D + r"\tokenizer.json", encoding="utf-8"))
vocab = json.load(open(D + r"\vocab.json", encoding="utf-8"))
tokens = [None] * len(vocab)
for t, i in vocab.items():
    tokens[i] = t
merges = [l.rstrip("\n") for l in open(D + r"\merges.txt", encoding="utf-8") if l.strip() and not l.startswith("#")]
types = json.load(open(D + r"\token_type.json", encoding="utf-8"))
tk = ST.Tokenizer(tokens, merges, types, pre="glm4")
tk._re = regex.compile(meta["pre_pattern"])

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
