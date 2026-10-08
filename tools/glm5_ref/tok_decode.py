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
for path in sys.argv[1:]:
    ids = [int(x) for x in open(path, encoding="utf-8").read().split(",") if x]
    print(path.split("\\")[-1].split("/")[-1], repr(tk.decode(ids)))
