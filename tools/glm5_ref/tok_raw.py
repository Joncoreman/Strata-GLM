# Encode raw text (special tokens such as <|user|> parsed) with the GLM-5.3 pack's tokenizer (2026-10-10).
# Usage: python tok_raw.py "<|user|>question<|assistant|>"   or   python tok_raw.py @file
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

text = sys.argv[1]
if text.startswith("@"):
    text = open(text[1:], encoding="utf-8").read()
print(",".join(str(i) for i in tk.encode(text, parse_special=True)))
