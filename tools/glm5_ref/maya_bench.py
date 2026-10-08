import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
doc = open(r"D:\AI\LLM\strata-glm-gpu\docs\MODELS.md", encoding="utf-8").read()[:4200]
cases = [
    ("chat", "Explain how a transformer language model generates text, step by step, for a high school student. "
             "Then write a short haiku about autumn in Japanese.", 400),
    ("chat2", "日本の四季の行事と食べ物を、季節ごとに短く説明してください。", 400),
    ("long", "Summarize the following document in Japanese, in five bullet points.\n\n" + doc, 200),
]
for name, msg, mx in cases:
    body = json.dumps({"model": "glm", "messages": [{"role": "user", "content": msg}], "max_tokens": mx,
                       "temperature": 0, "stream": False,
                       "chat_template_kwargs": {"reasoning_effort": "low"}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    dt = time.time() - t0
    u = r.get("usage", {})
    tm = r.get("timings") or {}
    print(f"[{name}] {dt:.1f} s  prompt {u.get('prompt_tokens')}  out {u.get('completion_tokens')}  "
          f"timings {json.dumps({k: tm.get(k) for k in ('prompt_per_second', 'predicted_per_second', 'prompt_ms', 'predicted_ms') if k in tm})}")
    m = r["choices"][0]["message"]
    print("   ", (m.get("content") or "")[:160].replace("\n", " "))
