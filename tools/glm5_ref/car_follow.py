# The quality yardstick (2026-10-10): run strata (not --serve) over a prompt and a fixed continuation
# (--spec-follow), each position's logits to OUT (STRATA_FOLLOW_LOGITS).
# Usage: python car_follow.py PROMPT_IDS CONT_IDS OUT [extra env as K=V ...]
import os, subprocess, sys, time
from pathlib import Path

WT = Path(r"D:\AI\LLM\strata-glm-v041")
EXE = WT / "build-engine" / "strata.exe"
PACK = "D:/AI/LLM/Strata-data/packs/glm53_iq4xs"
NATIVE = "D:/AI/LLM/GLM-5.3-Flash-GGUF/unsloth/GLM-5.3-Flash-GGUF/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf"
prompt, cont, out = sys.argv[1], sys.argv[2], sys.argv[3]
env = dict(os.environ, STRATA_GLM5_GO="1", STRATA_FOLLOW_LOGITS=out, STRATA_GLM5_PREFILL="4096", STRATA_GLM5_PF_DIRECT="1")
for kv in sys.argv[4:]:
    k, v = kv.split("=", 1)
    env[k] = v
n_cont = open(cont).read().count(",") + 1
args = ["--pack", PACK, "--native", NATIVE, "--prefill", "auto", "--spec", "6", "--max-context", "32768",
        "--pcie-frac", "0", "--short-read", "1000000000", "--prompt-cache", "0", "--pool-workers", "22",
        "--vram-reserve-mib", "3400", "--mmap-experts", "--resident-budget-gib", "98",
        "--expert-profile", PACK + "/expert-profile-learned-2026-10-08b.bin", "--expert-cache", "0",
        "--adapt-every", "1", "--adapt-swaps", "64",
        "--tokens-file", prompt, "--max-new", str(n_cont), "--spec-follow", cont]
# ORACLE=1: the continuation as the drafts too - the windows go wide (8 tokens, all accepted: a perfect drafter's
# speed); without it every window is one token (the yardstick's runs, 2026-10-10)
if os.environ.get("ORACLE") == "1":
    args += ["--spec-oracle", cont]
t0 = time.time()
r = subprocess.run([str(EXE)] + args, cwd=WT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
log = Path(out).with_suffix(".log")
log.write_text(r.stderr + "\n" + r.stdout, encoding="utf-8")
size = Path(out).stat().st_size if Path(out).exists() else 0
print(f"exit {r.returncode} in {time.time() - t0:.0f} s; logits {size / 1e6:.1f} MB; log {log}")
for line in r.stderr.splitlines():
    if any(s in line for s in ("tok/s", "car test", "follow", "error", "ERR", "cannot")):
        print(" ", line[:200])
