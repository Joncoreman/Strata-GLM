# GLM-5.3 speed: one PC (the standard setting) vs the 5090 + the 3060 stage (2026-10-08).  Same chat prompt, 256 tokens.
# Usage: python glm_ab_speed.py single|mn [K1 K2]
import os, subprocess, sys, time
from pathlib import Path

WT = Path(os.environ.get("WT", r"D:\AI\LLM\strata-glm-gpu"))   # the worktree whose build-engine runs
EXE = WT / "build-engine" / "strata.exe"
PACK = "D:/AI/LLM/Strata-data/packs/glm53_iq4xs"
NATIVE = "D:/AI/LLM/GLM-5.3-Flash-GGUF/unsloth/GLM-5.3-Flash-GGUF/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf"
OUT = WT / "build-engine" / "glm_ab"   # logs and tokens (beside the engine build, not in the source tree)
OUT.mkdir(exist_ok=True)
mode = sys.argv[1]
K1, K2 = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (30, 44)
NEW = 256
LONG = int(os.environ.get("LONG", "0"))   # N > 0: summarize the first N characters of docs (a long prompt)
msg_args = []
if LONG > 0:
    docs = "".join(p.read_text(encoding="utf-8", errors="replace") for p in sorted((WT / "docs").glob("*.md")))
    msg_args = ["Summarize the following documents in Japanese, in ten bullet points.\n\n" + docs[:LONG]]
    NEW = 64
ids = subprocess.run([sys.executable, str(Path(__file__).parent / "tok_chat.py")] + msg_args, capture_output=True,
                     text=True, check=True).stdout.strip()
print("prompt tokens:", ids.count(",") + 1, flush=True)
args = ["--serve", "--pack", PACK, "--native", NATIVE, "--prefill", "auto", "--spec", os.environ.get("SPEC", "6"), "--max-context", os.environ.get("CTX", "32768"),
        "--pcie-frac", "0", "--short-read", "1000000000", "--prompt-cache", "0", "--pool-workers", os.environ.get("WORKERS", "16"),
        "--vram-reserve-mib", os.environ.get("VRES", "2200"), "--mmap-experts", "--resident-budget-gib", os.environ.get("RES", "105"),
        "--expert-profile", PACK + "/expert-profile-learned-2026-10-08b.bin", "--expert-cache", os.environ.get("CACHE", "0"),
        "--adapt-every", os.environ.get("ADAPT_EVERY", "1"), "--adapt-swaps", os.environ.get("ADAPT", "64")]
args += os.environ.get("EXTRA", "").split()   # more engine flags for an A/B (space separated)
if mode == "mn":
    args += ["--remote-stage", os.environ.get("STAGE_ADDR", "KITT3060:9300"), "--layer-split", f"{K1},{K2}"]
elif mode == "mn2":   # both 3060s: GPU 0 runs [30, 37), GPU 1 [37, 44)
    args += ["--remote-stage", "KITT3060:9300,KITT3060:9302", "--layer-split", os.environ.get("SPLIT3", "30,37,44")]
env = dict(os.environ, STRATA_GLM5_GO="1", STRATA_TRACE="1")
TAG = os.environ.get("TAG", "")   # a suffix for the log and token files (A/B runs side by side)
err = open(OUT / f"head_{mode}{TAG}.err", "w", encoding="utf-8")
p = subprocess.Popen([str(EXE)] + args, cwd=WT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                     text=True, bufsize=1)
t0 = time.time()
toks, sent = [], 0
REPEAT = int(os.environ.get("REPEAT", "1"))   # 2: a second request (its reset makes the stage print the first's split)
for line in p.stdout:
    line = line.strip()
    if line.startswith("READY"):
        print(f"ready in {time.time() - t0:.0f} s", flush=True)
        p.stdin.write(f"GEN {NEW} {ids}\n"); p.stdin.flush(); sent = 1
    elif line.startswith("T "):
        toks.append(int(line[2:]))
    elif line.startswith("DONE") or line.startswith("ERR"):
        f = line.split()
        if f[0] == "DONE":
            n, prompt_ms, decode_ms = int(f[1]), float(f[3]), float(f[4])
            print(f"{mode}: prompt {int(f[2])} tokens in {prompt_ms / 1000:.1f} s; {n} tokens in {decode_ms / 1000:.1f} s "
                  f"= {1000 * n / decode_ms:.2f} tok/s", flush=True)
        else:
            print(line, flush=True)
        if f[0] == "DONE" and sent < REPEAT:
            p.stdin.write("GEN 8 " + ",".join(ids.split(",")[:20]) + "\n"); p.stdin.flush(); sent += 1
            continue
        break
try:
    p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait(timeout=120)
except Exception:
    p.kill()
(OUT / f"tokens_{mode}{TAG}.txt").write_text(",".join(map(str, toks)), encoding="utf-8")
