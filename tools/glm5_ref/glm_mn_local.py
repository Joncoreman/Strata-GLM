# GLM-5.3 multinode correctness on one PC (2026-10-08): a stage engine (layers K1..K2-1) and a head engine on the
# 5090, versus one engine.  Greedy; the generated token streams must match.
# Usage: python glm_mn_local.py single|mn [K1 K2]
import os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _glm5_paths as P  # noqa: E402

WT = Path(os.environ.get("WT", str(P.REPO)))
EXE = WT / "build-engine" / "strata.exe"
PACK = str(P.pack_dir()).replace("\\", "/")
NATIVE = str(P.native_gguf()).replace("\\", "/")
OUT = WT / "build-engine" / "glm_mn"   # logs and tokens (beside the engine build, not in the source tree)
OUT.mkdir(exist_ok=True)
mode = sys.argv[1]
K1, K2 = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (30, 44)
ids = [int(x) for x in subprocess.run([sys.executable, str(Path(__file__).parent / "tok_chat.py"),
                                       "Explain in three sentences why the sky is blue."],
                                      capture_output=True, text=True, check=True).stdout.strip().split(",")]
NEW = 40

def common(budget, cache, ctx=8192):
    return ["--serve", "--pack", PACK, "--native", NATIVE, "--prefill", "auto", "--spec", "6", "--max-context", str(ctx),
            "--pcie-frac", "0", "--short-read", "1000000000", "--prompt-cache", "0", "--pool-workers", "16",
            "--vram-reserve-mib", "2200", "--mmap-experts", "--resident-budget-gib", str(budget),
            "--expert-profile", PACK + "/expert-profile-learned-2026-10-08b.bin", "--expert-cache", str(cache)]

env = dict(os.environ, STRATA_GLM5_GO="1", STRATA_TRACE="1")
stage = None
if mode == "mn":
    serr = open(OUT / "stage.err", "w", encoding="utf-8")
    stage = subprocess.Popen([str(EXE)] + common(8, int(os.environ.get("STAGE_CACHE", "300"))) + ["--stage-listen", "9300", "--stage-layers", f"{K1},{K2}",
                             "--stage-allow", "127.0.0.1"], cwd=WT, env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=serr)
    t0 = time.time()
    while "listening on port" not in (OUT / "stage.err").read_text(encoding="utf-8", errors="replace"):
        if stage.poll() is not None or time.time() - t0 > 900:
            print("stage failed:", (OUT / "stage.err").read_text(encoding="utf-8", errors="replace")[-3000:])
            sys.exit(1)
        time.sleep(2)
    print(f"stage up in {time.time() - t0:.0f} s")
    head_args = common(60, 0) + ["--remote-stage", "127.0.0.1:9300", "--layer-split", f"{K1},{K2}"]
elif mode == "single2":   # the same engine, another expert placement (a smaller RAM tier and GPU cache)
    head_args = common(20, 500)
else:
    head_args = common(60, 0)
herr = open(OUT / f"head_{mode}.err", "w", encoding="utf-8")
head = subprocess.Popen([str(EXE)] + head_args, cwd=WT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=herr, text=True, bufsize=1)
toks, t_start = [], time.time()
for line in head.stdout:
    line = line.strip()
    if line.startswith("READY"):
        print(f"head ready in {time.time() - t_start:.0f} s")
        head.stdin.write(f"GEN {NEW} " + ",".join(map(str, ids)) + "\n")
        head.stdin.flush()
    elif line.startswith("T "):
        toks.append(int(line[2:]))
    elif line.startswith(("DONE", "ERR")):
        print(line)
        break
try:
    head.stdin.write("QUIT\n"); head.stdin.flush()
    head.wait(timeout=120)
except Exception:
    head.kill()
if stage is not None:
    stage.kill()
(OUT / f"tokens_{mode}.txt").write_text(",".join(map(str, toks)), encoding="utf-8")
print(mode, len(toks), "tokens:", toks)
