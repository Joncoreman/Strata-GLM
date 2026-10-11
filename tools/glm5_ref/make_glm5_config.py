"""Write strata-glm5.json (the GLM-5.3-Flash server's settings) for this PC and, when the pack is missing, build it
with tools/iq_pack.py and copy the learned expert profile beside it (2026-10-10).

  python tools\\glm5_ref\\make_glm5_config.py --gguf D:\\models\\UD-IQ4_XS\\GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
  options: --data-dir (default: Strata-data next to the repository), --ram-gib (default: total RAM - 30), --workers
           (default: physical cores - 4), --context 262144, --port 8097, --skip (turn the expert skipping on, 10%)
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROFILE_SRC = REPO / "tools" / "glm5_ref" / "profiles" / "expert-profile-learned-2026-10-08b.bin"


def total_ram_gib() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / 2**30
    except Exception:
        if os.name == "nt":
            out = subprocess.run(["wmic", "computersystem", "get", "TotalPhysicalMemory"], capture_output=True, text=True).stdout
            m = re.search(r"\d+", out)
            if m:
                return int(m.group()) / 2**30
        return 128.0


def physical_cores() -> int:
    try:
        import psutil
        return psutil.cpu_count(logical=False) or os.cpu_count() or 8
    except Exception:
        return os.cpu_count() or 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="shard 1 of the GLM-5.3-Flash GGUF (unsloth UD-IQ4_XS)")
    ap.add_argument("--data-dir", default=str(REPO.parent / "Strata-data"))
    ap.add_argument("--pack-name", default=None, help="default: glm53_<quant from the file name>")
    ap.add_argument("--ram-gib", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--context", type=int, default=262144)
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--vram-reserve-mib", type=int, default=3400)
    ap.add_argument("--skip", action="store_true", help="expert skipping on (STRATA_GLM5_SKIP_KEEP=2, MINW=0.10)")
    ap.add_argument("--exe", default=None, help="default: build-engine\\strata.exe")
    a = ap.parse_args()

    gguf = Path(a.gguf).resolve()
    if not gguf.exists():
        sys.exit(f"not found: {gguf}")
    m = re.search(r"UD-([A-Z0-9_]+)-00001", gguf.name) or re.search(r"-([A-Z0-9_]+)-00001", gguf.name)
    quant = (m.group(1) if m else "custom").lower().replace("_", "")
    pack = Path(a.data_dir) / "packs" / (a.pack_name or f"glm53_{quant}")
    exe = Path(a.exe) if a.exe else REPO / "build-engine" / "strata.exe"
    if not exe.exists():
        print(f"note: {exe} does not exist yet - run build-glm5.bat before START-GLM5.bat")

    if not (pack / "glm5-layout.json").exists():
        print(f"building the pack {pack} from the GGUF (reads the headers and the dense tensors, about a minute)")
        pack.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([sys.executable, str(REPO / "tools" / "iq_pack.py"), "--gguf", str(gguf), "--out", str(pack)])
        if r.returncode != 0:
            sys.exit("iq_pack.py failed")
    prof = pack / PROFILE_SRC.name
    if not prof.exists():
        if PROFILE_SRC.exists():
            shutil.copy2(PROFILE_SRC, prof)
        else:
            sys.exit(f"the expert profile is missing: {PROFILE_SRC}")

    ram = a.ram_gib if a.ram_gib else max(16, int(total_ram_gib()) - 30)
    workers = a.workers if a.workers else max(4, physical_cores() - 4)
    env = {"STRATA_GLM5_GO": "1", "STRATA_GLM5_PREFILL": "4096", "STRATA_GLM5_PF_DIRECT": "1",
           "STRATA_POOL_GUIDED": "1", "STRATA_POOL_ETAIL": "6",
           "STRATA_GLM5_SLOTS": "8", "STRATA_GLM5_SLOT_DIR": str(REPO / "build-engine" / "glm5-slots")}
    if a.skip:
        env.update({"STRATA_GLM5_SKIP_KEEP": "2", "STRATA_GLM5_SKIP_MINW": "0.10"})
    cfg = {
        "exe": str(exe),
        "args": ["--pack", str(pack), "--native", str(gguf), "--prefill", "auto", "--spec", "6",
                 "--max-context", str(a.context), "--pcie-frac", "0", "--mmap-experts",
                 "--resident-budget-gib", str(ram), "--expert-profile", str(prof), "--expert-cache", "0",
                 "--adapt-every", "1", "--adapt-swaps", "32", "--adapt-async", "1", "--pool-workers", str(workers),
                 "--vram-reserve-mib", str(a.vram_reserve_mib)],
        "env": env,
        "cwd": str(REPO),
        "tokenizer": str(pack / "tokenizer"),
        "model_name": "glm-5.3-flash-strata",
        "log": str(REPO / "build-engine" / "strata-glm5-serve.log"),
        "port": a.port,
    }
    out = REPO / "strata-glm5.json"
    if out.exists():
        bak = out.with_suffix(".json.bak")
        shutil.copy2(out, bak)
        print(f"the previous strata-glm5.json is kept as {bak.name}")
    out.write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}: pack {pack}, RAM tier {ram} GiB, {workers} pool workers, context {a.context}, port {a.port}, "
          f"expert skipping {'on' if a.skip else 'off'}")
    print("next: START-GLM5.bat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
