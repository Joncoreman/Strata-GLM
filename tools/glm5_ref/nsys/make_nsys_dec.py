# Writes build-engine\nsys\run_nsys_dec.bat: a short chat prompt, 64 tokens of decode under
# Nsight Systems (CUDA trace, graph nodes), the settings of 2026-10-10 (skip on, 20 workers).
import subprocess, sys
from pathlib import Path

WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT / "tools" / "glm5_ref"))
import _glm5_paths as P  # noqa: E402
msg = "Explain in three short paragraphs why the sky is blue and why sunsets are red."
ids = subprocess.run([sys.executable, str(WT / "tools" / "glm5_ref" / "tok_chat.py"), msg], capture_output=True,
                     text=True, check=True).stdout.strip()
print("prompt tokens", ids.count(",") + 1)
PACK = str(P.pack_dir())
NATIVE = str(P.native_gguf())
args = (f"--tokens {ids} --max-new 64 --pack {PACK} --native {NATIVE} --prefill auto --spec 6 --max-context 262144 "
        f"--pcie-frac 0 --mmap-experts --resident-budget-gib 98 --expert-profile {PACK}\\expert-profile-learned-2026-10-08b.bin "
        f"--expert-cache 0 --adapt-every 1 --adapt-swaps 64 --pool-workers 20 --vram-reserve-mib 3400")
out = WT / "build-engine" / "nsys"
out.mkdir(exist_ok=True)
(out / "run_nsys_dec.bat").write_text(
    "@echo off\r\n"
    'cd /d "%~dp0"\r\n'
    "set STRATA_GLM5_GO=1\r\nset STRATA_GLM5_PREFILL=4096\r\nset STRATA_GLM5_PF_DIRECT=1\r\n"
    "set STRATA_GLM5_SKIP_KEEP=2\r\nset STRATA_GLM5_SKIP_MINW=0.10\r\n"
    '"C:\\Program Files\\NVIDIA Corporation\\Nsight Systems 2024.6.2\\target-windows-x64\\nsys.exe" profile '
    "-t cuda --cuda-graph-trace=node --cuda-flush-interval=500 --sample=none --cpuctxsw=none -f true -o glm_dec "
    f'"{WT}\\build-engine\\strata.exe" {args} > run_nsys_dec.log 2>&1\r\n'
    '"C:\\Program Files\\NVIDIA Corporation\\Nsight Systems 2024.6.2\\target-windows-x64\\nsys.exe" export '
    "--type sqlite -f true -o glm_dec.sqlite glm_dec.nsys-rep >> run_nsys_dec.log 2>&1\r\n", encoding="ascii")
print("wrote", out / "run_nsys_dec.bat")
