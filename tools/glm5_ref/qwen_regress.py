# Qwen (Flash-Next IQ2_XS) regression for the GLM port (2026-10-08): the same greedy run on the engine before the
# port (strata-mn build-mn, commit 60bd1d3a) and after it (strata-glm-gpu build-engine); the token streams must match.
# Usage: python qwen_regress.py before|after <out.json>   (MN_TEST_DIR: the folder of the multinode test's mn_test.py)
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("MN_TEST_DIR", str(Path(__file__).resolve().parents[2] / "mn-test")))
import mn_test as M   # noqa: E402

which, out = sys.argv[1], sys.argv[2]
if which == "after":
    M.WT = Path(os.environ.get("WT", str(Path(__file__).resolve().parents[2])))   # the worktree whose build-engine runs
    M.EXE = M.WT / "build-engine" / "strata.exe"
print("engine:", M.EXE)
sys.argv = ["mn_test.py", "run", "ref", "--profile", str(M.BASE_PROFILE), "--max-new", "64", "--out", out]
M.main()
