"""tools/glm5_ref/_glm5_paths.py - where the GLM-5.3 tools find the engine, the pack and the GGUF (2026-10-10).

Order: the environment (PACK, NATIVE, STRATA_EXE), then strata-glm5.json at the repository root (the --pack / --native
arguments the server starts the engine with), then Strata-data next to the repository (setup.py's default data folder).
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "strata-glm5.json"


def _config_args() -> dict:
    try:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    args = cfg.get("args") or []
    found = {}
    for i, a in enumerate(args[:-1]):
        if a in ("--pack", "--native", "--expert-profile", "--resident-budget-gib", "--pool-workers"):
            found[a] = args[i + 1]
    if cfg.get("exe"):
        found["exe"] = cfg["exe"]
    return found


_CFG = _config_args()


def pack_dir() -> Path:
    p = os.environ.get("PACK") or _CFG.get("--pack")
    if p:
        return Path(p)
    cands = sorted((REPO.parent / "Strata-data" / "packs").glob("glm53*"))
    if cands:
        return cands[0]
    sys.exit("no GLM-5.3 pack found: set PACK=<pack folder>, or write strata-glm5.json (see strata-glm5.example.json)")


def native_gguf() -> Path:
    p = os.environ.get("NATIVE") or _CFG.get("--native")
    if p:
        return Path(p)
    sys.exit("no GLM-5.3 GGUF given: set NATIVE=<shard 1 .gguf>, or write strata-glm5.json")


def engine_exe() -> Path:
    p = os.environ.get("STRATA_EXE") or _CFG.get("exe")
    if p:
        return Path(p)
    for d in ("build-engine", "build"):
        e = REPO / d / "strata.exe"
        if e.exists():
            return e
    sys.exit("strata.exe not found: run build-glm5.bat, or set STRATA_EXE")


def expert_profile() -> Path:
    p = os.environ.get("PROFILE") or _CFG.get("--expert-profile")
    if p:
        return Path(p)
    return pack_dir() / "expert-profile-learned-2026-10-08b.bin"


def load_tokenizer():
    """The pack's tokenizer through tools/strata_tokenizer.py (GLM-4 pre-tokenizer, the pack's own pre_pattern)."""
    import regex
    sys.path.insert(0, str(REPO / "tools"))
    import strata_tokenizer as ST  # noqa: E402
    d = pack_dir() / "tokenizer"
    meta = json.load(open(d / "tokenizer.json", encoding="utf-8"))
    vocab = json.load(open(d / "vocab.json", encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = [l.rstrip("\n") for l in open(d / "merges.txt", encoding="utf-8") if l.strip() and not l.startswith("#")]
    types = json.load(open(d / "token_type.json", encoding="utf-8"))
    tk = ST.Tokenizer(tokens, merges, types, pre="glm4")
    tk._re = regex.compile(meta["pre_pattern"])
    return tk
