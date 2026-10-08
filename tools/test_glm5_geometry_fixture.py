"""Exercise the C++ geometry loader with captured real metadata and malformed headers.

Only tiny, newly created files under the selected build directory are written. No model weights.
Usage: python tools/test_glm5_geometry_fixture.py build-glm/glm5_geometry_test.exe
"""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def string(s):
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def encoded(value):
    if isinstance(value, str):
        return 8, string(value)
    if isinstance(value, bool):
        return 7, struct.pack("<?", value)
    if isinstance(value, float):
        return 6, struct.pack("<f", value)
    if isinstance(value, int):
        return (5, struct.pack("<i", value)) if value < 0 else (4, struct.pack("<I", value))
    if isinstance(value, list):
        if all(isinstance(v, int) for v in value):
            tag, fmt = (5, "i") if any(v < 0 for v in value) else (4, "I")
            raw = struct.pack("<" + fmt * len(value), *value)
        else:
            tag = 6
            raw = struct.pack("<" + "f" * len(value), *value)
        return 9, struct.pack("<IQ", tag, len(value)) + raw
    raise TypeError(type(value))


def write_header(path, md):
    blob = struct.pack("<IIQQ", 0x46554747, 3, 0, len(md))
    for key, value in md.items():
        tag, payload = encoded(value)
        blob += string(key) + struct.pack("<I", tag) + payload
    blob += bytes((-len(blob)) % 32)
    path.write_bytes(blob)


def main():
    executable = Path(sys.argv[1]).resolve()
    md = json.loads((ROOT / "tests/data/glm5-next-metadata.json").read_text())["metadata"]
    passed = 0
    cases = [(None, None, None),
             ("embedding_length", None, "missing glm5-next.embedding_length"),
             ("embedding_length", 0, "invalid integer: embedding_length"),
             ("embedding_length", -1, "invalid integer: embedding_length"),
             ("embedding_length", 4096.0, "expected integer: embedding_length"),
             ("attention.head_count_kv", 1, "must be a per-layer array"),
             ("attention.head_count_kv", [0] * 45, "length must equal block_count"),
             ("attention.head_count_kv", [0] * 45 + [-1], "invalid integer"),
             ("nextn_predict_layers", 46, "leaves no main layers"),
             ("rope.dimension_count", 1, "requires rope.dimension_count = 0"),
             ("expert_used_count", 289, "exceeds expert_count"),
             ("expert_feed_forward_length", 2048.5, "invalid array integer"),
             ("expert_feed_forward_length", -1, "invalid array integer"),
             ("swiglu_clamp_exp", [10.0] * 45, "wrong array length"),
             ("swiglu_clamp_shexp", [-1.0] * 46, "invalid swiglu_clamp_shexp"),
             ("attention.layer_norm_rms_epsilon", -1.0, "epsilons must be positive"),
             ("attention.layer_norm_rms_epsilon", "bad", "expected finite number")]
    with tempfile.TemporaryDirectory(prefix="glm5-geometry-", dir=executable.parent) as tmp:
        for i, (key, value, error) in enumerate(cases):
            changed = md.copy()
            if key is not None:
                key = "glm5-next." + key
                if value is None:
                    changed.pop(key)
                else:
                    changed[key] = value
            fixture = Path(tmp) / f"case-{i}.gguf"
            write_header(fixture, changed)
            result = subprocess.run([str(executable), str(fixture)], capture_output=True, text=True)
            if error is None:
                assert result.returncode == 0, result.stderr
                print(result.stdout.strip())
            else:
                assert result.returncode == 1 and error in result.stderr, (key, result.returncode, result.stderr)
            passed += 1
        for experts in (512, 128):  # canonical plus a synthetic pruned variant, same non-expert dimensions
            qwen = {"general.architecture": "qwen4exp", "qwen4exp.block_count": 48,
                    "qwen4exp.embedding_length": 2560, "qwen4exp.expert_count": experts,
                    "qwen4exp.expert_used_count": 10, "qwen4exp.attention.head_count": 24,
                    "qwen4exp.attention.head_count_kv": 2}
            fixture = Path(tmp) / f"qwen-{experts}.gguf"
            write_header(fixture, qwen)
            result = subprocess.run([str(executable), str(fixture)], capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            print(result.stdout.strip())
            passed += 1
    print(f"glm5_geometry_fixture: {passed} passed, 0 failed")


if __name__ == "__main__":
    main()
