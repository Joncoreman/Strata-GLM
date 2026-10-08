"""Read llama-eval-callback dumps (runs/dump_*.txt) into {name: [entries]}.

Each entry keeps the header's shape, the printed rows (only the first/last few values of each row are printed:
'...' marks the gap) and the sum. Names repeat (e.g. 'hc_mixes-0' appears once per sub-block), so every
occurrence is kept in order.
"""
import re

_HDR = re.compile(r"common_debug_cb_eval:\s+(.+?) = \((\w+)\)\s+(\S+)\((.*)\) = \{([\d, ]+)\}\s*$")
_NUM = re.compile(r"-?\d+\.\d+|-?inf|nan", re.I)


def parse(path):
    out = {}
    cur = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _HDR.match(line.rstrip("\n"))
            if m:
                name = m.group(1).strip()
                cur = {"name": name, "type": m.group(2), "op": m.group(3),
                       "shape": [int(v) for v in m.group(5).split(",")], "rows": [], "sum": None}
                out.setdefault(name, []).append(cur)
                continue
            if cur is None:
                continue
            s = line.strip()
            if s.startswith("sum ="):
                cur["sum"] = float(s.split("=")[1])
                cur = None
            elif s.startswith("[") and s.endswith("],") and any(c.isdigit() for c in s):
                body = s[1:-2]
                if "..." in body:
                    head, tail = body.split("...", 1)
                    cur["rows"].append(([float(v) for v in _NUM.findall(head)], [float(v) for v in _NUM.findall(tail)]))
                else:
                    cur["rows"].append(([float(v) for v in _NUM.findall(body)], []))
    return out


def compare(entry, full_rows, tol=None):
    """full_rows: 2-D array [rows, ne0] in the same order the dump prints (ne1 outer, ne0 inner).
    Returns (max abs diff over the printed values, our sum, dump sum)."""
    import numpy as np
    worst = 0.0
    for r, (head, tail) in enumerate(entry["rows"]):
        row = np.asarray(full_rows[r], dtype=np.float64)
        for i, v in enumerate(head):
            worst = max(worst, abs(row[i] - v))
        for i, v in enumerate(tail):
            worst = max(worst, abs(row[len(row) - len(tail) + i] - v))
    return worst, float(np.asarray(full_rows, dtype=np.float64).sum()), entry["sum"]
