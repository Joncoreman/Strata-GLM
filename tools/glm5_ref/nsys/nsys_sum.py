# Summarize a Nsight Systems sqlite export (2026-10-09): GPU busy share, kernels by name, copies, CUDA API time.
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
q = lambda s: db.execute(s).fetchall()
names = dict(q("select id, value from StringIds"))
k = q("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL")
m = q("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY")
t0 = min(min(a for a, _, _ in k), min(a for a, *_ in m)); t1 = max(max(b for _, b, _ in k), max(b for _, b, *_ in m))
span = (t1 - t0) / 1e9
def union(iv):
    iv = sorted(iv); tot = 0; cs = ce = None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None: tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return (tot + (ce - cs if cs is not None else 0)) / 1e9
print(f"window {span:.2f} s: kernels busy {union([(a, b) for a, b, _ in k]):.2f} s, copies busy {union([(a, b) for a, b, *_ in m]):.2f} s, "
      f"either {union([(a, b) for a, b, _ in k] + [(a, b) for a, b, *_ in m]):.2f} s")
by = {}
for a, b, n in k:
    nm = names.get(n, str(n)); t, c = by.get(nm, (0, 0)); by[nm] = (t + b - a, c + 1)
for nm, (t, c) in sorted(by.items(), key=lambda x: -x[1][0])[:12]:
    print(f"  {t/1e9:6.2f} s  n={c:7d}  avg {t/c/1e3:8.1f} us  {nm[:70]}")
h2d = [(a, b, by_) for a, b, by_, kind in m if kind == 1]
print(f"H2D copies {len(h2d)}, {sum(x[2] for x in h2d)/1e9:.1f} GB, {sum(b - a for a, b, _ in h2d)/1e9:.2f} s")
api = q("select r.nameId, count(*), sum(r.end - r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r group by r.nameId order by 3 desc limit 8")
for n, c, t in api:
    print(f"  api {t/1e9:6.2f} s  n={c:7d}  {names.get(n, n)}")
