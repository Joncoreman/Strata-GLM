# One decode window of a Nsight Systems sqlite export (2026-10-10): the kernels between two cudaGraphLaunch calls
# (the Nth and N+1th), grouped by name; the window's wall, GPU busy, waits and gaps.  Usage: nsys_win.py DB [N]
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
q = lambda s: db.execute(s).fetchall()
names = dict(q("select id, value from StringIds"))
gl = [r[0] for r in q("select r.start from CUPTI_ACTIVITY_KIND_RUNTIME r where r.nameId in "
                      "(select id from StringIds where value like 'cudaGraphLaunch%') order by r.start")]
print("graph launches", len(gl))
t0, t1 = gl[N], gl[N + 1]
k = q(f"select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL where start >= {t0} and start < {t1} order by start")
m = q(f"select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where start >= {t0} and start < {t1}")
wall = (t1 - t0) / 1e6
busy = sum(b - a for a, b, _ in k) / 1e6
wait = sum(b - a for a, b, n in k if names.get(n, "").startswith("wait_flag")) / 1e6
print(f"window {N}: wall {wall:.2f} ms, kernels {len(k)}, kernel time {busy:.2f} ms (waits {wait:.2f} ms, compute {busy - wait:.2f} ms), "
      f"copies {len(m)} {sum(x[2] for x in m) / 1e6:.1f} MB {sum(b - a for a, b, *_ in m) / 1e6:.2f} ms")
by = {}
for a, b, n in k:
    nm = names.get(n, str(n)); t, c = by.get(nm, (0, 0)); by[nm] = (t + b - a, c + 1)
for nm, (t, c) in sorted(by.items(), key=lambda x: -x[1][0])[:16]:
    print(f"  {t / 1e6:6.2f} ms  n={c:5d}  avg {t / c / 1e3:7.1f} us  {nm[:60]}")
# the sequence of one layer (from the 2nd wait_flag to the 5th): names in order
seq, w = [], 0
for a, b, n in k:
    nm = names.get(n, "")
    if nm.startswith("wait_flag"):
        w += 1
        if w == 5: break
    if w >= 2: seq.append(nm[:28])
print("one layer's kernels (" + str(len(seq)) + "):", " ".join(seq)[:1500])
# gaps: time between kernels with nothing running
gaps = 0.0; last = None
for a, b, _ in sorted(k):
    if last is not None and a > last: gaps += a - last
    last = max(last, b) if last else b
print(f"idle gaps between kernels {gaps / 1e6:.2f} ms")
