# The kernel sequence of one DSA (MLA) layer in a decode window: from the wait before the first mla_chunk_kernel's
# layer to the third wait after it.  Usage: nsys_dsa.py DB [N]
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
q = lambda s: db.execute(s).fetchall()
names = dict(q("select id, value from StringIds"))
gl = [r[0] for r in q("select r.start from CUPTI_ACTIVITY_KIND_RUNTIME r where r.nameId in "
                      "(select id from StringIds where value like 'cudaGraphLaunch%') order by r.start")]
t0, t1 = gl[N], gl[N + 1]
k = [(a, b, names.get(n, "")) for a, b, n in
     q(f"select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL where start >= {t0} and start < {t1} order by start")]
# split into layers at every 3rd wait_flag (waitA, waitB, waitCPU per layer)
layers, cur, w = [], [], 0
for a, b, nm in k:
    cur.append((a, b, nm))
    if nm.startswith("wait_flag"):
        w += 1
        if w % 3 == 0: layers.append(cur); cur = []
print("layers split:", len(layers))
for i, L in enumerate(layers):
    if any(nm.startswith("mla_chunk") for _, _, nm in L):
        c = collections.Counter(nm for _, _, nm in L)
        t = sum(b - a for a, b, _ in L) / 1e6
        print(f"DSA layer #{i}: {len(L)} kernels, {t:.2f} ms;", ", ".join(f"{n}x{v}" for v, n in c.most_common(8)))
        seq = [nm[:22] for _, _, nm in L]
        print("  ", " ".join(seq)[:2500])
        break
c = collections.Counter()
for L in layers:
    if any(nm.startswith("mla_chunk") for _, _, nm in L):
        c["dsa_small_mmvq"] += sum(1 for _, _, nm in L if nm.startswith("native_small_mmvq"))
        c["dsa"] += 1
    else:
        c["kda_small_mmvq"] += sum(1 for _, _, nm in L if nm.startswith("native_small_mmvq"))
        c["kda"] += 1
print(dict(c))
