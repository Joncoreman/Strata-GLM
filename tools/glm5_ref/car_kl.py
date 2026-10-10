# Compare two STRATA_FOLLOW_LOGITS files position by position: mean KL(ref || test) of the next-token distribution,
# top-1 agreement, and the worst positions.  Read in blocks (long continuations).  Usage: python car_kl.py REF TEST
import sys
import numpy as np

V = 154880
a = np.memmap(sys.argv[1], dtype=np.float32, mode="r")
b = np.memmap(sys.argv[2], dtype=np.float32, mode="r")
assert a.size % V == 0 and b.size % V == 0, "not a whole number of vocab rows"
a = a.reshape(-1, V)
b = b.reshape(-1, V)
n = min(len(a), len(b))


def logsoftmax(x):
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


kl = np.empty(n)
top1 = np.empty(n, dtype=bool)
for s in range(0, n, 64):
    x = np.asarray(a[s:s + 64], dtype=np.float64)
    y = np.asarray(b[s:s + 64], dtype=np.float64)
    lx, ly = logsoftmax(x), logsoftmax(y)
    kl[s:s + len(x)] = (np.exp(lx) * (lx - ly)).sum(axis=1)
    top1[s:s + len(x)] = x.argmax(axis=1) == y.argmax(axis=1)
print(f"positions {n} (vocab {V}): mean KL {kl.mean():.5f}  median {np.median(kl):.5f}  p95 {np.percentile(kl, 95):.4f}  "
      f"max {kl.max():.3f}  top-1 agreement {top1.mean():.4f}")
worst = np.argsort(-kl)[:5]
print("worst positions:", ", ".join(f"{i}:{kl[i]:.3f}" for i in worst))
