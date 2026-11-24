# Numerical check of the shared-minimiser claim on a well-specified model:
# K Gaussian classes with identity covariance (so a linear softmax model is exact), 5 nodes with Dirichlet(0.1) label skew.
import numpy as np
rng = np.random.default_rng(0)
K, d, N, m = 4, 6, 5, 40000
mu = rng.normal(0, 1.2, (K, d))
pis = rng.dirichlet(0.1 * np.ones(K), N) * 0.98 + 0.005          # node label distributions (kept away from 0)
data = []
for i in range(N):
    y = rng.choice(K, m, p=pis[i] / pis[i].sum()); x = mu[y] + rng.normal(0, 1, (m, d)); data.append((x, y))
def grad(W, b, x, y, off):
    z = x @ W.T + b + off; z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    p[np.arange(len(y)), y] -= 1
    return p.T @ x / len(y), p.mean(0)
def zeta2(W, b, offs):
    g = [np.concatenate([a.ravel(), c]) for a, c in (grad(W, b, *data[i], offs[i]) for i in range(N))]
    gbar = np.mean(g, 0); return float(np.mean([np.sum((gi - gbar) ** 2) for gi in g])), float(np.sum(gbar ** 2))
Wst, bst = mu.copy(), -0.5 * (mu ** 2).sum(1)                     # f*(x)_y = log p(x|y) + c(x)
la = [np.log(pis[i]) for i in range(N)]; ce = [np.zeros(K)] * N
pbar = pis.mean(0)
print('heterogeneity (1/n) sum ||grad_i - grad_bar||^2 at the population solution, and ||grad_bar||^2')
print('  logit-adjusted loss at theta*            :', zeta2(Wst, bst, la))
print('  plain CE at its pooled optimum (b+log pi):', zeta2(Wst, bst + np.log(pbar), ce))
# decentralised GD with local steps on a ring: does each method reach consensus on its optimum?
Wm = np.zeros((N, N))
for i in range(N):
    for j in (i - 1, i, i + 1): Wm[i, j % N] += 1 / 3
def run(offs, target_b, rounds=300, local=10, lr=0.2):
    Ws = [np.zeros((K, d)) for _ in range(N)]; bs = [np.zeros(K) for _ in range(N)]
    for r in range(rounds):
        for i in range(N):
            for _ in range(local):
                gW, gb = grad(Ws[i], bs[i], *data[i], offs[i]); Ws[i] -= lr * gW; bs[i] -= lr * gb
        Ws = [sum(Wm[i, j] * Ws[j] for j in range(N)) for i in range(N)]; bs = [sum(Wm[i, j] * bs[j] for j in range(N)) for i in range(N)]
    Wb, bb = np.mean(Ws, 0), np.mean(bs, 0)
    cons = np.mean([np.sum((Ws[i] - Wb) ** 2) + np.sum((bs[i] - bb) ** 2) for i in range(N)])
    bb0, tb0 = bb - bb.mean(), target_b - target_b.mean()                      # bias is defined up to a constant
    dist = np.sum((Wb - Wb.mean(0) - (Wst - Wst.mean(0))) ** 2) + np.sum((bb0 - tb0) ** 2)
    # balanced test accuracy of each node's model and of the average (fixed test sample, see below)
    acc = lambda W, b: float(((xt @ W.T + b).argmax(1) == yt).mean())
    return cons, dist, np.mean([acc(Ws[i], bs[i]) for i in range(N)]), acc(Wb, bb)
# one balanced test sample with its own seed, shared with figures/make_figures.py
rng_test = np.random.default_rng(1)
yt = rng_test.integers(0, K, 20000); xt = mu[yt] + rng_test.normal(0, 1, (20000, d)); print('Bayes accuracy (balanced):', float(((xt @ Wst.T + bst).argmax(1) == yt).mean()))
for name, offs, tb in (('logit-adjusted', la, bst), ('plain CE', ce, bst + np.log(pbar))):
    c, dist, ai, aa = run(offs, tb)
    print(f'{name:>15}: consensus distance {c:.2e}  distance to its optimum {dist:.2e}  node acc {ai:.4f}  avg-model acc {aa:.4f}')
