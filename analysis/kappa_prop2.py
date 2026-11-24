"""Reviewer Q3: kappa_i = max_y pi_i(y) / min_y pi_i(y) of the smoothed node priors (counts + 1), for the exact Dirichlet
partitions used in the experiments (CIFAR-10/100 label vectors: 5000 / 500 images per class, same partition code and seeds)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import lagossip as g
rows = []
for ds, K, per in (('cifar10', 10, 5000), ('cifar100', 100, 500)):
    y = np.repeat(np.arange(K), per)
    for n, alpha in ((5, 0.1), (5, 0.5), (5, 1.0), (16, 0.1), (32, 0.1)):
        if ds == 'cifar100' and (n != 5 or alpha != 0.1): continue
        ks = []
        for s in range(5):
            parts = g.dirichlet_partition(y, n, alpha, s, 64)
            for p in parts:
                c = np.bincount(y[p], minlength=K); pi = (c + 1) / (c.sum() + K); ks.append(pi.max() / pi.min())
        ks = np.array(ks)
        rows.append((ds, n, alpha, np.median(ks), ks.max(), np.mean(ks ** 2)))
        print(f'{ds:9s} n={n:2d} alpha={alpha}: kappa median {np.median(ks):8.0f}  max {ks.max():8.0f}  mean kappa^2 {np.mean(ks**2):.2e}')
