"""Round-3D analysis (review Q2, Q3, Q4, Q5). Input: round3D_lines.txt (notebook journal-round3-d-reviewer-experiments v1),
round2_combined_per_seed.csv (5-node CNN, alpha 0.1, seeds 0-4: la_bn, dpsgd, qgm at lr 0.1).
Outputs: round3D_per_run.csv, round3D_tests.csv. Paired tests by seed; Holm within each family."""
import json, os
import numpy as np, pandas as pd
from scipy import stats
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
L = [l.strip() for l in open(f'{D}/round3D_lines.txt')]
R = pd.DataFrame([json.loads(l[4:]) for l in L if l.startswith('RES|')]); R.to_csv(f'{D}/round3D_per_run.csv', index=False)
T2 = pd.DataFrame([json.loads(l[6:]) for l in L if l.startswith('TUNE2|')])
C = pd.read_csv(f'{D}/round2_combined_per_seed.csv')

def paired(a, b):
    i = a.index.intersection(b.index); d = (a[i] - b[i]).values; n = len(d)
    ci = stats.t.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return dict(n=n, diff=d.mean(), lo=d.mean() - ci, hi=d.mean() + ci, p=stats.ttest_rel(a[i], b[i]).pvalue, wins=int((d > 0).sum()))
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj
rows = []
def add(fam, name, r): rows.append(dict(family=fam, comparison=name, **r))

# ---- Q2: learning-rate grid above 0.1 (validation individual accuracy, mean of seeds 100-101) ----
print('=== Q2: validation individual accuracy by lr (TUNE2) ===\n' + T2.to_string(index=False))
q = R[(R.m == 'qgm') & (R.val == 0) & (R.lr == 0.3)].set_index('s')
c10 = C[C.d == 'cifar10']
la = c10[c10.m == 'la_bn'].set_index('s'); q01 = c10[c10.m == 'qgm'].set_index('s')
print('\nQGM lr 0.3 test seeds: ens %.2f +- %.2f, ind %.2f +- %.2f' % (q.e5.mean(), q.e5.std(), q.i5.mean(), q.i5.std()))
for c in ('i5', 'e5'):
    add('Q2', f'qgm(lr0.3) - qgm(lr0.1) [{c}]', paired(q[c], q01[c]))
    add('Q2', f'la_bn(lr0.1) - qgm(lr0.3) [{c}]', paired(la[c], q[c]))

# ---- Q3: zeta^2 and ECE at the averaged model (5-node CNN, seeds 0-4) ----
d1 = R[R.zeta2.notna()] if 'zeta2' in R else R.iloc[0:0]
print('\n=== Q3: diagnostics (mean +- sd over 5 seeds) ===')
print(d1.groupby('m')[['e5', 'i5', 'zeta2', 'grad_norm2', 'mean_grad_norm2', 'ece_node_mean', 'ece_ens']].agg(['mean', 'std']).round(3).T.to_string())
P = {m: g.set_index('s') for m, g in d1.groupby('m')}
fam = []
for y in ('dpsgd', 'fedlc'):
    for c in ('zeta2', 'ece_node_mean', 'ece_ens'):
        fam.append((f'la_bn - {y} [{c}]', paired(P['la_bn'][c], P[y][c])))
ratio = (P['la_bn'].zeta2 / P['dpsgd'].zeta2)
print('zeta2 ratio la/dpsgd per seed:', ratio.round(3).tolist(), 'geo-mean %.3f' % np.exp(np.log(ratio).mean()))
adj = holm(np.array([r['p'] for _, r in fam]))
for (n, r), a in zip(fam, adj): add('Q3', n, {**r, 'p_holm': a})

# ---- Q5: local epochs E, 16-node exponential graph, CNN, seeds 0-2 ----
d3 = R[(R.n == 16) & (R.topo == 'exp') & (R.model == 'cnn')]
print('\n=== Q5: local epochs (mean over 3 seeds) ===')
print(d3.groupby(['E', 'm'])[['e5', 'i5', 'cons']].mean().round(3).to_string())
fam = []
for E in (1, 2, 4):
    g = d3[d3.E == E]; A = g[g.m == 'la_bn'].set_index('s'); B = g[g.m == 'dpsgd'].set_index('s')
    for c in ('i5', 'e5'): fam.append((f'E={E} la_bn - dpsgd [{c}]', paired(A[c], B[c])))
adj = holm(np.array([r['p'] for _, r in fam]))
for (n, r), a in zip(fam, adj): add('Q5', n, {**r, 'p_holm': a})
# does the gain change with E? gain per seed and E, then E4 - E1
gain = {E: (d3[(d3.E == E) & (d3.m == 'la_bn')].set_index('s').i5 - d3[(d3.E == E) & (d3.m == 'dpsgd')].set_index('s').i5) for E in (1, 2, 4)}
for E in (2, 4):
    add('Q5', f'gain(E={E}) - gain(E=1) [i5]', paired(gain[E], gain[1]))

# ---- Q4: CIFAR-100, ResNet-18, 16-node ring, seeds 0-2 ----
d4 = R[(R.ds == 'cifar100') & (R.model == 'resnet18')]
print('\n=== Q4: CIFAR-100 ResNet-18 16-node ring ===')
print(d4.groupby('m')[['e5', 'i5', 'cons']].agg(['mean', 'std']).round(3).T.to_string())
A = d4[d4.m == 'la_bn'].set_index('s'); B = d4[d4.m == 'dpsgd'].set_index('s')
for c in ('i5', 'e5'): add('Q4', f'la_bn - dpsgd [{c}]', paired(A[c], B[c]))

O = pd.DataFrame(rows); O.to_csv(f'{D}/round3D_tests.csv', index=False)
print('\n=== paired tests ===\n' + O.round(4).to_string())
