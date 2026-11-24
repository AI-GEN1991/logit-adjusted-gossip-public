"""ResNet-18 sweep with 5 seeds (review Q4/W6). Inputs: sweep_full_per_seed.csv (seeds 0-2),
round2B_RES_lines_session1.txt + round2B_RES_lines_session2.txt (seeds 3-4, notebook journal-round2-b-sweep-seeds v1+v2).
Outputs: sweep5_per_seed.csv, sweep5_paired_tests.csv (LA - baseline, Holm over the 12 comparisons of each metric, as in sweep_paired_tests.csv)."""
import json, os
import numpy as np, pandas as pd
from scipy import stats
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
old = pd.read_csv(f'{D}/sweep_full_per_seed.csv')[['m', 's', 'n', 'topo', 'e5', 'i5', 'cons']]
new = [json.loads(l[4:]) for f in ('round2B_RES_lines_session1.txt', 'round2B_RES_lines_session2.txt') for l in open(f'{D}/{f}') if l.startswith('RES|')]
new = pd.DataFrame(new)[['m', 's', 'n', 'topo', 'e5', 'i5', 'cons']]
A = pd.concat([old, new]).drop_duplicates(['m', 's', 'n', 'topo'], keep='last').sort_values(['n', 'topo', 'm', 's'])
A.to_csv(f'{D}/sweep5_per_seed.csv', index=False)
print(A.groupby(['n', 'topo', 'm']).size().unstack())
print(A.groupby(['n', 'topo', 'm'])[['e5', 'i5', 'cons']].agg(['mean', 'std']).round(2).to_string())
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj
rows = []
for c in ('e5', 'i5'):
    block = []
    for (n, tp), g in A.groupby(['n', 'topo']):
        la = g[g.m == 'la_bn'].set_index('s')[c]
        for y in ('dpsgd', 'kgt', 'fedlc'):
            b = g[g.m == y].set_index('s')[c]; i = la.index.intersection(b.index); d = la[i] - b[i]; k = len(d)
            ci = stats.t.ppf(.975, k - 1) * d.std(ddof=1) / np.sqrt(k)
            block.append(dict(n=n, topo=tp, vs=y, c=c, diff=d.mean(), lo=d.mean() - ci, hi=d.mean() + ci, p=stats.ttest_rel(la[i], b[i]).pvalue, wins=int((d > 0).sum()), k=k))
    adj = holm(np.array([r['p'] for r in block]))
    for r, a in zip(block, adj): r['holm'] = a; rows.append(r)
T = pd.DataFrame(rows); T.to_csv(f'{D}/sweep5_paired_tests.csv', index=False)
print(T.round(4).to_string())
print('total paired wins:', int(T.wins.sum()), 'of', int(T.k.sum()))
