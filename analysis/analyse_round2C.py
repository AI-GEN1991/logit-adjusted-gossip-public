"""Round-2C analysis: per-node post-hoc control (tier 18, review Q1/W4).
Input : round2C_PHL_lines.txt (raw PHL| lines from notebook journal-round2-c-posthoc-control v1 log).
        round2_combined_per_seed.csv (la_bn, dpsgd from rounds 1-2, same seeds and settings).
Output: posthoc_local_per_seed.csv, posthoc_paired_tests.csv.
Fields: e5/i5 = ensemble / mean individual test accuracy over the last 5 rounds without correction;
        ind1/ens1, ind05/ens05 = same with each node's own smoothed log prior subtracted (tau = 1, 0.5)."""
import json, os
import numpy as np, pandas as pd
from scipy import stats
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
P = pd.DataFrame([json.loads(l.split('|', 1)[1]) for l in open(f'{D}/round2C_PHL_lines.txt') if l.startswith('PHL|')])
P = P.rename(columns={'ds': 'd'}).sort_values(['d', 's'])
P.to_csv(f'{D}/posthoc_local_per_seed.csv', index=False)
R = pd.read_csv(f'{D}/round2_combined_per_seed.csv')

def paired(a, b):
    d = a - b; n = len(d); ci = stats.t.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return dict(n=n, diff=d.mean(), lo=d.mean() - ci, hi=d.mean() + ci, p=stats.ttest_rel(a, b).pvalue, wins=int((d > 0).sum()))
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj

print('=== means +- sd (tier 18, D-PSGD) ===')
print(P.groupby('d')[['e5', 'i5', 'ind1', 'ens1', 'ind05', 'ens05']].agg(['mean', 'std']).round(2).T.to_string())
rows = []
for d in ('cifar10', 'cifar100'):
    p = P[P.d == d].set_index('s')
    la = R[(R.d == d) & (R.m == 'la_bn')].set_index('s').loc[p.index]
    dp = R[(R.d == d) & (R.m == 'dpsgd')].set_index('s').loc[p.index]
    # reproducibility of the uncorrected D-PSGD run against the earlier D-PSGD run (same seed)
    for c in ('e5', 'i5'): rows.append(dict(d=d, comparison=f'dpsgd(tier18) - dpsgd(earlier) [{c}]', **paired(p[c], dp[c]), p_holm=np.nan))
    fam = []
    for metric, la_c, cands in (('individual', 'i5', ('i5', 'ind1', 'ind05')), ('ensemble', 'e5', ('e5', 'ens1', 'ens05'))):
        for c in cands: fam.append((f'la_bn - dpsgd[{c}] ({metric})', paired(la[la_c], p[c])))
    adj = holm(np.array([r['p'] for _, r in fam]))
    for (name, r), a in zip(fam, adj): rows.append(dict(d=d, comparison=name, **r, p_holm=a))
T = pd.DataFrame(rows); T.to_csv(f'{D}/posthoc_paired_tests.csv', index=False)
print('\n=== paired tests (Holm within dataset over the 6 LA comparisons) ===')
print(T.round(4).to_string())
