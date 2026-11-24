"""Round-2 analysis: GUT baseline, CIFAR-100 at 5 seeds, FEMNIST. Inputs: sota_v2_per_seed.csv, round2A_compact.csv."""
import pandas as pd, numpy as np, os
from scipy import stats
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
old = pd.read_csv(f'{D}/sota_v2_per_seed.csv')                       # d,m,s,e5,i5,cons (CIFAR-10 seeds 0-4, CIFAR-100 seeds 0-2)
C, Fm = [], []
for line in open(f'{D}/round2A_compact.csv'):
    if line.startswith('C,'):
        k, d, m, s, val, mu, e5, i5, cons, mb = line.strip().split(',')
        if int(val) == 0: C.append(dict(d=d, m=m, s=int(s), e5=float(e5), i5=float(i5), cons=float(cons) if cons else np.nan))
    elif line.startswith('F,'):
        k, m, s, fa, *v = line.strip().split(','); v = list(map(float, v))
        Fm.append(dict(m=m, s=int(s), fa=float(fa), raw_ind_bal=v[0], raw_ens_bal=v[1], minus_ind_bal=v[2], minus_ens_bal=v[3],
                       raw_ind_nat=v[4], raw_ens_nat=v[5], plus_ind_nat=v[6], plus_ens_nat=v[7]))
new = pd.DataFrame(C); allc = pd.concat([old[['d', 'm', 's', 'e5', 'i5', 'cons']], new], ignore_index=True)
allc = allc.drop_duplicates(['d', 'm', 's'], keep='last')
allc.to_csv(f'{D}/round2_combined_per_seed.csv', index=False)

def paired(A, B):
    i = A.index.intersection(B.index); d = (A[i] - B[i]).values; n = len(d)
    ci = stats.t.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return dict(n=n, diff=d.mean(), lo=d.mean() - ci, hi=d.mean() + ci, p=stats.ttest_rel(A[i], B[i]).pvalue, wins=int((d > 0).sum()))
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj

print('=== mean +- sd per method (individual / ensemble) ===')
g = allc.groupby(['d', 'm']).agg(n=('i5', 'size'), ens=('e5', 'mean'), ens_sd=('e5', 'std'), ind=('i5', 'mean'), ind_sd=('i5', 'std'), cons=('cons', 'mean')).round(2)
print(g.to_string())
rows = []
for d in ('cifar10', 'cifar100'):
    sub = allc[allc.d == d]; piv = {c: sub.pivot(index='s', columns='m', values=c) for c in ('i5', 'e5')}
    for c in ('i5', 'e5'):
        base = [b for b in ('dpsgd', 'kgt', 'qgm', 'fedlc', 'fedrs', 'gut') if b in piv[c]]
        res = [paired(piv[c]['la_bn'], piv[c][b]) for b in base]
        adj = holm(np.array([r['p'] for r in res]))
        for b, r, a in zip(base, res, adj): rows.append(dict(d=d, metric=c, vs=b, **r, p_holm=a))
    for x, y in (('gut_la', 'gut'), ('gut_la', 'la_bn'), ('kgt_la', 'la_bn'), ('qgm_la', 'la_bn')):
        if x in piv['i5'] and y in piv['i5']:
            for c in ('i5', 'e5'): rows.append(dict(d=d, metric=c, vs=f'{x}-{y}', **paired(piv[c][x], piv[c][y]), p_holm=np.nan))
T = pd.DataFrame(rows); T.to_csv(f'{D}/round2_paired_tests.csv', index=False)
print('\n=== paired tests (LA - baseline unless named) ===')
print(T.round(4).to_string())

F = pd.DataFrame(Fm); F.to_csv(f'{D}/femnist_per_seed.csv', index=False)
print('\n=== FEMNIST means ===')
print(F.groupby(['fa', 'm']).mean(numeric_only=True).drop(columns='s').round(2).to_string())
