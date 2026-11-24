"""Section 'Negative results: changing the consensus step'. Inputs: tier1_cifar10_a0.1_per_seed.csv, tiers_2_4_5_per_seed.csv
(Fisher-gated consensus and controls, five-node CNN, CIFAR-10, alpha 0.1), cig_pilot_per_seed.jsonl (class-aware head averaging pilot,
GroupNorm, 3 seeds). Prints the means and paired differences quoted in the paper."""
import json, os
import pandas as pd
from scipy import stats
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
a = pd.read_csv(f'{D}/tier1_cifar10_a0.1_per_seed.csv'); b = pd.read_csv(f'{D}/tiers_2_4_5_per_seed.csv')
df = pd.concat([a[['d', 'a', 'm', 's', 'e5', 'i5']], b[['d', 'a', 'm', 's', 'e5', 'i5']]]).drop_duplicates(['d', 'a', 'm', 's'], keep='last')
d = df[df.a == 0.1]
print(d.groupby('m')[['e5', 'i5']].agg(['mean', 'count']).round(2).to_string())
def pair(x, y, c):
    A = d[d.m == x].set_index('s')[c]; B = d[d.m == y].set_index('s')[c]; i = A.index.intersection(B.index); dd = A[i] - B[i]
    return f'{dd.mean():+.2f} (p={stats.ttest_rel(A[i], B[i]).pvalue:.4f}, {int((dd > 0).sum())}/{len(dd)})'
print('\nFisher gate [0.10,0.20] (dfrc) - D-PSGD, ensemble:', pair('dfrc', 'dpsgd', 'e5'))
print('Fisher gate - same coefficients randomly permuted, ensemble:', pair('dfrc', 'dfrc_randgate', 'e5'))
print('wide Fisher gate - permuted gate, ensemble:', pair('dfrc_wide', 'dfrc_wide_rand', 'e5'), '| prox form:', pair('dfrc_prox', 'dfrc_prox_rand', 'e5'))
C = pd.DataFrame([json.loads(l) for l in open(f'{D}/cig_pilot_per_seed.jsonl')]).set_index(['m', 's'])
print('\nclass-aware head averaging (GN, 3 seeds), individual accuracy:')
print('  alone (chead_gn - dpsgd_gn): %+.2f' % (C.loc['chead_gn'].i5 - C.loc['dpsgd_gn'].i5).mean())
print('  on top of LA (cig_gn - la_gn): %+.2f' % (C.loc['cig_gn'].i5 - C.loc['la_gn'].i5).mean())

# Review round 2, Q3: is the gain of logit adjustment a batch-normalization effect? The pilot also ran D-PSGD and logit
# adjustment with GroupNorm (no batch statistics at all), five nodes on a ring, CIFAR-10, alpha 0.1, seeds 0-2.
la, dp = C.loc['la_gn'], C.loc['dpsgd_gn']; i = la.index.intersection(dp.index); dd = la.i5[i] - dp.i5[i]
ci = stats.t.ppf(.975, len(dd) - 1) * dd.std(ddof=1) / len(dd) ** 0.5
print('\nGroupNorm: D-PSGD ind %.2f ens %.2f | LA ind %.2f ens %.2f' % (dp.i5.mean(), dp.e5.mean(), la.i5.mean(), la.e5.mean()))
print('  LA - D-PSGD individual: %+.2f [%+.2f, %+.2f], p=%.4f, %d/%d seeds; ensemble: %+.2f'
      % (dd.mean(), dd.mean() - ci, dd.mean() + ci, stats.ttest_rel(la.i5[i], dp.i5[i]).pvalue, int((dd > 0).sum()), len(dd), (la.e5[i] - dp.e5[i]).mean()))
