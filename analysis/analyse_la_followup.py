"""Table 'Heterogeneity level and temperature' (CIFAR-10, five nodes, CNN). Input: la_followup_per_seed.jsonl
(la_bn = tau 1, la_bn_t05 = tau 0.5, la_bn_t2 = tau 2; alpha 0.1 with 5 seeds, alpha 0.5 and 1.0 with 3 seeds)."""
import json, os
import pandas as pd
from scipy import stats
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
F = pd.DataFrame([json.loads(l) for l in open(f'{D}/la_followup_per_seed.jsonl')])
F = F[F.d == 'cifar10']
for a, g in F.groupby('a'):
    base = g[g.m == 'dpsgd'].set_index('s')
    print(f'alpha={a}: D-PSGD ens {base.e5.mean():.2f} ind {base.i5.mean():.2f}')
    for m in ('la_bn_t05', 'la_bn', 'la_bn_t2'):
        x = g[g.m == m].set_index('s')
        if not len(x): continue
        i = x.index.intersection(base.index); dd = x.i5[i] - base.i5[i]
        print(f'   {m:10s} ens {x.e5.mean():.2f} ind {x.i5.mean():.2f}  delta ind {dd.mean():+.2f} (p={stats.ttest_rel(x.i5[i], base.i5[i]).pvalue:.3f}, {int((dd > 0).sum())}/{len(dd)})')
