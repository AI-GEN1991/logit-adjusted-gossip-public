"""LaTeX rows of the stronger post-hoc table (tab:posthoc2) and the numbers quoted in Sec. 5.6/5.7, from the outputs of
analyse_round3E.py (round3E_variants.csv, round3E_tests.csv, round3E_per_run.csv). Run analyse_round3E.py first.
Usage: python3 table_round3E.py
"""
import os
import numpy as np, pandas as pd
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
V = pd.read_csv(f'{D}/round3E_variants.csv'); T = pd.read_csv(f'{D}/round3E_tests.csv'); R = pd.read_csv(f'{D}/round3E_per_run.csv')
S = ['cifar10/cnn/n5', 'cifar100/cnn/n5', 'cifar10/resnet18/n16']
OWN = ['own0.1', 'own0.25', 'own0.5', 'own1.0']; MIX = ['mix1_0.5', 'mix1_1.0', 'mix2_0.5', 'mix2_1.0']

def val(s, m, v, metric):
    return float(V[(V.setting == s) & (V.method == m) & (V.variant == v)][metric].iloc[0])
def best(s, m, vs, metric):
    c = {v: val(s, m, v, metric) for v in vs}; b = max(c, key=c.get); return c[b], b
def test(s, name):
    r = T[(T.setting == s) & (T.comparison.str.startswith(name))]
    assert len(r) == 1, (s, name); return r.iloc[0]
f = lambda x: f'{x:.2f}'
sg = lambda x: f'${x:+.2f}$'.replace('+', '+').replace('-', '-')

# check: the best D-PSGD variant of the main comparison is the fitted bias everywhere
for s in S:
    for metric in ('ind', 'ens'):
        r = T[(T.setting == s) & T.comparison.str.contains(f'best D-PSGD variant') & T.comparison.str.endswith(f'[{metric}]')]
        assert '(fit)' in r.comparison.iloc[0], r.comparison.iloc[0]
print('% check passed: best D-PSGD variant = fitted bias in all six cells')

rows = [('D-PSGD, raw logits', 'dpsgd', ['raw']), ('D-PSGD, own prior (best $\\tau$)', 'dpsgd', OWN),
        ('D-PSGD, mixture prior (best)', 'dpsgd', MIX), ('D-PSGD, fitted bias', 'dpsgd', ['fit']),
        ('LA, raw logits', 'la_bn', ['raw']), ('LA, fitted bias', 'la_bn', ['fit'])]
cells = {}
for lab, m, vs in rows:
    for s in S:
        for metric in ('ind', 'ens'):
            cells[(lab, s, metric)] = best(s, m, vs, metric)
colmax = {(s, metric): max(cells[(lab, s, metric)][0] for lab, _, _ in rows) for s in S for metric in ('ind', 'ens')}
print('% ---- table rows ----')
for lab, m, vs in rows:
    out = []
    for s in S:
        for metric in ('ens', 'ind'):
            x = cells[(lab, s, metric)][0]; t = f(x)
            out.append(f'\\textbf{{{t}}}' if abs(x - colmax[(s, metric)]) < 1e-9 else t)
    print(f'{lab} & ' + ' & '.join(out) + ' \\\\')
    if len(vs) > 1:
        print('%   chosen: ' + ', '.join(f'{s}/{metric}={cells[(lab, s, metric)][1]}' for s in S for metric in ('ind', 'ens')))
print('\\midrule')
for lab, m in (('Averaged model, D-PSGD: raw / fitted', 'dpsgd'), ('Averaged model, LA: raw / fitted', 'la_bn')):
    print(f'{lab} & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{f(val(s, m, "avg_raw", "ind"))} / {f(val(s, m, "avg_fit", "ind"))}}}' for s in S) + ' \\\\')
print('\\midrule')
sg2 = lambda x: f'${x:.2f}$'
for lab, name in (('$\\Delta$ LA raw $-$ D-PSGD best label-count', 'LA raw - best label-count D-PSGD variant'),
                  ('$\\Delta$ LA raw $-$ D-PSGD fitted', 'LA raw - best D-PSGD variant'),
                  ('$\\Delta$ LA fitted $-$ D-PSGD fitted', 'LA fit - D-PSGD fit')):
    dd, ci, pp = [], [], []
    for s in S:
        for metric in ('ens', 'ind'):
            r = T[(T.setting == s) & T.comparison.str.startswith(name) & T.comparison.str.endswith(f'[{metric}]')].iloc[0]
            dd.append(f'{sg(r["diff"])} ({int(r["wins"])}/{int(r["n"])})'); ci.append(f'[{sg2(r["lo"])}, {sg2(r["hi"])}]'); pp.append(f'{r["p_holm"]:.3f}')
    print(f'{lab} & ' + ' & '.join(dd) + ' \\\\')
    print('\\quad 95\\% CI & ' + ' & '.join(ci) + ' \\\\')
    print('\\quad $p_{\\mathrm{Holm}}$ & ' + ' & '.join(pp) + ' \\\\')

print('\n% ---- numbers for the text ----')
for s in S:
    print(f'% {s}')
    for name in ('LA - D-PSGD, averaged model raw', 'LA - D-PSGD, averaged model with fitted bias'):
        r = test(s, name); print(f'%   {name}: {r["diff"]:+.2f} [{r["lo"]:+.2f}, {r["hi"]:+.2f}] wins {int(r["wins"])}/{int(r["n"])} p={r["p"]:.4f} p_holm={r["p_holm"]:.3f}')
    for metric in ('ind', 'ens'):
        for nm in ('LA raw - best label-count D-PSGD variant', 'LA raw - best D-PSGD variant'):
            r = T[(T.setting == s) & T.comparison.str.startswith(nm) & T.comparison.str.endswith(f'[{metric}]')].iloc[0]
            print(f'%   {r["comparison"]}: {r["diff"]:+.2f} [{r["lo"]:+.2f}, {r["hi"]:+.2f}] wins {int(r["wins"])}/{int(r["n"])} p={r["p"]:.4f} p_holm={r["p_holm"]:.3f}')
    raw, fit, la = val(s, 'dpsgd', 'raw', 'ind'), val(s, 'dpsgd', 'fit', 'ind'), val(s, 'la_bn', 'raw', 'ind')
    mix, mixv = best(s, 'dpsgd', MIX, 'ind'); own, ownv = best(s, 'dpsgd', OWN, 'ind')
    print(f'%   ind: D-PSGD raw {raw:.2f}, best own {own:.2f} ({ownv}), best mix {mix:.2f} ({mixv}), fit {fit:.2f}; LA raw {la:.2f}, LA fit {val(s, "la_bn", "fit", "ind"):.2f}')
    print(f'%   LA gain over raw {la - raw:.2f}; fitted bias gain over raw {fit - raw:.2f} ({100 * (fit - raw) / (la - raw):.0f}% of the LA gain); LA raw - best mix {la - mix:.2f}; LA raw - best own {la - own:.2f}')
    print(f'%   LA with fitted bias - LA raw: ind {val(s, "la_bn", "fit", "ind") - la:+.2f}')
    for who in ('D-PSGD model', 'LA model'):
        z = T[(T.setting == s) & (T.comparison == f'zeta2 LA loss - CE loss at the {who}')]
        if len(z):
            x = R[(R.setting == s) & (R.m == ('dpsgd' if who == 'D-PSGD model' else 'la_bn'))]
            ratio = x.zeta2_la / x.zeta2_ce; z = z.iloc[0]
            print(f'%   zeta2 at the {who}: LA {x.zeta2_la.mean():.2f} +- {x.zeta2_la.std():.2f}, CE {x.zeta2_ce.mean():.2f} +- {x.zeta2_ce.std():.2f}; '
                  f'diff {z["diff"]:.1f} [{z["lo"]:.1f}, {z["hi"]:.1f}] p={z["p"]:.4f} wins(LA>CE) {int(z["wins"])}/{int(z["n"])}; ratio geo {np.exp(np.log(ratio).mean()):.3f} '
                  f'(1/{1 / np.exp(np.log(ratio).mean()):.1f}), range {ratio.min():.3f}-{ratio.max():.3f}; ratio of means {x.zeta2_ce.mean() / x.zeta2_la.mean():.1f}')
            print(f'%     mean-gradient norm^2 at the {who}: LA {x.gbar2_la.mean():.2f}, CE {x.gbar2_ce.mean():.2f}')

# agreement of the new runs with Table 5 (tab:posthoc): D-PSGD from posthoc_local_per_seed.csv (notebook C),
# logit-adjusted gossip from round2_combined_per_seed.csv (5 seeds, as in Table 5)
pc = pd.read_csv(f'{D}/posthoc_local_per_seed.csv'); rc = pd.read_csv(f'{D}/round2_combined_per_seed.csv'); dm = 0
for ds in ('cifar10', 'cifar100'):
    for m, src in (('dpsgd', pc), ('la_bn', rc)):
        x = src[(src.d == ds) & (src.m == m)]; assert len(x) == 5, (ds, m, len(x))
        s = f'{ds}/cnn/n5'; de, di = val(s, m, 'raw', 'ens') - x.e5.mean(), val(s, m, 'raw', 'ind') - x.i5.mean()
        dm = max(dm, abs(de), abs(di)); print(f'% Table 5 check {s} {m}: ens {de:+.2f}, ind {di:+.2f}')
print(f'% largest |difference| from Table 5: {dm:.2f}')
