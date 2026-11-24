"""Round-3E analysis (review round 2, Q1 and Q2): stronger prediction-time controls and heterogeneity at a common model.
Input : round3E_PH2_lines.txt (PH2| lines of notebook journal-round3-E-controls; tiers 23-24).
Output: round3E_per_run.csv, round3E_variants.csv, round3E_tests.csv.

Per run (last five rounds, evaluated on 9,000 test images; 1,000 class-balanced test images are held out to fit biases):
  ind_<v>/ens_<v>: individual / ensemble accuracy of the node models with prediction-time variant v:
     raw; own<t> (minus t*log own smoothed prior, t in 0.1/0.25/0.5/1.0); mix1_<t>, mix2_<t> (minus t*log of the
     neighbourhood-mixture prior W pi and W^2 pi, t in 0.5/1.0); fit (per-node bias fitted on the held-out images).
  avg_raw / avg_fit: accuracy of the averaged model, raw and with a fitted bias.
  zeta2 (own loss), zeta2_ce, zeta2_la: gradient heterogeneity of cross-entropy / logit-adjusted losses at this run's
     averaged model (tier 23 only), so both losses are compared at a common model.

Comparisons (paired by seed; Holm correction over the eight comparisons of each setting): logit-adjusted gossip (raw logits)
minus the best D-PSGD variant that uses label counts only (raw/own/mix), and minus the best D-PSGD variant overall, per
metric; both methods with a fitted bias; the averaged model raw and with a fitted bias. The best variant is chosen on the
evaluation images themselves, which favours D-PSGD; these comparisons are therefore conservative for our method.
Usage: python3 analyse_round3E.py [path to PH2 lines]
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
SRC = sys.argv[1] if len(sys.argv) > 1 else f'{D}/round3E_PH2_lines.txt'
OUT = os.path.dirname(os.path.abspath(SRC)) if len(sys.argv) > 1 else D
R = pd.DataFrame([json.loads(l.split('|', 1)[1]) for l in open(SRC) if l.startswith('PH2|')])
R['setting'] = R['ds'] + '/' + R['model'] + '/n' + R['n'].astype(str)
R.to_csv(f'{OUT}/round3E_per_run.csv', index=False)
VARIANTS = ['raw', 'own0.1', 'own0.25', 'own0.5', 'own1.0', 'mix1_0.5', 'mix1_1.0', 'mix2_0.5', 'mix2_1.0', 'fit']

def paired(a, b):
    i = a.index.intersection(b.index); d = (a[i] - b[i]).values; n = len(d)
    ci = stats.t.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    p = stats.ttest_rel(a[i], b[i]).pvalue if n > 1 else np.nan
    return dict(n=n, diff=d.mean(), lo=d.mean() - ci, hi=d.mean() + ci, p=p, wins=int((d > 0).sum()))
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj

vrows, trows = [], []
for setting, g in R.groupby('setting'):
    dp = g[g.m == 'dpsgd'].set_index('s'); la = g[g.m == 'la_bn'].set_index('s')
    print(f'\n=== {setting}: D-PSGD n={len(dp)}, LA n={len(la)} ===')
    for who, x in (('dpsgd', dp), ('la_bn', la)):
        for v in VARIANTS:
            if f'ind_{v}' in x:
                vrows.append(dict(setting=setting, method=who, variant=v, ind=x[f'ind_{v}'].mean(), ind_sd=x[f'ind_{v}'].std(),
                                  ens=x[f'ens_{v}'].mean(), ens_sd=x[f'ens_{v}'].std()))
        for v in ('avg_raw', 'avg_fit'):
            if v in x: vrows.append(dict(setting=setting, method=who, variant=v, ind=x[v].mean(), ind_sd=x[v].std(), ens=np.nan, ens_sd=np.nan))
    V = pd.DataFrame([r for r in vrows if r['setting'] == setting])
    print(V.round(2).to_string(index=False))
    fam = []
    for metric in ('ind', 'ens'):
        cand = {v: dp[f'{metric}_{v}'].mean() for v in VARIANTS if f'{metric}_{v}' in dp}
        lc = {v: c for v, c in cand.items() if v != 'fit'}; best_lc = max(lc, key=lc.get)   # corrections from label counts only
        fam.append((f'LA raw - best label-count D-PSGD variant ({best_lc}) [{metric}]', paired(la[f'{metric}_raw'], dp[f'{metric}_{best_lc}'])))
        best = max(cand, key=cand.get)
        fam.append((f'LA raw - best D-PSGD variant ({best}) [{metric}]', paired(la[f'{metric}_raw'], dp[f'{metric}_{best}'])))
        fam.append((f'LA fit - D-PSGD fit [{metric}]', paired(la[f'{metric}_fit'], dp[f'{metric}_fit'])))
    fam.append(('LA - D-PSGD, averaged model raw [acc]', paired(la['avg_raw'], dp['avg_raw'])))
    fam.append(('LA - D-PSGD, averaged model with fitted bias [acc]', paired(la['avg_fit'], dp['avg_fit'])))
    adj = holm([r['p'] for _, r in fam])
    for (name, r), a in zip(fam, adj): trows.append(dict(setting=setting, comparison=name, **r, p_holm=a))
    if 'zeta2_ce' in dp and dp['zeta2_ce'].notna().any():
        for who, x in (('D-PSGD model', dp), ('LA model', la)):
            ce, lz = x['zeta2_ce'], x['zeta2_la']; ratio = lz / ce
            print(f'  zeta^2 at the {who}: CE {ce.mean():.3f} +- {ce.std():.3f}, LA {lz.mean():.3f} +- {lz.std():.3f}, '
                  f'ratio LA/CE geo-mean {np.exp(np.log(ratio).mean()):.3f} (range {ratio.min():.3f}-{ratio.max():.3f})')
            trows.append(dict(setting=setting, comparison=f'zeta2 LA loss - CE loss at the {who}', **paired(lz, ce), p_holm=np.nan))
T = pd.DataFrame(trows); T.to_csv(f'{OUT}/round3E_tests.csv', index=False)
pd.DataFrame(vrows).to_csv(f'{OUT}/round3E_variants.csv', index=False)
print('\n=== paired tests (Holm within setting; zeta rows uncorrected) ===')
print(T.round(4).to_string(index=False))
