"""Analysis of the final journal run (notebook fee037c52c): tuning, tuned 5-seed table, feature skew, K-GT LR."""
import json, os, numpy as np, csv
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
from scipy import stats
R = [json.loads(l[4:]) for l in open(f'{D}/final_exp_log_lines.txt') if l.startswith('RES|')]
def get(m, feat='none', val=0, lr=0.1, n=5, seeds=range(5)):
    d = {r['s']: r for r in R if r['m'] == m and r['feat'] == feat and r['val'] == val and r['lr'] == lr and r['n'] == n}
    return [d[s] for s in seeds]
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, (m - k) * ps[i])); adj[i] = run
    return adj
rows = []
print('== tuned 5-seed table (all methods selected lr=0.1 on validation) ==')
M = ['dpsgd', 'kgt', 'qgm', 'fedlc', 'fedrs', 'la_bn']
for m in M:
    x = get(m); e = np.array([r['e5'] for r in x]); i = np.array([r['i5'] for r in x]); c = np.array([r['cons'] for r in x])
    print(f"{m:7s} ens {e.mean():6.2f} ± {e.std(ddof=1):4.2f}   ind {i.mean():6.2f} ± {i.std(ddof=1):4.2f}   cons {c.mean():.4f}")
    rows.append(dict(table='tuned', method=m, ens_mean=round(e.mean(), 2), ens_sd=round(e.std(ddof=1), 2), ind_mean=round(i.mean(), 2), ind_sd=round(i.std(ddof=1), 2), cons=round(c.mean(), 4)))
print('== paired LA - baseline (5 seeds), Holm over 5 baselines per metric ==')
tests = []
for met in ['i5', 'e5']:
    la = np.array([r[met] for r in get('la_bn')]); ps = []; tmp = []
    for b in M[:-1]:
        bb = np.array([r[met] for r in get(b)]); d = la - bb
        t, p = stats.ttest_rel(la, bb); h = stats.t.ppf(0.975, 4) * d.std(ddof=1) / np.sqrt(5)
        tmp.append([met, b, d.mean(), d.mean() - h, d.mean() + h, int((d > 0).sum()), p]); ps.append(p)
    for row, a in zip(tmp, holm(np.array(ps))):
        row.append(a); tests.append(row)
        print(f"{row[0]} vs {row[1]:6s}: {row[2]:+.2f} [{row[3]:+.2f}, {row[4]:+.2f}] wins {row[5]}/5  p={row[6]:.4f}  holm={a:.4f}")
print('== feature skew (rotation by node) + label skew, 5 seeds ==')
for m in ['dpsgd', 'fedlc', 'la_bn']:
    x = get(m, feat='rot'); e = np.array([r['e5'] for r in x]); i = np.array([r['i5'] for r in x])
    print(f"{m:7s} ens {e.mean():6.2f} ± {e.std(ddof=1):4.2f}   ind {i.mean():6.2f} ± {i.std(ddof=1):4.2f}")
    rows.append(dict(table='rot', method=m, ens_mean=round(e.mean(), 2), ens_sd=round(e.std(ddof=1), 2), ind_mean=round(i.mean(), 2), ind_sd=round(i.std(ddof=1), 2), cons=round(np.mean([r['cons'] for r in x]), 4)))
for met in ['i5', 'e5']:
    la = np.array([r[met] for r in get('la_bn', feat='rot')])
    for b in ['dpsgd', 'fedlc']:
        bb = np.array([r[met] for r in get(b, feat='rot')]); d = la - bb; t, p = stats.ttest_rel(la, bb)
        h = stats.t.ppf(0.975, 4) * d.std(ddof=1) / np.sqrt(5)
        print(f"rot {met} LA vs {b}: {d.mean():+.2f} [{d.mean()-h:+.2f}, {d.mean()+h:+.2f}] wins {(d>0).sum()}/5 p={p:.4f}")
        tests.append(['rot_' + met, b, d.mean(), d.mean() - h, d.mean() + h, int((d > 0).sum()), p, float('nan')])
print('== K-GT, ResNet-18, 16-node ring, 100 rounds, lower learning rates (3 seeds) ==')
for lr in (0.05, 0.02):
    x = get('kgt', lr=lr, n=16, seeds=range(3)); e = np.array([r['e5'] for r in x]); i = np.array([r['i5'] for r in x])
    print(f"lr {lr}: ens {e.mean():.2f} ± {e.std(ddof=1):.2f} {list(e)}  ind {i.mean():.2f} ± {i.std(ddof=1):.2f} {list(i)}")
    rows.append(dict(table=f'kgt_rn18_16ring_lr{lr}', method='kgt', ens_mean=round(e.mean(), 2), ens_sd=round(e.std(ddof=1), 2), ind_mean=round(i.mean(), 2), ind_sd=round(i.std(ddof=1), 2), cons=round(np.mean([r['cons'] for r in x]), 3)))
with open(f'{D}/final_exp_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(f'{D}/final_exp_paired_tests.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['metric', 'baseline', 'mean_diff', 'ci_lo', 'ci_hi', 'wins_of_5', 'p', 'p_holm']); w.writerows(tests)
