"""Smoke test of the whole training pipeline on tiny synthetic data (CPU, about a minute): D-PSGD and logit-adjusted gossip,
2 rounds, with diagnostics and the post-hoc controls. Checks that every recorded quantity is present and finite."""
import json, math, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import lagossip as g

def fake_cifar10(ntr=1000, nte_per_class=110, K=10):
    torch.manual_seed(0)
    xtr = torch.randint(0, 255, (ntr, 3, 32, 32), dtype=torch.uint8); ytr = torch.arange(ntr) % K
    xte = torch.randint(0, 255, (nte_per_class * K, 3, 32, 32), dtype=torch.uint8); yte = torch.arange(nte_per_class * K) % K
    return g.GPUData('cifar10', xtr, ytr, xte, yte)

def main():
    data = fake_cifar10()
    for m in ('dpsgd', 'la_bn'):
        spec = dict(dataset='cifar10', alpha=0.1, method=m, seed=0, rounds=2, convs_per_block=1, amp=False, min_node_size=8, posthoc2=1, diag=2)
        r = g.run_experiment(spec, data=data, verbose=False)
        h = r['hist']
        assert len(h['ind_acc']) == 2 and len(h['post2']) == 2
        vals = [*h['ind_acc'], *h['ens_acc'], *h['post2'][-1].values(), h['diag']['zeta2'], h['diag_ce']['zeta2'], h['diag_la']['zeta2']]
        assert all(math.isfinite(v) for v in vals)
        json.dumps(r)
        print(f'{m:6s} ok: ind {h["ind_acc"][-1]:.1f}  ens {h["ens_acc"][-1]:.1f}  zeta2(own) {h["diag"]["zeta2"]:.3f}  post-hoc variants {len(h["post2"][-1])}')
    print('SMOKE TEST PASSED')

if __name__ == '__main__':
    main()
