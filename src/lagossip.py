# Logit-adjusted gossip: experiment library (all methods, partitions, graphs, training loop, diagnostics) and CLI.
import os, json, time, math, copy, random, glob
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.func import functional_call, vmap, grad

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    torch.backends.cudnn.benchmark = True

DATASETS = {
    'cifar10':  dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), classes=10,  ch=3),
    'cifar100': dict(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762), classes=100, ch=3),
    'fmnist':   dict(mean=(0.2860,), std=(0.3530,), classes=10, ch=1),
    # FEMNIST (LEAF, by writer; Hugging Face flwrlabs/femnist). Images inverted to a black background, padded to 32x32.
    # mean/std are overwritten with the values of the loaded data; no horizontal flips (characters).
    'femnist':  dict(mean=(0.1,), std=(0.3,), classes=62, ch=1, flip=False),
}

# Paper hyper-parameters (Table 1). Change here, not inside the functions.
DEFAULTS = dict(
    dataset='cifar10', alpha=0.1, seed=0, method='dfrc',
    nodes=5, rounds=50, local_epochs=1, batch=64,
    lr=0.1, momentum=0.9, wd=5e-4, clip=10.0, amp=True,
    # D-FRC
    beta=0.9, eps=1e-8, delta=1e-6, lam_min=0.10, lam_max=0.20, gamma=0.25, mu=1.0,
    warm=5, m_warm=1.0,            # Algorithm 1: pure Euclidean warm-up (m = 1). The text says 0.5: check your code.
    gate='bounded',                # 'bounded': lam_min + (lam_max-lam_min)*sigmoid  -> range [0.10, 0.20] (Table 1)
                                   # 'paper'  : sigmoid*lam_max + lam_min             -> range [0.10, 0.30] (Algorithm 1)
    gate_mode='fisher',            # 'fisher' | 'random' (ablation) | 'inverted' (ablation)
    n_fisher=512, fisher_chunk=128,
    # baselines
    lazy_m=0.16,                   # constant mixing weight = the mean lambda D-FRC reported
    prox_mu=0.01,                  # decentralized FedProx
    kgt_scale=1.0,                 # K-GT correction scale (1 = standard)
    rho=1.0,                       # gate='prox': m = rho / (F_tilde + rho), the closed form of the Fisher-Rao proximal step
    convs_per_block=2, min_node_size=64,
)

# CIG (class-aware information-weighted gossip) options
DEFAULTS.update(norm='bn',        # 'bn' BatchNorm | 'gn' GroupNorm(8)
                la_tau=0.0,       # logit adjustment: train on CE(z + tau * log pi_i, y); 0 = off
                head_mix='avg',   # 'avg' plain gossip | 'class': row c of the classifier averaged with weights W_ij * n_j(c)
                head_eps=1.0,     # pseudo-count added to every class count
                lc_tau=0.0,       # FedLC-style logit calibration: train on z_c - tau * (n_c + 1)^(-1/4); 0 = off
                rs_alpha=1.0,
                model='cnn',      # 'cnn' (small VGG-like) | 'resnet18' (CIFAR variant)
                eval_every=1,
                topo='ring',
                val=0,            # >0: hold out this many training samples (fixed split) as the evaluation set, for tuning
                feat_skew='none',
                fem_wpn=20,       # FEMNIST: writers per node (writer partition)
                fem_alpha=0.0,    # FEMNIST: >0 adds label skew: node i keeps a training sample of class c with prob p_i(c)/max p_i, p_i ~ Dir(fem_alpha)
                fem_test=0.2,
                gut_mu=0.9,
                posthoc_local=0,
                posthoc2=0,       # 1: stronger post-hoc controls (own prior, tau grid; neighbourhood-mixture prior; fitted bias), averaged-model accuracy
                diag=0)           # 1: after the last round, measure gradient heterogeneity at the averaged model and calibration (ECE)  # 1: also evaluate every node with its own log prior subtracted at prediction time (tau 1 and 0.5), last 5 rounds       # Global Update Tracking (Aketi et al., 2023): scaling factor of the tracking correction; 0 = D-PSGD     # FEMNIST: fraction of each writer's samples held out for the test set # 'rot': node i sees its images rotated by 90*(i mod 4) degrees; test images get all 4 rotations      # 'ring' (2 neighbours) | 'exp' (exponential graph: neighbours at distance 1, 2, 4, ...)     # evaluate every k rounds (always in the last 5); skipped rounds are stored as NaN     # FedRS-style restricted softmax: logits of locally absent classes scaled by alpha; 1 = off
# Method presets. The ablations reuse the D-FRC code path.
METHODS = {
    'dpsgd':         dict(),
    'dpsgd_lazy':    dict(),
    'dprox':         dict(),
    'kgt':           dict(),
    'fw_gossip':     dict(),
    'dfrc':          dict(),
    'dfrc_noema':    dict(beta=0.0),
    'dfrc_nowarm':   dict(warm=0),
    'dfrc_randgate': dict(gate_mode='random'),
    'dfrc_invgate':  dict(gate_mode='inverted'),
    # Tier 5: stronger-mixing D-FRC variants, each with a matched control that keeps the same gate values
    # but destroys the link between Fisher value and parameter (same mixing strength, no information).
    'dfrc_prox':      dict(gate='prox'),
    'dfrc_prox_rand': dict(gate='prox', gate_mode='random'),
    'dfrc_wide':      dict(lam_min=0.10, lam_max=1.00),
    'dfrc_wide_rand': dict(lam_min=0.10, lam_max=1.00, gate_mode='random'),
    # CIG pilot ('base' selects the communication code path)
    'dpsgd_gn':     dict(base='dpsgd', norm='gn'),
    'la_gn':        dict(base='dpsgd', norm='gn', la_tau=1.0),
    'chead_gn':     dict(base='dpsgd', norm='gn', head_mix='class'),
    'cig_gn':       dict(base='dpsgd', norm='gn', la_tau=1.0, head_mix='class'),
    'cig_bn':       dict(base='dpsgd', la_tau=1.0, head_mix='class'),
    'kgt_gn':       dict(base='kgt', norm='gn'),
    'fw_gossip_gn': dict(base='fw_gossip', norm='gn'),
    # logit-adjusted gossip (LA) with BatchNorm: follow-up to the pilot
    'la_bn':     dict(base='dpsgd', la_tau=1.0),
    'la_bn_t05': dict(base='dpsgd', la_tau=0.5),
    'la_bn_t2':  dict(base='dpsgd', la_tau=2.0),
    # state-of-the-art baselines and their combination with logit adjustment
    'kgt_la':    dict(base='kgt', la_tau=1.0),
    'qgm':       dict(base='qgm'),                    # quasi-global momentum (Lin et al., 2021), local-step version
    'qgm_la':    dict(base='qgm', la_tau=1.0),
    'fedlc':     dict(base='dpsgd', lc_tau=1.0),      # FedLC logit calibration (Zhang et al., 2022) inside gossip
    'fedrs':     dict(base='dpsgd', rs_alpha=0.5),    # FedRS restricted softmax (Li and Zhan, 2021) inside gossip
    'gut':       dict(base='gut'),                    # Global Update Tracking (Aketi et al., 2023), local-epoch version
    'gut_la':    dict(base='gut', la_tau=1.0),
}
LABELS = {'dpsgd': 'D-PSGD', 'dpsgd_lazy': 'D-PSGD (m=0.16)', 'dprox': 'D-FedProx', 'kgt': 'K-GT',
          'fw_gossip': 'Fisher-weighted gossip', 'dfrc': 'D-FRC', 'dfrc_noema': 'D-FRC w/o EMA',
          'dfrc_nowarm': 'D-FRC w/o warm-up', 'dfrc_randgate': 'D-FRC random gate',
          'dfrc_invgate': 'D-FRC inverted gate', 'dfrc_prox': 'D-FRC (prox gate)', 'dfrc_prox_rand': 'D-FRC (prox gate, shuffled)',
          'dfrc_wide': 'D-FRC (gate 0.1-1.0)', 'dfrc_wide_rand': 'D-FRC (gate 0.1-1.0, shuffled)'}
def base_method(m): return METHODS.get(m, {}).get('base') or ('dfrc' if m.startswith('dfrc') else m)

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# ---------------- data (kept on the GPU as uint8, augmented on the GPU) ----------------
def gpu_augment(x, pad=4, flip=True):
    # RandomCrop(32, padding=4) + RandomHorizontalFlip, applied per sample on the GPU
    B, C, H, W = x.shape; d = x.device
    xp = F.pad(x, (pad, pad, pad, pad))
    i = torch.randint(0, 2 * pad + 1, (B,), device=d); j = torch.randint(0, 2 * pad + 1, (B,), device=d)
    r = i[:, None] + torch.arange(H, device=d)
    xp = xp.gather(2, r[:, None, :, None].expand(B, C, H, W + 2 * pad))
    c = j[:, None] + torch.arange(W, device=d)
    xp = xp.gather(3, c[:, None, None, :].expand(B, C, H, W))
    if not flip: return xp
    fl = torch.rand(B, device=d) < 0.5
    return torch.where(fl[:, None, None, None], xp.flip(3), xp)

class GPUData:
    def __init__(self, name, xtr, ytr, xte, yte):
        st = DATASETS[name]; self.name = name; self.classes = st['classes']; self.ch = st['ch']; self.flip = st.get('flip', True)
        self.xtr, self.ytr = xtr.to(DEVICE), ytr.to(DEVICE)
        self.xte, self.yte = xte.to(DEVICE), yte.to(DEVICE)
        self.ytr_cpu = ytr.numpy()
        self.mean = torch.tensor(st['mean'], device=DEVICE).view(1, -1, 1, 1)
        self.std = torch.tensor(st['std'], device=DEVICE).view(1, -1, 1, 1)
    def prep(self, xb, augment):
        x = xb.float().div_(255)
        if augment: x = gpu_augment(x, flip=self.flip)
        return (x - self.mean) / self.std

_DATA = {}
def get_data(name, root=None):
    if name in _DATA: return _DATA[name]
    _DATA.clear(); torch.cuda.empty_cache() if DEVICE == 'cuda' else None
    import torchvision
    root = root or os.environ.get('DFRC_DATA_ROOT', './data')
    cls = {'cifar10': torchvision.datasets.CIFAR10, 'cifar100': torchvision.datasets.CIFAR100,
           'fmnist': torchvision.datasets.FashionMNIST}[name]
    def conv(ds):
        x = torch.as_tensor(np.asarray(ds.data))
        if x.ndim == 3: x = x.unsqueeze(-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        if x.shape[-1] == 28: x = F.pad(x, (2, 2, 2, 2))   # Fashion-MNIST 28x28 -> 32x32
        return x, torch.as_tensor(np.asarray(ds.targets), dtype=torch.long)
    xtr, ytr = conv(cls(root, train=True, download=True))
    xte, yte = conv(cls(root, train=False, download=True))
    _DATA[name] = GPUData(name, xtr, ytr, xte, yte)
    return _DATA[name]

_FEM = {}
def load_femnist(path=None):
    # all of FEMNIST as uint8 arrays on the CPU, cached as .npz next to the parquet file
    if 'X' in _FEM: return _FEM
    path = path or os.environ.get('FEMNIST_PARQUET', './data/femnist/train.parquet')
    cache = path.replace('.parquet', '_u8.npz')
    if os.path.exists(cache):
        z = np.load(cache); X, y, w = z['X'], z['y'], z['w']
    else:
        import io, pyarrow.parquet as pq
        from PIL import Image
        t = pq.read_table(path).to_pydict()
        X = np.stack([255 - np.asarray(Image.open(io.BytesIO(im['bytes'])).convert('L'), dtype=np.uint8) for im in t['image']])
        y = np.asarray(t['character'], dtype=np.int64)
        _, w = np.unique(np.asarray(t['writer_id']), return_inverse=True)
        np.savez(cache, X=X, y=y, w=w)
    X = np.pad(X, ((0, 0), (2, 2), (2, 2)))[:, None]                     # 28x28 -> 32x32, channel dim
    m = float(X.mean() / 255); sd = float(X.std() / 255)
    DATASETS['femnist'].update(mean=(m,), std=(sd,))
    _FEM.update(X=X, y=y, w=w)
    return _FEM

def femnist_view(cfg):
    # writer partition: node i holds fem_wpn whole writers; each writer's samples are split (1-fem_test)/fem_test into
    # train/test; the test set is the union of the held-out samples of all selected writers (natural label distribution).
    F_ = load_femnist(); X, y, w = F_['X'], F_['y'], F_['w']
    rng = np.random.default_rng(1000 + cfg['seed']); N, K = cfg['nodes'], 62
    cnt = np.bincount(w); elig = np.where(cnt >= 100)[0]
    writers = rng.choice(elig, N * cfg['fem_wpn'], replace=False)
    order = np.argsort(w, kind='stable'); starts = np.concatenate([[0], np.cumsum(cnt)])
    tr_parts, te_idx = [], []
    for i in range(N):
        tr = []
        for wr in writers[i * cfg['fem_wpn']:(i + 1) * cfg['fem_wpn']]:
            ix = order[starts[wr]:starts[wr + 1]].copy(); rng.shuffle(ix)
            nt = int(round(cfg['fem_test'] * len(ix))); te_idx.extend(ix[:nt].tolist()); tr.extend(ix[nt:].tolist())
        tr = np.array(tr)
        if cfg['fem_alpha'] > 0:
            for _ in range(1000):
                p = rng.dirichlet(cfg['fem_alpha'] * np.ones(K)); keep = rng.random(len(tr)) < (p / p.max())[y[tr]]
                if keep.sum() >= cfg['min_node_size']: break
            tr = tr[keep]
        tr_parts.append(np.sort(tr))
    all_tr = np.concatenate(tr_parts); te = np.array(sorted(te_idx))
    offs = np.cumsum([0] + [len(t) for t in tr_parts])
    parts = [np.arange(offs[i], offs[i + 1]) for i in range(N)]
    data = GPUData('femnist', torch.as_tensor(X[all_tr]), torch.as_tensor(y[all_tr]), torch.as_tensor(X[te]), torch.as_tensor(y[te]))
    return data, parts

def dirichlet_partition(y, n_nodes, alpha, seed, min_size=64):
    # For each class k: p_k ~ Dir(alpha * 1_N), split class-k samples proportionally (Hsu et al. 2019).
    # Re-draws until every node has >= min_size samples (state this in the paper).
    rng = np.random.default_rng(seed); y = np.asarray(y); K = int(y.max()) + 1
    for _ in range(1000):
        idx = [[] for _ in range(n_nodes)]
        for k in range(K):
            ik = np.where(y == k)[0]; rng.shuffle(ik)
            p = rng.dirichlet(alpha * np.ones(n_nodes))
            cuts = (np.cumsum(p) * len(ik)).astype(int)[:-1]
            for n, part in enumerate(np.split(ik, cuts)): idx[n].extend(part.tolist())
        if min(len(a) for a in idx) >= min_size: break
    return [np.array(sorted(a)) for a in idx]

# ---------------- model ----------------
class CNN(nn.Module):
    # VGG-like: 3 blocks (32, 64, 128) with BN + Dropout(0.25); Linear(2048,256)-BN-Dropout(0.5)-Linear.
    # NOTE: the paper reports 696,618 parameters. If you still have your original model class, paste it here
    # instead so the numbers stay comparable.
    def __init__(self, in_ch=3, n_classes=10, convs_per_block=2, width=(32, 64, 128), hidden=256, norm='bn'):
        super().__init__()
        layers, c = [], in_ch
        for w in width:
            for _ in range(convs_per_block):
                layers += [nn.Conv2d(c, w, 3, padding=1), nn.BatchNorm2d(w) if norm == 'bn' else nn.GroupNorm(8, w), nn.ReLU()]; c = w
            layers += [nn.MaxPool2d(2), nn.Dropout(0.25)]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c * 16, hidden), nn.BatchNorm1d(hidden) if norm == 'bn' else nn.GroupNorm(8, hidden),
                                        nn.ReLU(), nn.Dropout(0.5), nn.Linear(hidden, n_classes))
    def forward(self, x): return self.classifier(self.features(x))

class _Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sh = None if stride == 1 and cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
    def forward(self, x):
        o = F.relu(self.b1(self.c1(x))); o = self.b2(self.c2(o))
        return F.relu(o + (x if self.sh is None else self.sh(x)))

class ResNet18(nn.Module):
    # CIFAR variant: 3x3 stem, no max-pool, widths 64-128-256-512, two basic blocks per stage (about 11.2 M parameters)
    def __init__(self, in_ch=3, n_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        layers, c = [], 64
        for w, st in ((64, 1), (128, 2), (256, 2), (512, 2)):
            layers += [_Block(c, w, st), _Block(w, w, 1)]; c = w
        self.layers = nn.Sequential(*layers); self.fc = nn.Linear(512, n_classes)
    def forward(self, x):
        return self.fc(F.adaptive_avg_pool2d(self.layers(self.stem(x)), 1).flatten(1))

def make_model(cfg, data):
    if cfg['model'] == 'resnet18': return ResNet18(data.ch, data.classes)
    return CNN(data.ch, data.classes, cfg['convs_per_block'], norm=cfg['norm'])

def n_params(m): return sum(p.numel() for p in m.parameters())
def float_buffers(m): return [b for b in m.buffers() if b.dtype.is_floating_point]

# ---------------- topology ----------------
def ring_W(n):
    W = np.zeros((n, n))
    for i in range(n):
        for j in (i - 1, i, i + 1): W[i, j % n] += 1 / 3
    return W
def exp_W(n):
    # symmetric exponential graph: node i is linked to i +- 2^k (k = 0, 1, ...), uniform weights 1/(degree + 1)
    nb = set()
    k = 1
    while k < n:
        nb.add(k % n); nb.add((-k) % n); k *= 2
    nb.discard(0); deg = len(nb)
    W = np.zeros((n, n))
    for i in range(n):
        W[i, i] = 1 / (deg + 1)
        for o in nb: W[i, (i + o) % n] = 1 / (deg + 1)
    return W

def make_W(cfg):
    return exp_W(cfg['nodes']) if cfg['topo'] == 'exp' else ring_W(cfg['nodes'])

def spectral_gap(W):
    ev = sorted(np.abs(np.linalg.eigvals(W)), reverse=True); return float(1 - ev[1])

@torch.no_grad()
def mix_tensors(lists, Wt):
    # lists[i][k] = tensor k of node i  ->  out[i][k] = sum_j W_ij lists[j][k]
    N, out = len(lists), [[None] * len(lists[0]) for _ in range(len(lists))]
    for k in range(len(lists[0])):
        S = torch.stack([lists[j][k].float() for j in range(N)])
        M = torch.tensordot(Wt, S, dims=([1], [0]))
        for i in range(N): out[i][k] = M[i]
    return out

# ---------------- Fisher ----------------
def diag_fisher(model, data, idx, n_samples, chunk):
    # Monte-Carlo diagonal Fisher: y ~ p_theta(y|x), F = E[(d log p / d theta)^2], exact per-sample grads via vmap.
    model.eval()
    sel = idx[torch.randperm(len(idx), device=idx.device)[:n_samples]]
    names = [n for n, _ in model.named_parameters()]
    params = {n: p.detach() for n, p in model.named_parameters()}
    buffers = {n: b.detach() for n, b in model.named_buffers()}
    def loss_fn(prm, x, y):
        out = functional_call(model, (prm, buffers), (x.unsqueeze(0),))
        return F.cross_entropy(out, y.unsqueeze(0))
    per_sample_grad = vmap(grad(loss_fn), in_dims=(None, 0, 0))
    acc = {n: torch.zeros_like(p) for n, p in params.items()}
    for s in range(0, len(sel), chunk):
        x = data.prep(data.xtr[sel[s:s + chunk]], augment=False)
        with torch.no_grad():
            y = torch.multinomial(F.softmax(model(x).float(), 1), 1).squeeze(1)
        g = per_sample_grad(params, x, y)
        for n in acc: acc[n] += g[n].pow(2).sum(0)
    model.train()
    return [acc[n] / len(sel) for n in names]

def _flat(ts): return torch.cat([t.flatten() for t in ts])
def _unflat(v, like):
    out, o = [], 0
    for t in like: out.append(v[o:o + t.numel()].view_as(t)); o += t.numel()
    return out

def normalize_fisher(Fbar, eps, delta):
    med = _flat(Fbar).median()
    return [torch.clamp(f / (med + eps), min=delta) for f in Fbar]

def fisher_gate(Ft, cfg):
    if cfg['gate_mode'] == 'random':        # same distribution of values, Fisher-parameter link destroyed
        v = _flat(Ft); Ft = _unflat(v[torch.randperm(v.numel(), device=v.device)], Ft)
    if cfg['gate'] == 'prox':               # closed form of the proximal step: range (0, 1], median weight -> rho / (1 + rho)
        if cfg['gate_mode'] == 'inverted': return [f / (f + cfg['rho']) for f in Ft]
        return [cfg['rho'] / (f + cfg['rho']) for f in Ft]
    g = -cfg['gamma'] if cfg['gate_mode'] == 'inverted' else cfg['gamma']
    out = []
    for f in Ft:
        s = torch.sigmoid(-g * (f - cfg['mu']))
        out.append(s * cfg['lam_max'] + cfg['lam_min'] if cfg['gate'] == 'paper'
                   else cfg['lam_min'] + (cfg['lam_max'] - cfg['lam_min']) * s)
    return out

# ---------------- evaluation ----------------
@torch.no_grad()
def evaluate(models, data, cfg, bs=1000, offsets=None):
    # offsets: optional per-node logit offsets added at prediction time (used for the per-node post-hoc control)
    for m in models: m.eval()
    N = len(models); correct = torch.zeros(N, device=DEVICE); ens = torch.zeros((), device=DEVICE)
    for s in range(0, len(data.yte), bs):
        x = data.prep(data.xte[s:s + bs], False); y = data.yte[s:s + bs]; psum = 0
        for i, m in enumerate(models):
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=cfg['amp'] and DEVICE == 'cuda'):
                z = m(x).float()
            p = F.softmax(z if offsets is None else z + offsets[i], 1)
            correct[i] += (p.argmax(1) == y).sum(); psum = psum + p
        ens += (psum.argmax(1) == y).sum()
    for m in models: m.train()
    n = len(data.yte)
    return (correct / n * 100).tolist(), float(ens / n * 100)

@torch.no_grad()
def evaluate_ext(models, data, cfg, log_pg, bs=1000):
    # natural and class-balanced accuracy, individual (mean over nodes) and ensemble, for three prediction-time offsets:
    # 'raw' (logits as trained), 'plus' (+ log of the global training prior), 'minus' (- log of the global training prior).
    for m in models: m.eval()
    offs = {'raw': 0.0, 'plus': log_pg, 'minus': -log_pg}; K = data.classes; N = len(models)
    pred = {k: [[] for _ in range(N + 1)] for k in offs}
    for s in range(0, len(data.yte), bs):
        x = data.prep(data.xte[s:s + bs], False)
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=cfg['amp'] and DEVICE == 'cuda'):
            Z = [m(x).float() for m in models]
        for k, o in offs.items():
            ps = 0
            for i, z in enumerate(Z):
                p = F.softmax(z + o, 1); pred[k][i].append(p.argmax(1)); ps = ps + p
            pred[k][N].append(ps.argmax(1))
    for m in models: m.train()
    y = data.yte; nc = torch.bincount(y, minlength=K).float(); present = nc > 0; out = {}
    for k in offs:
        nat, bal = [], []
        for i in range(N + 1):
            pr = torch.cat(pred[k][i]); ok = (pr == y).float()
            nat.append(float(ok.mean() * 100))
            rc = torch.bincount(y, weights=ok, minlength=K)
            bal.append(float((rc[present] / nc[present]).mean() * 100))
        out[k] = dict(ind_nat=float(np.mean(nat[:N])), ens_nat=nat[N], ind_bal=float(np.mean(bal[:N])), ens_bal=bal[N])
    return out

# ---------------- one local epoch ----------------
def local_epoch(model, opt, scaler, data, idx, cfg, base, anchor=None, corr=None, prior=None, scale=None):
    model.train(); params = list(model.parameters())
    perm = idx[torch.randperm(len(idx), device=idx.device)]
    loss_sum, correct, seen, lr_sum = 0.0, 0, 0, 0.0
    use_amp = cfg['amp'] and DEVICE == 'cuda'
    for s in range(0, len(perm), cfg['batch']):
        b = perm[s:s + cfg['batch']]
        if len(b) < 2: continue                      # BatchNorm1d needs >1 sample
        x = data.prep(data.xtr[b], True); y = data.ytr[b]
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=use_amp):
            out = model(x); z = out if (prior is None and scale is None) else out.float()
            if scale is not None: z = z * scale
            if prior is not None: z = z + prior
            loss = F.cross_entropy(z, y)
        if base == 'dprox':
            loss = loss.float() + 0.5 * cfg['prox_mu'] * sum((p - a).pow(2).sum() for p, a in zip(params, anchor))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        if corr is not None:                         # K-GT tracking correction
            for p, c in zip(params, corr): p.grad.add_(c)
        torch.nn.utils.clip_grad_norm_(params, cfg['clip'])
        scaler.step(opt); scaler.update()
        lr_sum += opt.param_groups[0]['lr']
        loss_sum += float(loss.detach()) * len(b); correct += int((out.argmax(1) == y).sum()); seen += len(b)
    return loss_sum, correct, seen, lr_sum

@torch.no_grad()
def _avg_model(models):
    avg = copy.deepcopy(models[0])
    for name, t in avg.state_dict().items():
        if t.is_floating_point():
            t.copy_(torch.stack([m.state_dict()[name].float() for m in models]).mean(0).to(t.dtype))
    return avg

def diagnostics(models, data, idxs, cfg, priors, scales, n_per=2048, chunk=256, bins=15):
    """Empirical check of the theory (reviewer question): gradient heterogeneity of the local TRAINING losses at the
    averaged model, zeta^2 = mean_i ||g_i - mean_j g_j||^2 with g_i the gradient of node i's loss on up to n_per of its
    samples (no augmentation, eval-mode BatchNorm), and the expected calibration error (ECE, 15 bins) of the node models
    and of the ensemble on the balanced test set."""
    avg = _avg_model(models); avg.eval(); params = [p for p in avg.parameters()]
    G = []
    gen = torch.Generator(device='cpu').manual_seed(12345)
    for i, idx in enumerate(idxs):
        sel = idx[torch.randperm(len(idx), generator=gen)[:n_per].to(idx.device)]
        acc = [torch.zeros_like(p) for p in params]
        for s0 in range(0, len(sel), chunk):
            b = sel[s0:s0 + chunk]; x = data.prep(data.xtr[b], False); y = data.ytr[b]
            z = avg(x).float()
            if scales[i] is not None: z = z * scales[i]
            if priors[i] is not None: z = z + priors[i]
            loss = F.cross_entropy(z, y, reduction='sum') / len(sel)
            gr = torch.autograd.grad(loss, params)
            for a, g in zip(acc, gr): a.add_(g)
        G.append(torch.cat([a.flatten() for a in acc]))
    G = torch.stack(G); gm = G.mean(0)
    zeta2 = float(((G - gm) ** 2).sum(1).mean()); gn2 = float((G ** 2).sum(1).mean()); gbar2 = float((gm ** 2).sum())
    # calibration
    def ece_of(P, y):
        conf, pred = P.max(1); ok = (pred == y).float(); e = torch.zeros((), device=P.device)
        edges = torch.linspace(0, 1, bins + 1, device=P.device)
        for k in range(bins):
            m = (conf > edges[k]) & (conf <= edges[k + 1])
            if m.any(): e += m.float().mean() * (conf[m].mean() - ok[m].mean()).abs()
        return float(e * 100)
    with torch.no_grad():
        for m in models: m.eval()
        Ps = [[] for _ in models]; ys = []
        for s0 in range(0, len(data.yte), 1000):
            x = data.prep(data.xte[s0:s0 + 1000], False); ys.append(data.yte[s0:s0 + 1000])
            for i, m in enumerate(models): Ps[i].append(F.softmax(m(x).float(), 1))
        y = torch.cat(ys); Ps = [torch.cat(p) for p in Ps]
        ece_nodes = [ece_of(P, y) for P in Ps]; ece_ens = ece_of(torch.stack(Ps).mean(0), y)
        for m in models: m.train()
    return dict(zeta2=zeta2, grad_norm2=gn2, mean_grad_norm2=gbar2, ece_node_mean=float(np.mean(ece_nodes)), ece_ens=ece_ens)

_CAL = {}
def cal_split(data, per_class_total=1000, seed=4242):
    """Fixed, class-balanced split of the TEST set: per_class_total images (equal per class) to fit a per-node bias,
    the rest to evaluate every post-hoc variant. Training data are untouched, so runs stay paired with earlier tiers."""
    key = (data.name, len(data.yte))
    if key not in _CAL:
        y = data.yte.cpu().numpy(); K = data.classes; k = per_class_total // K; rng = np.random.default_rng(seed)
        cal = np.concatenate([rng.choice(np.where(y == c)[0], k, replace=False) for c in range(K)])
        ev = np.setdiff1d(np.arange(len(y)), cal)
        _CAL[key] = (torch.as_tensor(cal, device=DEVICE), torch.as_tensor(ev, device=DEVICE))
    return _CAL[key]

@torch.no_grad()
def _logits_all(model, data, cfg, bs=1000):
    model.eval(); out = []
    for s0 in range(0, len(data.yte), bs):
        x = data.prep(data.xte[s0:s0 + bs], False)
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=cfg['amp'] and DEVICE == 'cuda'):
            out.append(model(x).float())
    model.train()
    return torch.cat(out)

def _fit_bias(z, y, iters=100):
    """Per-node logit bias b (K numbers) minimising the cross-entropy of softmax(z + b) on the calibration images."""
    b = torch.zeros(z.shape[1], device=z.device, requires_grad=True)
    opt = torch.optim.LBFGS([b], lr=0.5, max_iter=iters, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); l = F.cross_entropy(z + b, y); l.backward(); return l
    with torch.enable_grad(): opt.step(closure)
    return b.detach()

def posthoc2_eval(models, data, cfg, class_hist, W):
    """Stronger prediction-time controls (review of the revision). For every node model, on the evaluation part of the test set:
    raw logits; minus tau*log(own smoothed prior), tau in {0.1, 0.25, 0.5, 1}; minus tau*log of the neighbourhood-mixture prior
    after one (W pi) and two (W^2 pi) gossip steps, tau in {0.5, 1}; plus a per-node bias fitted on 1000 balanced held-out images.
    Returns individual (mean over nodes) and ensemble accuracy per variant, and the accuracy of the averaged model."""
    cal, ev = cal_split(data); y = data.yte
    Z = [_logits_all(m, data, cfg) for m in models]
    pis = np.array([(np.array(h) + 1.0) / (np.sum(h) + len(h)) for h in class_hist])
    mix1 = W @ pis; mix2 = W @ mix1
    lg = lambda a: torch.log(torch.as_tensor(a, dtype=torch.float32, device=DEVICE))
    V = {'raw': [None] * len(models)}
    for t in (0.1, 0.25, 0.5, 1.0): V[f'own{t}'] = [-t * lg(pis[i]) for i in range(len(models))]
    for t in (0.5, 1.0):
        V[f'mix1_{t}'] = [-t * lg(mix1[i]) for i in range(len(models))]
        V[f'mix2_{t}'] = [-t * lg(mix2[i]) for i in range(len(models))]
    V['fit'] = [_fit_bias(Z[i][cal], y[cal]) for i in range(len(models))]
    rec = {}
    for name, offs in V.items():
        P = [F.softmax(Z[i][ev] if offs[i] is None else Z[i][ev] + offs[i], 1) for i in range(len(models))]
        rec['ind_' + name] = float(np.mean([float((p.argmax(1) == y[ev]).float().mean()) * 100 for p in P]))
        rec['ens_' + name] = float((torch.stack(P).mean(0).argmax(1) == y[ev]).float().mean()) * 100
    za = _logits_all(_avg_model(models), data, cfg)
    rec['avg_raw'] = float((za[ev].argmax(1) == y[ev]).float().mean()) * 100
    rec['avg_fit'] = float(((za[ev] + _fit_bias(za[cal], y[cal])).argmax(1) == y[ev]).float().mean()) * 100
    return rec

def _sync():
    if DEVICE == 'cuda': torch.cuda.synchronize()

# ---------------- full run ----------------
_VIEWS = {}
def val_view(data, n_val):
    # fixed held-out split of the training set (same for every seed and method); evaluation then uses it instead of the test set
    key = (data.name, n_val)
    if key not in _VIEWS:
        perm = np.random.default_rng(12345).permutation(len(data.ytr_cpu)); v, tr = perm[:n_val], perm[n_val:]
        vt, trt = torch.as_tensor(v, device=DEVICE), torch.as_tensor(tr, device=DEVICE)
        o = copy.copy(data); o.xtr, o.ytr, o.ytr_cpu = data.xtr[trt], data.ytr[trt], data.ytr_cpu[tr]
        o.xte, o.yte = data.xtr[vt], data.ytr[vt]; _VIEWS[key] = o
    return _VIEWS[key]

def rotate_view(data, parts):
    # feature skew: node i trains on images rotated by 90*(i mod 4); test image j is rotated by 90*(j mod 4)
    o = copy.copy(data); xtr = data.xtr.clone(); xte = data.xte.clone()
    for i, p in enumerate(parts):
        k = i % 4
        if k:
            ix = torch.as_tensor(p, device=DEVICE); xtr[ix] = torch.rot90(xtr[ix], k, dims=(2, 3))
    for k in (1, 2, 3): xte[k::4] = torch.rot90(xte[k::4], k, dims=(2, 3))
    o.xtr, o.xte = xtr, xte
    return o

def run_experiment(spec, data=None, verbose=True, ckpt_path=None, ckpt_every=5,
                   models_path=None, deadline=None, stop_after_round=None):
    # ckpt_path : mid-run checkpoint, written every ckpt_every rounds and resumed from if it exists
    # models_path: where the 5 final node models are saved
    # deadline  : unix time; when passed, the run checkpoints and returns None (to resume next session)
    cfg = {**DEFAULTS, **spec}; cfg = {**cfg, **METHODS[cfg['method']]}
    base = base_method(cfg['method']); N, T, E = cfg['nodes'], cfg['rounds'], cfg['local_epochs']
    seed_everything(cfg['seed'])
    if cfg['dataset'] == 'femnist':
        data, parts = femnist_view(cfg)
    else:
        data = data or get_data(cfg['dataset'])
        if cfg['val'] > 0: data = val_view(data, cfg['val'])
        parts = dirichlet_partition(data.ytr_cpu, N, cfg['alpha'], cfg['seed'], cfg['min_node_size'])
    if cfg['feat_skew'] == 'rot': data = rotate_view(data, parts)
    idxs = [torch.as_tensor(p, device=DEVICE) for p in parts]
    class_hist = [np.bincount(data.ytr_cpu[p], minlength=data.classes).tolist() for p in parts]
    # logit adjustment: tau * log pi_i(c), with pi_i the node's (smoothed) class distribution
    # per-node logit offsets used only in the training loss (prediction always uses the raw logits)
    def _offset(h):
        n = torch.tensor(h, dtype=torch.float32, device=DEVICE)
        if cfg['la_tau'] != 0: return cfg['la_tau'] * torch.log((n + 1) / (n.sum() + len(h)))     # logit adjustment
        if cfg['lc_tau'] != 0: return -cfg['lc_tau'] * (n + 1) ** -0.25                          # FedLC calibration
        return None
    priors = [_offset(h) for h in class_hist]
    scales = [None if cfg['rs_alpha'] == 1.0 else
              torch.where(torch.tensor(h, device=DEVICE) > 0, torch.tensor(1.0, device=DEVICE), torch.tensor(cfg['rs_alpha'], device=DEVICE))
              for h in class_hist]

    seed_everything(cfg['seed'])                     # same init + same partition for every method (paired design)
    proto = make_model(cfg, data).to(DEVICE)
    models = [copy.deepcopy(proto) for _ in range(N)]
    opts = [torch.optim.SGD(m.parameters(), lr=cfg['lr'], momentum=0.0 if base == 'qgm' else cfg['momentum'], weight_decay=cfg['wd']) for m in models]
    scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=T * E) for o in opts]
    scalers = [torch.amp.GradScaler(DEVICE, enabled=cfg['amp'] and DEVICE == 'cuda') for _ in models]
    W = make_W(cfg); Wt = torch.tensor(W, dtype=torch.float32, device=DEVICE)
    d = n_params(proto)
    Fbar = [None] * N
    corr = [[torch.zeros_like(p) for p in m.parameters()] for m in models] if base in ('kgt', 'qgm') else None
    if base == 'gut':
        # GUT state per node: x (own model; the network holds the mixed model s = sum_j W_ij x_j), u = eta*y (last tracked update),
        # D = eta*delta (last combined update), lrs (sum of learning rates of the last round, to rescale u and D when eta changes)
        corr = dict(x=[[p.detach().clone() for p in m.parameters()] for m in models], u=None, D=None, lrs=None)

    H = {k: [] for k in ['train_loss', 'train_acc', 'ind_acc', 'ens_acc', 'node_acc', 'lam_mean', 'lam_node',
                         'consensus', 't_train', 't_fisher', 't_gossip', 't_eval', 'ext']}
    gcount = np.sum(class_hist, 0); log_pg = torch.tensor(np.log((gcount + 1) / (gcount.sum() + len(gcount))), dtype=torch.float32, device=DEVICE)
    start_round, elapsed0 = 0, 0.0
    if ckpt_path and os.path.exists(ckpt_path):          # ---- resume ----
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        for m, sd in zip(models, ck['models']): m.load_state_dict(sd)
        for o, sd in zip(opts, ck['opts']): o.load_state_dict(sd)
        for s, sd in zip(scheds, ck['scheds']): s.load_state_dict(sd)
        for s, sd in zip(scalers, ck['scalers']): s.load_state_dict(sd)
        Fbar = [None if f is None else [x.to(DEVICE) for x in f] for f in ck['Fbar']]
        if ck['corr'] is None: corr = None
        elif isinstance(ck['corr'], dict):
            corr = {k: (v if v is None or k == 'lrs' else [[x.to(DEVICE) for x in c] for c in v]) for k, v in ck['corr'].items()}
        else: corr = [[x.to(DEVICE) for x in c] for c in ck['corr']]
        H, start_round, elapsed0 = ck['hist'], ck['round'] + 1, ck['elapsed']
        random.setstate(ck['rng']['py']); np.random.set_state(ck['rng']['np']); torch.set_rng_state(ck['rng']['torch'])
        if DEVICE == 'cuda' and ck['rng']['cuda'] is not None: torch.cuda.set_rng_state_all(ck['rng']['cuda'])
        if verbose: print(f"  resumed {cfg['method']} s{cfg['seed']} from round {start_round}")
    t_start = time.time()
    for t in range(start_round, T):
        # ---- local training ----
        _sync(); t0 = time.time()
        x0 = [[p.detach().clone() for p in m.parameters()] for m in models] if base in ('dprox', 'kgt', 'qgm', 'gut') else None
        L = C = S = 0; lr_sums = []
        for i in range(N):
            lr_sum = 0.0
            for _ in range(E):
                l, c, s, lr = local_epoch(models[i], opts[i], scalers[i], data, idxs[i], cfg, base,
                                          anchor=x0[i] if base == 'dprox' else None,
                                          corr=corr[i] if base in ('kgt', 'qgm') else None, prior=priors[i], scale=scales[i])
                scheds[i].step(); L += l; C += c; S += s; lr_sum += lr
            lr_sums.append(lr_sum)
        _sync(); t1 = time.time()

        # ---- Fisher estimation (D-FRC family and Fisher-weighted gossip only) ----
        if base in ('dfrc', 'fw_gossip'):
            for i in range(N):
                Fi = diag_fisher(models[i], data, idxs[i], cfg['n_fisher'], cfg['fisher_chunk'])
                if Fbar[i] is None or t < cfg['warm'] or cfg['beta'] == 0:
                    Fbar[i] = Fi
                else:
                    Fbar[i] = [cfg['beta'] * a + (1 - cfg['beta']) * b for a, b in zip(Fbar[i], Fi)]
        _sync(); t2 = time.time()

        # ---- consensus ----
        with torch.no_grad():
            P = [[p.detach() for p in m.parameters()] for m in models]
            Bf = [float_buffers(m) for m in models]
            nbP, nbB = mix_tensors(P, Wt), mix_tensors(Bf, Wt)
            if base == 'fw_gossip' and t >= cfg['warm']:
                Fe = [[f + cfg['eps'] for f in Fbar[i]] for i in range(N)]
                num = mix_tensors([[f * p for f, p in zip(Fe[i], P[i])] for i in range(N)], Wt)
                den = mix_tensors(Fe, Wt)
            new_P, new_B, lams = [], [], []
            for i in range(N):
                if base in ('dpsgd', 'dprox', 'kgt', 'qgm', 'gut') or (base == 'fw_gossip' and t < cfg['warm']):
                    np_i, nb_i, lam = nbP[i], nbB[i], 1.0
                elif base == 'fw_gossip':
                    np_i = [a / b for a, b in zip(num[i], den[i])]; nb_i, lam = nbB[i], 1.0
                elif base == 'dpsgd_lazy':
                    m_ = cfg['lazy_m']
                    np_i = [(1 - m_) * p + m_ * q for p, q in zip(P[i], nbP[i])]
                    nb_i = [(1 - m_) * p + m_ * q for p, q in zip(Bf[i], nbB[i])]; lam = m_
                else:  # D-FRC family
                    if t < cfg['warm']:
                        M = [torch.full_like(p, cfg['m_warm']) for p in P[i]]
                    else:
                        M = fisher_gate(normalize_fisher(Fbar[i], cfg['eps'], cfg['delta']), cfg)
                    np_i = [(1 - m_) * p + m_ * q for m_, p, q in zip(M, P[i], nbP[i])]
                    lam = float(_flat(M).mean())     # buffers (BN statistics) use the node's mean lambda
                    nb_i = [(1 - lam) * p + lam * q for p, q in zip(Bf[i], nbB[i])]
                new_P.append(np_i); new_B.append(nb_i); lams.append(lam)
            if cfg['head_mix'] == 'class':
                # class-aware head gossip: row c of the last layer is averaged with weights W_ij * n_j(c),
                # so the neighbours that hold class c decide its classifier row (the body uses plain gossip)
                cnt = torch.tensor(class_hist, dtype=torch.float32, device=DEVICE) + cfg['head_eps']   # [N, K]
                A = Wt[:, :, None] * cnt[None, :, :]; A = A / A.sum(1, keepdim=True)              # [i, j, c]
                Wh = torch.stack([P[j][-2].float() for j in range(N)]); Bh = torch.stack([P[j][-1].float() for j in range(N)])
                for i in range(N):
                    new_P[i] = list(new_P[i])
                    new_P[i][-2] = torch.einsum('jc,jck->ck', A[i], Wh)
                    new_P[i][-1] = torch.einsum('jc,jc->c', A[i], Bh)
            if base == 'kgt':
                # tracking update: c_i <- c_i - d_i + sum_j W_ij d_j, with d_i = (1 - momentum) (x_i^start - x_i^end) / sum(lr).
                # Heavy-ball SGD moves about lr / (1 - momentum) per unit of gradient, so without this factor the
                # correction is ~10x too large and K-GT diverges (this was the bug in the first runs).
                kg = (1 - cfg['momentum']) * cfg['kgt_scale']
                D = [[kg * (a - p) / max(lr_sums[i], 1e-12) for a, p in zip(x0[i], P[i])] for i in range(N)]
                mD = mix_tensors(D, Wt)
                for i in range(N): corr[i] = [c - di + mi for c, di, mi in zip(corr[i], D[i], mD[i])]
            if base == 'gut':
                # Global Update Tracking (Aketi et al., 2023, Alg. 1) with one local epoch per communication round.
                # Each round node i trains from the mixed model s_i = sum_j W_ij x_j (Alg. 1 evaluates the gradient there) to e_i.
                # With eta the step of the round and g_i = (s_i - e_i) / eta, Alg. 1 reads, in units u = eta*y and D = eta*delta:
                #   D_i = eta*delta_i = (s_i - e_i) - (s_i - x_i) = x_i - e_i
                #   u_i = D_i + mu * [ sum_j W_ij u_j^prev - (s_i - x_i) - D_i^prev ]     (sum_j W_ij (x_j - x_i) = s_i - x_i)
                #   x_i <- x_i - u_i,   and the network holds s_i = sum_j W_ij x_j for the next round and for evaluation.
                # u is the only quantity a node sends (1x communication, as D-PSGD). With mu = 0 this is exactly D-PSGD.
                # eta cancels; u^prev and D^prev are rescaled by eta_t / eta_{t-1} because the learning rate decays.
                mu_g = cfg['gut_mu']; X = corr['x']
                Dn = [[a - e for a, e in zip(X[i], P[i])] for i in range(N)]
                if corr['u'] is None:
                    U = Dn
                else:
                    r = [lr_sums[i] / max(corr['lrs'][i], 1e-12) for i in range(N)]
                    mU = mix_tensors(corr['u'], Wt)
                    U = [[dn + mu_g * (r[i] * mu_ - (s0 - x) - r[i] * dp)
                          for dn, mu_, s0, x, dp in zip(Dn[i], mU[i], x0[i], X[i], corr['D'][i])] for i in range(N)]
                X = [[x - u for x, u in zip(X[i], U[i])] for i in range(N)]
                new_P = mix_tensors(X, Wt)
                corr = dict(x=X, u=U, D=Dn, lrs=list(lr_sums))
            for i in range(N):
                for p, v in zip(models[i].parameters(), new_P[i]): p.copy_(v)
                for b, v in zip(float_buffers(models[i]), new_B[i]): b.copy_(v)
            if base == 'qgm':
                # quasi-global momentum: m_i <- beta m_i + (1 - beta) d_i, with d_i = (x_i^start - x_i^after gossip) / sum(lr);
                # during the next local epoch every gradient gets + beta * m_i (stored in corr as beta * m_i)
                bq = cfg['momentum']
                for i in range(N):
                    corr[i] = [bq * (bq * (c / bq) + (1 - bq) * (a - p.detach()) / max(lr_sums[i], 1e-12))
                               for c, a, p in zip(corr[i], x0[i], models[i].parameters())]
            avg = [torch.stack(ts).mean(0) for ts in zip(*[[p for p in m.parameters()] for m in models])]
            cons = float(np.mean([sum(float((p - a).pow(2).sum()) for p, a in zip(m.parameters(), avg)) for m in models]))
        _sync(); t3 = time.time()

        if (t + 1) % cfg['eval_every'] == 0 or t >= T - 5 or t == 0:
            node_acc, ens_acc = evaluate(models, data, cfg)
            if cfg['dataset'] == 'femnist' and t >= T - 5: H.setdefault('ext', []).append(evaluate_ext(models, data, cfg, log_pg))
            if cfg['posthoc_local'] and t >= T - 5:
                # per-node post-hoc correction: node i predicts with f(x) - tau * log pi_i (its own smoothed class prior)
                lp = [torch.log((torch.tensor(h, dtype=torch.float32, device=DEVICE) + 1) / (sum(h) + len(h))) for h in class_hist]
                rec = {}
                for tau, tag in ((1.0, '1'), (0.5, '05')):
                    na, ea = evaluate(models, data, cfg, offsets=[-tau * l for l in lp]); rec['ind' + tag] = float(np.mean(na)); rec['ens' + tag] = ea
                H.setdefault('post_local', []).append(rec)
            if cfg['posthoc2'] and t >= T - 5:
                H.setdefault('post2', []).append(posthoc2_eval(models, data, cfg, class_hist, W))
        else:
            node_acc, ens_acc = [float('nan')] * N, float('nan')
        _sync(); t4 = time.time()
        H['train_loss'].append(L / S); H['train_acc'].append(C / S * 100)
        H['ind_acc'].append(float(np.mean(node_acc))); H['ens_acc'].append(ens_acc); H['node_acc'].append(node_acc)
        H['lam_mean'].append(float(np.mean(lams))); H['lam_node'].append(lams); H['consensus'].append(cons)
        H['t_train'].append(t1 - t0); H['t_fisher'].append(t2 - t1); H['t_gossip'].append(t3 - t2); H['t_eval'].append(t4 - t3)
        if verbose and (t % 10 == 9 or t == T - 1 or t == 0):
            print(f"  [{cfg['method']:>13} s{cfg['seed']} a{cfg['alpha']}] round {t+1:>3}/{T}  "
                  f"ens {ens_acc:5.2f}  ind {np.mean(node_acc):5.2f}  train {C/S*100:5.2f}  "
                  f"lam {np.mean(lams):.3f}  {t4-t0:.1f}s/round", flush=True)
        out_of_time = deadline is not None and time.time() > deadline
        stop_now = out_of_time or (stop_after_round is not None and t + 1 >= stop_after_round)
        if ckpt_path and t + 1 < T and ((t + 1) % ckpt_every == 0 or stop_now):      # ---- checkpoint ----
            ck = dict(round=t, elapsed=elapsed0 + time.time() - t_start, cfg=cfg,
                      models=[m.state_dict() for m in models], opts=[o.state_dict() for o in opts],
                      scheds=[s.state_dict() for s in scheds], scalers=[s.state_dict() for s in scalers],
                      Fbar=Fbar, corr=corr, hist=H,
                      rng=dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state(),
                               cuda=torch.cuda.get_rng_state_all() if DEVICE == 'cuda' else None))
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(ck, ckpt_path + '.tmp'); os.replace(ckpt_path + '.tmp', ckpt_path)
        if stop_now and t + 1 < T:
            return None

    mult = 2 if base in ('kgt', 'fw_gossip') else 1   # K-GT also sends tracking vars; FW-gossip also sends Fisher
    deg = int((W[0] > 0).sum()) - 1
    meta = dict(n_params=d, spectral_gap=spectral_gap(W), node_sizes=[len(p) for p in parts],
                class_hist=class_hist, floats_sent_per_node_round=deg * d * mult,
                MB_sent_per_node_round=deg * d * mult * 4 / 2**20,
                extra_state_floats={'dfrc': d, 'fw_gossip': d, 'kgt': d, 'dprox': d, 'qgm': d, 'gut': 3 * d}.get(base, 0),
                wallclock_s=elapsed0 + time.time() - t_start,
                gpu=torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu', torch=torch.__version__)
    if cfg['diag']:
        H['diag'] = diagnostics(models, data, idxs, cfg, priors, scales)
        if cfg['diag'] == 2:   # both losses at this run's averaged model: plain cross-entropy and logit-adjusted (tau = 1)
            la_pr = [torch.log((torch.tensor(h, dtype=torch.float32, device=DEVICE) + 1) / (sum(h) + len(h))) for h in class_hist]
            H['diag_ce'] = diagnostics(models, data, idxs, cfg, [None] * N, [None] * N)
            H['diag_la'] = diagnostics(models, data, idxs, cfg, la_pr, [None] * N)
    if models_path:                                      # final node models
        os.makedirs(os.path.dirname(models_path), exist_ok=True)
        torch.save(dict(cfg=cfg, models=[m.state_dict() for m in models]), models_path)
    if ckpt_path and os.path.exists(ckpt_path): os.remove(ckpt_path)
    return dict(cfg=cfg, hist=H, meta=meta)

def run_id(spec):
    c = {**DEFAULTS, **spec}
    tag = '' if c['model'] == 'cnn' and c['nodes'] == 5 else f"_{c['model']}_n{c['nodes']}"
    if c['topo'] != 'ring': tag += f"_{c['topo']}"
    if c['rounds'] != 50: tag += f"_T{c['rounds']}"
    if c['lr'] != 0.1: tag += f"_lr{c['lr']}"
    if c['val'] > 0: tag += f"_val{c['val']}"
    if c['feat_skew'] != 'none': tag += f"_{c['feat_skew']}"
    if c['dataset'] == 'femnist': tag += f"_wpn{c['fem_wpn']}_fa{c['fem_alpha']}"
    if base_method(c['method']) == 'gut': tag += f"_gmu{c['gut_mu']}"
    if c.get('posthoc_local'): tag += '_phl'
    if c.get('local_epochs', 1) != 1: tag += f"_E{c['local_epochs']}"
    if c.get('diag'): tag += '_diag' if c['diag'] == 1 else f"_diag{c['diag']}"
    if c.get('posthoc2'): tag += '_ph2'
    return f"{c['dataset']}_a{c['alpha']}_{c['method']}_s{c['seed']}{tag}"


# ---------------- experiment plan ----------------
MAIN = ['dpsgd', 'dpsgd_lazy', 'dprox', 'kgt', 'fw_gossip', 'dfrc']
ABL  = ['dfrc_noema', 'dfrc_nowarm', 'dfrc_randgate', 'dfrc_invgate']
FIX  = ['kgt', 'dfrc_prox', 'dfrc_prox_rand', 'dfrc_wide', 'dfrc_wide_rand']
TIERS = {
    1: [dict(dataset='cifar10', alpha=0.1, method=m, seed=s) for s in range(5) for m in MAIN],          # significance
    2: [dict(dataset='cifar10', alpha=a, method=m, seed=s) for a in (0.5, 1.0) for s in range(3) for m in MAIN],  # heterogeneity
    3: [dict(dataset=ds, alpha=0.1, method=m, seed=s) for ds in ('cifar100', 'fmnist') for s in range(3) for m in MAIN],  # datasets
    4: [dict(dataset='cifar10', alpha=0.1, method=m, seed=s) for s in range(3) for m in ABL],           # ablations
    5: [dict(dataset='cifar10', alpha=0.1, method=m, seed=s) for s in range(5) for m in FIX],           # K-GT re-run + stronger-mixing D-FRC
}

PILOT = ['dpsgd', 'dpsgd_gn', 'la_gn', 'chead_gn', 'cig_gn', 'cig_bn', 'kgt_gn', 'fw_gossip_gn']
SOTA = ['dpsgd', 'la_bn', 'kgt', 'kgt_la', 'qgm', 'qgm_la', 'fedlc', 'fedrs']
RN = dict(model='resnet18', nodes=16, eval_every=5)
TUNE_M = ['dpsgd', 'la_bn', 'kgt', 'qgm', 'fedlc', 'fedrs']; LR_GRID = (0.1, 0.05, 0.02)
TIERS[11] = [dict(dataset='cifar10', alpha=0.1, method=m, seed=s, lr=lr, val=5000) for s in (100, 101) for lr in LR_GRID for m in TUNE_M]   # LR tuning on a validation split
TIERS[12] = [dict(dataset='cifar10', alpha=0.1, method=m, seed=s, feat_skew='rot') for s in range(5) for m in ('dpsgd', 'la_bn', 'fedlc')]   # label + feature skew
TIERS[13] = [dict(dataset='cifar10', alpha=0.1, method='kgt', seed=s, lr=lr, nodes=16, topo='ring', model='resnet18', eval_every=5, rounds=100)
             for lr in (0.05, 0.02) for s in range(3)]   # K-GT with smaller learning rates, ResNet-18, 16-node ring
SW = dict(model='resnet18', eval_every=5, rounds=100)
TIERS[10] = [dict(dataset='cifar10', alpha=0.1, method=m, seed=s, nodes=n, topo=tp, **SW)
             for ms in (['dpsgd', 'la_bn'], ['kgt', 'fedlc']) for s in range(3) for n in (16, 32) for tp in ('ring', 'exp') for m in ms]   # scale/topology sweep
TIERS[9] = [dict(dataset='cifar10', alpha=0.1, method=m, seed=s, **RN) for s in range(3) for m in ['dpsgd', 'la_bn']]   # 16-node ResNet-18 pilot
TIERS[8] = ([dict(dataset='cifar10', alpha=0.1, method=m, seed=s) for s in range(5) for m in SOTA]
            + [dict(dataset='cifar100', alpha=0.1, method=m, seed=s) for s in range(3) for m in SOTA])   # SOTA baselines
TIERS[7] = ([dict(dataset='cifar10', alpha=0.1, method=m, seed=s) for s in range(5) for m in ['dpsgd', 'la_bn', 'la_bn_t05', 'la_bn_t2']]
            + [dict(dataset='cifar10', alpha=a, method=m, seed=s) for a in (0.5, 1.0) for s in range(3) for m in ['dpsgd', 'la_bn']]
            + [dict(dataset='cifar100', alpha=0.1, method=m, seed=s) for s in range(3) for m in ['dpsgd', 'la_bn']])   # LA follow-up
FEM = dict(dataset='femnist', nodes=16, topo='ring', model='cnn', rounds=50, eval_every=5)
TIERS[14] = [dict(method=m, seed=s, fem_alpha=fa, **FEM) for fa in (0.0, 0.5) for s in range(5) for m in ('dpsgd', 'la_bn', 'fedlc')]   # FEMNIST, writer partition (+ label skew)
TIERS[6] = [dict(dataset='cifar10', alpha=0.1, method=m, seed=s) for s in range(3) for m in PILOT]   # CIG pilot
# ---- journal revision, round 2 ----
# mu = 0.9 (the paper's value) diverges on our ring with weights 1/3: in the bias-correction form of GUT (Aketi et al., Eq. 15) the
# consensus part is x+ = W x + mu (2W - I)(x - x_prev), and for an eigenvalue -0.2 of W this has a root of modulus > 1 when mu >= ~0.5.
GUT_MU = (0.9, 0.5, 0.2, 0.1, 0.05)
TIERS[15] = [dict(dataset='cifar10', alpha=0.1, method='gut', seed=s, gut_mu=g, val=5000) for s in (100, 101) for g in GUT_MU]   # GUT: tune mu on the validation split
def gut_tier(mu):   # GUT and GUT + logit adjustment at the selected mu, 5 seeds on CIFAR-10 and CIFAR-100
    return [dict(dataset=ds, alpha=0.1, method=m, seed=s, gut_mu=mu) for ds in ('cifar10', 'cifar100') for s in range(5) for m in ('gut', 'gut_la')]
TIERS[16] = [dict(dataset='cifar100', alpha=0.1, method=m, seed=s) for s in (3, 4) for m in SOTA]   # CIFAR-100 seeds 3 and 4 (seeds 0-2 in sota_v2)
# ---- round 3 (reviewer comments) ----
TIERS[19] = [dict(dataset='cifar10', alpha=0.1, method=m, seed=s, lr=lr, val=5000, **({'gut_mu': 0.2} if m == 'gut' else {}))
             for s in (100, 101) for lr in (0.2, 0.3) for m in TUNE_M + ['gut']]          # Q2: learning-rate grid extended above 0.1
TIERS[20] = [dict(dataset='cifar10', alpha=0.1, method=m, seed=s, nodes=16, topo='exp', local_epochs=E, eval_every=5)
             for E in (1, 2, 4) for s in range(3) for m in ('dpsgd', 'la_bn')]            # Q5: local drift on a well-connected graph
TIERS[21] = [dict(dataset='cifar10', alpha=0.1, method=m, seed=s, diag=1) for s in range(5) for m in ('dpsgd', 'la_bn', 'fedlc')]   # Q3: zeta^2 and calibration
TIERS[22] = [dict(dataset='cifar100', alpha=0.1, method=m, seed=s, nodes=16, topo='ring', model='resnet18', eval_every=5, rounds=100)
             for s in range(3) for m in ('dpsgd', 'la_bn')]                               # Q4: harder dataset at scale
TIERS[23] = [dict(dataset=ds, alpha=0.1, method=m, seed=s, posthoc2=1, diag=2) for ds in ('cifar10', 'cifar100') for s in range(5) for m in ('dpsgd', 'la_bn')]   # stronger post-hoc controls + zeta^2 at a common model (5-node CNN)
TIERS[24] = [dict(dataset='cifar10', alpha=0.1, method=m, seed=s, posthoc2=1, nodes=16, topo='ring', model='resnet18', eval_every=5, rounds=100)
             for s in range(3) for m in ('dpsgd', 'la_bn')]                               # stronger post-hoc controls at scale (16-node ring)
TIERS[18] = [dict(dataset=ds, alpha=0.1, method='dpsgd', seed=s, posthoc_local=1) for ds in ('cifar10', 'cifar100') for s in range(5)]   # per-node post-hoc control
TIERS[17] = [dict(dataset='cifar10', alpha=0.1, method=m, seed=s, nodes=n, topo=tp, **SW)
             for ms in (['dpsgd', 'la_bn'], ['kgt', 'fedlc']) for s in (3, 4) for n in (16, 32) for tp in ('ring', 'exp') for m in ms]   # sweep seeds 3 and 4


def download_dataset(name, root=None):
    import torchvision
    root = root or os.environ.get('DFRC_DATA_ROOT', './data')
    cls = {'cifar10': torchvision.datasets.CIFAR10, 'cifar100': torchvision.datasets.CIFAR100,
           'fmnist': torchvision.datasets.FashionMNIST}[name]
    cls(root, train=True, download=True); cls(root, train=False, download=True)

def free_gpu():
    _DATA.clear()
    if DEVICE == 'cuda': torch.cuda.empty_cache()

def worker_main(tiers, out, deadline=None, ckpt_every=5, save_models=True, overrides=None, lock_dir='/tmp/dfrc_locks', specs=None):
    # Several workers (one per GPU) can run this at once: each run is claimed with a lock file.
    os.makedirs(lock_dir, exist_ok=True)
    for sub_ in ('runs', 'ckpt', 'models'): os.makedirs(f'{out}/{sub_}', exist_ok=True)
    todo = list(specs) if specs is not None else sorted([s for k in tiers for s in TIERS[k]], key=lambda s: s['dataset'])
    for spec in todo:
        rid = run_id(spec); path = f'{out}/runs/{rid}.json'
        if os.path.exists(path): continue
        if deadline is not None and time.time() > deadline - 420:      # leave ~7 min of margin for one more run
            print('Time budget reached; remaining runs are left for the next session.', flush=True); break
        try:
            os.close(os.open(f'{lock_dir}/{rid}.lock', os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue                                                    # another worker has it
        print(f'=== {rid} ===', flush=True)
        try:
            res = run_experiment({**spec, **(overrides or {})}, ckpt_path=f'{out}/ckpt/{rid}.pt', ckpt_every=ckpt_every,
                                 models_path=f'{out}/models/{rid}.pt' if save_models else None, deadline=deadline)
            if res is None:
                print(f'   paused {rid} at a checkpoint (out of time)', flush=True); break
            with open(path + '.tmp', 'w') as f: json.dump(res, f)
            os.replace(path + '.tmp', path)
        except Exception as e:
            import traceback; traceback.print_exc(); print('   FAILED:', rid, repr(e), flush=True)
    print('Worker finished.', flush=True)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiers', type=int, nargs='*', default=[]); ap.add_argument('--specs', default=None); ap.add_argument('--out', required=True)
    ap.add_argument('--deadline', type=float, default=None); ap.add_argument('--ckpt-every', type=int, default=5)
    ap.add_argument('--no-models', action='store_true')
    a = ap.parse_args()
    print('Worker on', torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu', flush=True)
    specs = json.load(open(a.specs)) if a.specs else None
    worker_main(a.tiers, a.out, a.deadline, a.ckpt_every, not a.no_models, specs=specs)
