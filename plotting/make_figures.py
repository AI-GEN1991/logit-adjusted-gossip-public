# Builds every figure of Q1_manuscript.tex from the per-seed files in ../results/.
# Usage: python3 code/make_figures.py   (reads results/, writes paper/figures/*.pdf and *.png)
import json, csv, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
from matplotlib import font_manager
for _f in glob.glob(os.path.expanduser('~/Library/Fonts/cmun*.ttf')) + glob.glob('/usr/share/fonts/**/cmun*.ttf', recursive=True):
    font_manager.fontManager.addfont(_f)   # CMU Serif = Computer Modern, the font of the manuscript
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "figures")

# Validated categorical palette (fixed order, one colour per method in every figure).
COL = {"dpsgd": "#2a78d6", "la_bn": "#eb6834", "kgt": "#1baf7a", "qgm": "#eda100",
       "fedlc": "#e87ba4", "fedrs": "#008300"}
MRK = {"dpsgd": "o", "la_bn": "D", "kgt": "s", "qgm": "^", "fedlc": "v", "fedrs": "P"}
NAME = {"dpsgd": "D-PSGD", "la_bn": "LA gossip (ours)", "kgt": "K-GT", "qgm": "QGM",
        "fedlc": "FedLC", "fedrs": "FedRS"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["CMU Serif", "cmr10"],  # Computer Modern, as in the manuscript
    "mathtext.fontset": "cm", "axes.unicode_minus": True, "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "legend.frameon": False, "savefig.dpi": 300,
    "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", name)


def panel(ax, letter, x=-0.16):
    ax.text(x, 1.04, f"({letter})", transform=ax.transAxes, fontweight="bold", fontstretch="expanded", fontsize=9, va="bottom")


def ci95(d):
    d = np.asarray(d, float); m = d.mean(); h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return m, m - h, m + h


# ---------------------------------------------------------------- data
fin = [json.loads(l.split("|", 1)[1]) for l in open(os.path.join(RES, "final_exp_log_lines.txt")) if l.startswith("RES|")]
tune = [json.loads(l.split("|", 1)[1]) for l in open(os.path.join(RES, "final_exp_log_lines.txt")) if l.startswith("TUNE|")]
tuned = [r for r in fin if r["val"] == 0 and r["feat"] == "none" and r["model"] == "cnn" and r["lr"] == 0.1]
rot = [r for r in fin if r["feat"] == "rot"]
follow = [json.loads(l) for l in open(os.path.join(RES, "la_followup_per_seed.jsonl"))]
sweep = list(csv.DictReader(open(os.path.join(RES, "sweep5_per_seed.csv"))))
METHODS = ["dpsgd", "kgt", "qgm", "fedlc", "fedrs", "la_bn"]


def by(rows, m, key="i5"):
    return {r["s"]: float(r[key]) for r in rows if r["m"] == m}


# ================================================================ Fig. 1  synthetic check
def fig_synthetic():
    rng = np.random.default_rng(0)
    K, d, N, m = 4, 6, 5, 40000
    mu = rng.normal(0, 1.2, (K, d))
    pis = rng.dirichlet(0.1 * np.ones(K), N) * 0.98 + 0.005
    data = []
    for i in range(N):
        y = rng.choice(K, m, p=pis[i] / pis[i].sum()); x = mu[y] + rng.normal(0, 1, (m, d)); data.append((x, y))

    def grad(W, b, x, y, off):
        z = x @ W.T + b + off; z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        p[np.arange(len(y)), y] -= 1
        return p.T @ x / len(y), p.mean(0)

    Wm = np.zeros((N, N))
    for i in range(N):
        for j in (i - 1, i, i + 1): Wm[i, j % N] += 1 / 3
    Wst, bst = mu.copy(), -0.5 * (mu ** 2).sum(1)
    rng_test = np.random.default_rng(1)  # same test sample as results/theory_numerical_check.py
    yt = rng_test.integers(0, K, 20000); xt = mu[yt] + rng_test.normal(0, 1, (20000, d))
    acc = lambda W, b: float(((xt @ W.T + b).argmax(1) == yt).mean())
    bayes = acc(Wst, bst)
    la = [np.log(pis[i]) for i in range(N)]; ce = [np.zeros(K)] * N
    hist = {}
    for name, offs in (("la_bn", la), ("dpsgd", ce)):
        Ws = [np.zeros((K, d)) for _ in range(N)]; bs = [np.zeros(K) for _ in range(N)]
        cons, accs = [], []
        for r in range(300):
            for i in range(N):
                for _ in range(10):
                    gW, gb = grad(Ws[i], bs[i], *data[i], offs[i]); Ws[i] -= 0.2 * gW; bs[i] -= 0.2 * gb
            Ws = [sum(Wm[i, j] * Ws[j] for j in range(N)) for i in range(N)]
            bs = [sum(Wm[i, j] * bs[j] for j in range(N)) for i in range(N)]
            if r % 5 == 0 or r == 299:
                Wb, bb = np.mean(Ws, 0), np.mean(bs, 0)
                cons.append((r + 1, np.mean([np.sum((Ws[i] - Wb) ** 2) + np.sum((bs[i] - bb) ** 2) for i in range(N)])))
                accs.append((r + 1, np.mean([acc(Ws[i], bs[i]) for i in range(N)])))
        hist[name] = (np.array(cons), np.array(accs))

    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.15), gridspec_kw={"width_ratios": [0.85, 1, 1]})
    ax = axs[0]
    im = ax.imshow(pis / pis.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(K)); ax.set_xticklabels([f"{k+1}" for k in range(K)])
    ax.set_yticks(range(N)); ax.set_yticklabels([f"node {i+1}" for i in range(N)])
    ax.set_xlabel("class"); ax.grid(False); ax.set_title("node label priors $\\pi_i$", loc="left")
    for i in range(N):
        for k in range(K):
            v = pis[i, k] / pis[i].sum()
            ax.text(k, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.55 else INK)
    panel(ax, "a")
    ax = axs[1]
    for n in ("dpsgd", "la_bn"):
        c = hist[n][0]; ax.semilogy(c[:, 0], c[:, 1], color=COL[n], label="cross-entropy" if n == "dpsgd" else "logit-adjusted")
    ax.set_xlabel("round"); ax.set_ylabel("consensus distance"); ax.set_title("node models agree only with LA", loc="left")
    ax.legend(loc="center right"); panel(ax, "b")
    ax = axs[2]
    for n in ("dpsgd", "la_bn"):
        a = hist[n][1]; ax.plot(a[:, 0], 100 * a[:, 1], color=COL[n])
    ax.axhline(100 * bayes, color=INK2, lw=0.8, ls="--")
    ax.text(300, 100 * bayes + 0.6, f"Bayes {100*bayes:.1f}%", ha="right", va="bottom", fontsize=7, color=INK2)
    a = hist["dpsgd"][1]; ax.text(300, 100 * a[-1, 1] - 1.2, f"CE {100*a[-1,1]:.1f}%", ha="right", va="top", fontsize=7, color=INK2)
    ax.set_xlabel("round"); ax.set_ylabel("node-model accuracy (%)")
    ax.set_title("LA reaches the Bayes classifier", loc="left"); panel(ax, "c")
    fig.tight_layout(w_pad=1.6); save(fig, "fig_synthetic")
    return {n: (hist[n][0][-1, 1], hist[n][1][-1, 1]) for n in hist}, bayes


# ================================================================ Fig. 2  tuned re-run
def fig_tuned():
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [0.9, 1.25, 1.05]})
    ax = axs[0]
    lrs = [0.02, 0.05, 0.1]
    for t in tune:
        ax.plot(range(3), [t[str(l)] for l in lrs], color=COL[t["m"]], marker=MRK[t["m"]], ms=4.5, lw=1.3,
                mec="white", mew=0.6)
        dy = {"fedlc": 0.45, "fedrs": -0.45}.get(t["m"], 0)
        ax.text(2.08, t["0.1"] + dy, NAME[t["m"]].replace(" (ours)", ""), va="center", fontsize=6.5, color=INK2)
    ax.set_xticks(range(3)); ax.set_xticklabels([str(l) for l in lrs]); ax.set_xlim(-0.15, 2.9)
    ax.set_xlabel("learning rate"); ax.set_ylabel("validation individual acc. (%)")
    ax.set_title("every method selects 0.1", loc="left"); panel(ax, "a")

    ax = axs[1]
    la = by(tuned, "la_bn")
    for k, m in enumerate(METHODS):
        v = by(tuned, m); seeds = sorted(v)
        xs = k + np.linspace(-0.18, 0.18, len(seeds))
        ax.scatter(xs, [v[s] for s in seeds], s=14, color=COL[m], marker=MRK[m], edgecolor="white", lw=0.5, zorder=3)
        mean = np.mean(list(v.values()))
        ax.plot([k - 0.3, k + 0.3], [mean, mean], color=INK, lw=1.2, zorder=4)
        if m == "la_bn": ax.text(k - 0.34, mean, f"{mean:.1f}", ha="right", va="center", fontsize=6.8, color=INK)
        else: ax.text(k, mean + 1.0, f"{mean:.1f}", ha="center", fontsize=6.8, color=INK)
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels([NAME[m].replace(" gossip (ours)", "") for m in METHODS], fontsize=6)
    ax.set_ylabel("test individual accuracy (%)"); ax.set_title("tuned re-run, 5 seeds (bar = mean)", loc="left")
    panel(ax, "b")

    ax = axs[2]
    base = ["dpsgd", "kgt", "qgm", "fedlc", "fedrs"]
    for j, (key, off, lab) in enumerate((("i5", -0.13, "individual"), ("e5", 0.13, "ensemble"))):
        L = by(tuned, "la_bn", key)
        for k, b in enumerate(base):
            B = by(tuned, b, key); dd = [L[s] - B[s] for s in sorted(L)]
            mn, lo, hi = ci95(dd)
            y = k + off
            ax.plot([lo, hi], [y, y], color=COL[b], lw=1.6 if key == "i5" else 1.0, alpha=1 if key == "i5" else 0.75)
            ax.scatter([mn], [y], color=COL[b] if key == "i5" else "white", edgecolor=COL[b], marker=MRK[b], s=20, zorder=3, lw=1)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(base))); ax.set_yticklabels([NAME[b] for b in base]); ax.invert_yaxis()
    ax.set_xlabel("LA $-$ baseline (points), 95% CI")
    ax.scatter([], [], color=INK2, marker="o", s=16, label="individual"); ax.scatter([], [], color="white", edgecolor=INK2, marker="o", s=16, label="ensemble")
    ax.set_ylim(4.5, -1.1); ax.legend(loc="upper right", ncol=2, handletextpad=0.1, columnspacing=0.6, fontsize=6.8)
    ax.set_title("paired gains", loc="left"); panel(ax, "c")
    fig.tight_layout(w_pad=1.4); save(fig, "fig_tuned")


# ================================================================ Fig. 3  heterogeneity and temperature
def fig_alpha_tau():
    fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.2))
    ax = axs[0]
    alphas = [0.1, 0.5, 1.0]
    for a_i, a in enumerate(alphas):
        D = {r["s"]: r["i5"] for r in follow if r["d"] == "cifar10" and r["a"] == a and r["m"] == "dpsgd"}
        L = {r["s"]: r["i5"] for r in follow if r["d"] == "cifar10" and r["a"] == a and r["m"] == "la_bn"}
        dd = [L[s] - D[s] for s in sorted(L)]
        ax.scatter([a_i] * len(dd) + np.linspace(-0.08, 0.08, len(dd)), dd, s=13, color=COL["la_bn"], marker="D", edgecolor="white", lw=0.5, zorder=3)
        mn = np.mean(dd); ax.plot([a_i - 0.22, a_i + 0.22], [mn, mn], color=INK, lw=1.2)
        ax.text(a_i + 0.26, mn, f"{mn:+.1f}", va="center", fontsize=7)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"$\\alpha$={a}" for a in alphas]); ax.set_xlim(-0.4, 2.6)
    ax.set_ylabel("LA $-$ D-PSGD, individual (points)"); ax.set_title("gain shrinks as data approach IID", loc="left")
    ax.set_xlabel("Dirichlet concentration (more IID $\\rightarrow$)"); panel(ax, "a", -0.36)
    ax = axs[1]
    taus = [("dpsgd", 0.0), ("la_bn_t05", 0.5), ("la_bn", 1.0), ("la_bn_t2", 2.0)]
    for key, lab, mk, col in (("i5", "individual", "D", COL["la_bn"]), ("e5", "ensemble", "o", COL["dpsgd"])):
        means = [np.mean([r[key] for r in follow if r["d"] == "cifar10" and r["a"] == 0.1 and r["m"] == m]) for m, _ in taus]
        ax.plot([t for _, t in taus], means, color=col, marker=mk, ms=5, mec="white", mew=0.6, label=lab)
    ax.set_xticks([0, 0.5, 1, 2]); ax.set_xticklabels(["0\n(CE)", "0.5", "1", "2"])
    ax.set_xlabel("temperature $\\tau$"); ax.set_ylabel("accuracy (%)")
    ax.set_title("$\\tau=1$ is a safe default ($\\alpha=0.1$)", loc="left"); ax.legend(loc="lower right"); panel(ax, "b")
    fig.tight_layout(w_pad=1.8); save(fig, "fig_alpha_tau")


# ================================================================ Fig. 4  scale and topology
GAP = {("16", "ring"): 0.051, ("16", "exp"): 0.50, ("32", "ring"): 0.013, ("32", "exp"): 0.40}


def fig_scale():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.45), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axs[0]
    graphs = [("16", "exp"), ("16", "ring"), ("32", "exp"), ("32", "ring")]
    ms = ["kgt", "dpsgd", "fedlc", "la_bn"]
    w = 0.19
    for gi, g in enumerate(graphs):
        for mi, m in enumerate(ms):
            v = [float(r["i5"]) for r in sweep if r["m"] == m and (r["n"], r["topo"]) == g]
            x = gi + (mi - 1.5) * w
            ax.bar(x, np.mean(v), width=w - 0.025, color=COL[m], label=NAME[m] if gi == 0 else None)
            ax.errorbar(x, np.mean(v), yerr=np.std(v, ddof=1), color=INK2, lw=0.7, capsize=1.5)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{n} nodes, {'exp.' if t=='exp' else 'ring'}\n(gap {GAP[(n,t)]})" for n, t in graphs], fontsize=7)
    ax.set_ylabel("individual accuracy (%)"); ax.set_ylim(0, 92)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), columnspacing=0.9, handlelength=1.1)
    ax.text(0, -0.30, "", transform=ax.transAxes)
    panel(ax, "a")
    ax = axs[1]
    for g in graphs:
        D = {r["s"]: float(r["i5"]) for r in sweep if r["m"] == "dpsgd" and (r["n"], r["topo"]) == g}
        L = {r["s"]: float(r["i5"]) for r in sweep if r["m"] == "la_bn" and (r["n"], r["topo"]) == g}
        dd = [L[s] - D[s] for s in sorted(L)]
        mn, lo, hi = ci95(dd)
        mk = "o" if g[0] == "16" else "s"
        ax.errorbar(GAP[g], mn, yerr=[[mn - lo], [hi - mn]], fmt=mk, color=COL["la_bn"], ms=5, mec="white", mew=0.6, lw=1, capsize=2)
        off, ha = ((-7, 2), "right") if g == ("32", "exp") else ((7, 2), "left")
        ax.annotate(f"{g[0]}, {'exp.' if g[1]=='exp' else 'ring'}", (GAP[g], mn), xytext=off, textcoords="offset points", fontsize=6.8, color=INK2, ha=ha)
    D = by(tuned, "dpsgd"); L = by(tuned, "la_bn"); dd = [L[s] - D[s] for s in sorted(L)]
    mn, lo, hi = ci95(dd)
    ax.errorbar(0.461, mn, yerr=[[mn - lo], [hi - mn]], fmt="^", color=INK2, ms=5, mec="white", mew=0.6, lw=1, capsize=2)
    ax.annotate("5, ring (CNN)", (0.461, mn), xytext=(-6, -14), textcoords="offset points", fontsize=6.8, color=INK2, ha="right")
    ax.set_xscale("log"); ax.set_xlim(0.009, 0.9)
    ax.set_xlabel("spectral gap $1-|\\lambda_2(W)|$ (log scale)"); ax.set_ylabel("LA $-$ D-PSGD, individual (points)")
    ax.set_title("gain grows as connectivity falls", loc="left"); ax.set_ylim(0, 20); panel(ax, "b", -0.27)
    fig.tight_layout(w_pad=1.6); save(fig, "fig_scale")


# ================================================================ Fig. 5  mechanism
def fig_mechanism():
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.6))
    sets = [(axs[0], [(r["m"], r["e5"], r["i5"]) for r in tuned], "five nodes, CNN (tuned re-run)", (60, 88)),
            (axs[1], [(r["m"], float(r["e5"]), float(r["i5"])) for r in sweep], "ResNet-18, 16 and 32 nodes", (25, 88))]
    for k, (ax, pts, title, lim) in enumerate(sets):
        ax.plot(lim, lim, color=INK2, lw=0.8, ls="--")
        present = [m for m in METHODS if any(p[0] == m for p in pts)]
        for m in present:
            P = np.array([(e, i) for mm, e, i in pts if mm == m])
            ax.scatter(P[:, 0], P[:, 1], s=16 if m != "la_bn" else 20, color=COL[m], marker=MRK[m], edgecolor="white", lw=0.5,
                       label=NAME[m], zorder=4 if m == "la_bn" else 3)
        ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
        ax.set_xlabel("ensemble accuracy (%)"); ax.set_ylabel("individual accuracy (%)")
        ax.set_title(title, loc="left"); panel(ax, "ab"[k])
        ax.legend(loc="upper left", fontsize=6.5, handletextpad=0.1, borderaxespad=0.2)
    fig.tight_layout(w_pad=2.0); save(fig, "fig_mechanism")


# ================================================================ Fig. 6  feature skew
def fig_rotation():
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ms = ["dpsgd", "fedlc", "la_bn"]
    for k, m in enumerate(ms):
        a = np.mean(list(by(tuned, m).values())); b = np.mean(list(by(rot, m).values()))
        ax.plot([a, b], [k, k], color=COL[m], lw=1.6, zorder=2)
        ax.scatter([a], [k], s=34, color=COL[m], marker=MRK[m], edgecolor="white", lw=0.6, zorder=3)
        ax.scatter([b], [k], s=34, color="white", marker=MRK[m], edgecolor=COL[m], lw=1.2, zorder=3)
        ax.text(a + 0.8, k, f"{a:.1f}", va="center", fontsize=7); ax.text(b - 0.8, k, f"{b:.1f}", va="center", ha="right", fontsize=7)
    ax.set_yticks(range(3)); ax.set_yticklabels([NAME[m] for m in ms]); ax.invert_yaxis()
    ax.set_xlim(40, 84); ax.set_xlabel("individual accuracy (%), CIFAR-10, 5 seeds")
    ax.scatter([], [], color=INK2, s=20, label="label skew only"); ax.scatter([], [], color="white", edgecolor=INK2, s=20, label="+ rotation (feature skew)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.25), ncol=2, columnspacing=0.8, handletextpad=0.2)
    fig.tight_layout(); save(fig, "fig_rotation")


# ================================================================ Fig.  learning curves (ResNet-18 sweep, seeds 3-4)
def fig_curves():
    D = json.load(open(os.path.join(RES, "round2B_curves.json")))["data"]
    graphs = [(16, "ring"), (16, "exp"), (32, "ring"), (32, "exp")]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.1), sharey=True)
    for k, (ax, (n, tp)) in enumerate(zip(axs, graphs)):
        for m in ("dpsgd", "la_bn"):
            runs = [c for c in D if c["m"] == m and c["n"] == n and c["topo"] == tp]
            r = np.array(runs[0]["r"]) + 1
            ind = np.mean([c["ind"] for c in runs], 0); ens = np.mean([c["ens"] for c in runs], 0)
            ax.plot(r, ind, color=COL[m], lw=1.5, label=f"{NAME[m]}, individual")
            ax.plot(r, ens, color=COL[m], lw=1.0, ls="--", label=f"{NAME[m]}, ensemble")
        ax.set_title(f"{n} nodes, {'exp.' if tp == 'exp' else 'ring'}", fontsize=8.5)
        ax.set_xlabel("round"); ax.set_xlim(0, 100); ax.set_ylim(0, 90)
        panel(ax, "abcd"[k], x=-0.12)
    axs[0].set_ylabel("test accuracy (%)")
    axs[0].legend(loc="upper center", bbox_to_anchor=(2.3, 1.42), ncol=4, columnspacing=0.9, handlelength=1.6)
    save(fig, "fig_curves")


if __name__ == "__main__":
    syn, bayes = fig_synthetic()
    print("synthetic final (consensus, node acc):", syn, "bayes", bayes)
    fig_tuned(); fig_alpha_tau(); fig_scale(); fig_mechanism(); fig_rotation(); fig_curves()
