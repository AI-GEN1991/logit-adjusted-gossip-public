# Seven further figures (Figs. 4, 6, 9, 12, 13, 14, 16 of the paper), built only from the per-seed and test files in ../results/.
# Usage: python3 plotting/make_figures_results.py   (writes figures/fig_*.pdf/.png and prints the numbers
# quoted in the captions so that they can be checked against the tables of the manuscript).
import csv, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
from matplotlib import font_manager
for _f in glob.glob(os.path.expanduser('~/Library/Fonts/cmun*.ttf')) + glob.glob('/usr/share/fonts/**/cmun*.ttf', recursive=True):
    font_manager.fontManager.addfont(_f)   # CMU Serif = Computer Modern, the font of the manuscript
from matplotlib.lines import Line2D
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)

COL = {"dpsgd": "#2a78d6", "la_bn": "#eb6834", "kgt": "#1baf7a", "qgm": "#eda100",
       "fedlc": "#e87ba4", "fedrs": "#008300", "gut": "#7b5cd6"}
NAME = {"dpsgd": "D-PSGD", "la_bn": "LA gossip (ours)", "kgt": "K-GT", "qgm": "QGM",
        "fedlc": "FedLC", "fedrs": "FedRS", "gut": "GUT"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ALONE = "#9d9c97"  # bars of a communication rule without logit adjustment (fig_combo)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["CMU Serif", "cmr10"],  # Computer Modern, as in the manuscript
    "mathtext.fontset": "cm", "axes.unicode_minus": True, "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", name)


def panel(ax, letter, x=-0.16):
    ax.text(x, 1.04, f"({letter})", transform=ax.transAxes, fontweight="bold", fontstretch="expanded", fontsize=9, va="bottom")


def rows(f):
    return list(csv.DictReader(open(os.path.join(RES, f))))


comb = rows("round2_combined_per_seed.csv")
BASE = ["dpsgd", "kgt", "gut", "qgm", "fedlc", "fedrs"]


def mean_of(d, m, key):
    v = [float(r[key]) for r in comb if r["d"] == d and r["m"] == m]
    return np.mean(v), np.std(v, ddof=1), v


# ================================================================ paired forest plot (Table paired as a figure)
def fig_forest():
    pt = rows("round2_paired_tests.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, d, letter in zip(axes, ["cifar10", "cifar100"], "ab"):
        for k, b in enumerate(BASE):
            for metric, off, filled in (("i5", -0.15, True), ("e5", 0.15, False)):
                r = next(r for r in pt if r["d"] == d and r["metric"] == metric and r["vs"] == b)
                y = k + off; m, lo, hi = float(r["diff"]), float(r["lo"]), float(r["hi"])
                ax.plot([lo, hi], [y, y], color=COL[b], lw=1.4)
                ax.plot(m, y, marker="o", ms=5, color=COL[b], mfc=COL[b] if filled else "white", mew=1.2)
                print(f"forest {d} {metric} vs {b}: {m:+.2f} [{lo:+.2f}, {hi:+.2f}] wins {r['wins']} pHolm {float(r['p_holm']):.3f}")
        ax.axvline(0, color=INK2, lw=0.8, ls="--")
        ax.set_yticks(range(len(BASE))); ax.set_yticklabels([NAME[b] for b in BASE]); ax.invert_yaxis()
        ax.set_xlabel("LA gossip $-$ baseline (points)")
        ax.set_title("CIFAR-10" if d == "cifar10" else "CIFAR-100"); panel(ax, letter, -0.05 if letter == "b" else -0.3)
    h = [Line2D([], [], marker="o", color=INK2, ls="", mfc=INK2, label="individual accuracy"),
         Line2D([], [], marker="o", color=INK2, ls="", mfc="white", label="ensemble accuracy")]
    fig.legend(handles=h, loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.06))
    fig.subplots_adjust(bottom=0.24)
    save(fig, "fig_forest")


# ================================================================ accuracy against communication
def fig_comm():
    ORDER = [("dpsgd", 1), ("qgm", 1), ("fedrs", 1), ("fedlc", 1), ("gut", 1), ("qgm_la", 1), ("la_bn", 1), ("gut_la", 1),
             ("kgt", 2), ("kgt_la", 2)]
    LBL = dict(NAME, la_bn="LA gossip", kgt_la="K-GT\n+ LA", gut_la="GUT\n+ LA", qgm_la="QGM\n+ LA")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, d, letter in zip(axes, ["cifar10", "cifar100"], "ab"):
        ax.axvspan(7.5, 9.6, color=GRID, alpha=0.6, lw=0)
        for x, (m, c) in enumerate(ORDER):
            mu, sd, v = mean_of(d, m, "i5")
            ours = m == "la_bn" or m.endswith("_la")
            col = COL["la_bn"] if ours else COL[m]
            ax.errorbar(x, mu, yerr=sd, fmt="D" if ours else "o", color=col, ms=5, capsize=2, lw=0.9,
                        mfc=col if ours else "white", mew=1.1)
            print(f"comm {d} {m}: comm {c}x, ind {mu:.2f} +- {sd:.2f}")
        ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([LBL[m] for m, _ in ORDER], rotation=60, fontsize=6.5)
        ax.set_xlim(-0.6, 9.6); ax.grid(axis="x", visible=False)
        yl = ax.get_ylim(); ax.text(3.5, yl[1], "$1\\times$ communication", ha="center", va="top", fontsize=7, color=INK2)
        ax.text(8.55, yl[1], "$2\\times$", ha="center", va="top", fontsize=7, color=INK2)
        ax.set_ylabel("individual accuracy (%)")
        ax.set_title("CIFAR-10" if d == "cifar10" else "CIFAR-100", pad=10); panel(ax, letter)
    save(fig, "fig_comm")


# ================================================================ combination with tracking / momentum
def fig_combo():
    pt = rows("round2_paired_tests.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=False)
    for ax, d, letter in zip(axes, ["cifar10", "cifar100"], "ab"):
        xs = np.arange(4); w = 0.36
        groups = [("dpsgd", "la_bn", "D-PSGD"), ("kgt", "kgt_la", "K-GT"), ("gut", "gut_la", "GUT"), ("qgm", "qgm_la", "QGM")]
        for k, (b, bla, lab) in enumerate(groups):
            for j, (m, col) in enumerate(((b, ALONE), (bla, COL["la_bn"]))):
                mu, sd, v = mean_of(d, m, "i5")
                ax.bar(k + (j - 0.5) * w, mu, w * 0.92, color=col, alpha=0.85, yerr=sd, capsize=2,
                       error_kw=dict(lw=0.8, ecolor=INK2))
                ax.scatter(np.full(len(v), k + (j - 0.5) * w), v, s=6, color=INK, zorder=3)
            print(f"combo {d} {lab}: base {mean_of(d, b, 'i5')[0]:.2f}, +LA {mean_of(d, bla, 'i5')[0]:.2f}")
        ax.set_xticks(xs); ax.set_xticklabels([g[2] for g in groups])
        lo = min(float(r["i5"]) for r in comb if r["d"] == d) - 3
        ax.set_ylim(lo, None); ax.set_ylabel("individual accuracy (%)")
        ax.set_title("CIFAR-10" if d == "cifar10" else "CIFAR-100"); panel(ax, letter)
    h = [plt.Rectangle((0, 0), 1, 1, color=ALONE, alpha=0.85, label="communication rule alone"),
         plt.Rectangle((0, 0), 1, 1, color=COL["la_bn"], alpha=0.85, label="with logit adjustment"),
         Line2D([], [], marker="o", color=INK, ls="", ms=3, label="one seed")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.22)
    save(fig, "fig_combo")


# ================================================================ sweep: gain over every baseline in every graph
def fig_sweep_gain():
    pt = rows("sweep5_paired_tests.csv")
    GR = [("16", "ring", "16 ring\n(gap 0.051)"), ("16", "exp", "16 exp.\n(gap 0.50)"),
          ("32", "ring", "32 ring\n(gap 0.013)"), ("32", "exp", "32 exp.\n(gap 0.40)")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, c, letter, title in zip(axes, ["i5", "e5"], "ab", ["individual accuracy", "ensemble accuracy"]):
        for k, (n, t, lab) in enumerate(GR):
            for j, b in enumerate(["dpsgd", "fedlc", "kgt"]):
                r = next(r for r in pt if r["n"] == n and r["topo"] == t and r["vs"] == b and r["c"] == c)
                x = k + (j - 1) * 0.22; m, lo, hi = float(r["diff"]), float(r["lo"]), float(r["hi"])
                ax.plot([x, x], [lo, hi], color=COL[b], lw=1.4)
                sig = float(r["holm"]) < 0.05
                ax.plot(x, m, "o", color=COL[b], ms=5, mfc=COL[b] if sig else "white", mew=1.2)
                print(f"sweep {c} {n}/{t} vs {b}: {m:+.2f} [{lo:+.2f}, {hi:+.2f}] wins {r['wins']}/{r['k']} pHolm {float(r['holm']):.3f}")
        ax.axhline(0, color=INK2, lw=0.8, ls="--")
        ax.set_xticks(range(4)); ax.set_xticklabels([g[2] for g in GR], fontsize=7)
        ax.set_title(title); panel(ax, letter, -0.12 if letter == "a" else -0.04)
    axes[0].set_ylabel("LA gossip $-$ baseline (points)")
    h = [Line2D([], [], marker="o", color=COL[b], ls="-", label=NAME[b]) for b in ["dpsgd", "fedlc", "kgt"]]
    h.append(Line2D([], [], marker="o", color=INK2, ls="", mfc="white", label="$p_{\\mathrm{Holm}}\\geq0.05$"))
    axes[1].legend(handles=h, loc="upper right", fontsize=7)
    save(fig, "fig_sweep_gain")


# ================================================================ gradient heterogeneity at a common model + own models
def fig_zeta():
    e = rows("round3E_per_run.csv")
    dd = rows("round3D_per_run.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    # (a), (b): both losses at the same averaged model, per seed
    for ax, ds, letter in zip(axes[:2], ["cifar10", "cifar100"], "ab"):
        for m, mk in (("dpsgd", "o"), ("la_bn", "D")):
            sel = [r for r in e if r["ds"] == ds and r["model"] == "cnn" and r["m"] == m]
            ce = [float(r["zeta2_ce"]) for r in sel]; la = [float(r["zeta2_la"]) for r in sel]
            for a, b in zip(ce, la):
                ax.plot([0, 1], [a, b], color=COL[m], lw=0.8, alpha=0.7)
            ax.plot(np.zeros(len(ce)), ce, mk, color=COL[m], ms=4.5)
            ax.plot(np.ones(len(la)), la, mk, color=COL[m], ms=4.5, mfc="white", mew=1.1)
            ratio = np.exp(np.mean(np.log(np.array(la) / np.array(ce))))
            print(f"zeta {ds} at {m} model: CE {np.mean(ce):.2f} LA {np.mean(la):.2f} geo-ratio {ratio:.3f} (n={len(ce)})")
        ax.set_yscale("log"); ax.set_xticks([0, 1]); ax.set_xticklabels(["cross-entropy", "logit-adjusted"])
        ax.set_xlim(-0.35, 1.35); ax.set_ylabel(r"$\zeta^2$ at the averaged model")
        ax.set_title("CIFAR-10" if ds == "cifar10" else "CIFAR-100"); panel(ax, letter, -0.3)
    h = [Line2D([], [], marker="o", color=COL["dpsgd"], label="D-PSGD model"),
         Line2D([], [], marker="D", color=COL["la_bn"], label="LA model")]
    axes[1].legend(handles=h, loc="upper right", fontsize=6.5)
    # (c): each method, own loss at own model, CIFAR-10 five nodes: zeta2 and ECE
    ax = axes[2]
    for k, m in enumerate(["dpsgd", "fedlc", "la_bn"]):
        sel = [r for r in dd if r["ds"] == "cifar10" and r["m"] == m and r["zeta2"] != "" and r["n"] == "5"]
        z = [float(r["zeta2"]) for r in sel]
        ax.bar(k, np.mean(z), 0.6, color=COL[m], alpha=0.8)
        ax.scatter(np.full(len(z), k), z, s=7, color=INK, zorder=3)
        print(f"zeta own {m}: {np.mean(z):.2f} +- {np.std(z, ddof=1):.2f} (n={len(z)}); ECE ens {np.mean([float(r['ece_ens']) for r in sel]):.2f} node {np.mean([float(r['ece_node_mean']) for r in sel]):.2f}")
    ax.set_xticks(range(3)); ax.set_xticklabels(["D-PSGD", "FedLC", "LA"]); ax.set_ylabel(r"$\zeta^2$, own loss and model")
    ax.set_title("CIFAR-10, own objective"); panel(ax, "c", -0.3)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_zeta")


# ================================================================ prediction-time controls as a figure
def fig_posthoc():
    v = rows("round3E_variants.csv")
    SET = [("cifar10/cnn/n5", "CIFAR-10, CNN, 5 nodes"), ("cifar100/cnn/n5", "CIFAR-100, CNN, 5 nodes"),
           ("cifar10/resnet18/n16", "CIFAR-10, ResNet-18, 16 nodes")]
    BARS = [("D-PSGD raw", "dpsgd", ["raw"], 0.35), ("D-PSGD own prior", "dpsgd", "own", 0.5),
            ("D-PSGD mixture prior", "dpsgd", "mix", 0.65), ("D-PSGD fitted bias\n(oracle)", "dpsgd", ["fit"], 0.95),
            ("LA raw", "la_bn", ["raw"], 0.8), ("LA fitted bias\n(oracle)", "la_bn", ["fit"], 1.0)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=False)
    for ax, (s, title), letter in zip(axes, SET, "abc"):
        vals = []
        for lab, m, var, a in BARS:
            cand = [r for r in v if r["setting"] == s and r["method"] == m and
                    (r["variant"] in var if isinstance(var, list) else r["variant"].startswith(var))]
            best = max(cand, key=lambda r: float(r["ind"]))
            vals.append((lab, m, float(best["ind"]), float(best["ind_sd"]), a, best["variant"]))
        for k, (lab, m, mu, sd, a, var) in enumerate(vals):
            ax.barh(k, mu, 0.7, color=COL[m], alpha=a, xerr=sd, capsize=2, error_kw=dict(lw=0.8, ecolor=INK2))
            ax.text(mu + sd + 0.4, k, f"{mu:.1f}", va="center", fontsize=6.5, color=INK)
            print(f"posthoc {s} {lab.replace(chr(10), ' ')} ({var}): {mu:.2f}")
        ax.set_yticks(range(len(vals))); ax.invert_yaxis()
        ax.set_yticklabels([x[0] for x in vals] if letter == "a" else [""] * len(vals), fontsize=7)
        lo = min(x[2] for x in vals) - 6; hi = max(x[2] + x[3] for x in vals) + 4
        ax.set_xlim(lo, hi); ax.set_xlabel("individual accuracy (%)"); ax.set_title(title, fontsize=8)
        panel(ax, letter, -0.05 if letter != "a" else -0.75)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_posthoc")


# ================================================================ FEMNIST
def fig_femnist():
    f = rows("femnist_per_seed.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=False)
    for ax, (target, cols, labs), letter in zip(axes, [
            ("balanced", ("raw_ind_bal", "minus_ind_bal"), ("raw logits", r"$-\log\pi_{\mathrm{glob}}$")),
            ("natural", ("raw_ind_nat", "plus_ind_nat"), ("raw logits", r"$+\log\pi_{\mathrm{glob}}$"))], "ab"):
        for g, (fa, glab) in enumerate((("0.0", "A: writers"), ("0.5", "B: writers + label skew"))):
            for k, m in enumerate(["dpsgd", "fedlc", "la_bn"]):
                for j, c in enumerate(cols):
                    vals = [float(r[c]) for r in f if r["m"] == m and r["fa"] == fa]
                    x = g * 4 + k * 1.15 + (j - 0.5) * 0.5
                    ax.bar(x, np.mean(vals), 0.46, color=COL[m], alpha=0.45 if j == 0 else 0.9, hatch=None)
                    ax.scatter(np.full(len(vals), x), vals, s=4, color=INK, zorder=3)
                    print(f"femnist {target} setting {fa} {m} {c}: {np.mean(vals):.2f} (n={len(vals)})")
            ax.text(g * 4 + 1.15, ax.get_ylim()[1] if False else 0, "", fontsize=7)
        ax.set_xticks([1.15, 5.15]); ax.set_xticklabels(["A: writers", "B: writers + label skew"])
        allv = [float(r[c]) for r in f for c in cols]
        ax.set_ylim(min(allv) - 4, max(allv) + 3)
        ax.set_ylabel(f"individual {target} accuracy (%)"); ax.set_title(f"{target.capitalize()} target"); panel(ax, letter)
        h = [plt.Rectangle((0, 0), 1, 1, color=COL[m], label=NAME[m]) for m in ["dpsgd", "fedlc", "la_bn"]]
        h += [plt.Rectangle((0, 0), 1, 1, color=INK2, alpha=0.4, label=labs[0]),
              plt.Rectangle((0, 0), 1, 1, color=INK2, alpha=0.9, label=labs[1])]
        ax.legend(handles=h if letter == "b" else h[3:], loc="upper right", fontsize=6.5, ncol=1)
    save(fig, "fig_femnist")


if __name__ == "__main__":
    fig_forest(); fig_comm(); fig_combo(); fig_sweep_gain(); fig_zeta(); fig_posthoc(); fig_femnist()
