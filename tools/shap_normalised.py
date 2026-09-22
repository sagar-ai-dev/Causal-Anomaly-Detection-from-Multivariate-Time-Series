"""
Put the SHAP attributions on a scale that can be read across panels.

The supervisor's note on the results chapter was that the SHAP values look low
overall. They do, but the number itself is not meaningful: GradientExplainer
returns attributions in the units of the input, so a panel's magnitude is set by
whatever the sensors happen to be scaled in. Across the nine robot panels the
largest attribution ranges from 0.22 to 227, which is three orders of magnitude
of axis, not three orders of magnitude of explanation.

Normalising each panel by its own total attribution removes that. What is left
is the share of the explanation a variable carries, which is comparable across
models and faults and against the flat baseline of 1/244 that a model spreading
attribution evenly would give.
"""
import csv, io, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "Baselines/outputs/shap_full.csv"
OUT = "Baselines/outputs/shap_full_normalised.csv"
FIG = "Thesis/figures/baseline_shap_normalised_ledscontrol.png"
TOP = 12


def load():
    by = collections.defaultdict(list)
    for r in csv.DictReader(io.open(SRC, encoding="utf-8")):
        by[(r["Model Variant"], r["Fault Name"])].append(
            (int(r["Rank"]), r["Variable"], float(r["Score"])))
    for k in by:
        by[k].sort()
    return by


def main():
    by = load()

    out = [["Dataset", "Model Variant", "Fault Name", "Rank", "Variable",
            "Score", "Share (%)", "Cumulative Share (%)"]]
    for (mv, fn), ranked in sorted(by.items()):
        tot = sum(s for _, _, s in ranked)
        cum = 0.0
        for rank, var, s in ranked:
            share = 100.0 * s / tot if tot else 0.0
            cum += share
            out.append(["Robotics (Pepper)", mv, fn, rank, var,
                        round(s, 6), round(share, 4), round(cum, 3)])
    with io.open(OUT, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(out)
    print("  wrote %s  (%d rows)" % (OUT, len(out) - 1))

    models = ["LSTM Autoencoder", "TCN Autoencoder", "Transformer Autoencoder"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True)
    for ax, mv in zip(axes, models):
        ranked = by[(mv, "LedsControl")]
        tot = sum(s for _, _, s in ranked)
        top = ranked[:TOP][::-1]
        names = [v for _, v, _ in top]
        shares = [100.0 * s / tot for _, _, s in top]
        # A variable is coloured as on-target when it belongs to the subsystem
        # the fault was injected into, which for this fault is the LEDs and the
        # tablet, following the supervisor's definition.
        on = [n.lower().startswith(("led", "tablet")) for n in names]
        ax.barh(range(len(top)), shares,
                color=["#2b6cb0" if o else "#cbd5e0" for o in on])
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(names, fontsize=8)
        flat = 100.0 / len(ranked)
        ax.axvline(flat, color="#c53030", ls="--", lw=1)
        # Log scale: the panels differ by two orders of magnitude in how much
        # they concentrate, and on a linear axis the flatter two collapse
        # against the TCN's single dominant bar.
        ax.set_xscale("log")
        ax.set_xlim(0.004, 150)
        ax.set_title("%s  (top-3 = %.0f%%)"
                     % (mv, sum(x[2] for x in ranked[:3]) / tot * 100),
                     fontsize=10)
        ax.set_xlabel("share of total attribution (%, log scale)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(flat, len(top) - 0.2, " flat 0.41%", color="#c53030",
                fontsize=7.5, va="bottom")
    fig.suptitle("Normalised SHAP attribution, LedsControl fault "
                 "(blue = LED or tablet variable)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG, dpi=300, bbox_inches="tight")
    print("  wrote %s" % FIG)


if __name__ == "__main__":
    main()
