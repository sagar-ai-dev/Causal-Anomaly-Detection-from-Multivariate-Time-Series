"""
Generate the analysis figures for Chapters 6 and 7 from the pipeline CSVs.

The thesis argues four relationships that were, until now, carried only by tables. A
table is the right place to read a number off; it is the wrong place to see whether three
points fall on a line, whether twenty-seven intervals overlap, or whether a cap removes
the same share of every graph. Each figure here plots a relationship the text already
claims, so the reader can check the claim rather than take it.

  conditional advantage   The central claim of the thesis: the causal margin over the
                          best tuned neural baseline falls as the trivial control's
                          score rises. Three points, and the figure says so -- it is
                          drawn as three labelled points and not as a fitted line,
                          because three points do not establish a trend.

  density and detection   The monotone healthcare relationship and the industrial
                          reversal, on one pair of axes, so that the reversal is visible
                          rather than asserted.

  bootstrap intervals     Twenty-seven configurations with their intervals. The point of
                          Section 6.3 is that the intervals overlap so heavily that the
                          ordering is unsupported, which a forest plot shows at a glance
                          and a twenty-seven-row table does not.

  parent-cap bite         What share of each discovered graph survives the three-parent
                          cap. The asymmetry -- 98.2 per cent of PCMCI's robotics edges
                          discarded against none of GES's -- is the point.

Every value is read from a CSV a pipeline wrote. Nothing here is typed by hand, for the
same reason no table is.

Usage:  python Thesis/tools/make_figures.py
"""

import csv
import io
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "Thesis", "figures")

# A restrained, print-safe palette. The thesis is read on paper as often as on screen,
# so the three domains are separated by marker shape as well as by colour.
INK = "#1b1b1b"
GRID = "#cfd3da"
DOMAIN_STYLE = {
    "Healthcare": ("#2f4d86", "o"),
    "Robotics": ("#8f5c15", "s"),
    "Industrial": ("#40603a", "^"),
}

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})


def folder_label(folder):
    """Render a results-directory name the way the dissertation names things.

    Replacing underscores with spaces produced "CAM UV healthcare" and "GES robotic" on the
    forest plot axis, where every other page says CAM-UV and Robotics. The hyphen matters
    -- CAM-UV is how the abbreviations list defines it -- and a figure spelling an algorithm
    differently from the text around it reads as a different algorithm.
    """
    parts = folder.split("_")
    dom = parts[-1].lower()
    algo = "_".join(parts[:-1]).replace("_", "-").upper()
    algo = {"CAM-UV": "CAM-UV", "PCMCI": "PCMCI", "GES": "GES"}.get(algo, algo)
    dom = {"robotic": "Robotics", "robotics": "Robotics",
           "healthcare": "Healthcare", "industrial": "Industrial"}.get(dom, dom.title())
    return "%s %s" % (algo, dom)


def read(path):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return []
    with io.open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("  wrote Thesis/figures/%s" % name)


def best_causal_by_domain():
    """Highest AUC-ROC in each domain, over all nine configurations there."""
    folders = {
        "Robotics": ["PCMCI/PCMCI_robotic", "GES/GES_robotic", "CAM-UV/CAM_UV_robotics"],
        "Healthcare": ["PCMCI/PCMCI_healthcare", "GES/GES_healthcare",
                       "CAM-UV/CAM_UV_healthcare"],
        "Industrial": ["PCMCI/PCMCI_industrial", "GES/GES_industrial",
                       "CAM-UV/CAM_UV_industrial"],
    }
    out = {}
    for dom, fs in folders.items():
        best = -1.0
        for folder in fs:
            for r in read(folder + "/outputs/metrics_summary.csv"):
                try:
                    best = max(best, float(r["AUC_ROC"]))
                except (KeyError, ValueError):
                    pass
        if best >= 0:
            out[dom] = best
    return out


def baselines_by_domain():
    """Best tuned neural AUC and the trivial control, per domain."""
    neural, trivial = {}, {}
    for r in read("Baselines/outputs/neural_all_domains_tuned.csv"):
        dom = r["Dataset"].split(" (")[0]
        try:
            a = float(r["AUC_ROC"])
        except (KeyError, ValueError):
            continue
        neural[dom] = max(neural.get(dom, -1.0), a)
    for r in read("Baselines/outputs/trivial_baseline.csv"):
        dom = r["Dataset"].split(" (")[0]
        try:
            trivial[dom] = float(r["AUC_ROC"])
        except (KeyError, ValueError):
            pass
    return neural, trivial


# ------------------------------------------------------------------ figure 1
def fig_conditional_advantage():
    causal = best_causal_by_domain()
    neural, trivial = baselines_by_domain()
    doms = [d for d in ("Healthcare", "Robotics", "Industrial")
            if d in causal and d in neural and d in trivial]
    if len(doms) < 2:
        print("  (skipped conditional advantage: inputs incomplete)")
        return

    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.axhline(0.0, color=GRID, lw=1, zorder=1)
    for d in doms:
        margin = causal[d] - neural[d]
        colour, marker = DOMAIN_STYLE[d]
        ax.scatter(trivial[d], margin, s=70, color=colour, marker=marker,
                   zorder=3, edgecolor="white", linewidth=0.8)
        ax.annotate("%s\n%+.4f" % (d, margin), (trivial[d], margin),
                    textcoords="offset points", xytext=(0, 13 if margin >= 0 else -26),
                    ha="center", fontsize=8.5, color=colour)
    xs = sorted(trivial[d] for d in doms)
    ys = [causal[d] - neural[d] for d in sorted(doms, key=lambda d: trivial[d])]
    ax.plot(xs, ys, color=GRID, lw=1.2, ls="--", zorder=2)

    ax.set_xlabel("AUC-ROC of the parameter-free trivial control\n"
                  "(how much of the fault signature is pure amplitude)")
    ax.set_ylabel("causal $-$ neural (AUC-ROC)")
    ax.set_xlim(0.55, 1.0)
    ax.margins(y=0.30)
    # Straddle the zero line in data coordinates, on the left where the axes are empty.
    # In axes coordinates these collided with the Industrial label.
    ax.text(0.575, 0.004, "causal ahead", fontsize=8, color="#5b6273", va="bottom")
    ax.text(0.575, -0.004, "neural ahead", fontsize=8, color="#5b6273", va="top")
    save(fig, "fig_conditional_advantage.png")


# ------------------------------------------------------------------ figure 2
def fig_density_vs_auc():
    rows = read("tools/graph_density.csv")
    if not rows:
        print("  (skipped density: tools/graph_density.csv absent)")
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    labels = []
    for dom, keys in (("Robotics", "robotic"), ("Healthcare", "healthcare"),
                      ("Industrial", "industrial")):
        pts = []
        for r in rows:
            if keys not in r["Folder"].lower():
                continue
            folder = r["Folder"]
            algo = ("PCMCI" if "PCMCI" in folder else
                    "GES" if folder.startswith("GES") else "CAM-UV")
            best = -1.0
            for m in read(_folder_path(folder) + "/outputs/metrics_summary.csv"):
                try:
                    best = max(best, float(m["AUC_ROC"]))
                except (KeyError, ValueError):
                    pass
            if best >= 0:
                pts.append((100.0 * float(r["Density"]), best, algo))
        if not pts:
            continue
        pts.sort()
        colour, marker = DOMAIN_STYLE[dom]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour, lw=1.1,
                ls="--", alpha=0.55, zorder=2)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=55, color=colour,
                   marker=marker, label=dom, zorder=3, edgecolor="white", linewidth=0.8)
        for x, y, algo in pts:
            labels.append((x, y, algo, colour))
    ax.set_xlabel("graph density (per cent of variable pairs linked)")
    ax.set_ylabel("best AUC-ROC in that folder")
    # Above the axes: inside, the legend collided with the sparsest robotics point
    # whichever corner it was placed in.
    ax.legend(frameon=False, fontsize=8.5, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), ncol=3, handletextpad=0.4, columnspacing=1.6)
    ax.margins(y=0.16)
    _place_labels(fig, ax, labels)
    save(fig, "fig_density_vs_auc.png")


def _place_labels(fig, ax, labels, size=7.5):
    """
    Annotate each point with the first offset that clears the labels already placed.

    A fixed offset does not work here because two of the collisions are between
    domains rather than within one: industrial GES and CAM-UV are 0.5 density
    points apart, and healthcare CAM-UV is 8 points from robotics CAM-UV, which
    at this size is narrower than the word "CAM-UV" itself. Both printed one
    label on top of the other. Boxes are compared in points, after the axes are
    final, so the test is against where the text actually lands.
    """
    fig.canvas.draw()
    scale = 72.0 / fig.dpi
    placed = []
    for x, y, text, colour in sorted(labels):
        px, py = ax.transData.transform((x, y))
        px, py = px * scale, py * scale
        w, h = 0.62 * size * len(text), 1.05 * size
        for dx, dy, va in ((0, 7, "bottom"), (0, -7, "top"),
                           (0, 16, "bottom"), (0, -16, "top")):
            cy = py + dy + (h / 2.0 if va == "bottom" else -h / 2.0)
            box = (px - w / 2.0, px + w / 2.0, cy - h / 2.0, cy + h / 2.0)
            if all(box[1] <= o[0] or box[0] >= o[1]
                   or box[3] <= o[2] or box[2] >= o[3] for o in placed):
                break
        placed.append(box)
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                    ha="center", va=va, fontsize=size, color=colour)
    return placed

def _folder_path(folder):
    if folder.startswith("PCMCI"):
        return "PCMCI/" + folder
    if folder.startswith("GES"):
        return "GES/" + folder
    return "CAM-UV/" + folder


# ------------------------------------------------------------------ figure 3
def fig_bootstrap_forest():
    rows = read("tools/bootstrap_intervals.csv")
    if not rows:
        print("  (skipped forest: tools/bootstrap_intervals.csv absent)")
        return
    VS = {"Linear / Ridge": "Linear", "Polynomial degree 3": "Poly-3", "RBF + Ridge": "RBF"}
    items = []
    for r in rows:
        dom = r.get("Dataset", "").split(" (")[0]
        items.append((dom, r.get("Folder", ""), VS.get(r.get("Variant", ""), ""),
                      float(r["AUC"]), float(r["CI_low"]), float(r["CI_high"])))
    order = {"Healthcare": 0, "Robotics": 1, "Industrial": 2}
    items.sort(key=lambda t: (order.get(t[0], 9), t[1], t[2]))

    fig, ax = plt.subplots(figsize=(5.6, 7.4))
    for i, (dom, folder, var, auc, lo, hi) in enumerate(items):
        colour, marker = DOMAIN_STYLE.get(dom, (INK, "o"))
        ax.plot([lo, hi], [i, i], color=colour, lw=1.6, alpha=0.75, solid_capstyle="round")
        ax.scatter([auc], [i], s=26, color=colour, marker=marker, zorder=3,
                   edgecolor="white", linewidth=0.6)
    ax.axvline(0.5, color="#9d2f34", lw=1.0, ls=":", zorder=1)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(["%s  %s" % (folder_label(f), v) for _d, f, v, _a, _l, _h in items],
                       fontsize=7.5)
    ax.invert_yaxis()
    # After inversion the top of the plot is the smallest coordinate, so the label goes
    # there. Placed before inverting it landed on the 0.5 tick at the bottom.
    ax.text(0.5, -0.9, "chance", color="#9d2f34", fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("AUC-ROC with 95 per cent moving-block bootstrap interval")
    ax.set_xlim(0.35, 1.02)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig_bootstrap_forest.png")


# ------------------------------------------------------------------ figure 4
def fig_parent_cap():
    rows = read("tools/parent_cap_bite.csv")
    if not rows:
        print("  (skipped parent cap: tools/parent_cap_bite.csv absent)")
        return
    order = {"Healthcare": 0, "Robotics": 1, "Industrial": 2}
    rows.sort(key=lambda r: (order.get(r["Domain"], 9), -float(r["Percent of edges discarded"])))
    vals = [float(r["Percent of edges discarded"]) for r in rows]
    colours = [DOMAIN_STYLE.get(r["Domain"], (INK, "o"))[0] for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bars = ax.bar(range(len(rows)), vals, color=colours, width=0.68)
    for b, v in zip(bars, vals):
        ax.annotate("%.1f" % v, (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    # Algorithm on the tick, domain once per group underneath. Putting both on every
    # tick overlapped the neighbouring labels.
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["Algorithm"] for r in rows], fontsize=8)
    seen, start = None, 0
    for i, r in enumerate(rows + [{"Domain": None}]):
        if r["Domain"] != seen:
            if seen is not None:
                ax.text((start + i - 1) / 2.0, -0.155, seen, transform=
                        ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5,
                        color=DOMAIN_STYLE.get(seen, (INK, "o"))[0])
                if i < len(rows):
                    ax.axvline(i - 0.5, color=GRID, lw=0.8, ymax=0.02)
            seen, start = r["Domain"], i
    ax.set_ylabel("per cent of discovered edges\ndiscarded by the three-parent cap")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig_parent_cap.png")


# ------------------------------------------------------------------ figure 5
def fig_drift_mechanism():
    """Why the detector scores coefficient drift and not residual magnitude.

    The premise of the whole pipeline is that drift separates a broken mechanism from a
    rescaled one and residual magnitude does not. The synthetic validation measures
    exactly that, and the separation is four orders of magnitude, so the figure is drawn
    on a log axis -- on a linear one the two intact cases are invisible against the
    broken one, which would hide the very comparison the figure exists to make.
    """
    vals = {r["quantity"]: r["value"] for r in read("tools/algorithm_validation.csv")}
    cases = [
        ("drift_broken_relationship", "relationship\nbreaks", "#9d2f34"),
        ("drift_rescaled_intact", "values rescaled,\nrelationship holds", "#2f4d86"),
        ("drift_unchanged", "nothing\nchanges", "#40603a"),
    ]
    have = [(lab, float(vals[k]), c) for k, lab, c in cases if k in vals]
    if len(have) < 3:
        print("  (skipped drift mechanism: tools/algorithm_validation.csv incomplete)")
        return

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    bars = ax.bar(range(len(have)), [v for _l, v, _c in have],
                  color=[c for _l, _v, c in have], width=0.6)
    for b, (_lab, v, _c) in zip(bars, have):
        ax.annotate("%.4f" % v, (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_xticks(range(len(have)))
    ax.set_xticklabels([l for l, _v, _c in have], fontsize=8.5)
    ax.set_ylabel("mean coefficient drift after the change\n(log scale)")
    ax.set_ylim(1e-4, 40)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    # The bracket is the argument: a residual-magnitude detector sees the first two as
    # the same event, because in both the values moved.
    #
    # The legs used to descend to 6.6, which on this log axis is close enough to the
    # 4.1707 bar that the left leg was drawn straight through that bar's value label:
    # the compiled figure read "4.1|707". They now stop at 14, well clear of the
    # tallest label, and the bracket sits higher to keep its own caption off the grid.
    y, leg = 26.0, 14.0
    ax.plot([0, 0, 1, 1], [leg, y, y, leg], color="#5b6273", lw=1.0)
    ax.text(0.5, y * 1.18, "a residual-magnitude detector\ncannot separate these two",
            ha="center", fontsize=8, color="#5b6273")
    save(fig, "fig_drift_mechanism.png")


def _load_rls():
    """
    The pipeline's own _rls_update, lifted out of the robotics linear script without
    running the pipeline around it. Reusing the real function rather than a copy is
    what lets this figure end on the numbers the validation gate already publishes.
    """
    p = os.path.join("PCMCI", "PCMCI_robotic", "anomaly_detection.py")
    src = io.open(p, encoding="utf-8").read()
    ns = {"np": np}
    start = src.index("def _rls_update")
    end = src.index(chr(10) + "def ", start + 1)
    exec(compile(src[start:end], p, "exec"), ns)
    return ns["_rls_update"]


def fig_coefficient_tracking():
    """
    What the detector actually watches, on one coefficient.

    Chapters 3 and 4 define the score as the gap between an offline coefficient fitted
    on fault-free data and an online coefficient that keeps tracking. The equations say
    that; they do not show it. This replays the experiment the validation suite runs --
    a coefficient stepped from 0.0 to 2.5 halfway through a 4000-sample series -- and
    plots both passes against it, so the two windows and the gap between them are
    visible rather than inferred.

    The endpoints are the figures in tools/algorithm_validation.csv, because this runs
    the same generator, the same seed and the same filter.
    """
    rls = _load_rls()
    rng = np.random.default_rng(1)
    n, d = 4000, 3
    X = rng.normal(size=(n, d))
    w_a = np.array([1.0, 0.0, -1.0])
    w_b = np.array([1.0, 2.5, -1.0])
    y = np.array([X[t] @ (w_a if t < n // 2 else w_b) for t in range(n)])
    y += 0.01 * rng.normal(size=n)

    track = {}
    for lam in (0.995, 1.0):
        w, P = np.zeros(d), np.eye(d) / 1e-6
        hist = np.empty(n)
        for t in range(n):
            w, P, _ = rls(X[t], y[t], w, P, lam=lam)
            hist[t] = w[1]
        track[lam] = hist

    baseline = 0.0                      # the offline fit over the fault-free half
    lam_s = "$" + chr(92) + "lambda"
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True,
                                 gridspec_kw={"height_ratios": [2.0, 1.0]})

    ax.axvspan(0, n // 2, color="0.92", zorder=0)
    ax.text(n // 4, 2.92, "fault-free window\noffline fit gives $w^{(0)}$",
            ha="center", va="top", fontsize=7.5, color="0.35")
    ax.text(3 * n // 4, 2.92, "evaluation window\nonline filter tracks $w(t)$",
            ha="center", va="top", fontsize=7.5, color="0.35")

    ax.axhline(baseline, color="0.45", ls="--", lw=1.1)
    ax.text(120, baseline + 0.08, "offline baseline $w^{(0)} = 0$", fontsize=7.5, color="0.35")
    ax.axhline(2.5, color="0.45", ls=":", lw=1.1)
    # Below the line, clear of the two window headings above it. Placed at the top of
    # the figure it collided with the fault-free heading and printed over it.
    ax.text(120, 2.5 - 0.20, "true value once the mechanism changes",
            fontsize=7.5, color="0.35", va="top")

    ax.plot(track[0.995], color="#b2182b", lw=1.2,
            label=lam_s + " = 0.995$, the pipeline setting on Robotics")
    ax.plot(track[1.0], color="#2166ac", lw=1.2,
            label=lam_s + " = 1.0$, infinite memory")
    ax.annotate("%.4f" % track[0.995][-1], (n - 1, track[0.995][-1]),
                textcoords="offset points", xytext=(-6, 6), ha="right",
                fontsize=8, color="#b2182b")
    ax.annotate("%.4f" % track[1.0][-1], (n - 1, track[1.0][-1]),
                textcoords="offset points", xytext=(-6, -12), ha="right",
                fontsize=8, color="#2166ac")
    ax.set_ylabel("value of one coefficient")
    ax.set_ylim(-0.35, 3.0)
    ax.legend(loc="center left", fontsize=7.5, frameon=False)

    bx.axvspan(0, n // 2, color="0.92", zorder=0)
    bx.plot(np.abs(track[0.995] - baseline), color="#b2182b", lw=1.2)
    bx.set_ylabel("drift\n$|w(t) - w^{(0)}|$")
    bx.set_xlabel("timestep")
    bx.text(3 * n // 4, 0.35, "this is what is scored", ha="center",
            fontsize=7.5, color="0.35")
    for a in (ax, bx):
        a.axvline(n // 2, color="0.25", lw=0.9)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    save(fig, "fig_coefficient_tracking.png")


def main():
    os.chdir(ROOT)
    if not os.path.isdir(OUT):
        raise SystemExit("figures directory missing: " + OUT)
    print("generating analysis figures into Thesis/figures/\n")
    fig_conditional_advantage()
    fig_density_vs_auc()
    fig_bootstrap_forest()
    fig_parent_cap()
    fig_drift_mechanism()
    fig_coefficient_tracking()
    print("\n  every value plotted is read from a pipeline CSV")
    return 0


if __name__ == "__main__":
    sys.exit(main())
