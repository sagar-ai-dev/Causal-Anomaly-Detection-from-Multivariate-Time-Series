"""
How much does a SHAP explanation move when nothing about the model changes?

This exists because of an accident. The two SHAP scripts were writing figures at
matplotlib's default resolution instead of 300 dpi, and regenerating them to fix that
changed the attribution results -- on a rerun of seeded training, with no code change
touching the model. The LSTM's top variable for the LedsControl fault moved from LedEarR3
to LedFaceBR5, and the TCN's second variable for JointControl moved from SonarB to
LShoulderPitch.

The cause is that shap.GradientExplainer samples the background distribution, and it drew
those samples from whatever RNG state training happened to leave behind. That is now seeded,
so the pipeline is repeatable. But repeatable is not stable, and the interesting question is
the one the accident raised: holding the trained model completely fixed, how much does the
explanation depend on the explainer's own randomness?

The experiment isolates that. One model is trained once. The explainer is then run K times
against it with nothing varying but its seed, and the top-3 attribution is recorded each
time. Any variation is attributable to the explainer alone -- not to training, not to
initialisation, not to data order.

The comparison this is for: the causal attribution is deterministic by construction. It
reads the ranking off residuals of a fitted model against a cached graph, so the same inputs
give the same three variables every time, and there is no seed to vary. If SHAP's ranking
moves under reseeding and the causal ranking cannot, that is a concrete argument for
structural attribution over post-hoc attribution -- and it is a measurement rather than a
preference.

Usage:  python tools/shap_stability.py            # 10 explainer seeds, LSTM
        python tools/shap_stability.py --runs 20
        python tools/shap_stability.py --arch "TCN Autoencoder"
"""

import csv
import io
import os
import sys
import warnings


warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("Baselines"))


def main():
    runs = 10
    if "--runs" in sys.argv:
        runs = int(sys.argv[sys.argv.index("--runs") + 1])
    arch = "LSTM Autoencoder"
    if "--arch" in sys.argv:
        arch = sys.argv[sys.argv.index("--arch") + 1]

    import torch
    import neural_baselines as nb

    print("SHAP stability under reseeding: %s, %d explainer seeds\n" % (arch, runs))
    print("The model is trained once and never touched again. Only the explainer seed varies.\n")

    scaler, train_raw, heldout_raw, attacks, input_dim, var_names, sub = nb.load_data()
    L = nb.SEQ_LENGTHS[0]
    x_all = torch.tensor(nb.sequences(scaler.transform(train_raw), L), dtype=torch.float32)
    n_val = max(1, int(nb.VAL_FRAC * len(x_all)))
    x_tr, x_val = x_all[:-n_val], x_all[-n_val:]

    cls = nb.ARCHITECTURES[arch]
    model, _, _ = nb.train(cls, x_tr, x_val, input_dim, nb.SEEDS[0],
                           **{"hidden_dim": 64, "num_layers": 1, "lr": 0.001})
    model.eval()
    print("  trained once: %s, %d train windows, %d variables\n"
          % (arch, len(x_tr), input_dim))

    rows = []
    for fault, raw in attacks.items():
        w = torch.tensor(nb.sequences(scaler.transform(raw), L), dtype=torch.float32)
        if len(w) == 0:
            continue
        idx = slice(0, min(len(w), 200))
        tops = []
        for s in range(runs):
            nb.SHAP_SEED = 1000 + s          # the only thing that changes
            top = nb.shap_attribution(model, x_tr[:50], w[idx], var_names,
                                      "stability_probe",
                                      outdir=os.path.join("tools", "_shap_tmp"))
            if top and str(top[0]).startswith("SHAP unavailable"):
                print("  %s: %s" % (fault, top[0]))
                tops = []
                break
            tops.append(tuple(top))
        if not tops:
            continue

        top1 = [t[0] for t in tops]
        distinct1 = sorted(set(top1))
        union3 = sorted({v for t in tops for v in t})
        # how often the modal top-1 actually wins
        modal = max(distinct1, key=top1.count)
        agree = top1.count(modal) / len(top1)

        print("  %s" % fault)
        print("    top-1 over %d seeds : %s" % (runs, ", ".join(distinct1)))
        print("    modal top-1        : %s (%d/%d = %.0f%%)"
              % (modal, top1.count(modal), len(top1), 100 * agree))
        print("    union of top-3     : %d distinct variables for 3 slots" % len(union3))
        print()
        rows.append({"Fault": fault, "Architecture": arch, "Seeds": runs,
                     "Distinct top-1": len(distinct1),
                     "Top-1 variables": " | ".join(distinct1),
                     "Modal top-1": modal,
                     "Modal agreement": "%.4f" % agree,
                     "Union of top-3": len(union3)})

    if not rows:
        raise SystemExit("no attributions produced; is shap installed?")

    out = "tools/shap_stability.csv"
    with io.open(out, "w", encoding="utf-8", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w_.writeheader()
        w_.writerows(rows)
    print("  wrote %s" % out)

    tmp = os.path.join("tools", "_shap_tmp")
    if os.path.isdir(tmp):
        for fn in os.listdir(tmp):
            os.remove(os.path.join(tmp, fn))
        os.rmdir(tmp)

    unstable = [r for r in rows if r["Distinct top-1"] > 1]
    print()
    if unstable:
        print("  %d of %d faults gave more than one top-1 variable across seeds."
              % (len(unstable), len(rows)))
        print("  The model was identical throughout. Only the explainer seed moved.")
    else:
        print("  Every fault gave the same top-1 variable across all %d seeds." % runs)
        print("  The explanation is stable under reseeding for this model.")
    print("\n  The causal attribution has no seed to vary: it reads its ranking off residuals")
    print("  of a fitted model against a cached graph, so it returns the same three")
    print("  variables every time by construction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
