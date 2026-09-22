"""
SHAP attribution for the neural baselines on the healthcare and industrial data.

The supervisor asked for the causal attribution and the post-hoc neural one to be
compared on every dataset, not only on the robot. The original baseline script is
wired to the Pepper telemetry throughout: its loader reads the PCMCI robotics
cache, its faults are the three injected attacks, and the dataset label is fixed.
Rather than rewire it and put the published robotics numbers at risk, this script
reuses its architectures, its training loop and its SHAP routine, and supplies the
other two domains' data through loaders that mirror the causal pipelines exactly.

Mirroring matters more than convenience here. The neural baselines have to see the
same matrix the causal pipeline saw -- same subsampling, same constant-column
exclusion, same fault-free split -- or the attributions are not being scored on the
same problem. Each loader therefore reads the subsample stride and the non-constant
column set out of that domain's own PCMCI cache instead of recomputing them.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import neural_baselines as NB

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HC_DIR = os.path.join(ROOT, "PCMCI", "PCMCI_healthcare", "Healthcare dataset")
TEP_DIR = os.path.join(ROOT, "PCMCI", "PCMCI_industrial", "TEP")

ECG_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]
TEP_META = ["faultNumber", "simulationRun", "sample"]
SIM_RUN = 1
FAULT_ONSET_SAMPLE = 160


def _cache(path):
    f = np.load(path, allow_pickle=True)
    return int(f["subsample"]), f["nonconst"], f["var"]


def load_healthcare():
    """Twelve ECG leads, twenty diagnosed records, as the causal pipeline reads them."""
    sub, nonconst, _ = _cache(os.path.join(HC_DIR, "healthcare_normal.npz"))
    normal = pd.read_pickle(os.path.join(HC_DIR, "healthcare_normal_train.pkl"))
    leads = [c for c in ECG_LEADS if c in normal.columns]
    normal = normal[leads]
    var_names = [leads[i] for i in nonconst]

    vals = normal.values
    train_raw = np.nan_to_num(vals[:int(NB.TRAINING_FRAC * len(vals))][::sub][:, nonconst])
    full_raw = np.nan_to_num(vals[::sub][:, nonconst])
    heldout_raw = full_raw[len(train_raw):]

    attacks = {}
    d = pd.read_pickle(os.path.join(HC_DIR, "healthcare_attacks.pkl"))
    for k in d:
        cols = [c for c in ECG_LEADS if c in d[k].columns]
        attacks[k] = np.nan_to_num(d[k][cols].values[::sub][:, nonconst])
    return train_raw, heldout_raw, attacks, var_names, sub


def _tep(path):
    df = pd.read_pickle(path)
    if "simulationRun" in df.columns:
        df = df[df["simulationRun"] == SIM_RUN]
    if "sample" in df.columns:
        df = df.sort_values("sample")
    df = df.drop(columns=[c for c in TEP_META if c in df.columns])
    if len(df.columns) == 52:
        df.columns = ([f"XMEAS_{i}" for i in range(1, 42)]
                      + [f"XMV_{i}" for i in range(1, 12)])
    return df.reset_index(drop=True)


def load_industrial():
    """Fifty-two plant sensors, twenty simulated faults, one simulation run."""
    sub, nonconst, _ = _cache(os.path.join(TEP_DIR, "tep_normal.npz"))
    normal = _tep(os.path.join(TEP_DIR, "TEP_FaultFree_Training_run1.pkl"))
    var_names = [normal.columns[i] for i in nonconst]

    train_raw = np.nan_to_num(normal.values[::sub][:, nonconst])
    heldout = _tep(os.path.join(TEP_DIR, "TEP_FaultFree_Testing_run1.pkl"))
    heldout_raw = np.nan_to_num(heldout.values[::sub][:, nonconst])

    attacks = {}
    for i in range(1, 21):
        p = os.path.join(TEP_DIR, f"TEP_Faulty_Testing_fault{i}.pkl")
        if not os.path.exists(p):
            continue
        # TEP injects the fault at sample 160. The causal pipeline attributes over
        # post-onset windows only, so passing the whole file here would dilute every
        # attribution with 160 genuinely fault-free samples.
        a = np.nan_to_num(_tep(p).values[::sub][:, nonconst])
        attacks[f"Fault_{i}"] = a[FAULT_ONSET_SAMPLE // max(sub, 1):]
    return train_raw, heldout_raw, attacks, var_names, sub


DOMAINS = {
    "Healthcare (PTB-XL)": load_healthcare,
    "Industrial (TEP)": load_industrial,
}


def run(dataset, loader):
    train_raw, heldout_raw, attacks, var_names, sub = loader()
    L = NB.SEQ_LENGTHS[0]
    print("=" * 78)
    print(f" {dataset}: {len(var_names)} variables, subsample {sub}, "
          f"{len(train_raw)} fault-free train, {len(attacks)} faults")
    print("=" * 78)

    scaler = StandardScaler().fit(train_raw)
    x_tr, x_val = NB.split(scaler, train_raw, L)

    for arch, cls in NB.ARCHITECTURES.items():
        print(f"\n  {arch}")
        model, _, _ = NB.train(cls, x_tr, x_val, len(var_names), seed=0,
                               hidden_dim=64, num_layers=1, lr=1e-3)
        for fault in sorted(attacks,
                            key=lambda s: int("".join(c for c in s if c.isdigit()) or 0)):
            w = NB.sequences(scaler.transform(attacks[fault]), L)
            if len(w) == 0:
                print(f"    {fault}: too short for a window, skipped")
                continue
            x = torch.tensor(w, dtype=torch.float32)
            top = NB.shap_attribution(model, x_tr[:50], x, var_names,
                                      f"{arch}_{fault}", dataset=dataset,
                                      arch=arch, fault=fault)
            print(f"    {fault}: " + ", ".join(top))


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for dataset, loader in DOMAINS.items():
        if only and only.lower() not in dataset.lower():
            continue
        run(dataset, loader)
    print("\ndone")


if __name__ == "__main__":
    main()
