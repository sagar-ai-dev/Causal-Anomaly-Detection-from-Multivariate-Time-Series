"""
Add an opt-in eight-lead mode to the nine healthcare configurations.

Four of the twelve standard ECG leads are exact linear functions of the other eight:

    III = II - I,   aVR = -(I + II)/2,   aVL = I - II/2,   aVF = II - I/2

A causal graph learned over all twelve therefore recovers algebra as well as physiology,
and the healthcare result -- the strongest in the thesis -- cannot separate the two. The
reviewer asked for a rerun on the independent leads only, as the last experiment if time
allowed. This makes that rerun possible without a second copy of the pipeline.

The mode is off unless ECG_INDEPENDENT_LEADS=1 is set in the environment, and when it is
on, the cache and the output directory both take a _leads8 suffix, so an ablation run
cannot overwrite the headline results.

Two places need the restriction, not one. The normal data flows through read_data, but the
attack frames are read straight from the pickle and then indexed positionally by the
`nonconst` column list derived from the *normal* frame. Restricting one without the other
would silently misalign the columns and produce plausible, wrong numbers -- so both are
patched, and the script refuses to write a file where it cannot find both anchors.

Usage:
    python tools/enable_ecg_lead_ablation.py           # apply
    python tools/enable_ecg_lead_ablation.py --check   # report only
"""

import io
import os
import re
import sys

SCRIPTS = [
    "PCMCI/PCMCI_healthcare/anomaly_detection.py",
    "PCMCI/PCMCI_healthcare/anomaly_detection_polynomial.py",
    "PCMCI/PCMCI_healthcare/anomaly_detection_rbf.py",
    "GES/GES_healthcare/anomaly_detection_ges.py",
    "GES/GES_healthcare/anomaly_detection_ges_polynomial.py",
    "GES/GES_healthcare/anomaly_detection_ges_rbf.py",
    "CAM-UV/CAM_UV_healthcare/anomaly_detection.py",
    "CAM-UV/CAM_UV_healthcare/anomaly_detection_polynomial.py",
    "CAM-UV/CAM_UV_healthcare/anomaly_detection_rbf.py",
]

MARKER = "ECG_INDEPENDENT_LEADS"

BLOCK = '''
# ============================================================================
#            ECG INDEPENDENT-LEAD ABLATION  (off unless requested)
# ============================================================================
# Einthoven's and Goldberger's relations make four of the twelve standard leads
# exact linear functions of the other eight:
#
#     III = II - I,  aVR = -(I + II)/2,  aVL = I - II/2,  aVF = II - I/2
#
# A causal graph over all twelve therefore recovers algebra alongside
# physiology. Setting ECG_INDEPENDENT_LEADS=1 restricts the pipeline to the
# eight independent leads so the two can be told apart. Cache and output paths
# are suffixed, so an ablation run cannot overwrite the headline results.
INDEPENDENT_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
ABLATION_8_LEADS = os.environ.get("ECG_INDEPENDENT_LEADS") == "1"
ABL = "_leads8" if ABLATION_8_LEADS else ""


def restrict_leads(df):
    """Keep only the eight independent leads, when the ablation is enabled."""
    if not ABLATION_8_LEADS or not hasattr(df, "columns"):
        return df
    keep = [c for c in INDEPENDENT_LEADS if c in df.columns]
    return df[keep] if len(keep) == len(INDEPENDENT_LEADS) else df

'''


def patch_file(path, apply=True):
    text = io.open(path, encoding="utf-8").read()
    if MARKER in text:
        return "already patched", text

    notes = []
    out = text

    # 1. the ablation block, inserted immediately before the data loader
    anchor = re.search(r"\ndef read_data\(", out)
    if not anchor:
        return "FAILED: no read_data", text
    out = out[:anchor.start()] + "\n" + BLOCK + out[anchor.start():]
    notes.append("block")

    # 2. normal data -- wrap read_data's result
    m = re.search(r"\ndef read_data\(path: str, task: str\)[^\n]*\n", out)
    if not m:
        return "FAILED: no read_data signature", text
    out = (out[:m.start()] + "\ndef _read_data_unrestricted(path: str, task: str):\n"
           + out[m.end():])
    # append the wrapper right after the original function body
    nxt = re.search(r"\n\n\n(?=[A-Za-z@#])", out[m.start():])
    ins = m.start() + (nxt.end() - 1 if nxt else 0)
    wrapper = ("\ndef read_data(path, task):\n"
               "    return restrict_leads(_read_data_unrestricted(path, task))\n\n")
    out = out[:ins] + wrapper + out[ins:]
    notes.append("read_data")

    # 3. attack frames -- PCMCI subsets by ECG_LEADS, GES and CAM-UV do not subset at all
    before = out
    out = out.replace(
        "attack_dfs = [attack_data_dict[k][[c for c in ECG_LEADS "
        "if c in attack_data_dict[k].columns]] for k in attack_names]",
        "attack_dfs = [restrict_leads(attack_data_dict[k][[c for c in ECG_LEADS "
        "if c in attack_data_dict[k].columns]]) for k in attack_names]")
    if out != before:
        notes.append("attacks(pcmci)")
    else:
        before = out
        out = out.replace("        attack_frames = [attack_dict[k] for k in attack_names]",
                          "        attack_frames = [restrict_leads(attack_dict[k]) "
                          "for k in attack_names]")
        if out != before:
            notes.append("attacks(ges/camuv)")

    if not any(n.startswith("attacks") for n in notes):
        return "FAILED: no attack anchor", text

    # 4. cache path and output directory take the suffix
    before = out
    out = re.sub(r'(os\.path\.join\(PREFIX,\s*f"\{TASK\}_normal[^"]*?)\.npz"',
                 r'\1{ABL}.npz"', out)
    if out != before:
        notes.append("cache")
    before = out
    out = out.replace('os.path.join(BASE_DIR, "outputs")',
                      'os.path.join(BASE_DIR, "outputs" + ABL)')
    if out != before:
        notes.append("outputs")

    if "cache" not in notes or "outputs" not in notes:
        return "FAILED: path anchors (%s)" % ",".join(notes), text

    if apply:
        io.open(path, "w", encoding="utf-8", newline="\n").write(out)
    return "patched: " + ", ".join(notes), out


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    os.chdir(root)
    apply = "--check" not in sys.argv
    bad = 0
    for p in SCRIPTS:
        if not os.path.exists(p):
            print("  %-52s MISSING" % p)
            bad += 1
            continue
        status, _ = patch_file(p, apply=apply)
        print("  %-52s %s" % (p.split("/", 1)[1], status))
        bad += status.startswith("FAILED")
    print()
    print("%d script(s) failed" % bad if bad else
          "all nine healthcare scripts support ECG_INDEPENDENT_LEADS=1")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
