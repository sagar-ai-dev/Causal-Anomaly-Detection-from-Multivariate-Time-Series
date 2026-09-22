# Data

The code in this repository reads three datasets. None of them is mirrored
here: two are public benchmarks that should be fetched from their own sources,
and the third is not mine to redistribute.

| Domain | Dataset | Where it comes from |
| --- | --- | --- |
| Healthcare | PTB-XL, 12-lead ECG | [physionet.org/content/ptb-xl](https://physionet.org/content/ptb-xl/) |
| Industrial | Tennessee Eastman process | [Rieth et al. 2017, Harvard Dataverse](https://doi.org/10.7910/DVN/6C3JR1) |
| Robotics | Pepper service-robot recordings | Recorded for the Explainable AI course laboratory, University of Verona. Not redistributed here -- contact the course staff. |

## Where each dataset goes

Place the files so that each algorithm-domain folder finds them at the paths
the scripts already expect:

```
<ALGORITHM>/<ALGORITHM>_robotic/pepper_csv/        normal.csv, LedsControl.csv,
                                                   JointControl.csv, WheelsControl.csv
<ALGORITHM>/<ALGORITHM>_healthcare/Healthcare dataset/
<ALGORITHM>/<ALGORITHM>_industrial/TEP/
```

`.gitignore` keeps those three directory names out of version control, so a
local copy of the data will not be committed by accident.

## What is published

Everything downstream of the data: the discovery, regression and scoring code
for all 27 configurations, the result CSVs and figures each run wrote, the
verification tools, and the full LaTeX source of the thesis under `Thesis/`.
Every number quoted in the manuscript is regenerated from these outputs and
checked by the scripts in `tools/`.
