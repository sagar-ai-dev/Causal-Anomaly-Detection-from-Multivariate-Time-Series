# RQ3 against tuned baselines

The neural baselines were searched on robotics only. Healthcare and industrial trained one
fixed configuration, so every cross-domain comparison in the draft set a tuned causal
pipeline against an untuned neural one — the criticism this thesis makes of the published
literature, sitting inside its own evidence.

The identical twelve-configuration grid has now been run on all three domains, selected on
fault-free validation loss so no attack label is consulted. `neural_all_domains_tuned.csv`.

## What the search chose

Healthcare and industrial had genuinely been left at the wrong settings — seven of the nine
architecture/domain cells moved off the fixed configuration:

| Domain | Architecture | Fixed | Searched |
|---|---|---|---|
| Robotics | LSTM | h64 l1 lr.001 | h64 l1 lr.001 — unchanged |
| Robotics | TCN | h64 l1 lr.001 | **h64 l1 lr.01** |
| Robotics | Transformer | h64 l1 lr.001 | **h64 l1 lr.01** |
| Healthcare | LSTM | h64 l1 lr.001 | **h64 l3 lr.01** |
| Healthcare | TCN | h64 l1 lr.001 | **h32 l1 lr.001** |
| Healthcare | Transformer | h64 l1 lr.001 | h64 l1 lr.001 — unchanged |
| Industrial | LSTM | h64 l1 lr.001 | **h32 l2 lr.01** |
| Industrial | TCN | h64 l1 lr.001 | **h64 l1 lr.01** |
| Industrial | Transformer | h64 l1 lr.001 | **h64 l2 lr.01** |

## The comparison that matters

Best AUC-ROC per domain, all four method families:

| Domain | Best causal | Neural untuned | **Neural tuned** | Trivial control |
|---|---:|---:|---:|---:|
| Robotics | 0.8752 | 0.8809 | **0.8856** | 0.8139 |
| Healthcare | **0.8114** | 0.6815 | 0.7077 | 0.6136 |
| Industrial | 0.9151 | 0.9996 | **0.9997** | 0.9286 |

**The headline: tuning changed no conclusion anywhere.** Healthcare moved 0.6815 → 0.7077,
robotics 0.8809 → 0.8856, industrial 0.9996 → 0.9997. The untuned-baseline flaw was real and
worth fixing, but it was not holding up any result. That is worth stating plainly, because it
is the kind of thing a reader assumes was swept away rather than checked.

## What each domain actually says

**Healthcare — the claim survives, and this is the thesis's one clean win.** GES linear at
0.8114 against the best tuned neural at 0.7077, a margin of 0.104, with the trivial control
far below at 0.6136. Tuning closed only a quarter of the gap.

**Robotics — causal loses, and it lost before tuning too.** 0.8752 against 0.8809 untuned.
Any sentence claiming causal superiority on Pepper is unsupported by the data in this
repository, and was already unsupported before the search ran. The honest framing is that the
three families are within roughly 0.01 of each other here, which is inside the bootstrap
interval, so the domain separates nothing.

**Industrial — causal loses to everything, including a method with no learned parameters.**
0.9151 against a tuned TCN at 0.9997 and a z-score baseline at 0.9286. TEP faults are large
and abrupt, so reconstruction error finds them trivially and structure buys nothing. This
belongs in the discussion as a scope limit on when causal discovery is worth its cost, and it
is a more interesting result than a win would have been.

## One caveat that needs a significance test

The healthcare win narrows sharply when both corrections apply at once. Against the eight
independent leads — dropping the four that Einthoven and Goldberger make exact linear
functions of the others — GES falls to 0.7418, and the tuned neural baseline is 0.7077:

| Comparison | Causal | Neural tuned | Margin |
|---|---:|---:|---:|
| 12 leads, untuned neural | 0.8114 | 0.6815 | 0.1299 |
| 12 leads, tuned neural | 0.8114 | 0.7077 | 0.1037 |
| **8 leads, tuned neural** | **0.7418** | **0.7077** | **0.0341** |

0.0341 is small enough that it must be tested, not asserted. Run DeLong on the paired scores
(`delong1988comparing` is in `refs.bib`) — Hanley–McNeil gives an interval for one AUC and
cannot answer whether two differ. Until that test is run, the defensible claim is the
twelve-lead one, with the eight-lead figure reported alongside it as a robustness check whose
margin is not yet established.
