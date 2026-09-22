# Thesis workspace

Everything needed to write the dissertation in Overleaf, with every table
generated from the pipelines' own output so no number is ever hand-typed.

```
Thesis/
├── main.tex              LaTeX root -- compile this
├── refs.bib              ~30 core references (VERIFY each before submitting)
├── frontmatter/          title page, abstract, acknowledgements, abbreviations
├── chapters/             eight chapters, scaffolded
├── appendices/           four appendices, scaffolded
├── tables/               10 GENERATED tables -- never edit by hand
├── figures/              32 figures, renamed LaTeX-safe
└── tools/make_tables.py  regenerates tables/ from the pipeline CSVs
```

## Getting it into Overleaf

1. Zip the `Thesis/` folder.
2. Overleaf → New Project → Upload Project.
3. Set the compiler to **pdfLaTeX** and the main document to `main.tex`.
4. It will compile immediately — the scaffold is valid LaTeX. Every chapter
   is currently comments, so you get a correctly formatted skeleton with an
   empty body, and you fill it in.

No changes needed after upload. All `\input` and `\includegraphics` paths
resolve the way LaTeX resolves them — relative to `main.tex`, not to the file
doing the including — and `Thesis/tools/check_latex.py` verifies this before
every upload:

```bash
python Thesis/tools/check_latex.py
```

It checks brace balance, environment matching, and that every `\input` and
graphics target exists. Run it whenever the compiler reports something you
cannot locate.

## Regenerating the tables

After any pipeline rerun:

```bash
python Thesis/tools/make_tables.py
```

Ten tables are rewritten from the CSVs. This is the same guarantee
`tools/verify_docs.py` gives the README, extended to the dissertation: a
result that changes changes the thesis, rather than leaving a stale number
behind for an examiner to find.

---

## What the University of Verona actually scores

From the Department of Computer Science thesis regulation
(`di.univr.it/documenti/CorsoStudi/descrizione/desc353067.pdf`), section 2.1.
The supervisor, any co-supervisors and the *controrelatore* each score seven
criteria, plus presentation quality, each from 0 to 1 in tenths. The sum is
added to your weighted exam average.

| # | Criterion (original) | What it means | Where you earn it |
|---|---|---|---|
| 1 | *livello di approfondimento* | depth relative to the state of the art | Ch. 2 — the gap argument, not the summaries |
| 2 | *avanzamento conoscitivo o tecnologico* | what the work advances | Ch. 6 RQ3 and RQ5 — the conditional claim and the power result |
| 3 | *impegno critico* | critical engagement | Ch. 7 — declaring the confounds before an examiner finds them |
| 4 | *impegno sperimentale* | experimental effort | Ch. 5 and 6 — 27 configurations, baselines at two capacity levels |
| 5 | *autonomia di lavoro* | working independently | your supervisor scores this; the *controrelatore* does not |
| 6 | *significatività delle metodologie* | soundness of method | Ch. 4 and §5.5 — the verification infrastructure |
| 7 | *accuratezza dello svolgimento e della scrittura* | accuracy of execution and writing | generated tables, verified figures, a clean bibliography |

Three consequences worth knowing:

- **Above five points, a *controrelatore* is mandatory.** If your supervisor
  believes the thesis merits more than a five-point increment he must propose
  one, and that person writes an independent written evaluation. Assume a
  hostile expert reader who did not watch you do the work. That reader is
  exactly who §5.5 and Ch. 7 are written for.
- **Exceptional theses can earn up to ten points**, but the supervisor must
  declare exceptionality at submission, and it triggers a public seminar a
  week before the defence plus an external supervisor.
- **Defence length**: 12 + 3 minutes without a *controrelatore*, **15 + 5 with
  one**. Build the slides for 15.

---

## On plagiarism and AI detection

You asked for a thesis free of both. The honest answer:

**Plagiarism** is fully solvable and the method is simple. Read a source,
close it, and write what it said from memory in your own words. Then reopen it
only to check the numbers and get the citation right. Never rewrite someone's
sentence with synonyms — that is what similarity checkers catch, and it is
also worse writing than what you would produce from your own understanding.
Every technical claim gets a citation; every number in your results chapter
comes from your own CSVs.

**AI detection** cannot be solved by editing AI text — only by not using it.
This is why the chapters in this folder contain *structure and guidance*, not
prose. Every `.tex` file tells you what the section must argue, which of your
numbers proves it, and what an examiner will attack. The sentences are yours
to write.

That is not me being unhelpful. It is the only version that survives the
defence: a *controrelatore* will ask you to explain a paragraph, and there is
no recovery from not being able to. You did this work. The results are real.
Writing about your own experiments in your own voice is the easiest writing
you will ever do — far easier than defending sentences you did not compose.

The three things that make writing genuinely yours, which you already have:

1. **Your data.** Every figure traces to a CSV your pipelines wrote.
2. **Your mistakes.** The dead leakage code, the unscaled ridge, the missing
   intercept, the force-included tablet channels — these are in the repository
   history and nobody else could write about them. They are also the best
   evidence of criterion 3 that exists. Use them.
3. **Your judgement calls.** Why 35 sensors. Why the 95th percentile. Why you
   kept the scaling fix that *cost* AUC. Nobody else can explain those.

---

## Writing order

Not chapter order. Write in this sequence:

| # | What | Why this order | Est. |
|---|------|----------------|------|
| 1 | Ch. 5 Experimental Setup | pure description, no argument needed — builds momentum | 3 d |
| 2 | Ch. 6 Results | the tables already exist; you are writing what they say | 4 d |
| 3 | Ch. 4 Methodology | now you know what needs explaining | 3 d |
| 4 | Ch. 3 Background | now you know exactly which theory is load-bearing | 3 d |
| 5 | Ch. 7 Discussion | needs 6 finished; the hardest and highest-value chapter | 4 d |
| 6 | Ch. 2 Related Work | needs 7, so you know what gap you actually filled | 4 d |
| 7 | Ch. 8 Conclusion | mechanical once 6 and 7 exist | 1 d |
| 8 | Ch. 1 Introduction | you cannot introduce findings you have not written | 2 d |
| 9 | Abstract | last, always | 0.5 d |
| 10 | Appendices, figures, bibliography | | 2 d |
| 11 | Full read-through aloud | catches everything silent reading misses | 1 d |

About four weeks of real work. Starting with Chapter 1 is the single most
common way to lose a fortnight.

---

## The five things that will most raise the mark

1. **Lead with the industrial result, don't bury it.** A parameter-free
   control (0.9286) beats every causal configuration (0.9151) on TEP. Putting
   that in the abstract signals to a *controrelatore* that nothing else is
   being hidden. Concealing it invites the assumption that other things are.

2. **Declare the scoring-resolution confound in Chapter 5, not Chapter 7.**
   PCMCI scores at 74/7/1 while GES and CAM-UV are fixed at 10, so within a
   domain the three algorithms are not scoring the same series. This is the
   first thing a reader of the code finds. Declare it, quantify it with the
   two ablations, and bound the conclusion. Found-and-declared is worth more
   than the point it costs; found-by-the-examiner costs far more.

3. **Give §5.5 the space it deserves.** Almost no Master's thesis can prove
   its own experimental claims mechanically. Fourteen uniformity checks, four
   numerical validations against closed-form answers, 459 figures traced to
   source. This is criterion 6, and it is unusual enough to be memorable.

4. **Draw the architecture figure.** Five boxes: preprocessing, discovery,
   baseline fit, online scoring, attribution — with the two variable stages
   shaded. It is the single highest-value figure you can add and it does not
   exist yet.

5. **Write the mechanism, not the ranking.** "GES beat PCMCI on ECG" is a
   result. "Causal structure pays where the fault is relational, and the
   trivial control's score tells you in advance how much room there is" is a
   contribution. The second is what gets cited, and it is what your data
   actually supports.

---

## Defence preparation

Build for 15 + 5 minutes. Roughly 12 slides:

1. Title
2. The problem — the ECG lead-identity example, one figure
3. The gap — comparisons vary everything at once
4. The design — 27 configurations, three axes, one detector
5. The verification — 14 checks (this slide surprises people)
6. Results: the main table
7. Results: causal vs. baselines — **including the industrial row**
8. Results: attribution, 9/9 and 8/9
9. Results: the TEP power sweep
10. The conditional claim
11. Limitations — resolution, separability, one dataset per domain
12. Conclusions and future work

Questions to rehearse until the answers are automatic:

- *Why lag 1?* Comparability: CAM-UV has no native lag notion.
- *Why 35 sensors?* Elbow at K=32, 94.55% variance at K=35, 160 variables did
  not converge in 66 minutes.
- *Your causal method loses to a z-score on TEP. Why is this thesis useful?*
  Because it says **when** to use which, and the z-score's own score is the
  diagnostic that tells you.
- *Is the LED-to-tablet result circular?* No — name-agnostic selection,
  variance ranks 2 and 7 of 249. An earlier version was circular; it was found
  and removed.
- *Why is PCMCI at a different subsample rate?* Adaptive Nyquist-based rule
  from the tigramite workflow. Declared as the primary threat to internal
  validity; ablations bound it at roughly 0.10–0.19 AUC.
- *Is any of your ranking statistically significant?* No winner is separable
  from its runner-up. The ordering is stable; the ranking is not established.

Rehearse the third and the sixth most. They are the ones a *controrelatore*
asks, and answering them calmly — because you already wrote them down — is
what distinguishes a defended thesis from a survived one.
