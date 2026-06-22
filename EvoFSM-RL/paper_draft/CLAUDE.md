# CLAUDE.md — paper_draft/ working context

The EvoFSM-RL paper draft. See `../CLAUDE.md` for the project; this is the
paper-writing context.

## Framing
- Three contributions: (1) test-time adaptation for unseen apps, (2) the B4
  joint prompt+LoRA adaptation algorithm, (3) a multi-level benchmark partition.
- `experiment.tex` is ONE chapter with two levels — within-benchmark
  (AndroidWorld+) and cross-benchmark (AW+ → MobileWorld). Each level restates
  its own data structure (they differ); the two levels are peers, not
  pilot-vs-main.
- Motivation is anchored on arXiv 2603.07432 (`gu2026generalization`);
  AndroidWorld+ = its AndroidWorld-Generalization benchmark.

## Data-number discipline (important)
- Part-1 (Level-1) numbers are locked to the old `text_draft/eval.tex` lineage
  (B1 38.6 / B2 43.3→45.2 / B3 48.1). The `docs/results/experiments.md` lineage
  (B3 46.7) is NOT used here.
- Level-1 **B4 = 52.9% is a PROVISIONAL placeholder** — a per-tier cherry-pick
  (Tier-B 70.4 from v3-C + Tier-C 34.3 from v3-B), see the `% NOTE` in
  `experiment.tex`. Replace with a clean single-model rerun before submission.
- Level-2 B3/B4 are `\emph{ongoing}` (MobileWorld TTA not finished).

## Don't
- Don't put weakening material in `limitation.tex` — it is defensive only
  (inherent method/data/tool limitations + how we minimize them). Statistical
  power, the weight-channel null result, and the 52.9 cherry-pick do NOT go
  there.
- Don't invent citations — only keys already in `reference.bib`.
- Don't downgrade the joint (prompt + weight) selling point in the framing.
