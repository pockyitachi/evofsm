# CLAUDE.md — docs/ working context

All project documentation. See `../CLAUDE.md` for the project-wide picture;
this file is only what you need when reading or writing docs in this folder.

## Map
- `design/` — paper-ready specs. `dataset.md` (splits, counts, packages,
  T_adapt/T_eval protocol, citations) and `algorithm.md` (E-SPL + two-layer FSM
  + B1–B4). These are the distilled, cite-checked versions.
- `results/` — experiment reports. `experiments.md` is the main B1→B4 table;
  `b1_b2_static_baselines.md`, `b3_evolution.md` are the per-rung writeups; the
  `b4_*` files (diagnosis / index / training-data-format) + `history_length_ablation.md`
  + `trajectory_audit.md` are deep-dives; `figures/` holds the paper PNGs.
- `plan/` — raw working docs. `adr/00{1,2,3}-*.md` (emulator path, base model,
  snapshot schema), `linear_tickets.md` (Epic→Story→Task), `spike_verl_migration.md`.
- Top-level: `PROJECT_OVERVIEW.md` (intro), `method_report.md` (one-read
  synthesis), `project_progress_2026_05.md` (dated progress, Chinese).

## Source of truth
- `design/` is **paper-ready** — when it conflicts with anything else, trust it.
  Its upstream raw drafts live at `../plan/algorithm_design.md` + `../plan/project_plan.md`
  (lower priority).
- `plan/` here is **raw working material** (decisions + tickets), not vetted prose.
- For authoritative counts/packages/splits, the machine-readable truth is
  `../configs/splits.yaml`, not any prose doc.

## Don't
- **Don't quote B4 = 52.9% as a real result.** The 52.9% headline in
  `results/experiments.md` is a per-tier CHERRY-PICKED oracle — Tier-B taken
  from the v3-C checkpoint, Tier-C from v3-B, i.e. two different models stitched
  together. The clean single-model B4 = **48.1%** (the `b4_k4_teval` row, which
  equals B3 under that protocol). Use 48.1; if you must mention 52.9, flag it as
  an oracle upper bound.
- Don't treat `results/` reports as a single consistent snapshot — they are
  dated and were written across the project; check the date line and prefer the
  newest for a given rung.
- Don't invent taxonomies, splits, or numbers — this is a paper; everything
  must trace to `design/`, `configs/`, or a results report.
- Don't create docs outside `EvoFSM-RL/docs/`. Place new docs in the matching
  subfolder (`design/` / `results/` / `plan/`).
