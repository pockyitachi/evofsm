# CLAUDE.md — configs/ working context

MobileWorld cross-benchmark split config. See `../CLAUDE.md` for the project
picture. This file is only what you need when
working in this dir.

## File map
- `mobileworld_splits.yaml` — **authoritative MW split.** Per-task `adapt`/`eval`
  assignment + per-task tier over 161 GUI-only tasks; task-disjoint (a task is
  wholly in adapt or eval). Has `meta.version` (`v1.0-MW`), `meta.adapt_fraction`
  (0.4), `meta.rule`, `meta.totals` (161 / 51 / 110), `meta.per_tier`. Generated
  (hand-written lines for layout) — do not hand-edit.
- `gen_mobileworld_splits.py` — the deterministic generator. Reads MW task defs
  under `MobileWorld/src/mobile_world/tasks/definitions`, keeps GUI-only (no
  `agent-mcp` tag), tiers by Play-category (`CAT`/`NOVEL` maps at top), Tier-A →
  all eval, Tier-B/C → stratify by app-combo and take `round(0.4*n)` adapt. No
  RNG; sorts by class name. Run: `python EvoFSM-MW/configs/gen_mobileworld_splits.py`
  from the repo root.
- `teval_tasklist.txt` — the 110 eval tasks, derived from the yaml's `eval`
  records, for the harness.

## Conventions
- **Tier is CATEGORY-level, not app-level** — by Play-category membership of a
  task's apps (6 MW system apps share packages; unseen-app tiering was dropped).
  Tier-B = all categories seen in AW+, Tier-C = novel only (Social, Shopping),
  Tier-A = mixed. Tier-A is all-eval by construction.
- Task names are PascalCase MobileWorld class names (`CheckEventTimeTask`).
- `mobileworld_splits.yaml` is upstream; `teval_tasklist.txt` is downstream of
  its `eval` records.

## Don't
- **Don't hand-edit `mobileworld_splits.yaml`.** Change the split via the rules
  in `gen_mobileworld_splits.py`, rerun, and **bump `meta.version`**. The yaml
  header says so and reproducibility depends on it.
- **Don't read `teval_tasklist.txt` as one-task-per-line.** It is a SINGLE line,
  COMMA-separated, no trailing newline — `split(',')`, never `readlines()`.
- Don't hand-edit `teval_tasklist.txt` to add/drop a task — fix the yaml via the
  generator and regenerate, so the source of truth stays single.
- Don't introduce randomness into the generator (no `random`, no set-iteration
  order in output) — the byte-stable, sort-by-class-name output is the point.
- Don't add model / hyperparam configs here — that surface lives on the
  EvoFSM-RL side, not in the MW split dir.
