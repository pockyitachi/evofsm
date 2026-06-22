# CLAUDE.md — configs/ working context

Declarative run/split/model configs for EvoFSM-RL. See `../CLAUDE.md` for the
project-wide picture; this file is only what you need when editing configs here.

## File map
- `splits.yaml` — **authoritative split definition.** K=1 baseline. Three blocks:
  `source_pool` (12 apps / 96 templates), `tier_B_held_out` (6 apps, near-transfer),
  `tier_C_held_out` (7 apps, far-transfer). Each held-out app has `T_adapt` /
  `T_eval` template lists. Loaded by `evofsm_rl/splits.py`. Has `meta.version`
  (now `v1.1-K1`), `meta.revision_notes`, and derived counts in `meta.counts`.
  (Key names `tier_B_held_out` / `tier_C_held_out` are legacy synonyms for
  `tier_B_target` / `tier_C_target`; the rename is a pending pass — see `../CLAUDE.md`.)
- `multi_seed_config.yaml` — K overlay on `splits.yaml`: `K_source=5`, `K_adapt=5`,
  `K_eval=3`; `task_random_seeds` (adapt 30–34, eval 40–42); per-tier episode
  budgets; degenerate-app handling (`simple_draw_pro` N=1). Does NOT define membership.
- `model.yaml` — base-model single source of truth: repo id `Qwen/Qwen3-VL-8B-Instruct`,
  `revision` SHA, dtype, Mac vs A100 `max_pixels`. Read by `evofsm_rl/model/loader.py`.
- `model_fingerprint.lock.json` — loader-written fingerprint; compared on load, fails
  loud on mismatch. Regenerate by deleting it and re-running `scripts/model_smoke.py`.
- `task_categories.csv` — `task_name → app, tier, play_category, task_type`. Backs `taxonomy.py`.
- `source_pool_96.txt`, `baseline_50task.txt`, `t_eval/teval_v01.txt` — flat task lists
  (`Template:tier`), each derived from `splits.yaml`. Consumed by `../scripts/`.

## Conventions
- App label = snake_case (`simple_calendar_pro`); template = PascalCase (`MarkorCreateNote`).
  These match the registry / `splits.yaml`, not AndroidWorld internal label strings.
- `splits.yaml` is upstream; the `.txt` lists and `task_categories.csv` counts are
  downstream of it. Edit membership in `splits.yaml`, then regenerate the derived lists —
  never the other way around.
- Every template in `splits.yaml` must exist in the AndroidWorld registry, or the harness
  raises `KeyError` (this is what `v1.1-K1` fixed).
- Counts to sanity-check against: 25 apps, 191 templates, 96 source / 50 Tier-B / 46 Tier-C.

## Don't
- Don't change `splits.yaml` without bumping `meta.version` AND adding a `meta.revision_notes`
  entry AND noting it in `../CLAUDE.md`. Reproducibility hinges on this.
- Don't hardcode model name / revision / dtype anywhere in code — change `model.yaml` instead.
- Don't hand-edit `model_fingerprint.lock.json`; let the loader regenerate it.
- Don't hand-edit the derived `.txt` lists or `task_categories.csv` to add/drop a template —
  fix `splits.yaml` and regenerate, so the source of truth stays single.
- Don't add a template that isn't in the registry (verify against AndroidWorld upstream first).
