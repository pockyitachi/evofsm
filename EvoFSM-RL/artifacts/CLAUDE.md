# CLAUDE.md — artifacts/ working context

The static FSM / L_C symbolic prior. See `../CLAUDE.md` for the project-wide
picture; this file is only what you need when touching this directory.

## Version map (use the active dirs unless told otherwise)
- `static_fsms_v2/` (25 apps) + `L_C_v2/` (12 categories) — **active**, paper
  draft's version. Built 2026-06-08: 19 FSMs from `static_fsms_om/` (OpenMobile
  CoT) + 2 from `static_fsms_b4/` (maps_me, wikipedia) + 4 reused source-pool.
- `L_C_v3/` (12 categories) — exclude-system variant, **experimental** (newer).
  MW B-series can target it via the `--lc-dir` override.
- `static_fsms/` (12 apps, `.json`+`.txt`) + `L_C/` (6 categories) — **superseded**
  original seed layer (2026-05-02). B2/B3-era tooling still hard-codes these paths.
- `static_fsms_om/` (19), `static_fsms_b4/` (2) — intermediate **source pools**
  that fed v2; not consumed at runtime.

## File schema
- FSM: `{version, app, layer1, layer2, metadata}`. `layer1` = app-specific
  states/transitions/strategies/dead_ends (non-transferable). `layer2` = generic.
- L_C: `{category, layer2.categories[...]}`, each block = `name`, `precondition`,
  `abstract_steps`, `failure_modes`, `verification_checklist`. L_C is the
  **aggregation** of per-app FSM Layer-2 across a category — not error-correction.

## How injection consumes these (3-tier)
`evofsm_rl/fsm/injection.resolve_app_guidance(app, splits_yaml, l_c_dir, fsm_dir)`
resolves most→least specific:
1. **app** — `{fsm_dir}/{app}.json` exists ⇒ inject the full app FSM.
2. **category** — else app's Play-category has `{l_c_dir}/{slug}.json` ⇒ inject
   category Layer-2 (`resolve_l_c_for_app`).
3. **bootstrap** — neither ⇒ `(None, "bootstrap")`; caller bootstraps from the
   target app's own trajectories.
Returns `(prompt_text|None, tier)`; the tier label is for logging/analysis.
MW consumer `EvoFSM-MW/harness/gen_b2_guidance.py` defaults to `static_fsms_v2` +
`L_C_v2` and pre-bakes frozen guidance JSONs (`--mode full|app-l2|category-only`).

## Don't
- Don't inject Layer-1 across benchmarks — it's empirically net-harmful on MW
  (B2 vs B2'); the MW static prior must be Layer-2 / L_C only.
- Don't treat `static_fsms_v2`/`L_C_v2` as regenerable training output — they're
  hand-built knowledge; edit deliberately, don't overwrite from a sweep.
- Don't quote `static_fsms/` or `L_C/` (orig 6-cat) as current — superseded by v2.
- Don't look here for PRM / value-head / LoRA checkpoints — moved to
  `../archive/ppo_prm/`; nothing trainable lives in this dir.
- `_cleanup_summary.json` is a stale linter log — don't parse it as knowledge.
