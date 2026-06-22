# CLAUDE.md — scripts/ working context

CLI entry points (not importable modules) that drive the EvoFSM-RL pipeline.
See `../CLAUDE.md` for the project-wide picture; this file is only what you need
when working in this directory. The reusable logic lives in `../evofsm_rl/` —
these files glue it to the model + emulator + Anthropic API.

## Map
- **FSM synthesis** — `build_all_fsms.py` (per-app `F^0_a` → `artifacts/static_fsms/`),
  `build_L_C.py` (per-app FSMs → per-category `L_C` → `artifacts/L_C/`). Both lint;
  both call the Anthropic API (`build_all_fsms.py --dry-run` = compress + token-estimate only).
- **Baselines** — `run_b2_eval.py` (static `L_C` eval), `run_b3_evolution.py` +
  `run_b3_teval.py` (evolve Layer-2 + frozen eval), `run_b4_evolution.py` +
  `run_b4_teval.py` (evolution + GRPO LoRA + frozen joint eval),
  `run_phase1_pretraining.py` (shared-LoRA GRPO pretrain → `π^pre_θ`). Drivers:
  `run_b4_sweep.sh`, `run_b4_k4_dense_sweep.sh`.
- **Probes** — `run_rollout.py` (1 task), `baseline_10task.py` (N-task sweep),
  `coord_probe.py` (pixel-coord calibration), `model_smoke.py` (forward pass +
  fingerprint).
- **Analysis** (`_`-prefixed, pure offline, no model/emulator) —
  `_aggregate_b1_vs_b2_k3.py`, `_fsm_quality_digest.py`, `_probe_lint_violations.py`,
  `_relint_L_C.py`.
- **Env setup** (legacy) — `bootstrap_avd.sh`, `install_plus_apks.sh`.
- **Misc** — `clean_layer2.py` (LLM Layer-2 rewrite), `plot_convergence.py`
  (also imported by the evolution scripts), `patch_fsm_story22.py`,
  `generate_task_categories_csv.py`.

## Conventions
- Run from the **repo root** with `PYTHONPATH=android_world_plus:EvoFSM-RL`;
  hard-coded paths inside scripts are `EvoFSM-RL/...` relative to that root.
- `--app` takes a snake_case label (`pro_expense`); `--template` / templates
  are PascalCase (`MarkorCreateNote`). Don't invent either — they come from
  `../configs/splits.yaml` and the AndroidWorld registry.
- Emulator scripts take `--console-port/--grpc-port/--adb-path`; canonical pair
  is `5710 / 8710` against the AWAvd2 `apps_ready_dec2025` snapshot.
- The B3/B4 evolution + `*_teval` scripts **resume** from on-disk state
  (`population.json` / `results.csv`); re-running is cheap and idempotent
  (`--no-resume` to force fresh).
- The docstring at the top of each file is the source of truth for flags — read
  it before guessing argument names.

## Don't
- Don't look here for `run_ppo_*`, `run_prm_*`, `run_rft_*`, or
  `label_step_rewards` — those PPO/PRM/RFT entry points were abandoned and moved
  to `../archive/` (`ppo_prm/`, `rft/`, `b4_init_ablation/`). Main-line RL is
  GRPO only.
- Don't quote numbers from `traces/b4_evolution/` (v1, ABORTED, 4× OOM) or treat
  `b4_evolution_v2` as showing learning — it's pipeline-stable but learning-null.
  See `../CLAUDE.md` "B4 sweep lineage".
- Don't run `bootstrap_avd.sh` / `install_plus_apks.sh` for normal work — they're
  the legacy fresh-AVD path, superseded by the pre-baked AWAvd2 snapshot (boot
  read-only, never `emulator_setup=True`).
- Don't forget the API key for `build_all_fsms.py` / `build_L_C.py` /
  `clean_layer2.py` — they fail without `ANTHROPIC_API_KEY`. The sweep `.sh`
  drivers intentionally run **without** `set -u` so the missing-key error comes
  from Python with a clear message, not bash "unbound variable".
- `patch_fsm_story22.py` writes *aspirational* strategies (target templates have
  no successful trajectories) — it's a one-shot historical fix, not a re-runnable
  builder.
