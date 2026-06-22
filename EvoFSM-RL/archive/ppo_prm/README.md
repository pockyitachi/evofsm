# Archived route: PPO + Process Reward Model (abandoned)

This directory archives an **abandoned** research route — PPO with a Process
Reward Model (PRM) — explored before the project settled on GRPO (now
verl/SkyRL) as the main-line RL algorithm.

## Why it was abandoned
- The main-line RL converged on GRPO (`evofsm_rl/rl/grpo.py`, later replaced by
  the verl/SkyRL framework).
- The PPO+PRM sweep was unstable. The largest run
  (`ppo_prm_sweep.unstable_grad_2026-05-21`, ~23 GB) diverged on gradient
  instability; a smaller `ppo_prm_sweep` (~2.2 GB) preceded it.

## Contents
- `rl_ppo/` — the PPO/PRM package (7 modules: `trainer`, `ppo_loss`, `gae`,
  `prm_scorer`, `value_head`, `evolution_loop`, `__init__`).
- `scripts/` — entry points: `run_ppo_evolution.py`, `run_ppo_value_pretrain.py`,
  `run_prm_train.py`.
- `tests/` — `test_gae.py`, `test_ppo_components.py`.
- `artifacts/` — PRM and value-head checkpoints (21 files, ~340 KB).
- `traces_backup.zip` — original sweep trajectories, both `ppo_prm_sweep` and
  `ppo_prm_sweep.unstable_grad_2026-05-21` (~25 GB raw, 1.5 GB zipped; the sweep
  was log/json-heavy and highly compressible; **gitignored**). The original
  `traces/ppo_prm_sweep*` directories were deleted after this zip was verified.

## Status
Self-contained: the main-line code never imported `rl_ppo`. Fully restorable by
moving the directories back if this route is ever revisited.
