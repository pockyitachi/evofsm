# evofsm

EvoFSM — a GUI agent that adapts to a new app at test time by jointly evolving
an app-specific FSM (symbolic) and fine-tuning policy LoRA weights (sub-symbolic),
reusing a per-category abstract-action library `L_C` learned during pretraining.

## Repo layout (this is an umbrella repo of our own code)

| Dir | What |
|---|---|
| `EvoFSM-RL/` | Main project: method implementation, configs, docs. Start at `EvoFSM-RL/CLAUDE.md`. |
| `EvoFSM-MW/` | Cross-benchmark formal-experiment workspace (train on android_world_plus, eval on MobileWorld GUI-only). Imports `EvoFSM-RL`. See `EvoFSM-MW/README.md`. |
| `build_evofsm_tasks/` | Docker patch context that builds `androidworld:evofsm-tasks193` (193-task training env). |

## Not in this repo (referenced, not vendored)

- **Forks (own GitHub repos):** [SkyRL-AndroidWorld](https://github.com/pockyitachi/SkyRL-AndriodWorld) (verl training backend), [MobileWorld](https://github.com/pockyitachi/MobileWorld) (eval benchmark).
- **Regenerable / large:** `traces/`, `apks/`, `data/`, `.venv/`, `android-sdk/`, the AVD — kept out of git (see `.gitignore`).
- **`android_world_plus/`** — the 193-task env extension that `EvoFSM-RL` imports at runtime; currently excluded (add later if you want the repo self-runnable; check upstream android_world license first).
