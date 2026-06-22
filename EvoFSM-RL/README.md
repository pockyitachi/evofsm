# EvoFSM-RL

Test-time-adaptive mobile GUI agent: deployed to an unseen app, it adapts on a
small budget by **jointly** evolving a two-layer FSM prior (symbolic) and
fine-tuning a LoRA adapter (sub-symbolic). Built on `android_world_plus` — it
imports the benchmark's task registry, evaluators, and emulator harness, so treat
the parent repo as the runtime and keep everything project-specific under
`EvoFSM-RL/`.

## What's where

Every folder has its own `README.md` (what it is); code/doc folders also have a
`CLAUDE.md` (working context). Top level:

| Folder | What |
|---|---|
| `evofsm_rl/` | Core package — agent, env harness, two-layer FSM, model/LoRA, GRPO |
| `scripts/` | Entry-point CLIs — FSM synthesis, B1–B4 eval/sweep, Phase-1 pretraining |
| `tests/` | Unit tests (pytest-style, also runnable as plain scripts) |
| `configs/` | `splits.yaml` (source of truth) + seeds + model spec |
| `docs/` | `design/` (dataset, algorithm) · `results/` (B1–B4 reports) · `plan/` |
| `paper_draft/` | The paper, one `.tex` per section (no `main.tex`) |
| `artifacts/` | Static FSM / `L_C` knowledge layer (versioned) |
| `archive/` | Abandoned routes (PPO+PRM, RFT), trajectories zipped |
| `apks/` · `docker/` · `presentations/` · `traces/` | App APKs · emulator image · slide exports · run outputs (gitignored) |

## Key concepts

- **Source pool** (12 apps) → Phase-1 pretraining of per-app L1 FSMs + per-category `L_C`.
- **Tier-B** (near-transfer) / **Tier-C** (far-transfer) → target apps whose Play category is / isn't represented in the source pool.
- **`T_adapt` / `T_eval`** → template-disjoint adaptation vs frozen evaluation within each target app.

Full inventory and split protocol: `docs/design/dataset.md`. Method:
`docs/design/algorithm.md`.

## Setup & run

The canonical environment, server/AVD bootstrap, and gotchas live in `CLAUDE.md`.
Quick check:

```bash
cd /shared/linqiang/evofsm_project
source .venv/bin/activate
PYTHONPATH=android_world_plus:EvoFSM-RL \
  python -c "from evofsm_rl.model import resolve_device; print(resolve_device())"   # -> cuda
```
