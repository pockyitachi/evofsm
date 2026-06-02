# EvoFSM-MW — cross-benchmark formal experiment workspace

**Created 2026-06-02.** This is the workspace for the formal cross-benchmark
EvoFSM experiment. It is a **thin orchestration + new-harness layer that REUSES
(imports) the symbolic core of `../EvoFSM-RL`** — it does NOT fork or copy that
code. (Folder name `EvoFSM-MW` is provisional; rename freely.)

> Status: scaffold only. Many design points still open — see "Open items".
> Full background lives in `../EvoFSM-RL/CLAUDE.md` → the "### 2026-06-01 (session 2)"
> entry and memory `project_mobileworld_pivot`. Read those first.

## The experiment (one line)

Train Phase-1 π^pre on **android_world_plus** (193 tasks), test-time-adapt
(FSM + LoRA), and **evaluate on MobileWorld GUI-only tasks** — a pure-vision,
coordinate-grounded GUI agent whose symbolic half (FSM / `L_C`) is injected
into the prompt and transfers across benchmarks by Play-Store category.

## What lives where (reuse boundary — do NOT duplicate)

| Concern | Where | Reuse mode |
|---|---|---|
| Symbolic core: `L_C`/FSM builder, evolution loop, taxonomy, splits | `../EvoFSM-RL/evofsm_rl/` | **import via PYTHONPATH** (don't copy) |
| RL training backend: verl GRPO on AW+ | `../SkyRL-AndroidWorld/` | separate repo; drive via its server |
| Training env (193-task emulator) | docker image `androidworld:evofsm-tasks193` | run container; HTTP `/step` |
| Eval benchmark + reward | `../MobileWorld/` | use its harness + reward as-is |
| **NEW: pure-vision harness** (qwen3vl `mobile_use` rollout, action translation, L_C-injection prompt) | `harness/` (here) | new code |
| **NEW: outer-loop orchestration** (evolve L_C ↔ train ↔ eval) | `orchestration/` (here) | new code |
| Formal configs / results | `configs/`, `results/` (here) | new |

## Settled decisions (from 2026-06-01/02 sessions)

- **Harness = Qwen3-VL native `mobile_use` format** (base on `../MobileWorld/src/mobile_world/agents/utils/prompts/qwen3vl.py`). Pretrained schema → strongest zero-shot grounding → least RL. Drop a11y; drop MCP (GUI-only).
- **Env supports coordinate actions natively** (`evofsm-tasks193` `actuation.py`: click/long_press/double_tap via x,y; swipe via `direction=[x,y,x2,y2]`). Only a thin `mobile_use`→`JSONAction` translation needed in the harness — NO env changes.
- **Online RL**: rollouts are the data; no pre-prepared dataset. Old a11y RL trajectories not reusable (off-policy + bbox not persisted). `L_C`/FSM (abstract NL) carries over as prompt injection.
- **FSM/L_C symbolic thesis preserved** — injected into the general prompt; pure-vision change does not touch it.

## Open items (resolve before/while building)

1. **T-adapt domain** — confirm adapt runs on MobileWorld target apps (likely) vs AW+.
2. **Novel categories** — Mastodon/Mattermost (Social), Taodian (Shopping) have no same-cat pretraining → pure app-level adapt; position as a result, or add pretraining categories?
3. **Multi-app MobileWorld tasks** (e.g. Calendar+Messages) — how `L_C` category knowledge applies across 2 apps in one task.
4. **GPU** — M1 (verl Phase-1 training) blocked; usable GPU 4 & 7 occupied. See CLAUDE.md.

## Layout

```
EvoFSM-MW/
├── README.md            ← this file
├── harness/             # NEW pure-vision Qwen3-VL mobile_use harness
│   ├── prompt.py        # qwen3vl mobile_use template + L_C/FSM injection slot
│   ├── qwen3vl_agent.py # rollout agent (replaces EvoFSM-RL's M3A-clone Qwen3VLAgent)
│   └── action_translation.py  # mobile_use JSON <-> AndroidWorld JSONAction
├── orchestration/       # NEW outer loop: symbolic evolution ↔ verl train ↔ MW eval
├── configs/             # formal experiment configs
├── results/             # experiment outputs
└── notes/               # design notes / decisions
```
