<div align="center">

# EvoFSM

**Test-time adaptation for mobile GUI agents on _unseen_ apps**

Deployed to a new app, the agent adapts on a small budget by **jointly** evolving
a two-layer FSM prior (symbolic) and fine-tuning a LoRA policy (sub-symbolic),
reusing a per-category abstract-action library `L_C` learned during pretraining.

</div>

---

## 🎯 The problem

Mobile GUI agents are reliable on apps they were trained on but brittle on apps
they have never seen. Today's 60–80% benchmark numbers are measured where train
and test share the same task templates — they say little about the
deployment-relevant question: *when a user installs a banking app, a niche notes
tool, or a ride-hailing service the agent has never encountered, can it become
reliable from a handful of adaptation examples?* That unseen-app, small-budget
regime is what EvoFSM targets.

## 🧩 Approach

The agent's prior is a **two-layer finite state machine**:

- **Layer 1** — app-specific UI graph (screens, transitions, visual cues), non-transferable.
- **Layer 2** — category-generic workflow archetypes (`ADD_ENTRY`, `QUERY_INFO`, …), written with no app-specific token; a linter enforces this, so it is the structurally guaranteed unit of cross-app transfer. Same-category apps aggregate Layer 2 into a per-category library `L_C`.

At deployment, one loop co-adapts **two channels**: a frozen reference LLM mutates
the upper-layer FSM (prompt channel) while a LoRA adapter is updated by
group-relative policy optimization (weight channel). The same loop runs at
source-pool pretraining and at target-app deployment.

## 🧪 Two studies

EvoFSM is evaluated at two levels of generalization — one within a benchmark, one
across benchmarks:

| | [`EvoFSM-RL/`](EvoFSM-RL) | [`EvoFSM-MW/`](EvoFSM-MW) |
|---|---|---|
| **Setting** | within-benchmark | cross-benchmark |
| **Train → test** | AndroidWorld+ (held-out apps) | AndroidWorld+ → MobileWorld (GUI-only) |
| **Tests** | cross-app / cross-category transfer | transfer to an independently-authored benchmark |
| **Headline** | B1 38.6 → B2 (+9.3) → B3 (+3.7) → B4 **52.9** | symbolic TTA ≈ static prior on two models (8B + MAI-UI); joint **B4** (π^pre + lessons-only) under evaluation |

Each subproject's README has the full results, splits, and reproduce steps.

## 📁 Repository

This is an umbrella repo of **our own code**; benchmarks and the training backend
are referenced, not vendored.

| In git | What |
|---|---|
| [`EvoFSM-RL/`](EvoFSM-RL) | Method implementation + within-benchmark study. Start at `EvoFSM-RL/CLAUDE.md`. |
| [`EvoFSM-MW/`](EvoFSM-MW) | Cross-benchmark workspace (imports `EvoFSM-RL`'s symbolic core). |
| `build_evofsm_tasks/` | Docker patch that builds `androidworld:evofsm-tasks193` (the 193-task training env). |

| Referenced (not vendored) | Where |
|---|---|
| Training backend (verl GRPO) | fork [SkyRL-AndroidWorld](https://github.com/pockyitachi/SkyRL-AndriodWorld) |
| Eval benchmark | fork [MobileWorld](https://github.com/pockyitachi/MobileWorld) |
| 193-task env extension | `android_world_plus/` — imported at runtime, excluded from git (check upstream license before vendoring) |
| Large / regenerable | `traces/`, `apks/`, `.venv/`, `android-sdk/`, the AVD — gitignored |

## Where to start

- **Method + within-benchmark results** → [`EvoFSM-RL/README.md`](EvoFSM-RL/README.md) (project context in [`EvoFSM-RL/CLAUDE.md`](EvoFSM-RL/CLAUDE.md))
- **Cross-benchmark study** → [`EvoFSM-MW/README.md`](EvoFSM-MW/README.md)
