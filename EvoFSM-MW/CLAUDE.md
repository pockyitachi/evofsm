# CLAUDE.md — EvoFSM-MW

The cross-benchmark experiment workspace: pretrain on AndroidWorld+, deploy on
MobileWorld. **Every folder also has its own `CLAUDE.md` and `README.md`** — read
those when working in a folder. This file is the global picture, MW-specific
gotchas, and current status. Work principles (Rule 0) and the symbolic-core
internals live in [`../EvoFSM-RL/CLAUDE.md`](../EvoFSM-RL/CLAUDE.md) — EvoFSM-MW is
its cross-benchmark sibling.

## What this is

A thin orchestration + new-harness layer that **imports** the symbolic core of
`../EvoFSM-RL/evofsm_rl/` (FSM/`L_C`, evolution, taxonomy, splits) — it does NOT
fork it. The experiment: train Phase-1 `π^pre` on AndroidWorld+ (193 tasks),
evaluate on MobileWorld GUI-only (110-task `T_eval`), with the FSM/`L_C` symbolic
prior injected into a pure-vision Qwen3-VL `mobile_use` prompt and transferring by
Play-Store category. Ladder: B1 zero-shot → B2 static prior → B3 symbolic
evolution → B4 joint symbolic + weight.

## Repository navigation

| Need | Go to |
|---|---|
| Work inside a folder | that folder's `CLAUDE.md` + `README.md` |
| B-series results | `docs/qwen3_8b_res.md` (8B), `docs/mai_ui_8b_res.md` (MAI-UI + cross-model + B3) |
| B3 lesson-memory line | `docs/b3_lesson_memory_{design,results}.md` |
| B3/B4 TTA design | `docs/b3_b4_mw_tta_design.md` |
| Split / tiers | `configs/mobileworld_splits.yaml`, `docs/dataset_tiers.md` |
| The eval harness | `harness/` |
| The symbolic core (imported, don't copy) | `../EvoFSM-RL/evofsm_rl/` |

## Global gotchas (do not re-question)

- **MobileWorld containers do NOT reset app state** between tasks — never reuse a
  container across runs; every run gets fresh containers (image
  `mobile_world:reset`, network `mwnet`, since the host's default docker bridge is
  gone).
- **The real eval results live in `../MobileWorld/traj_logs/<run>/`** (per-task
  `result.txt`; `score: 1.0` = success, denominator 110). The `.log` files under
  `artifacts/` are stdout transcripts, NOT the source of truth for numbers.
- **`configs/teval_tasklist.txt` is ONE line, COMMA-separated** (110 tasks) —
  split on commas, not newlines.
- **Cross-benchmark Layer-1 injection is strictly harmful** — the best static
  config is **B2′** (app-Layer-2 + category `L_C`, no Layer-1).
- MCP-401 noise at startup is harmless for the GUI-only 110-task list.

## Current status

- **B1–B3 complete on both models** (`docs/qwen3_8b_res.md`,
  `docs/mai_ui_8b_res.md`): symbolic TTA — both B2 static and B3 evolution — tops
  out at ≈ the static prior on the weak EvoFSM-8B base (8.2 → 9.2 → ~10) and on the
  near-ceiling MAI-UI-8B (26.2 → 26.4 → ~26). Static-injection benefit is
  capability-dependent (qwen +1.0, MAI +0.2).
- **`lessons-only`** is the one reportable positive — distilled lessons match the
  prior at ~1/20 the text (efficiency, not accuracy).
- **B4 (joint symbolic + weight) is the open headroom** — bridge built, gated on
  Phase-1 `π^pre`. This is the thesis the cross-benchmark setting sharpens.

Session-level detail: auto-memory `MEMORY.md` + the docs above.
