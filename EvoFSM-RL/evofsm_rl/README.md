# evofsm_rl/ — core package

The Python package implementing the EvoFSM-RL method: a two-layer FSM prior plus
a joint evolution + GRPO test-time-adaptation loop for mobile GUI agents.

## Layout

| Subpackage | Role |
|---|---|
| `agent/` | M3A-style Qwen3-VL agent — `prompts`, `action` (JSON→JSONAction parser with error recovery), `a11y` (UI-element view + SoM), `rollout` (`step()` mirrors M3A + `save_episode()`) |
| `env/` | `harness.py` — thin AndroidWorld env wrapper |
| `fsm/` | Two-layer FSM machinery — `builder` (LLM synthesis), `aggregator` (per-app Layer-2 → category `L_C`), `linter` (enforce Layer-2 transferability), `injection` (3-tier prompt inject), `evolution` / `mutation` / `diff` / `population` (online evolve, TrueSkill), `schema` |
| `model/` | `loader` (base VLM), `lora` (LoRA wrap/unwrap) |
| `rl/` | `grpo.py` — group-relative policy optimization (main-line RL) |
| `splits.py` | source / Tier-B / Tier-C + `T_adapt` / `T_eval` loader (reads `configs/splits.yaml`) |
| `taxonomy.py` | app → Play-Store-category labels |

## Usage

Not a standalone CLI — imported by the entry points in `../scripts/`
(e.g. `run_b3_evolution.py`, `run_b4_evolution.py`). Run from the repo root with
`PYTHONPATH=android_world_plus:EvoFSM-RL`.

See `CLAUDE.md` in this directory for working context (conventions, pitfalls),
and `../CLAUDE.md` for the project-wide picture.
