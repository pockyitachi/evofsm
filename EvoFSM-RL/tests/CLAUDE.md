# CLAUDE.md — tests/ working context

Unit tests for the `evofsm_rl` package. See `../CLAUDE.md` for the project-wide
picture; this file is only what you need when working inside `tests/`.

## What each test covers
- **FSM** — `test_fsm_schema` (JSON round-trip + version policy), `test_fsm_linter`
  (Layer-2 transferability rules), `test_fsm_diff` (parse/apply/validate online
  diffs), `test_fsm_builder` (trajectory compression + prompt assembly),
  `test_fsm_aggregator` (per-app L2 → category `L_C` merge + `lint_L_C`),
  `test_mutation` (reflection/diff prompts + `mutate_fsm`), `test_population`
  (TrueSkill + sliding window + champion), `test_evolution` (B3 loop end-to-end).
- **Agent** — `test_action_parser` (`parse_action` recovery matrix), `test_a11y`
  (element filtering + M3A UI list + SoM), `test_agent_logprob` (B4 log-prob
  surface: `collect_log_probs` / `reset` / `get_trajectory_data`).
- **RL** — `test_grpo` (pure reward/advantage math + replay cleanup),
  `test_dense_reward` (opt-in partial-credit counting helpers).
- **Other** — `test_l_c_injection` (B2 prompt injection, zero-regression vs B1),
  `test_lora` (peft wrap/freeze/save/load on a synthetic module),
  `test_taxonomy_splits` (app/template counts + pool disjointness),
  `test_smoke_emulator` (Story 1.1 env wiring, needs a live AVD).

## Conventions
- pytest-style: bare `test_*` functions + `assert`, no test classes. No
  `conftest.py` — fixtures are local module-level helpers (`_element`, `_traj`, …).
- Every file except `test_dense_reward.py` also runs as a plain script via an
  `if __name__ == "__main__":` block (pytest-less fallback) — keep that mirror
  intact when adding tests.
- Run from repo root with `PYTHONPATH=android_world_plus:EvoFSM-RL`.
- Offline by design: API calls are monkeypatched at the seam
  (`fsm.mutation._call_claude`, `fsm.builder._call_anthropic`, aggregator's
  Claude call); real model / emulator paths are deferred to integration smokes,
  not unit-tested here.
- Tests that read real artifacts (`configs/splits.yaml`, `artifacts/L_C/*.json`)
  `pytest.skip` gracefully when absent.

## Don't
- Don't add `test_gae.py` / `test_ppo_components.py` or any PPO/PRM/value-head/GAE
  test here — that route is abandoned and its tests live at
  `../archive/ppo_prm/tests/`. Main-line RL is `rl/grpo.py` only.
- Don't reference the removed single-shot v0 symbols (`SYSTEM_PROMPT_V0`,
  `build_messages`, `build_user_turn`, `format_history`, …) — fix the caller, not
  the test.
- Don't make a unit test require a real GPU, the base VLM, or a network API
  call; if it can't run on CPU offline in seconds, it belongs in an integration
  smoke (`scripts/run_b4_evolution.py`), not here.
- Don't hard-code split counts that duplicate `splits.yaml`; assert via the
  `splits`/`taxonomy` loaders so a `splits.yaml` version bump stays single-source.
