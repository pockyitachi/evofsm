# tests/ — unit tests for the evofsm_rl package

Fast, mostly offline pytest-style unit tests for the FSM machinery, the M3A
agent surface, and the main-line GRPO math. No emulator, model, or real API
call required (the one emulator smoke test is opt-in).

## Test files

### FSM (two-layer machinery)
| File | Covers |
|---|---|
| `test_fsm_schema.py` | `fsm.schema` round-trip (`to_json`/`from_json`) at every nesting level, version policy, `to_prompt_text` invariants. |
| `test_fsm_linter.py` | `fsm.linter` Layer-2 transferability rules (app-name / resource-hint / package-name / app-state-id leaks). |
| `test_fsm_diff.py` | `fsm.diff` — parse, apply (states / transitions / strategies / categories), integrity validation, LLM-response extraction. |
| `test_fsm_builder.py` | `fsm.builder` — trajectory compression, prompt assembly, JSON extraction (Anthropic call itself not covered). |
| `test_fsm_aggregator.py` | `fsm.aggregator` — per-app Layer-2 → category `L_C` merge + `lint_L_C` (Claude call mocked). |
| `test_mutation.py` | `fsm.mutation` — reflection/diff prompts, diff parsing, `mutate_fsm` (all API mocked). |
| `test_population.py` | `fsm.population` — TrueSkill updates, sliding window, champion selection, JSON round-trip. |
| `test_evolution.py` | `fsm.evolution` — the B3 evolve loop end-to-end with mock rollouts (no emulator/model/API). |

### Agent (Qwen3-VL M3A)
| File | Covers |
|---|---|
| `test_action_parser.py` | `agent.action.parse_action` error-recovery matrix (fenced/prose JSON, coercion, coord clamping). |
| `test_a11y.py` | `agent.a11y` element filtering, M3A UI-element text list, SoM drawing, center-xy helpers. |
| `test_agent_logprob.py` | `Qwen3VLAgent` B4 log-prob surface (`collect_log_probs`, `reset`, `get_trajectory_data`) — no `step()`. |

### RL
| File | Covers |
|---|---|
| `test_grpo.py` | `rl.grpo` pure math — `compute_reward`, `compute_advantages`, replay cleanup (`grpo_step` deferred to integration). |
| `test_dense_reward.py` | Opt-in dense-reward (partial-credit) counting helpers + harness `--use-dense-reward` plumbing. |

### Other
| File | Covers |
|---|---|
| `test_l_c_injection.py` | B2 `L_C` prompt injection — zero-regression vs B1, section placement, per-tier `resolve_l_c_for_app`. |
| `test_lora.py` | `model.lora` peft wrap/freeze/save/load/count on a tiny synthetic module (no real VLM). |
| `test_taxonomy_splits.py` | `taxonomy` + `splits` sanity — app/template counts, pool disjointness, T_adapt/T_eval split. |
| `test_smoke_emulator.py` | Story 1.1 wiring — boot AVD → connect → init/eval tasks with no agent (needs a running emulator). |

## How to run

From the repo root, with the package on `PYTHONPATH`:

```bash
# all (offline) tests
PYTHONPATH=android_world_plus:EvoFSM-RL python -m pytest EvoFSM-RL/tests/ -v

# a single file
PYTHONPATH=android_world_plus:EvoFSM-RL python -m pytest EvoFSM-RL/tests/test_fsm_diff.py -v
```

Every file except `test_dense_reward.py` also runs as a plain script (no pytest):

```bash
PYTHONPATH=android_world_plus:EvoFSM-RL python EvoFSM-RL/tests/test_population.py
```

`test_smoke_emulator.py` is the only test that needs live infrastructure — a
booted AVD (see `../CLAUDE.md` "Canonical AVD"); the rest run on CPU in seconds.
