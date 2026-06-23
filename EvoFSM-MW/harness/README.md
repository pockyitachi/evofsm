# harness/ — pure-vision MobileWorld eval harness

The pure-vision (screenshot-only) Qwen3-VL `mobile_use` harness for the
**cross-benchmark** EvoFSM experiment: train π^pre on AndroidWorld+ (193 tasks),
then **evaluate on MobileWorld GUI-only** (110-task split), injecting the
symbolic FSM / `L_C` layer into the prompt so it transfers across benchmarks by
Play-Store category. This directory holds the eval agents, the shared
`mobile_use` prompt/action plumbing, the offline guidance renderers (the frozen
injection artifacts), and the run-specific driver chains that produced the
B1–B3 numbers.

## Scripts

### Agents (loaded by `mw eval --agent_type <file>`)
| File | Role |
|---|---|
| `qwen3vl_agent.py` | Base pure-vision rollout agent — screenshot-only, single forward/step via `prompt.py`, pixel-coord `mobile_use` actions. **Scaffold (`NotImplementedError`)** — eval reuses MobileWorld's own Qwen3VL agent. |
| `qwen3vl_b2_agent.py` | The B2-family eval agent: a file-path subclass of MobileWorld's stock `Qwen3VLAgentMCP` that reads a frozen guidance JSON (`EVOFSM_B2_GUIDANCE`) and splices the per-task block into the prompt before `# Response format`. Empty guidance ⇒ byte-identical to B1. |
| `mai_ui_b2_agent.py` | Same injection, subclassing MobileWorld's `MAIUINaivigationAgent` (splices before `## Note` via a `system_prompt` property override). The MAI-UI-8B B2 agent. |
| `smoke_b2_agent.py` | Offline smoke (no emulator/LLM): loads the B2 agent the way `mw eval` does and asserts one-and-only-one guidance section, no cross-step stacking, and byte-identical degradation on empty guidance. |

### Action / prompt (shared train + eval `mobile_use` plumbing)
| File | Role |
|---|---|
| `prompt.py` | The patched MobileWorld qwen3vl prompt — Qwen `mobile_use` tool schema + a `{{ app_guidance }}` injection slot (the one EvoFSM addition) + `open` app-launch. 999×999 coord space. |
| `action_parse.py` | Parse a `mobile_use` model turn (`Thought:`/`Action:`/`<tool_call>`) → AndroidWorld `/step` action dict (device pixels). The **training-side** translation. |
| `action_translation.py` | `mobile_use` JSON ↔ AndroidWorld `JSONAction` field mapper. **Scaffold (`NotImplementedError`).** |

### Guidance generators (offline → frozen `artifacts/*_guidance.json`)
| File | Role |
|---|---|
| `gen_b2_guidance.py` | Static (B2) renderer: per-app resolve → dedup category blocks → multi-block text. `--mode full` (B2, full app FSM) / `app-l2` (B2', Layer-2 only) / `category-only` (B2'', category `L_C`, the original EvoFSM-RL recipe). |
| `gen_b2ppp_guidance.py` | B2''' — entry-level retrieval on top of the frozen B2' artifact: keep ≤3 task-relevant entries per block (a Qwen3-VL text-only retriever picks them). |
| `gen_b3_guidance.py` | B3 — render an adapt run's round-1 evolved FSM **champions** into eval-guidance shape (Tier-C bootstrap apps now contribute non-empty blocks). |
| `gen_b3_lessons_guidance.py` | B3 lesson-memory — B2' prior + retrieved distilled lessons from an adapt run's `lessons/{app}.json` (`--all-lessons` / `--lessons-only` / `--top-k`). |

### Tools (offline analysis / artifact surgery)
| File | Role |
|---|---|
| `cmp_b3_composition.py` | Composition diff: which tasks B3 (evolved injection) passes vs B1 / B2', from `result.txt` scores. |
| `merge_fsm_lessons.py` | Merge an FSM-champion guidance + a lessons-only guidance into one `fsm+lessons` JSON. |

### Drivers (run-specific unattended chains — mostly historical)
One-off tmux chains that brought up fresh containers, gated on a prior stage, ran
an eval/adapt, and reported. Kept for provenance; not a reusable API.
| File(s) | Role |
|---|---|
| `launch_b2.sh`, `launch_b2p.sh`, `launch_b2pp.sh` | Single B2 / B2' / B2'' eval launches (agent file + log dir + `EVOFSM_B2_GUIDANCE` + container prefix). |
| `launch_rep.sh` | Parametrized single run (`<config> <rep>`) for the 5× variance study on the `lq_rep` container set. |
| `auto_chain_b2.sh`, `auto_chain_b2variants.sh`, `auto_chain_b2ppp.sh`, `auto_chain_repeats.sh` | qwen3-VL B2-family chains: B1→B2, B2→B2'→B2'', B2''' tail, and the 16-run variance study. |
| `auto_chain_b3*.sh`, `run_b3eval.sh` | qwen3-VL B3 adapt + eval chains (FSM-champion, lesson-memory, lesson-only, how-to-use-lessons preamble). |
| `auto_chain_b3_mai*.sh`, `auto_chain_b3mai*_eval.sh`, `auto_chain_maib2v3_eval.sh`, `auto_chain_pi350_maiui.sh` | MAI-UI-8B counterparts: B3 adapt/eval, lesson eval, the tightened `L_C_v3` B2' eval, and the π^pre-350 + MAI tail chain. |
| `eval_pi200_b1.sh`, `eval_pi250_b1.sh`, `eval_pi300_b1.sh`, `eval_mai_pre100_b1.sh`, `eval_mai_pre175_b1.sh` | Off-chain B1 evals of specific Phase-1 π^pre LoRA checkpoints (×5 reps). |
| `auto_chain_tonight.sh`, `auto_eval_mai_nodes.sh` | Misc one-night top-up chains (lesson reps + fsm+lessons; π^pre node eval). |

## Usage

The agents are **not** a CLI — they are loaded by MobileWorld's
`mw eval --agent_type <this-file>` (run from the `../../MobileWorld` venv).
The generators import the `EvoFSM-RL` symbolic core, so run them from the project
root with the **main** venv and `PYTHONPATH=EvoFSM-RL` (some B3 ones also need
`SkyRL-AndroidWorld/skyrl-agent`); each script's top docstring is the source of
truth for its exact command and output path. Frozen guidance JSONs land in
`../artifacts/`; eval traces in `../../MobileWorld/traj_logs/`.

See `CLAUDE.md` here for working context (conventions, gotchas), and
`../README.md` for the project-wide picture. Results writeups:
`../docs/qwen3_8b_res.md` (qwen3-VL B-series) and `../docs/mai_ui_8b_res.md` (MAI-UI).
