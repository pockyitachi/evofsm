# CLAUDE.md — harness/ working context

See `../CLAUDE.md` for the project-wide picture; this file is only what you need
when working in this directory.
The pure-vision MobileWorld eval harness: eval agents, the shared `mobile_use`
prompt/action plumbing, the offline guidance renderers, and the run-specific
driver chains. The symbolic content (FSM / `L_C`) is **imported** from
`../../EvoFSM-RL/evofsm_rl/` — never copied here.

## Map
- **Agents** (`mw eval --agent_type <file>`) — `qwen3vl_b2_agent.py`
  (subclasses MobileWorld's stock `Qwen3VLAgentMCP`), `mai_ui_b2_agent.py`
  (subclasses `MAIUINaivigationAgent`). Both read a frozen guidance JSON
  (`EVOFSM_B2_GUIDANCE`) and splice the per-task block into the prompt.
  `qwen3vl_agent.py` (base rollout) and `action_translation.py` are **scaffolds**
  (`raise NotImplementedError`) — eval reuses MobileWorld's own agent.
  `smoke_b2_agent.py` is the offline (no emulator/LLM) injection assertion.
- **Action / prompt** (shared train+eval) — `prompt.py` (the qwen3vl
  `mobile_use` template + the `{{ app_guidance }}` injection slot),
  `action_parse.py` (model turn → AndroidWorld `/step` dict, training side).
- **Guidance generators** (offline → `../artifacts/*_guidance.json`) —
  `gen_b2_guidance.py` (`--mode full|app-l2|category-only` = B2/B2'/B2''),
  `gen_b2ppp_guidance.py` (B2''' entry-level retrieval), `gen_b3_guidance.py`
  (B3 evolved champions), `gen_b3_lessons_guidance.py`
  (`--all-lessons`/`--lessons-only`/`--top-k`).
- **Tools** — `cmp_b3_composition.py` (which tasks B3 moved vs B1/B2'),
  `merge_fsm_lessons.py` (champion + lessons-only → fsm+lessons JSON).
- **Drivers** (`auto_chain_*.sh`, `launch_*.sh`, `eval_*_b1.sh`,
  `run_b3eval.sh`) — run-specific tmux chains, **mostly historical**. Each
  bakes in container ports, GPU/`:port` of the vLLM, and a PID/COMPLETE gate for
  one past run. Read the top comment; don't treat them as a reusable API.

## Conventions
- **Two venvs.** Agents/smoke run in the **MobileWorld** venv
  (`../../MobileWorld/.venv`) — they import `mobile_world`, never `evofsm_rl`,
  so that venv stays clean. Generators run from the **project root** in the
  **main** venv with `PYTHONPATH=EvoFSM-RL` (B3 ones add
  `SkyRL-AndroidWorld/skyrl-agent`) — they import the symbolic core.
- The guidance JSON is the **only** train→eval handoff: `{"meta": {...},
  "tasks": {task_name: {"text": str, "blocks": [...]}}}`. The agent reads
  `tasks[name].text` (+ `meta.stats`); it never imports `evofsm_rl`. Regenerate
  the JSON, not the agent, to change injection.
- app label = snake_case; template/`task_name` = PascalCase (`CVEmailTask`).
  They come from `../configs/mobileworld_splits.yaml` and the MobileWorld
  registry — don't invent either.
- The top docstring of each file is the source of truth for flags, ports, and
  the exact run command — read it before guessing.
- Eval traces land in `../../MobileWorld/traj_logs/<run>/<task>/result.txt`
  (`score: 1.0` = pass); guidance + chain logs in `../artifacts/`.

## Don't
- **Don't reuse MobileWorld containers across runs.** MW does NOT reset app
  state between tasks, so a container that already ran a 110-task eval carries
  dirty state — every run gets FRESH containers on network `mwnet`. This is why
  the drivers bring up new `lq_*` pools each time with disjoint port bands.
- **Don't expect the B2-family agents to do anything special with empty
  guidance** — by design, empty guidance ⇒ a byte-identical prompt to the stock
  B1 agent (the intended degradation for pure-novel-category tasks).
  `smoke_b2_agent.py` asserts exactly this; keep it green after agent edits.
- **Don't be alarmed by MCP-401 noise at startup** — it's harmless for the
  GUI-only 110-task list (the manifest excludes agent-MCP tasks and the runner
  clears tools for non-MCP tasks). Not a failure.
- **Don't inject Layer-1.** Cross-benchmark, Layer-1 (app states / visual cues)
  is empirically *harmful* — that's why B2' (`--mode app-l2`, Layer-2 only) is
  the strongest static config. Don't "fix" B2' by adding it back.
- **Don't read `qwen3vl_agent.py` / `action_translation.py` as live code** —
  they're scaffolds that raise on import. The working rollout agent is
  MobileWorld's own; `action_parse.py` is the live parser.
- **Don't re-run a driver `.sh` expecting it to just work** — each is wired to a
  specific past run (baked PID, container ports, vLLM `:port`/GPU, a prior
  COMPLETE gate). They are provenance, not tooling. To run a new eval, copy the
  matching `launch_*.sh` and re-point its ports/guidance, or call `mw eval`
  directly.
- **Container kill discipline:** the chains only tear down `linqiang`-owned
  `lq_*` containers (verified via `/proc`). Honor that on a shared host — never
  `docker rm` a pool you didn't create.
