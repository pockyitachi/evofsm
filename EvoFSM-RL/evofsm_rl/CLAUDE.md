# CLAUDE.md — evofsm_rl/ working context

Core package implementing the EvoFSM-RL method. See `../CLAUDE.md` for the
project-wide picture; this file is only what you need when working inside this
package.

## Module map
- `agent/` — M3A agent (action + summary). Edit prompts in `prompts.py`; JSON
  action parsing (with error recovery) in `action.py`; trajectory dump via
  `rollout.save_episode()`.
- `fsm/` — the two-layer FSM. **Layer-2 transferability is enforced by
  `linter.lint_layer2`** (rejects any Layer-2 block or online diff containing an
  app name or a source-pool resource id); injection is 3-tier
  (`injection.resolve_app_guidance`: app FSM → category `L_C` → bootstrap);
  online evolution = `evolution` + `mutation` + `population` (TrueSkill).
- `rl/grpo.py` — main-line GRPO. Group key = (FSM, task), advantage within group;
  do not remove the `1/T_j` length normalization (without it the learning signal
  drowns in a trajectory-length artifact).
- `model/lora.py` — LoRA must run with `offload_policy=False` (peft + fsdp2 +
  CPU-offload leaves rotary buffers on CPU → crash).

## Conventions
- app label = snake_case (`simple_calendar_pro`); template = PascalCase
  (`MarkorCreateNote`).
- Change split membership in `../configs/splits.yaml` and bump `meta.version` —
  never hard-code splits in code.

## Don't
- Don't import or "restore" `rl_ppo` / PRM / `value_head` / `gae` — abandoned,
  archived under `../archive/ppo_prm/`. Main-line RL is only `rl/grpo.py`
  (large-scale training runs on external verl/SkyRL).
- Don't re-add the single-shot v0 symbols (`SYSTEM_PROMPT_V0`, `build_messages`,
  `build_user_turn`, `format_history`, …); the Story 1.5 rewrite removed them —
  fix the caller instead.
