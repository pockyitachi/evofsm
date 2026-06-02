"""Pure-vision Qwen3-VL rollout agent for EvoFSM-MW.

SCAFFOLD — not implemented yet.

Replaces EvoFSM-RL's M3A-clone `Qwen3VLAgent` (a11y tree + 2-phase action+summary
+ element-index actions). This one is:
  - screenshot-only input (no a11y),
  - single prompt / single forward per step (uses `prompt.py`),
  - pixel-coordinate actions via `mobile_use` (translated by `action_translation.py`),
  - history = last-N screenshots + responses (qwen3vl style).

Reuses: base-model loader from `EvoFSM-RL/evofsm_rl/model/` (imported), and the
LoRA wrap from `evofsm_rl/model/lora.py`. The symbolic L_C/FSM content comes from
EvoFSM-RL's symbolic layer and is injected via `prompt.py`.

Used by: orchestration rollouts on `androidworld:evofsm-tasks193` (training, AW+)
and by MobileWorld eval (via MobileWorld's harness, which already has a qwen3vl agent).
"""

raise NotImplementedError("EvoFSM-MW harness.qwen3vl_agent — scaffold only")
