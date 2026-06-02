"""Pure-vision prompt for the EvoFSM-MW harness.

SCAFFOLD — not implemented yet.

Plan:
- Base on Qwen3-VL's native `mobile_use` tool-call template:
  `../../MobileWorld/src/mobile_world/agents/utils/prompts/qwen3vl.py`
  (`MOBILE_QWEN3VL_PROMPT`; <tools>/<tool_call> XML, 999x999 coords,
  actions: click/long_press/swipe/type/answer/system_button/wait/terminate).
- Strip the MCP / tool-injection part (we are GUI-only).
- Add the EvoFSM symbolic injection slot: render the per-(category,app) `L_C`
  guidance + current FSM hints into a section of the prompt. This is the ONE
  EvoFSM-specific addition. Source the L_C content from EvoFSM-RL's symbolic
  layer (imported, not copied).

Open: where exactly the L_C/FSM block sits (system preamble vs per-step user
turn), and how step history is carried (qwen3vl uses last-N screenshots+responses).
"""

raise NotImplementedError("EvoFSM-MW harness.prompt — scaffold only")
