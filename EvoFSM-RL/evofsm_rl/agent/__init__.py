"""Agent module — prompt construction, action parsing, rollout loop.

Only the lightweight, dependency-free pieces are re-exported here.
Rollout lives one import deeper because it pulls the android_world env
runtime, and we want the parser + prompts to be testable without that.

    from evofsm_rl.agent import parse_action            # always works
    from evofsm_rl.agent.rollout import Qwen3VLAgent    # needs AW + torch

Story 1.5: Qwen3-VL-M3A agent. The legacy single-shot ACTION_SCHEMAS /
SYSTEM_PROMPT_V0 are gone — see prompts.py for M3A's PROMPT_PREFIX,
GUIDANCE, and the action / summary prompt builders.
"""

from evofsm_rl.agent.action import (
    V0_ALLOWED_ACTION_TYPES,
    V0_ALLOWED_DIRECTIONS,
    V0_ALLOWED_GOAL_STATUSES,
    ParseResult,
    action_to_history_dict,
    parse_action,
)
from evofsm_rl.agent.prompts import (
    ACTION_SELECTION_PROMPT_TEMPLATE,
    GUIDANCE,
    PROMPT_PREFIX,
    SUMMARY_PROMPT_TEMPLATE,
    build_action_messages,
    build_action_prompt,
    build_summary_messages,
    build_summary_prompt,
)

__all__ = [
    "ACTION_SELECTION_PROMPT_TEMPLATE",
    "GUIDANCE",
    "ParseResult",
    "PROMPT_PREFIX",
    "SUMMARY_PROMPT_TEMPLATE",
    "V0_ALLOWED_ACTION_TYPES",
    "V0_ALLOWED_DIRECTIONS",
    "V0_ALLOWED_GOAL_STATUSES",
    "action_to_history_dict",
    "build_action_messages",
    "build_action_prompt",
    "build_summary_messages",
    "build_summary_prompt",
    "parse_action",
]
