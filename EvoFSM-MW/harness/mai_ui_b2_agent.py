"""MAI-UI B2-variant agent — official MAIUINaivigationAgent + static app_guidance.

Loaded by MobileWorld's file-path agent mechanism (runner passes task_name= to
file-loaded agents via the EvoFSM runner patch):

    EVOFSM_B2_GUIDANCE=<winner guidance json> \
    .venv/bin/mw eval --agent_type .../mai_ui_b2_agent.py --model_name MAI-UI-8B ...

Injection: the guidance text for the task is spliced into the MAI system
prompt immediately before the "## Note" section (rfind anchor — analogous to
the qwen3vl "# Response format" splice). Empty guidance ⇒ byte-identical
prompt to the stock MAI-UI agent (designed degradation). No MobileWorld fork
changes needed: `system_prompt` is a property, overridden here.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from loguru import logger

# Attribute access keeps MAIUINaivigationAgent out of this module's namespace
# so load_agent_from_file finds exactly one agent class.
from mobile_world.agents.implementations import mai_ui_agent as _mai


@lru_cache(maxsize=1)
def _guidance_map(path: str) -> dict:
    with open(path) as fh:
        data = json.load(fh)
    logger.info(
        "MAI-UI B2 guidance loaded from {}: {} tasks, stats={}",
        path,
        len(data["tasks"]),
        data["meta"].get("stats"),
    )
    return data["tasks"]


class MAIUIB2Agent(_mai.MAIUINaivigationAgent):
    def __init__(self, *args, env=None, tools=None, task_name=None, **kwargs):
        if tools is None:
            tools = env.tools if env is not None else []
        super().__init__(*args, tools=tools, env=env, **kwargs)

        guidance_path = os.environ.get(
            "EVOFSM_B2_GUIDANCE",
            str(Path(__file__).resolve().parents[1] / "artifacts" / "b2p_guidance.json"),
        )
        entry = _guidance_map(guidance_path).get(task_name)
        if task_name is None:
            logger.warning("MAI-UI B2 agent created without task_name — no guidance injected")
        elif entry is None:
            logger.warning("MAI-UI B2 agent: task '{}' not in guidance map — no guidance", task_name)
        self.app_guidance = (entry or {}).get("text", "")
        logger.info(
            "MAI-UI B2 guidance for task '{}': {} chars",
            task_name,
            len(self.app_guidance),
        )

    @property
    def system_prompt(self) -> str:
        base = _mai.MAIUINaivigationAgent.system_prompt.fget(self)
        guidance = getattr(self, "app_guidance", "")
        if not guidance:
            return base
        idx = base.rfind("\n## Note")
        if idx != -1:
            return base[:idx] + "\n" + guidance + "\n" + base[idx:]
        return base + "\n\n" + guidance
