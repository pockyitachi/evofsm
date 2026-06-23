"""B2 agent for MobileWorld eval — stock Qwen3VL agent + static app_guidance.

Loaded by MobileWorld's file-path agent mechanism (no registry edit needed):

    EVOFSM_B2_GUIDANCE=/shared/linqiang/evofsm_project/EvoFSM-MW/artifacts/b2_guidance.json \
    .venv/bin/mw eval --agent_type /shared/linqiang/evofsm_project/EvoFSM-MW/harness/qwen3vl_b2_agent.py ...

B2 == B1 (Qwen3VLAgentMCP, stock prompt, same vLLM weights) + one spliced
"# App guidance" section per task, pre-rendered offline by gen_b2_guidance.py.
Tasks whose guidance text is empty (pure Tier-C) produce a byte-identical
prompt to B1 — the designed degradation.

Requires the 2-line MobileWorld patches:
  - runner passes task_name= to file-loaded agents,
  - Qwen3VLAgentMCP.predict splices self.app_guidance before "# Response format".
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from loguru import logger

# Attribute access (not `from ... import`) keeps Qwen3VLAgentMCP out of this
# module's namespace, so load_agent_from_file finds exactly one agent class.
from mobile_world.agents.implementations import qwen3vl as _qwen3vl


@lru_cache(maxsize=1)
def _guidance_map(path: str) -> dict:
    with open(path) as fh:
        data = json.load(fh)
    logger.info(
        "B2 guidance loaded from {}: {} tasks, stats={}",
        path,
        len(data["tasks"]),
        data["meta"]["stats"],
    )
    return data["tasks"]


class Qwen3VLAgentB2(_qwen3vl.Qwen3VLAgentMCP):
    def __init__(self, *args, env=None, tools=None, task_name=None, **kwargs):
        if tools is None:
            tools = env.tools if env is not None else []
        super().__init__(*args, tools=tools, env=env, **kwargs)

        guidance_path = os.environ.get(
            "EVOFSM_B2_GUIDANCE",
            str(Path(__file__).resolve().parents[1] / "artifacts" / "b2_guidance.json"),
        )
        entry = _guidance_map(guidance_path).get(task_name)
        if task_name is None:
            logger.warning("B2 agent created without task_name — no guidance injected")
        elif entry is None:
            logger.warning("B2 agent: task '{}' not in guidance map — no guidance injected", task_name)
        self.app_guidance = (entry or {}).get("text", "")
        logger.info(
            "B2 guidance for task '{}': {} chars, blocks={}",
            task_name,
            len(self.app_guidance),
            [(b.get("tier", "?"), ",".join(b["apps"])) for b in (entry or {}).get("blocks", [])],
        )
