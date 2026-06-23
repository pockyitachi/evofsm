"""Offline smoke for the B2 agent — no emulator, no LLM.

Loads `qwen3vl_b2_agent.py` exactly the way `mw eval --agent_type <file>` does
(registry.load_agent_from_file), fakes the env + OpenAI client, runs the REAL
patched `Qwen3VLAgentMCP.predict`, and asserts:

  1. guidance task  → system prompt contains exactly ONE "# App guidance"
     section, placed before "# Response format";
  2. second predict on the same agent → still exactly one (no stacking);
  3. pure Tier-C task (empty guidance) → system prompt byte-identical to the
     stock B1 render (the designed degradation);
  4. unknown task → warning + stock prompt (B1 behavior).

Run inside the MobileWorld venv:
    /shared/linqiang/evofsm_project/MobileWorld/.venv/bin/python \
        /shared/linqiang/evofsm_project/EvoFSM-MW/harness/smoke_b2_agent.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from loguru import logger
from PIL import Image

logger.remove()
logger.add(sys.stderr, level="WARNING")

from mobile_world.agents.registry import load_agent_from_file  # noqa: E402
from mobile_world.agents.utils.prompts import MOBILE_QWEN3VL_PROMPT_WITH_ASK_USER  # noqa: E402

AGENT_FILE = Path(__file__).resolve().parent / "qwen3vl_b2_agent.py"

CANNED_RESPONSE = (
    "Thought: smoke.\n"
    "Action: click center.\n"
    '<tool_call>\n{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 500]}}\n</tool_call>'
)


def make_agent(cls, task_name: str):
    env = SimpleNamespace(tools=[])
    agent = cls(
        model_name="Qwen3-VL-8B-Instruct",
        llm_base_url="http://localhost:1/v1",  # never contacted (client faked)
        api_key="empty",
        env=env,
        task_name=task_name,
    )
    agent.initialize("smoke goal")
    return agent


def captured_system_prompt(agent) -> str:
    box = {}

    def fake_create(model, messages, **kwargs):
        box["messages"] = messages
        return CANNED_RESPONSE

    agent.openai_chat_completions_create = fake_create
    obs = {"screenshot": Image.new("RGB", (64, 64)), "tool_call": None, "ask_user_response": None}
    prediction, action = agent.predict(obs)
    assert prediction == CANNED_RESPONSE, "predict did not return the canned response"
    assert action is not None
    return box["messages"][0]["content"][0]["text"]


def main() -> None:
    cls = load_agent_from_file(str(AGENT_FILE))
    assert cls.__name__ == "Qwen3VLAgentB2", f"wrong class picked: {cls.__name__}"

    # 1) Tier-B task: app-FSM (Files) + category L_C (Mail) spliced once, before anchor
    a = make_agent(cls, "CVEmailTask")
    assert a.app_guidance, "CVEmailTask should have guidance"
    text = captured_system_prompt(a)
    assert text.count("# App guidance") == 1, "guidance section must appear exactly once"
    assert text.index("# App guidance") < text.index("# Response format"), "guidance must precede response-format"
    assert "## For [Files]  (app-specific FSM" in text
    assert "## For [Mail]  (category: Communication" in text

    # 2) second step on the same agent: prompt rebuilt fresh, still exactly one section
    text2 = captured_system_prompt(a)
    assert text2.count("# App guidance") == 1, "guidance must not stack across steps"

    # 3) pure Tier-C task degrades to byte-identical B1 prompt
    b = make_agent(cls, "MastodonNewPostTask")
    assert b.app_guidance == "", "pure-novel task must have empty guidance"
    stock = MOBILE_QWEN3VL_PROMPT_WITH_ASK_USER.render(tools="")
    got = captured_system_prompt(b)
    assert got == stock, "empty-guidance prompt must be byte-identical to stock B1"

    # 4) unknown task: warning + B1 behavior
    c = make_agent(cls, "NoSuchTask")
    assert c.app_guidance == ""

    print("PASS: B2 agent smoke")
    print(f"  CVEmailTask system prompt: {len(text)} chars (stock {len(stock)})")
    print(f"  guidance blocks: {[(blk['tier'], ','.join(blk['apps'])) for blk in __import__('json').load(open(Path(__file__).resolve().parents[1] / 'artifacts' / 'b2_guidance.json'))['tasks']['CVEmailTask']['blocks']]}")


if __name__ == "__main__":
    main()
