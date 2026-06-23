"""Pure-vision `mobile_use` prompt for the EvoFSM-MW harness (train + eval shared).

Derived from MobileWorld's `MOBILE_QWEN3VL_ORIGINAL_PROMPT`
(`../../MobileWorld/src/mobile_world/agents/utils/prompts/qwen3vl.py`), which
itself wraps Qwen's official `mobile_use` tool schema (Alibaba `qwen-agent`;
Qwen3-VL was grounding-trained on it → strongest zero-shot grounding).

EvoFSM-MW deltas vs that base (kept identical on BOTH the AndroidWorld training
side and the MobileWorld eval side, for train/eval consistency):
  + `open`              — app-launch by name; both envs support it (AndroidWorld
                           tasks need it). Named `open` (not `open_app`) to match
                           MobileWorld's eval parser; action_parse maps it to AW open_app.
  + `{{ app_guidance }}` — the ONE EvoFSM addition: an injection slot for the
                           per-(category, app) L_C abstract steps + FSM hints,
                           rendered from EvoFSM-RL's symbolic layer (imported).
  - `ask_user`          — excluded (autonomous tasks have no user; ORIGINAL base
                           already omits it).
  - `system_button=Menu` — excluded (MobileWorld's qwen3vl agent rejects Menu;
                           AndroidWorld has no Menu need). Only Back/Home/Enter.

Coordinate space is 999x999 (Qwen native); the agent scales to device pixels.
Response format (three parts, parsed downstream): `Thought:` / `Action:` /
a single `<tool_call>{...}</tool_call>` block.
"""

from jinja2 import Template

# --- System prompt: mobile_use tool definition + response format + L_C slot ---
SYSTEM_PROMPT = Template(
    """# Tools

You may call one function to control the mobile device.

You are provided with the function signature within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen's resolution is 999x999.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for the specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinate2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `open`: Open the app named by `text` (e.g. \\"Settings\\").\\n* `answer`: Output the answer to the user.\\n* `system_button`: Press a system button.\\n* `wait`: Wait for the specified seconds for the screen to change.\\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["click", "long_press", "swipe", "type", "open", "answer", "system_button", "wait", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): pixels from the left and top edges. Required by `click`, `long_press`, and the start of `swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): the end point. Required by `swipe`.", "type": "array"}, "text": {"description": "Required by `type` (text to input), `open` (app name), and `answer` (the answer).", "type": "string"}, "time": {"description": "The seconds to wait. Required by `long_press` and `wait`.", "type": "number"}, "button": {"description": "Back returns to the previous interface, Home returns to the desktop, Enter presses enter. Required by `system_button`.", "enum": ["Back", "Home", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required by `terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"]}}}
</tools>

For the function call, return a json object with the function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": "mobile_use", "arguments": <args-json-object>}
</tool_call>
{% if app_guidance %}
# App guidance
The following abstract strategy was learned for this app / category. Use it to plan; it describes WHAT to do at a high level, not exact coordinates — you still must look at the screenshot to ground each action.
{{ app_guidance }}
{% endif %}
# Response format

Response format for every step (all three parts, in this order):
1) Thought: one concise sentence explaining the next move (no multi-step reasoning).
2) Action: a short imperative describing what to do.
3) A single <tool_call>...</tool_call> block containing only the JSON.

Rules:
- Output exactly in the order: Thought, Action, <tool_call>.
- Be brief: one sentence for Thought, one for Action.
- Do not output anything outside those three parts.
- When the task is done, use mobile_use with action=terminate.""".strip()
)

# --- Per-step user turn (the current screenshot is attached alongside this) ---
USER_TEMPLATE = Template(
    """The user query: {{ instruction }}
Task progress (operations done so far on the device): {{ steps }}""".strip()
)


def build_system_prompt(app_guidance: str = "") -> str:
    """Render the system prompt. `app_guidance` is the rendered L_C/FSM block
    (empty string = no injection, e.g. B1 zero-shot baseline)."""
    return SYSTEM_PROMPT.render(app_guidance=app_guidance.strip() if app_guidance else "")


def build_user_turn(instruction: str, steps: str = "") -> str:
    return USER_TEMPLATE.render(instruction=instruction, steps=steps)
