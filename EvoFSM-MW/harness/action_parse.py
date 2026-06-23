"""Parse mobile_use model output → AndroidWorld `/step` action dict (training side).

Shared model output format (both train & eval), produced under `prompt.py`:
    Thought: <one sentence>
    Action: <one sentence>
    <tool_call>{"name": "mobile_use", "arguments": {...}}</tool_call>

EVAL side (MobileWorld) translates the same output via its own
`parsing_response_to_andoid_world_env_action`. This module is the TRAINING-side
translation: mobile_use args → the dict that `evofsm-tasks193`'s skyrl_server
`/step` accepts (it then does `JSONAction(**action)` after converting any
`touch_point`). We emit device-pixel `x,y` (and a 4-px `direction` for swipe)
directly, which JSONAction takes natively.

Coordinate convention mirrors MobileWorld qwen3vl: SCALE_FACTOR = 999. Raw model
coord (0..999) → /999 → *device-screen dim → device pixel.
"""

import json
import re

SCALE_FACTOR = 999  # matches MobileWorld qwen3vl (prompt declares 999x999)


class ActionParseError(ValueError):
    pass


# mobile_use system_button → AndroidWorld action_type
_SYSTEM_BUTTON = {
    "Back": "navigate_back",
    "Home": "navigate_home",
    "Enter": "keyboard_enter",
}


def parse_model_output(text: str):
    """Split a model turn into (thought, mobile_use_args).

    Raises ActionParseError if no parseable <tool_call> is present.
    """
    thought = ""
    m = re.search(r"Thought:\s*(.*?)(?:\n\s*Action:|\n?\s*<tool_call>)", text, re.S)
    if m:
        thought = m.group(1).strip()

    tc = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    if not tc:
        raise ActionParseError(f"no <tool_call> JSON found in model output: {text[:200]!r}")
    try:
        obj = json.loads(tc.group(1))
    except json.JSONDecodeError as e:
        raise ActionParseError(f"bad tool_call JSON: {e}: {tc.group(1)[:200]!r}")

    args = obj.get("arguments", obj)  # tolerate missing top-level wrapper
    if "action" not in args:
        raise ActionParseError(f"tool_call has no 'action': {args!r}")
    return thought, args


def to_aw_action(args: dict, screen_w: int, screen_h: int) -> dict:
    """Translate mobile_use arguments → AndroidWorld /step action dict (device px).

    `screen_w/screen_h` are the device logical screen dimensions (the /reset or
    /step observation image shape gives them).
    """
    action = args.get("action")

    def to_px(coord):
        if not coord or len(coord) != 2:
            raise ActionParseError(f"expected 2-element coordinate, got {coord!r}")
        x = round(float(coord[0]) / SCALE_FACTOR * screen_w)
        y = round(float(coord[1]) / SCALE_FACTOR * screen_h)
        return x, y

    if action == "click":
        x, y = to_px(args.get("coordinate"))
        return {"action_type": "click", "x": x, "y": y}

    if action == "long_press":
        x, y = to_px(args.get("coordinate"))
        return {"action_type": "long_press", "x": x, "y": y}

    if action == "swipe":
        x1, y1 = to_px(args.get("coordinate"))
        x2, y2 = to_px(args.get("coordinate2"))
        # AndroidWorld JSONAction swipe takes a 4-element pixel list in `direction`.
        return {"action_type": "swipe", "direction": [x1, y1, x2, y2]}

    if action == "type":
        return {"action_type": "input_text", "text": args.get("text", "")}

    if action in ("open", "open_app"):  # accept both; AW uses open_app + app_name
        return {"action_type": "open_app", "app_name": args.get("text", "")}

    if action == "answer":
        return {"action_type": "answer", "text": args.get("text", "")}

    if action == "system_button":
        button = args.get("button")
        if button not in _SYSTEM_BUTTON:
            raise ActionParseError(f"unsupported system_button: {button!r}")
        return {"action_type": _SYSTEM_BUTTON[button]}

    if action == "wait":
        return {"action_type": "wait"}

    if action == "terminate":
        ok = args.get("status") == "success"
        return {"action_type": "status", "goal_status": "complete" if ok else "infeasible"}

    raise ActionParseError(f"unknown mobile_use action: {action!r}")


def parse_and_translate(text: str, screen_w: int, screen_h: int):
    """Convenience: model text → (thought, aw_action_dict)."""
    thought, args = parse_model_output(text)
    return thought, to_aw_action(args, screen_w, screen_h)
