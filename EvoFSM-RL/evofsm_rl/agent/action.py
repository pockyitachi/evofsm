"""Action parser for EvoFSM-RL agent — Story 1.4.2.

Takes raw VLM text output and returns a validated AndroidWorld ``JSONAction``
(or a structured error that can be reinjected into the next turn as a
``<system-reminder>``).

Design choices (intentional; matches Claude-Code-style error recovery):

    * **Never raise.**  A parse failure must not crash the rollout loop.
      Return a ``ParseResult`` with ``action=None`` + a human-readable
      ``error`` string.  The rollout writes that string into the next
      user turn verbatim.
    * **Tolerate sloppy output.**  8B VLMs occasionally wrap JSON in
      ```` ```json ```` fences or emit a leading "Let me..." prose blurb
      despite the system prompt's instructions.  We extract the first
      balanced ``{...}`` rather than demanding the entire response be JSON.
      This is pragmatism, not permission — the agent prompt still tells
      the model "entire response must be JSON", and per-turn errors for
      format violations still get surfaced so fine-tuning can learn the
      strict contract.
    * **Whitelist v0 action set.**  ``JSONAction.__post_init__`` accepts
      all 14 AndroidWorld action types.  Our baseline v0 system prompt
      only teaches 8, so we reject the others here — otherwise the model
      can "slip out of distribution" by emitting ``double_tap`` or
      ``open_app`` and the dataclass happily constructs it.
    * **Clamp coordinates.**  Models occasionally emit ``x=9999`` because
      they multiplied the wrong axis.  Clamping avoids blowing up the
      env on an out-of-screen tap — the env's touch dispatcher no-ops
      off-screen coords but clamping gives us a consistent dataset.
    * **Actionable error messages.**  Every error string is written to
      be directly useful as ``<system-reminder>`` content: it says what
      was wrong AND what to emit instead.

Usage:

    from evofsm_rl.agent.action import parse_action

    result = parse_action(vlm_output, screen_width=1080, screen_height=2400)
    if result.ok:
        obs, reward, done, info = env.step(result.action)
    else:
        # Feed result.error back into next turn's prompt.
        next_prompt = build_user_turn(..., prev_error=result.error)
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Optional

from android_world.env.json_action import (
    ANSWER,
    CLICK,
    INPUT_TEXT,
    KEYBOARD_ENTER,
    LONG_PRESS,
    NAVIGATE_BACK,
    NAVIGATE_HOME,
    OPEN_APP,
    SCROLL,
    STATUS,
    WAIT,
    JSONAction,
)
from android_world.env import representation_utils

from evofsm_rl.agent import a11y

# ─────────────────────────────────────────────────────────────────────────
# Action whitelist (Story 1.5 — Qwen3-VL-M3A)
# ─────────────────────────────────────────────────────────────────────────
# Mirrors the action set M3A teaches in PROMPT_PREFIX. AndroidWorld defines
# 14 action_types total; we reject the ones M3A doesn't expose
# (``double_tap``, ``swipe``, ``unknown``) so a malformed model output can't
# slip an out-of-distribution action through env.execute_action. If a future
# ablation adds a new action, update BOTH this set AND prompts.py.
V0_ALLOWED_ACTION_TYPES: frozenset[str] = frozenset({
    CLICK,
    INPUT_TEXT,
    SCROLL,
    LONG_PRESS,
    NAVIGATE_BACK,
    NAVIGATE_HOME,
    WAIT,
    STATUS,
    OPEN_APP,
    ANSWER,
    KEYBOARD_ENTER,
})

# Alias table for common model drift. Baseline Qwen3-VL-8B emits these
# synonyms despite the system prompt:
#   * 'swipe' for 'scroll'   (observed ~80% of the time on SystemBrightnessMax)
#   * 'type'  for 'input_text'
#   * 'back' / 'home'        (short-form nav)
# We normalize to the canonical name before whitelist enforcement and
# remember the original on ParseResult.aliased_from so rollout metrics
# can track hit rate without us silently swallowing the drift.
_ACTION_TYPE_ALIASES: dict[str, str] = {
    "swipe": SCROLL,
    "type": INPUT_TEXT,
    "back": NAVIGATE_BACK,
    "home": NAVIGATE_HOME,
}

V0_ALLOWED_DIRECTIONS: frozenset[str] = frozenset({"up", "down", "left", "right"})
V0_ALLOWED_GOAL_STATUSES: frozenset[str] = frozenset({"complete", "infeasible"})


# Per-action required fields. Values are (required_keys, optional_keys).
# Optional keys are validated-if-present but not required.
#
# Note: click / long_press / input_text take EITHER ``index`` OR ``(x, y)``;
# that XOR is enforced separately below (mirrors JSONAction.__post_init__).
# ``index`` lets the model pick a SoM-labelled element instead of pixel coords —
# see a11y.build_ui_elements_view for how the index is assigned. scroll also
# accepts an optional ``index`` to scroll within a specific scrollable element.
_REQUIRED_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    CLICK:          (frozenset(),                        frozenset({"x", "y", "index"})),
    INPUT_TEXT:     (frozenset({"text"}),                frozenset({"x", "y", "index"})),
    SCROLL:         (frozenset({"direction"}),           frozenset({"index"})),
    LONG_PRESS:     (frozenset(),                        frozenset({"x", "y", "index"})),
    NAVIGATE_BACK:  (frozenset(),                        frozenset()),
    NAVIGATE_HOME:  (frozenset(),                        frozenset()),
    WAIT:           (frozenset(),                        frozenset()),
    STATUS:         (frozenset({"goal_status"}),         frozenset()),
    OPEN_APP:       (frozenset({"app_name"}),            frozenset()),
    ANSWER:         (frozenset({"text"}),                frozenset()),
    KEYBOARD_ENTER: (frozenset(),                        frozenset()),
}


# Actions for which a target locator (index or x,y) is mandatory.
_LOCATOR_REQUIRED: frozenset[str] = frozenset({CLICK, LONG_PRESS})


# ─────────────────────────────────────────────────────────────────────────
# Return type
# ─────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class ParseResult:
    """Outcome of ``parse_action``.

    Exactly one of ``action`` (success) or ``error`` (failure) is populated.

    Attributes:
        action: Validated ``JSONAction`` on success, else None.
        raw_text: The original model output (for logging / debugging).
        extracted_json: The substring we parsed as JSON (may differ from
            ``raw_text`` when the model wrapped it in fences or prose).
        error: Human-readable failure reason, safe to inject into the
            next ``<system-reminder>``. None on success.
        clamped: True if any coordinate was clamped to screen bounds.
            Useful for metrics — a high rate indicates prompt tuning is
            needed around coordinate accuracy.
        aliased_from: Original action_type string if an alias substitution
            was applied (e.g. "swipe" → "scroll"), else None. Track the
            rate in metrics — a high rate means the prompt's action list
            isn't reaching the model; zero means we can drop the alias.
    """

    action: Optional[JSONAction]
    raw_text: str
    extracted_json: Optional[str]
    error: Optional[str]
    clamped: bool = False
    aliased_from: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.action is not None and self.error is None


# ─────────────────────────────────────────────────────────────────────────
# JSON extraction — tolerate wrapped output
# ─────────────────────────────────────────────────────────────────────────


# Match ```json ... ``` or ``` ... ``` fence; DOTALL so the body spans lines.
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Pull JSON body out of a ```json … ``` fence, else return unchanged."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1)
    return text


def _extract_first_json_object(text: str) -> Optional[str]:
    """Return the first top-level balanced ``{...}`` substring.

    Why not just ``re.search(r"\\{.*\\}", text, re.DOTALL)``?  Greedy regex
    breaks on concatenated objects (e.g. the model hallucinating two) and
    unbalanced quotes.  Manual scan handles:

        * Strings with escaped quotes:  ``"text": "say \\"hi\\""``
        * Nested objects:                ``{"args": {"x": 1, "y": 2}}``
        * Leading prose / trailing prose
        * Nothing at all  → returns None
    """
    # Find the first opening brace that's not inside a string.
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        i += 1

    return None


# ─────────────────────────────────────────────────────────────────────────
# Field-level validation
# ─────────────────────────────────────────────────────────────────────────


def _coerce_int(value: Any) -> Optional[int]:
    """Coerce a JSON value to int; return None if impossible."""
    if isinstance(value, bool):
        # Stop ``True → 1`` silently slipping through.
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            try:
                f = float(value.strip())
                if f.is_integer():
                    return int(f)
            except ValueError:
                return None
    return None


def _clamp(value: int, lo: int, hi: int) -> tuple[int, bool]:
    """Return (clamped_value, was_clamped)."""
    if value < lo:
        return lo, True
    if value > hi:
        return hi, True
    return value, False


def _format_allowed(items: frozenset[str]) -> str:
    """Stable, sorted display of allowed values for error messages."""
    return ", ".join(repr(x) for x in sorted(items))


# ─────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────


def parse_action(
    raw_text: str,
    screen_width: int = 1080,
    screen_height: int = 2400,
) -> ParseResult:
    """Parse one VLM turn's output into a validated ``JSONAction``.

    Args:
        raw_text: Decoded generation from the model (already has the
            chat-template prefix stripped by the caller).
        screen_width, screen_height: Used for coordinate clamping.
            Defaults match Pixel 6.

    Returns:
        ``ParseResult``. Never raises — on any failure, ``action`` is None
        and ``error`` contains an actionable message ready for
        ``<system-reminder>`` reinjection.
    """
    if raw_text is None or not raw_text.strip():
        return ParseResult(
            action=None,
            raw_text=raw_text or "",
            extracted_json=None,
            error="Empty response. Emit one JSON object for the action.",
        )

    # Step 1: strip markdown fences if present (tolerant).
    unfenced = _strip_fences(raw_text)

    # Step 2: find the first balanced JSON object.
    extracted = _extract_first_json_object(unfenced)
    if extracted is None:
        return ParseResult(
            action=None,
            raw_text=raw_text,
            extracted_json=None,
            error=(
                "Response does not contain a JSON object. Your entire "
                "reply must be exactly one JSON object like "
                '{"action_type": "click", "x": 540, "y": 1850}.'
            ),
        )

    # Step 3: json.loads.
    try:
        obj = json.loads(extracted)
    except json.JSONDecodeError as e:
        return ParseResult(
            action=None,
            raw_text=raw_text,
            extracted_json=extracted,
            error=(
                f"JSON decode failed: {e.msg} at char {e.pos}. "
                "Emit exactly one valid JSON object; check quoting and commas."
            ),
        )

    if not isinstance(obj, dict):
        return ParseResult(
            action=None,
            raw_text=raw_text,
            extracted_json=extracted,
            error=(
                "Top-level JSON must be an object, not a "
                f"{type(obj).__name__}. "
                'Example: {"action_type": "click", "x": 540, "y": 1850}.'
            ),
        )

    # Step 4: action_type must be present and in v0 whitelist.
    action_type = obj.get("action_type")
    # Tolerate the common ``"action": "click"`` miss — but still flag it
    # so the model learns the correct key.
    if action_type is None and "action" in obj:
        return ParseResult(
            action=None,
            raw_text=raw_text,
            extracted_json=extracted,
            error=(
                "Field must be named 'action_type', not 'action'. "
                'Example: {"action_type": "click", "x": 540, "y": 1850}.'
            ),
        )
    if not isinstance(action_type, str):
        return ParseResult(
            action=None,
            raw_text=raw_text,
            extracted_json=extracted,
            error=(
                "Missing or non-string 'action_type'. Allowed values: "
                f"{_format_allowed(V0_ALLOWED_ACTION_TYPES)}."
            ),
        )

    aliased_from: Optional[str] = None
    if action_type in _ACTION_TYPE_ALIASES:
        aliased_from = action_type
        action_type = _ACTION_TYPE_ALIASES[action_type]

    if action_type not in V0_ALLOWED_ACTION_TYPES:
        return ParseResult(
            action=None,
            raw_text=raw_text,
            extracted_json=extracted,
            error=(
                f"action_type={action_type!r} is not in the allowed set. "
                f"Allowed: {_format_allowed(V0_ALLOWED_ACTION_TYPES)}."
            ),
        )

    # Step 5: required fields per action_type.
    required, optional = _REQUIRED_FIELDS[action_type]
    present = {k for k in obj.keys() if k != "action_type"}
    missing = required - present
    if missing:
        return ParseResult(
            action=None,
            raw_text=raw_text,
            extracted_json=extracted,
            error=(
                f"action_type={action_type!r} is missing required field(s): "
                f"{', '.join(repr(m) for m in sorted(missing))}."
            ),
        )

    # Unexpected fields are warnings-not-errors for now — the dataclass will
    # ignore unknown keys below via our selective extraction. We keep this
    # lenient because a model emitting a harmless extra key shouldn't fail
    # the whole turn.

    # Step 6: per-field validation + coord clamping.
    clamped_any = False

    kwargs: dict[str, Any] = {"action_type": action_type}

    has_index_key = "index" in obj
    has_xy_key = ("x" in obj) or ("y" in obj)

    # index XOR (x, y) — mirrors JSONAction.__post_init__. We enforce this
    # before JSONAction construction so the error message is actionable in
    # a <system-reminder> ("pick one, not both").
    if has_index_key and has_xy_key:
        return ParseResult(
            action=None,
            raw_text=raw_text,
            extracted_json=extracted,
            error=(
                "Provide either 'index' OR 'x'+'y', not both. Use 'index' "
                "(the number on the labelled box) when the element is "
                "visible in the UI element list; use 'x','y' only as a "
                "fallback for unlabelled regions."
            ),
        )

    # Locator is mandatory for click / long_press.
    if action_type in _LOCATOR_REQUIRED and not (has_index_key or has_xy_key):
        return ParseResult(
            action=None,
            raw_text=raw_text,
            extracted_json=extracted,
            error=(
                f"{action_type!r} requires either 'index' (preferred — from "
                "the UI element list) or 'x'+'y' (pixel coords)."
            ),
        )

    if has_index_key:
        idx = _coerce_int(obj.get("index"))
        if idx is None or idx < 0:
            return ParseResult(
                action=None,
                raw_text=raw_text,
                extracted_json=extracted,
                error=(
                    f"'index' must be a non-negative integer, got "
                    f"{obj.get('index')!r}."
                ),
            )
        kwargs["index"] = idx

    if has_xy_key:
        x = _coerce_int(obj.get("x"))
        y = _coerce_int(obj.get("y"))
        if action_type in (CLICK, LONG_PRESS):
            if x is None or y is None:
                return ParseResult(
                    action=None,
                    raw_text=raw_text,
                    extracted_json=extracted,
                    error=(
                        f"{action_type!r} requires integer 'x' and 'y' in "
                        f"pixels (screen is {screen_width}x{screen_height}), "
                        "or use 'index' from the UI element list."
                    ),
                )
        if x is not None:
            x, c1 = _clamp(x, 0, screen_width - 1)
            clamped_any = clamped_any or c1
            kwargs["x"] = x
        if y is not None:
            y, c2 = _clamp(y, 0, screen_height - 1)
            clamped_any = clamped_any or c2
            kwargs["y"] = y

    if "text" in obj:
        text = obj["text"]
        # JSONAction post-init coerces non-str, but we'd rather catch obvious
        # model confusion (e.g. ``text: {"foo": 1}``) here.
        if isinstance(text, (dict, list)):
            return ParseResult(
                action=None,
                raw_text=raw_text,
                extracted_json=extracted,
                error=(
                    "'text' must be a string, not a "
                    f"{type(text).__name__}."
                ),
            )
        kwargs["text"] = str(text)

    if "direction" in obj:
        direction = obj["direction"]
        if not isinstance(direction, str) or direction not in V0_ALLOWED_DIRECTIONS:
            return ParseResult(
                action=None,
                raw_text=raw_text,
                extracted_json=extracted,
                error=(
                    f"direction={direction!r} is invalid. "
                    f"Allowed: {_format_allowed(V0_ALLOWED_DIRECTIONS)}."
                ),
            )
        kwargs["direction"] = direction

    if "goal_status" in obj:
        gs = obj["goal_status"]
        if not isinstance(gs, str) or gs not in V0_ALLOWED_GOAL_STATUSES:
            return ParseResult(
                action=None,
                raw_text=raw_text,
                extracted_json=extracted,
                error=(
                    f"goal_status={gs!r} is invalid. "
                    f"Allowed: {_format_allowed(V0_ALLOWED_GOAL_STATUSES)}."
                ),
            )
        kwargs["goal_status"] = gs

    if "app_name" in obj:
        app_name = obj["app_name"]
        if not isinstance(app_name, str) or not app_name.strip():
            return ParseResult(
                action=None,
                raw_text=raw_text,
                extracted_json=extracted,
                error=(
                    "'app_name' must be a non-empty string for open_app, "
                    f"got {app_name!r}."
                ),
            )
        kwargs["app_name"] = app_name

    # Step 7: hand the sanitized kwargs to AndroidWorld's dataclass so
    # downstream env.step() sees the exact object the official baseline
    # uses.  __post_init__ does a second round of validation; if it
    # somehow raises we convert to a ParseResult rather than propagating.
    try:
        action = JSONAction(**kwargs)
    except (ValueError, TypeError) as e:
        return ParseResult(
            action=None,
            raw_text=raw_text,
            extracted_json=extracted,
            error=(
                f"JSONAction construction failed: {e}. "
                "Re-check field types and values."
            ),
        )

    return ParseResult(
        action=action,
        raw_text=raw_text,
        extracted_json=extracted,
        error=None,
        clamped=clamped_any,
        aliased_from=aliased_from,
    )


# ─────────────────────────────────────────────────────────────────────────
# Index → pixel resolution (Story 1.5)
# ─────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class IndexResolveResult:
    """Outcome of ``resolve_index_to_xy``.

    Attributes:
        action: The rewritten ``JSONAction``. On success for click /
            long_press / input_text, ``index`` is None and ``x,y`` are the
            bbox center of the selected element. For scroll with index we
            pass the action through unchanged — AW's env natively handles
            scroll-with-index, and there's no sensible single x,y for
            "scroll inside region".
        error: Non-None on failure (index out of range, element has no
            bbox, etc.). Safe to inject into a <system-reminder>.
    """

    action: Optional[JSONAction]
    error: Optional[str]

    @property
    def ok(self) -> bool:
        return self.action is not None and self.error is None


# Scroll w/ index is executed natively by the env (see actuation.execute_adb_action
# branch for scroll) — the env scales the stroke to the element's bbox.
# Converting it to an x,y here would lose that element-scoped behavior.
_INDEX_PASS_THROUGH: frozenset[str] = frozenset({SCROLL})


def resolve_index_to_xy(
    action: JSONAction,
    ui_elements: list[representation_utils.UIElement],
    logical_screen_size: tuple[int, int],
) -> IndexResolveResult:
    """Rewrite an indexed action to use pixel coords.

    For click / long_press / input_text:
        * Look up ``ui_elements[action.index]``.
        * Compute its bbox center in logical pixels.
        * Return a NEW ``JSONAction`` with ``index=None`` and ``x,y`` set.
          (JSONAction.__post_init__ rejects having both set simultaneously,
          so we always go index→xy, never copy both.)

    For scroll: pass through (the env's actuation layer handles it natively).

    For actions without ``index`` (or index=None): pass through unchanged.
    """
    if action.index is None:
        return IndexResolveResult(action=action, error=None)

    if action.action_type in _INDEX_PASS_THROUGH:
        return IndexResolveResult(action=action, error=None)

    idx = action.index
    if not isinstance(idx, int):
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return IndexResolveResult(
                action=None,
                error=(
                    f"'index' is not an integer: {action.index!r}. "
                    "Use a number from the UI element list."
                ),
            )

    if idx < 0 or idx >= len(ui_elements):
        return IndexResolveResult(
            action=None,
            error=(
                f"index={idx} is out of range — the UI element list has "
                f"{len(ui_elements)} entries (valid: 0..{len(ui_elements) - 1})."
            ),
        )

    element = ui_elements[idx]
    xy = a11y.element_center_xy(element, logical_screen_size)
    if xy is None:
        return IndexResolveResult(
            action=None,
            error=(
                f"index={idx} refers to an element with no bounding box; "
                "pick a different element."
            ),
        )
    cx, cy = xy

    # Re-construct so the dataclass' __post_init__ re-validates under the
    # new (x, y) regime and the XOR invariant holds.
    fields = action.as_dict(skip_none=True)
    fields.pop("index", None)
    fields["x"] = cx
    fields["y"] = cy
    try:
        new_action = JSONAction(**fields)
    except (ValueError, TypeError) as e:  # pragma: no cover — defensive
        return IndexResolveResult(
            action=None,
            error=f"Failed to rewrite action with x,y from index={idx}: {e}",
        )
    return IndexResolveResult(action=new_action, error=None)


# ─────────────────────────────────────────────────────────────────────────
# History helper — consumed by Rollout (Task 1.4.3)
# ─────────────────────────────────────────────────────────────────────────


def action_to_history_dict(
    step: int,
    result: ParseResult,
    *,
    exec_status: str = "ok",
) -> dict[str, Any]:
    """Compact per-step record for ``format_history`` in prompts.py.

    Args:
        step: 1-based step index within the episode.
        result: The ``ParseResult`` returned by ``parse_action``.
        exec_status: One of ``"ok"``, ``"failed_parse"``, ``"failed_exec"``,
            ``"no_screen_change"`` — the rollout decides this based on
            the env response, not the parser.
    """
    if result.ok:
        action_repr = result.action.as_dict(skip_none=True)
    else:
        # Preserve a short trace of the malformed attempt so the history
        # shows the model what it did wrong two turns ago.
        action_repr = {"_parse_error": result.error}
    return {"step": step, "action": action_repr, "status": exec_status}


__all__ = [
    "IndexResolveResult",
    "ParseResult",
    "V0_ALLOWED_ACTION_TYPES",
    "V0_ALLOWED_DIRECTIONS",
    "V0_ALLOWED_GOAL_STATUSES",
    "_ACTION_TYPE_ALIASES",
    "action_to_history_dict",
    "parse_action",
    "resolve_index_to_xy",
]
