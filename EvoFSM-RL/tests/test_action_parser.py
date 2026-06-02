"""Tests for evofsm_rl.agent.action.parse_action — Story 1.4.2.

Covers the full error-recovery matrix we expect from an 8B VLM:
    * happy-path JSON for each baseline v0 action
    * tolerant extraction: fenced, prose-wrapped, trailing whitespace
    * hard failures: wrong key name, missing required field, unknown action
    * coordinate clamping: out-of-bounds x/y
    * type coercion: floats, string-numbers, bools
    * history helper round-trip

Run:
    python -m pytest tests/test_action_parser.py -v
or  python tests/test_action_parser.py   (no pytest required)
"""

from __future__ import annotations

from evofsm_rl.agent.action import (
    _ACTION_TYPE_ALIASES,
    V0_ALLOWED_ACTION_TYPES,
    ParseResult,
    action_to_history_dict,
    parse_action,
)


# ─────────────────────────────────────────────────────────────────────────
# Happy paths — one per baseline v0 action
# ─────────────────────────────────────────────────────────────────────────


def test_click_happy_path():
    r = parse_action('{"action_type": "click", "x": 540, "y": 1850}')
    assert r.ok, r.error
    assert r.action.action_type == "click"
    assert r.action.x == 540
    assert r.action.y == 1850
    assert not r.clamped


def test_input_text_with_coords():
    r = parse_action(
        '{"action_type": "input_text", "text": "hello", "x": 100, "y": 200}'
    )
    assert r.ok, r.error
    assert r.action.action_type == "input_text"
    assert r.action.text == "hello"
    assert r.action.x == 100


def test_input_text_without_coords_is_allowed():
    r = parse_action('{"action_type": "input_text", "text": "hi"}')
    assert r.ok, r.error
    assert r.action.text == "hi"
    assert r.action.x is None and r.action.y is None


def test_scroll_down():
    r = parse_action('{"action_type": "scroll", "direction": "down"}')
    assert r.ok, r.error
    assert r.action.direction == "down"


def test_long_press():
    r = parse_action('{"action_type": "long_press", "x": 1, "y": 2}')
    assert r.ok, r.error


def test_navigate_back_no_args():
    r = parse_action('{"action_type": "navigate_back"}')
    assert r.ok, r.error


def test_navigate_home_no_args():
    r = parse_action('{"action_type": "navigate_home"}')
    assert r.ok, r.error


def test_wait_no_args():
    r = parse_action('{"action_type": "wait"}')
    assert r.ok, r.error


def test_status_complete():
    r = parse_action('{"action_type": "status", "goal_status": "complete"}')
    assert r.ok, r.error
    assert r.action.goal_status == "complete"


def test_status_infeasible():
    r = parse_action('{"action_type": "status", "goal_status": "infeasible"}')
    assert r.ok, r.error


def test_all_v0_actions_covered():
    """Protects against silently dropping a test if we add/remove actions.

    Story 1.5: whitelist now mirrors M3A's PROMPT_PREFIX action set
    (added open_app, answer, keyboard_enter).
    """
    expected = {
        "click", "input_text", "scroll", "long_press",
        "navigate_back", "navigate_home", "wait", "status",
        "open_app", "answer", "keyboard_enter",
    }
    assert V0_ALLOWED_ACTION_TYPES == expected


def test_open_app_happy_path():
    r = parse_action(
        '{"action_type": "open_app", "app_name": "Chrome"}'
    )
    assert r.ok, r.error
    assert r.action.action_type == "open_app"
    assert r.action.app_name == "Chrome"


def test_open_app_missing_app_name():
    r = parse_action('{"action_type": "open_app"}')
    assert not r.ok
    assert "app_name" in r.error


def test_open_app_empty_app_name_rejected():
    r = parse_action('{"action_type": "open_app", "app_name": ""}')
    assert not r.ok
    assert "app_name" in r.error


def test_answer_happy_path():
    r = parse_action(
        '{"action_type": "answer", "text": "Friday at 3pm."}'
    )
    assert r.ok, r.error
    assert r.action.action_type == "answer"
    assert r.action.text == "Friday at 3pm."


def test_answer_missing_text():
    r = parse_action('{"action_type": "answer"}')
    assert not r.ok
    assert "text" in r.error


def test_keyboard_enter_no_args():
    r = parse_action('{"action_type": "keyboard_enter"}')
    assert r.ok, r.error
    assert r.action.action_type == "keyboard_enter"


# ─────────────────────────────────────────────────────────────────────────
# Tolerant extraction — real 8B VLM output tends to be messy
# ─────────────────────────────────────────────────────────────────────────


def test_strips_markdown_fence():
    raw = '```json\n{"action_type": "click", "x": 1, "y": 2}\n```'
    r = parse_action(raw)
    assert r.ok, r.error
    assert r.action.x == 1


def test_strips_plain_fence():
    raw = '```\n{"action_type": "wait"}\n```'
    r = parse_action(raw)
    assert r.ok, r.error


def test_tolerates_leading_prose():
    raw = 'Here is my action:\n{"action_type": "navigate_home"}'
    r = parse_action(raw)
    assert r.ok, r.error
    assert r.action.action_type == "navigate_home"


def test_tolerates_trailing_prose():
    raw = '{"action_type": "wait"}\nLet me know if that worked.'
    r = parse_action(raw)
    assert r.ok, r.error


def test_handles_whitespace_only():
    r = parse_action("   \n  \t ")
    assert not r.ok
    assert "Empty" in r.error


def test_handles_none():
    r = parse_action(None)  # type: ignore[arg-type]
    assert not r.ok


def test_handles_nested_args_object_with_quotes_in_text():
    """Balanced-brace parser must survive escaped quotes inside strings."""
    raw = r'{"action_type": "input_text", "text": "say \"hi\""}'
    r = parse_action(raw)
    assert r.ok, r.error
    assert r.action.text == 'say "hi"'


def test_picks_first_of_two_objects():
    """Model hallucinating two objects — we use the first."""
    raw = (
        '{"action_type": "click", "x": 1, "y": 2} '
        '{"action_type": "wait"}'
    )
    r = parse_action(raw)
    assert r.ok, r.error
    assert r.action.action_type == "click"


# ─────────────────────────────────────────────────────────────────────────
# Hard failures — errors must be actionable in a <system-reminder>
# ─────────────────────────────────────────────────────────────────────────


def test_no_json_object_at_all():
    r = parse_action("I think we should click somewhere.")
    assert not r.ok
    assert "JSON" in r.error


def test_malformed_json():
    r = parse_action('{"action_type": "click", "x": 1, "y": 2,}')  # trailing comma
    assert not r.ok
    assert "JSON decode failed" in r.error


def test_wrong_field_name_action_instead_of_action_type():
    r = parse_action('{"action": "click", "x": 1, "y": 2}')
    assert not r.ok
    assert "action_type" in r.error


def test_unknown_action_type():
    r = parse_action('{"action_type": "jump"}')
    assert not r.ok
    assert "jump" in r.error
    assert "Allowed" in r.error


def test_disallowed_action_type_double_tap():
    """``double_tap`` is a real AW action but M3A doesn't teach it; reject."""
    r = parse_action('{"action_type": "double_tap", "x": 1, "y": 2}')
    assert not r.ok


def test_disallowed_action_type_swipe_without_direction_isnt_resurrected():
    """``swipe`` aliases to ``scroll`` only with a valid direction."""
    r = parse_action('{"action_type": "swipe"}')
    assert not r.ok


def test_click_missing_coordinates():
    r = parse_action('{"action_type": "click"}')
    assert not r.ok
    assert "x" in r.error or "y" in r.error


def test_click_missing_one_coordinate():
    r = parse_action('{"action_type": "click", "x": 540}')
    assert not r.ok


def test_scroll_missing_direction():
    r = parse_action('{"action_type": "scroll"}')
    assert not r.ok
    assert "direction" in r.error


def test_scroll_invalid_direction():
    r = parse_action('{"action_type": "scroll", "direction": "diagonal"}')
    assert not r.ok
    assert "diagonal" in r.error


def test_status_missing_goal_status():
    r = parse_action('{"action_type": "status"}')
    assert not r.ok
    assert "goal_status" in r.error


def test_status_invalid_goal_status():
    r = parse_action('{"action_type": "status", "goal_status": "maybe"}')
    assert not r.ok
    assert "maybe" in r.error


def test_input_text_missing_text():
    r = parse_action('{"action_type": "input_text", "x": 1, "y": 2}')
    assert not r.ok
    assert "text" in r.error


def test_input_text_with_dict_text_rejected():
    r = parse_action('{"action_type": "input_text", "text": {"foo": 1}}')
    assert not r.ok


def test_top_level_is_array():
    r = parse_action('[{"action_type": "click", "x": 1, "y": 2}]')
    # array contains a balanced {...} — extractor pulls it out.
    # This is a feature, not a bug (tolerant), so expect success.
    assert r.ok, r.error


def test_top_level_scalar():
    """A bare number has no object → extractor returns None."""
    r = parse_action("42")
    assert not r.ok


# ─────────────────────────────────────────────────────────────────────────
# Coordinate clamping & coercion
# ─────────────────────────────────────────────────────────────────────────


def test_clamp_negative_x():
    r = parse_action(
        '{"action_type": "click", "x": -10, "y": 100}',
        screen_width=1080,
        screen_height=2400,
    )
    assert r.ok, r.error
    assert r.action.x == 0
    assert r.clamped


def test_clamp_oversized_y():
    r = parse_action(
        '{"action_type": "click", "x": 100, "y": 99999}',
        screen_width=1080,
        screen_height=2400,
    )
    assert r.ok, r.error
    assert r.action.y == 2399
    assert r.clamped


def test_no_clamp_when_in_bounds():
    r = parse_action(
        '{"action_type": "click", "x": 100, "y": 200}',
        screen_width=1080,
        screen_height=2400,
    )
    assert r.ok and not r.clamped


def test_float_coordinates_coerced():
    r = parse_action('{"action_type": "click", "x": 540.0, "y": 1850.0}')
    assert r.ok, r.error
    assert r.action.x == 540


def test_string_numeric_coordinates_coerced():
    r = parse_action('{"action_type": "click", "x": "540", "y": "1850"}')
    assert r.ok, r.error
    assert r.action.x == 540


def test_bool_coordinates_rejected():
    """``True`` is technically numeric in Python but nonsense as a coord."""
    r = parse_action('{"action_type": "click", "x": true, "y": 1}')
    assert not r.ok


# ─────────────────────────────────────────────────────────────────────────
# History helper
# ─────────────────────────────────────────────────────────────────────────


def test_history_dict_on_success():
    r = parse_action('{"action_type": "click", "x": 1, "y": 2}')
    entry = action_to_history_dict(step=1, result=r)
    assert entry["step"] == 1
    assert entry["status"] == "ok"
    assert entry["action"] == {"action_type": "click", "x": 1, "y": 2}


def test_history_dict_on_failure_records_parse_error():
    r = parse_action("{bogus}")
    entry = action_to_history_dict(step=3, result=r, exec_status="failed_parse")
    assert entry["step"] == 3
    assert entry["status"] == "failed_parse"
    assert "_parse_error" in entry["action"]


def test_history_dict_exec_status_forwarded():
    r = parse_action('{"action_type": "wait"}')
    entry = action_to_history_dict(step=5, result=r, exec_status="no_screen_change")
    assert entry["status"] == "no_screen_change"


# ─────────────────────────────────────────────────────────────────────────
# Action-type aliases — accept common model drift, keep canonical output
# ─────────────────────────────────────────────────────────────────────────


def test_alias_swipe_maps_to_scroll():
    r = parse_action('{"action_type": "swipe", "direction": "up"}')
    assert r.ok, r.error
    assert r.action.action_type == "scroll"
    assert r.action.direction == "up"
    assert r.aliased_from == "swipe"


def test_alias_type_maps_to_input_text():
    r = parse_action('{"action_type": "type", "text": "hello"}')
    assert r.ok, r.error
    assert r.action.action_type == "input_text"
    assert r.action.text == "hello"
    assert r.aliased_from == "type"


def test_alias_back_maps_to_navigate_back():
    r = parse_action('{"action_type": "back"}')
    assert r.ok, r.error
    assert r.action.action_type == "navigate_back"
    assert r.aliased_from == "back"


def test_alias_home_maps_to_navigate_home():
    r = parse_action('{"action_type": "home"}')
    assert r.ok, r.error
    assert r.action.action_type == "navigate_home"
    assert r.aliased_from == "home"


def test_alias_swipe_missing_direction_still_fails():
    """Aliasing must not bypass per-action required-field checks."""
    r = parse_action('{"action_type": "swipe"}')
    assert not r.ok
    assert "direction" in r.error


def test_alias_swipe_invalid_direction_still_fails():
    r = parse_action('{"action_type": "swipe", "direction": "diagonal"}')
    assert not r.ok
    assert "diagonal" in r.error


def test_alias_does_not_resurrect_disallowed_actions():
    """double_tap / unknown must stay rejected — they're not in any alias."""
    r = parse_action('{"action_type": "double_tap", "x": 1, "y": 2}')
    assert not r.ok
    r = parse_action('{"action_type": "unknown"}')
    assert not r.ok


def test_aliased_from_none_when_not_aliased():
    r = parse_action('{"action_type": "scroll", "direction": "down"}')
    assert r.ok, r.error
    assert r.aliased_from is None


def test_alias_table_targets_are_all_in_whitelist():
    """Protects against typo'd aliases that would then 404 on whitelist."""
    for target in _ACTION_TYPE_ALIASES.values():
        assert target in V0_ALLOWED_ACTION_TYPES, target


def test_alias_table_sources_not_in_whitelist():
    """Sanity: alias keys must NOT also be canonical names (would be dead code)."""
    for source in _ACTION_TYPE_ALIASES:
        assert source not in V0_ALLOWED_ACTION_TYPES, source


# ─────────────────────────────────────────────────────────────────────────
# Index field (Story 1.5) — a11y tree + SoM input path
# ─────────────────────────────────────────────────────────────────────────


def test_click_with_index_ok():
    r = parse_action('{"action_type": "click", "index": 7}')
    assert r.ok, r.error
    assert r.action.action_type == "click"
    assert r.action.index == 7
    assert r.action.x is None and r.action.y is None


def test_long_press_with_index_ok():
    r = parse_action('{"action_type": "long_press", "index": 2}')
    assert r.ok, r.error
    assert r.action.index == 2


def test_input_text_with_index_ok():
    r = parse_action(
        '{"action_type": "input_text", "text": "hi", "index": 3}'
    )
    assert r.ok, r.error
    assert r.action.text == "hi"
    assert r.action.index == 3


def test_scroll_with_index_ok():
    r = parse_action(
        '{"action_type": "scroll", "direction": "down", "index": 5}'
    )
    assert r.ok, r.error
    assert r.action.direction == "down"
    assert r.action.index == 5


def test_click_rejects_both_index_and_xy():
    r = parse_action(
        '{"action_type": "click", "index": 1, "x": 10, "y": 20}'
    )
    assert not r.ok
    assert "index" in r.error.lower()


def test_click_with_neither_index_nor_xy():
    """A click action must provide a locator — index OR x,y."""
    r = parse_action('{"action_type": "click"}')
    assert not r.ok
    assert "index" in r.error.lower() or "x" in r.error


def test_index_must_be_non_negative_int():
    r = parse_action('{"action_type": "click", "index": -1}')
    assert not r.ok
    assert "index" in r.error.lower()


def test_index_rejects_non_numeric():
    r = parse_action('{"action_type": "click", "index": "foo"}')
    assert not r.ok


def test_index_coerces_float():
    """Models sometimes emit 7.0 — coerce cleanly."""
    r = parse_action('{"action_type": "click", "index": 7.0}')
    assert r.ok, r.error
    assert r.action.index == 7


# ─────────────────────────────────────────────────────────────────────────
# resolve_index_to_xy
# ─────────────────────────────────────────────────────────────────────────


def _make_ui_element(x_min, y_min, x_max, y_max, *, visible=True):
    """Tiny UIElement factory for index-resolution tests."""
    from android_world.env import representation_utils

    return representation_utils.UIElement(
        text="",
        bbox_pixels=representation_utils.BoundingBox(
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
        ),
        is_visible=visible,
        is_clickable=True,
    )


def test_resolve_index_sets_xy_and_clears_index():
    from evofsm_rl.agent.action import resolve_index_to_xy
    from android_world.env.json_action import JSONAction

    elements = [
        _make_ui_element(0, 0, 100, 100),
        _make_ui_element(200, 400, 400, 500),
    ]
    action = JSONAction(action_type="click", index=1)
    res = resolve_index_to_xy(action, elements, (1080, 2400))
    assert res.ok, res.error
    # Center of bbox (200,400)-(400,500) = (300, 450)
    assert res.action.x == 300
    assert res.action.y == 450
    assert res.action.index is None


def test_resolve_index_preserves_text_for_input_text():
    from evofsm_rl.agent.action import resolve_index_to_xy
    from android_world.env.json_action import JSONAction

    elements = [_make_ui_element(0, 0, 200, 100)]
    action = JSONAction(action_type="input_text", text="hello world", index=0)
    res = resolve_index_to_xy(action, elements, (1080, 2400))
    assert res.ok
    assert res.action.text == "hello world"
    assert res.action.x == 100 and res.action.y == 50
    assert res.action.index is None


def test_resolve_index_out_of_range():
    from evofsm_rl.agent.action import resolve_index_to_xy
    from android_world.env.json_action import JSONAction

    elements = [_make_ui_element(0, 0, 100, 100)]
    action = JSONAction(action_type="click", index=5)
    res = resolve_index_to_xy(action, elements, (1080, 2400))
    assert not res.ok
    assert "5" in res.error
    assert "range" in res.error.lower()


def test_resolve_index_no_bbox():
    from evofsm_rl.agent.action import resolve_index_to_xy
    from android_world.env.json_action import JSONAction
    from android_world.env import representation_utils

    elements = [representation_utils.UIElement(text="x", is_visible=True)]
    action = JSONAction(action_type="click", index=0)
    res = resolve_index_to_xy(action, elements, (1080, 2400))
    assert not res.ok
    assert "bounding box" in res.error.lower()


def test_resolve_passes_through_when_no_index():
    from evofsm_rl.agent.action import resolve_index_to_xy
    from android_world.env.json_action import JSONAction

    action = JSONAction(action_type="click", x=50, y=60)
    res = resolve_index_to_xy(action, [], (1080, 2400))
    assert res.ok
    assert res.action is action
    assert res.action.x == 50 and res.action.y == 60


def test_resolve_scroll_passes_through_with_index():
    """Scroll with index is handled natively by env.execute_action."""
    from evofsm_rl.agent.action import resolve_index_to_xy
    from android_world.env.json_action import JSONAction

    elements = [_make_ui_element(0, 0, 100, 100)]
    action = JSONAction(action_type="scroll", direction="down", index=0)
    res = resolve_index_to_xy(action, elements, (1080, 2400))
    assert res.ok
    # Scroll should retain its index — the env uses it to scope the stroke.
    assert res.action.index == 0
    assert res.action.direction == "down"
    assert res.action.x is None and res.action.y is None


def test_resolve_center_clamped_to_screen():
    """A bbox that extends past screen edge should still resolve in-bounds."""
    from evofsm_rl.agent.action import resolve_index_to_xy
    from android_world.env.json_action import JSONAction

    # bbox center would be (1090, 50) on a 1080×2400 screen → clamp x to 1079.
    elements = [_make_ui_element(1000, 0, 1180, 100)]
    action = JSONAction(action_type="click", index=0)
    res = resolve_index_to_xy(action, elements, (1080, 2400))
    assert res.ok
    assert res.action.x == 1079


# ─────────────────────────────────────────────────────────────────────────
# Never-raise contract
# ─────────────────────────────────────────────────────────────────────────


def test_parse_action_never_raises_on_random_garbage():
    samples = [
        "",
        "{",
        "}",
        "}{",
        '{"a": [1, 2, 3',
        '{"action_type": }',
        '\x00\x01{"action_type": "click"}',
        "null",
        "true",
        'Let me think... {"action_type": "click", "x": 1, "y": 2}.',
    ]
    for s in samples:
        r = parse_action(s)
        assert isinstance(r, ParseResult), f"non-ParseResult for {s!r}"


# ─────────────────────────────────────────────────────────────────────────
# Standalone runner for no-pytest environments
# ─────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import traceback

    ns = dict(globals())
    tests = [(name, fn) for name, fn in ns.items()
             if name.startswith("test_") and callable(fn)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  ✔ {name}")
        except Exception:
            failed += 1
            print(f"  ✘ {name}")
            traceback.print_exc()
    print(f"\n{passed}/{passed + failed} tests passed")
    raise SystemExit(0 if failed == 0 else 1)
