"""Tests for evofsm_rl.fsm.diff — Story 3.2.

The 31 tests from the story brief, grouped by concern:

  Parse (1-6):
    1.  FSMDiff.from_json(diff.to_json()) == diff
    2.  Case-insensitive op values
    3.  Pluralized target values
    4.  add op missing 'value' is skipped at parse time
    5.  Unknown layer / target is skipped
    6.  Empty ops list is valid

  Apply — states (7-12):
    7.  add state lands in layer1.states
    8.  add on existing state id upgrades to modify + warning
    9.  modify patches only fields present in value
    10. modify on missing state upgrades to add + warning
    11. remove cascades to transitions referencing the state
    12. remove on missing state is skipped, not crashed

  Apply — transitions (13-16):
    13. add lands in layer1.transitions
    14. modify changes the action
    15. remove by "from->to" key
    16. add with unknown states still applied with warning

  Apply — strategies (17-19):
    17. add lands in layer1.strategies
    18. modify with a full steps list fully replaces
    19. remove by name

  Apply — categories (20-22):
    20. add lands in layer2.categories
    21. modify's failure_modes list REPLACES (not merges)
    22. remove by name

  Cross-cutting (23-25, validate 26-27):
    23. Original FSM is not mutated
    24. ApplyResult tracks applied vs skipped
    25. Multiple ops applied in order
    26. validate_fsm_integrity flags dangling transition
    27. validate passes on clean FSM

  LLM response extraction (28-31):
    28. Bare JSON parses
    29. Fenced ```json block extracts
    30. JSON embedded in prose extracts first balanced {}
    31. No JSON at all raises ValueError

Run::
    python -m pytest tests/test_fsm_diff.py -v
"""

from __future__ import annotations

import copy

import pytest

from evofsm_rl.fsm.diff import (
    ApplyResult,
    DiffOp,
    FSMDiff,
    apply_diff,
    parse_diff_from_llm_response,
    validate_fsm_integrity,
)
from evofsm_rl.fsm.schema import (
    AbstractCategory,
    FSM,
    Layer1,
    Layer2,
    State,
    Strategy,
    Transition,
)


# ─────────────────────────────────────────────────────────────────────
# Fixture helpers — small fully-populated FSM reused across tests
# ─────────────────────────────────────────────────────────────────────


def _sample_fsm() -> FSM:
    """FSM with 2 states, 1 transition, 1 strategy, 1 dead_end, 1 L2 cat."""
    return FSM(
        app="markor",
        layer1=Layer1(
            app="markor",
            category="Productivity",
            states=[
                State(id="home", description="home screen",
                      visual_cues=["floating + button"],
                      resource_hints=["fab_main"]),
                State(id="note_editor", description="editor",
                      visual_cues=["title field"],
                      resource_hints=["edit_text"]),
            ],
            transitions=[
                Transition(from_state="home", to_state="note_editor",
                           action="click(fab_main)"),
            ],
            strategies=[
                Strategy(name="CREATE_NOTE",
                         preconditions="home visible",
                         steps=["click fab", "type title", "save"],
                         success_signal="note in list",
                         fallback=""),
            ],
            dead_ends=[
                {"state": "home", "failed_action": "long_press(fab_main)",
                 "note": "long press does nothing here"},
            ],
        ),
        layer2=Layer2(
            categories=[
                AbstractCategory(
                    name="ADD_ENTRY",
                    precondition="entry list visible",
                    abstract_steps=["locate primary affordance", "save"],
                    failure_modes=["confirming too early"],
                    verification_checklist=["new entry visible"],
                ),
            ],
        ),
    )


# ═════════════════════════════════════════════════════════════════════
# Parse tests
# ═════════════════════════════════════════════════════════════════════


def test_from_json_round_trip_identity():
    """Test 1: to_json → from_json reproduces the diff exactly."""
    diff = FSMDiff(
        ops=[
            DiffOp(layer="layer1", target="state", op="add",
                   key="search", value={"id": "search", "description": "search screen",
                                         "visual_cues": [], "resource_hints": []}),
            DiffOp(layer="layer2", target="category", op="modify",
                   key="ADD_ENTRY",
                   value={"failure_modes": ["x", "y"]}),
            DiffOp(layer="layer1", target="state", op="remove",
                   key="stale", value=None),
        ],
        reflection_summary="rationale goes here",
        layer_tag="both",
    )
    rt = FSMDiff.from_json(diff.to_json())
    assert rt == diff


def test_from_json_case_insensitive_op_and_layer():
    """Test 2: 'ADD' / 'Layer1' / 'CATEGORY' all normalize to lowercase."""
    data = {
        "ops": [
            {"layer": "Layer1", "target": "STATE", "op": "ADD",
             "key": "x", "value": {"id": "x"}},
        ],
    }
    diff = FSMDiff.from_json(data)
    assert len(diff.ops) == 1
    op = diff.ops[0]
    assert op.layer == "layer1"
    assert op.target == "state"
    assert op.op == "add"


def test_from_json_pluralized_target():
    """Test 3: 'states', 'categories', 'strategies', 'dead_ends' normalize."""
    data = {
        "ops": [
            {"layer": "layer1", "target": "states", "op": "remove", "key": "a"},
            {"layer": "layer1", "target": "transitions", "op": "remove", "key": "a->b"},
            {"layer": "layer1", "target": "strategies", "op": "remove", "key": "S"},
            {"layer": "layer1", "target": "dead_ends", "op": "remove", "key": "d"},
            {"layer": "layer2", "target": "categories", "op": "remove", "key": "C"},
        ],
    }
    diff = FSMDiff.from_json(data)
    assert [o.target for o in diff.ops] == [
        "state", "transition", "strategy", "dead_end", "category",
    ]


def test_from_json_add_missing_value_is_skipped_at_parse():
    """Test 4: 'add' with no value is dropped before it reaches the applier."""
    data = {
        "ops": [
            {"layer": "layer1", "target": "state", "op": "add", "key": "x"},
            {"layer": "layer1", "target": "state", "op": "remove", "key": "x"},
        ],
    }
    diff = FSMDiff.from_json(data)
    # The add is dropped; the remove (no value required) survives.
    assert len(diff.ops) == 1
    assert diff.ops[0].op == "remove"


def test_from_json_unknown_layer_or_target_is_skipped():
    """Test 5: unrecognized enum values are quietly dropped."""
    data = {
        "ops": [
            {"layer": "layer3", "target": "state", "op": "add",
             "key": "x", "value": {"id": "x"}},  # bad layer
            {"layer": "layer1", "target": "gizmo", "op": "add",
             "key": "x", "value": {"id": "x"}},  # bad target
            {"layer": "layer2", "target": "state", "op": "add",
             "key": "x", "value": {"id": "x"}},  # target not valid for layer
            {"layer": "layer1", "target": "state", "op": "add",
             "key": "good", "value": {"id": "good"}},  # the survivor
        ],
    }
    diff = FSMDiff.from_json(data)
    assert [o.key for o in diff.ops] == ["good"]


def test_from_json_empty_ops_is_valid():
    """Test 6: zero-op diff is a legal no-op."""
    diff = FSMDiff.from_json({"ops": []})
    assert diff.ops == []
    assert diff.reflection_summary == ""
    assert diff.layer_tag == "both"


# ═════════════════════════════════════════════════════════════════════
# Apply — states
# ═════════════════════════════════════════════════════════════════════


def test_apply_add_state():
    """Test 7: add state appears in layer1.states."""
    fsm = _sample_fsm()
    diff = FSMDiff(ops=[DiffOp(
        layer="layer1", target="state", op="add", key="search",
        value={"id": "search", "description": "search screen",
               "visual_cues": ["magnifier icon"], "resource_hints": ["search_src"]},
    )])
    r = apply_diff(fsm, diff)
    assert any(s.id == "search" for s in r.fsm.layer1.states)
    added = next(s for s in r.fsm.layer1.states if s.id == "search")
    assert added.description == "search screen"
    assert added.visual_cues == ["magnifier icon"]
    assert len(r.applied) == 1
    assert r.skipped == []


def test_apply_add_duplicate_state_upgrades_to_modify_with_warning():
    """Test 8: re-adding a known state id is treated as modify + warning."""
    fsm = _sample_fsm()
    diff = FSMDiff(ops=[DiffOp(
        layer="layer1", target="state", op="add", key="home",
        value={"id": "home", "description": "home (edited)"},
    )])
    r = apply_diff(fsm, diff)
    home = next(s for s in r.fsm.layer1.states if s.id == "home")
    assert home.description == "home (edited)"
    assert len(r.applied) == 1
    assert any("already exists" in w for w in r.warnings)


def test_apply_modify_state_patches_single_field():
    """Test 9: modify with one field in value leaves other fields untouched."""
    fsm = _sample_fsm()
    diff = FSMDiff(ops=[DiffOp(
        layer="layer1", target="state", op="modify", key="home",
        value={"description": "rewritten home"},
    )])
    r = apply_diff(fsm, diff)
    home = next(s for s in r.fsm.layer1.states if s.id == "home")
    assert home.description == "rewritten home"
    # Untouched fields stay.
    assert home.visual_cues == ["floating + button"]
    assert home.resource_hints == ["fab_main"]


def test_apply_modify_missing_state_upgrades_to_add_with_warning():
    """Test 10: modify of unknown state id creates the state, warns."""
    fsm = _sample_fsm()
    diff = FSMDiff(ops=[DiffOp(
        layer="layer1", target="state", op="modify", key="ghost",
        value={"id": "ghost", "description": "materialized"},
    )])
    r = apply_diff(fsm, diff)
    assert any(s.id == "ghost" for s in r.fsm.layer1.states)
    assert any("not found for modify" in w for w in r.warnings)


def test_apply_remove_state_cascades_to_transitions():
    """Test 11: removing a state also drops transitions touching it."""
    fsm = _sample_fsm()
    diff = FSMDiff(ops=[DiffOp(
        layer="layer1", target="state", op="remove", key="note_editor",
    )])
    r = apply_diff(fsm, diff)
    assert all(s.id != "note_editor" for s in r.fsm.layer1.states)
    # Original had home->note_editor; that transition must be gone.
    assert all(
        t.to_state != "note_editor" and t.from_state != "note_editor"
        for t in r.fsm.layer1.transitions
    )
    assert any("removed" in w and "transition" in w for w in r.warnings)


def test_apply_remove_missing_state_is_skipped():
    """Test 12: remove on an unknown id ends up in skipped, doesn't crash."""
    fsm = _sample_fsm()
    diff = FSMDiff(ops=[DiffOp(
        layer="layer1", target="state", op="remove", key="nope",
    )])
    r = apply_diff(fsm, diff)
    assert r.applied == []
    assert len(r.skipped) == 1
    assert r.skipped[0][0].key == "nope"


# ═════════════════════════════════════════════════════════════════════
# Apply — transitions
# ═════════════════════════════════════════════════════════════════════


def test_apply_add_transition():
    """Test 13: add transition lands in layer1.transitions."""
    fsm = _sample_fsm()
    diff = FSMDiff(ops=[DiffOp(
        layer="layer1", target="transition", op="add",
        key="note_editor->home",
        value={"from_state": "note_editor", "to_state": "home",
               "action": "click(back)"},
    )])
    r = apply_diff(fsm, diff)
    assert any(
        t.from_state == "note_editor" and t.to_state == "home"
        for t in r.fsm.layer1.transitions
    )


def test_apply_modify_transition_changes_action():
    """Test 14: modify patches the action field."""
    fsm = _sample_fsm()
    diff = FSMDiff(ops=[DiffOp(
        layer="layer1", target="transition", op="modify",
        key="home->note_editor",
        value={"action": "tap(fab_main) OR tap(menu)"},
    )])
    r = apply_diff(fsm, diff)
    t = next(t for t in r.fsm.layer1.transitions
              if t.from_state == "home" and t.to_state == "note_editor")
    assert t.action == "tap(fab_main) OR tap(menu)"


def test_apply_remove_transition_by_key():
    """Test 15: remove by 'from->to' key deletes the edge."""
    fsm = _sample_fsm()
    diff = FSMDiff(ops=[DiffOp(
        layer="layer1", target="transition", op="remove",
        key="home->note_editor",
    )])
    r = apply_diff(fsm, diff)
    assert r.fsm.layer1.transitions == []


def test_apply_add_transition_with_unknown_state_warns_but_applies():
    """Test 16: forward-reference transition gets a warning, still applied."""
    fsm = _sample_fsm()
    diff = FSMDiff(ops=[DiffOp(
        layer="layer1", target="transition", op="add",
        key="home->phantom",
        value={"from_state": "home", "to_state": "phantom",
               "action": "swipe_up"},
    )])
    r = apply_diff(fsm, diff)
    assert any(
        t.from_state == "home" and t.to_state == "phantom"
        for t in r.fsm.layer1.transitions
    )
    assert any("unknown state" in w for w in r.warnings)


# ═════════════════════════════════════════════════════════════════════
# Apply — strategies
# ═════════════════════════════════════════════════════════════════════


def test_apply_add_strategy():
    """Test 17: add strategy lands in layer1.strategies."""
    fsm = _sample_fsm()
    diff = FSMDiff(ops=[DiffOp(
        layer="layer1", target="strategy", op="add", key="DELETE_NOTE",
        value={"name": "DELETE_NOTE",
               "preconditions": "note_editor open",
               "steps": ["open menu", "confirm"],
               "success_signal": "note gone"},
    )])
    r = apply_diff(fsm, diff)
    assert any(s.name == "DELETE_NOTE" for s in r.fsm.layer1.strategies)


def test_apply_modify_strategy_steps_replaces_list():
    """Test 18: supplying 'steps' in modify fully replaces the list."""
    fsm = _sample_fsm()
    diff = FSMDiff(ops=[DiffOp(
        layer="layer1", target="strategy", op="modify", key="CREATE_NOTE",
        value={"steps": ["ONE", "TWO"]},
    )])
    r = apply_diff(fsm, diff)
    s = next(s for s in r.fsm.layer1.strategies if s.name == "CREATE_NOTE")
    assert s.steps == ["ONE", "TWO"]
    # Other fields untouched.
    assert s.success_signal == "note in list"


def test_apply_remove_strategy_by_name():
    """Test 19: remove by name drops the strategy."""
    fsm = _sample_fsm()
    diff = FSMDiff(ops=[DiffOp(
        layer="layer1", target="strategy", op="remove", key="CREATE_NOTE",
    )])
    r = apply_diff(fsm, diff)
    assert r.fsm.layer1.strategies == []


# ═════════════════════════════════════════════════════════════════════
# Apply — categories
# ═════════════════════════════════════════════════════════════════════


def test_apply_add_category():
    """Test 20: add category appears in layer2.categories."""
    fsm = _sample_fsm()
    diff = FSMDiff(ops=[DiffOp(
        layer="layer2", target="category", op="add", key="SEARCH_ENTRY",
        value={
            "name": "SEARCH_ENTRY",
            "precondition": "list surface visible",
            "abstract_steps": ["invoke search", "enter query", "select result"],
            "failure_modes": ["misreading autocomplete"],
            "verification_checklist": ["target entry visible"],
        },
    )])
    r = apply_diff(fsm, diff)
    names = [c.name for c in r.fsm.layer2.categories]
    assert "SEARCH_ENTRY" in names


def test_apply_modify_category_failure_modes_replaces_list():
    """Test 21: modify with a new 'failure_modes' list replaces wholesale."""
    fsm = _sample_fsm()
    diff = FSMDiff(ops=[DiffOp(
        layer="layer2", target="category", op="modify", key="ADD_ENTRY",
        value={"failure_modes": ["A", "B"]},
    )])
    r = apply_diff(fsm, diff)
    c = next(c for c in r.fsm.layer2.categories if c.name == "ADD_ENTRY")
    # Replacement, not merge: original "confirming too early" is gone.
    assert c.failure_modes == ["A", "B"]
    # Other fields untouched.
    assert c.precondition == "entry list visible"


def test_apply_remove_category_by_name():
    """Test 22: remove by name drops the category."""
    fsm = _sample_fsm()
    diff = FSMDiff(ops=[DiffOp(
        layer="layer2", target="category", op="remove", key="ADD_ENTRY",
    )])
    r = apply_diff(fsm, diff)
    assert r.fsm.layer2.categories == []


# ═════════════════════════════════════════════════════════════════════
# Cross-cutting apply tests
# ═════════════════════════════════════════════════════════════════════


def test_apply_does_not_mutate_original_fsm():
    """Test 23: the original FSM is untouched — apply works on a deep copy."""
    fsm = _sample_fsm()
    pristine = copy.deepcopy(fsm)
    diff = FSMDiff(ops=[
        DiffOp(layer="layer1", target="state", op="remove", key="note_editor"),
        DiffOp(layer="layer2", target="category", op="remove", key="ADD_ENTRY"),
    ])
    r = apply_diff(fsm, diff)
    # Mutated result must differ...
    assert r.fsm.to_json() != pristine.to_json()
    # ...but the input must not.
    assert fsm.to_json() == pristine.to_json()


def test_apply_result_tracks_applied_vs_skipped():
    """Test 24: ApplyResult partitions ops correctly."""
    fsm = _sample_fsm()
    diff = FSMDiff(ops=[
        # This one applies (modify existing).
        DiffOp(layer="layer2", target="category", op="modify", key="ADD_ENTRY",
               value={"precondition": "rewritten"}),
        # This one skips (remove non-existent).
        DiffOp(layer="layer1", target="state", op="remove", key="missing"),
    ])
    r = apply_diff(fsm, diff)
    assert len(r.applied) == 1 and r.applied[0].key == "ADD_ENTRY"
    assert len(r.skipped) == 1 and r.skipped[0][0].key == "missing"


def test_apply_multiple_ops_in_declared_order():
    """Test 25: multiple ops apply in the order given; later ops see earlier changes."""
    fsm = _sample_fsm()
    diff = FSMDiff(ops=[
        DiffOp(layer="layer1", target="state", op="add", key="settings",
               value={"id": "settings"}),
        DiffOp(layer="layer1", target="transition", op="add",
               key="home->settings",
               value={"from_state": "home", "to_state": "settings",
                      "action": "click(gear)"}),
        # Now delete settings; the transition cascade should trigger.
        DiffOp(layer="layer1", target="state", op="remove", key="settings"),
    ])
    r = apply_diff(fsm, diff)
    assert all(s.id != "settings" for s in r.fsm.layer1.states)
    assert all(
        t.to_state != "settings" and t.from_state != "settings"
        for t in r.fsm.layer1.transitions
    )
    # All three ops report as applied.
    assert len(r.applied) == 3


def test_validate_fsm_integrity_flags_dangling_transition():
    """Test 26: a transition to a non-existent state produces a warning."""
    fsm = _sample_fsm()
    # Add a transition whose to_state does not exist in layer1.states.
    fsm.layer1.transitions.append(
        Transition(from_state="home", to_state="void", action="teleport")
    )
    warnings = validate_fsm_integrity(fsm)
    assert any("does not exist" in w and "void" in w for w in warnings)


def test_validate_fsm_integrity_passes_on_clean_fsm():
    """Test 27: the sample fixture passes integrity check."""
    assert validate_fsm_integrity(_sample_fsm()) == []


# ═════════════════════════════════════════════════════════════════════
# LLM response extraction
# ═════════════════════════════════════════════════════════════════════


_VALID_DIFF_OBJECT = {
    "ops": [
        {"layer": "layer1", "target": "state", "op": "remove", "key": "old_state"},
    ],
    "reflection_summary": "drop an obsolete screen",
    "layer_tag": "layer1",
}


def test_parse_bare_json_response():
    """Test 28: bare JSON parses into an FSMDiff."""
    import json as _json
    raw = _json.dumps(_VALID_DIFF_OBJECT)
    diff = parse_diff_from_llm_response(raw)
    assert len(diff.ops) == 1
    assert diff.ops[0].key == "old_state"
    assert diff.reflection_summary == "drop an obsolete screen"


def test_parse_fenced_json_block():
    """Test 29: ```json ... ``` fence extracts correctly."""
    import json as _json
    raw = (
        "Sure, here is the diff:\n\n"
        "```json\n"
        + _json.dumps(_VALID_DIFF_OBJECT)
        + "\n```\n\n"
        "Let me know if you'd like me to adjust."
    )
    diff = parse_diff_from_llm_response(raw)
    assert diff.ops[0].key == "old_state"


def test_parse_json_embedded_in_prose():
    """Test 30: falls back to first balanced {...} when no fence."""
    import json as _json
    raw = (
        "Reasoning first. "
        + _json.dumps(_VALID_DIFF_OBJECT)
        + " — and that's the full diff."
    )
    diff = parse_diff_from_llm_response(raw)
    assert diff.ops[0].key == "old_state"


def test_parse_no_json_raises_value_error():
    """Test 31: text without any JSON object raises ValueError."""
    with pytest.raises(ValueError):
        parse_diff_from_llm_response("No JSON here, just prose about the app.")


# ─────────────────────────────────────────────────────────────────────
# Pytest-less standalone runner
# ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import inspect
    import traceback

    ns = dict(globals())
    tests = [(name, fn) for name, fn in ns.items()
             if name.startswith("test_") and callable(fn)]
    passed = failed = 0
    for name, fn in tests:
        try:
            sig = inspect.signature(fn)
            if any(p in sig.parameters for p in ("monkeypatch", "tmp_path")):
                continue  # none of these tests need them
            fn()
            passed += 1
            print(f"  ok    {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    total = passed + failed
    print(f"\n{passed}/{total} passed")
    raise SystemExit(0 if failed == 0 else 1)
