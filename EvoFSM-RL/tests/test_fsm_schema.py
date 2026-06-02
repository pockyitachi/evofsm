"""Tests for evofsm_rl.fsm.schema — Story 2.2.1.

Coverage matrix:

    * Round-trip equality FSM == FSM.from_json(fsm.to_json()) at every
      level of nesting (State / Transition / Strategy / Layer1 /
      AbstractCategory / Layer2 / FSM).
    * Round-trip preserves Unicode in user-facing strings.
    * Round-trip preserves list ordering (no sets / no sorting).
    * Empty edge cases: zero states, zero transitions, zero strategies,
      zero dead_ends, zero categories, empty metadata.
    * Optional fields default cleanly when absent in JSON.
    * Strict validation: missing required field → ValueError with
      actionable message.
    * Version policy: major-version mismatch raises SchemaVersionError;
      minor-version drift is accepted; future additive fields are ignored.
    * to_prompt_text format invariants: required headers present, layer
      separation visible, byte-stable output for identical input.

Run:
    python -m pytest tests/test_fsm_schema.py -v
or  python tests/test_fsm_schema.py   (no pytest required)
"""

from __future__ import annotations

import json

from evofsm_rl.fsm import (
    SCHEMA_VERSION,
    AbstractCategory,
    FSM,
    Layer1,
    Layer2,
    SchemaVersionError,
    State,
    Strategy,
    Transition,
)


# ─────────────────────────────────────────────────────────────────────────
# Fixtures — small fully-populated and minimal FSMs we reuse
# ─────────────────────────────────────────────────────────────────────────


def _full_fsm() -> FSM:
    """A small but fully-populated FSM: every field non-default, every
    list non-empty. The reference object for round-trip tests."""
    layer1 = Layer1(
        app="markor",
        category="Productivity",
        states=[
            State(
                id="home",
                description="file list view",
                visual_cues=["floating + button bottom-right", "bottom nav bar"],
                resource_hints=["net.gsantner.markor:id/fab_main"],
            ),
            State(
                id="note_editor",
                description="text editor for one note",
                visual_cues=["full-screen text area", "save icon top-right"],
                resource_hints=["net.gsantner.markor:id/document_edit_text"],
            ),
        ],
        transitions=[
            Transition(
                from_state="home",
                to_state="note_editor",
                action="click(fab_main)",
                precondition="app is open at home",
                postcondition="text area is focused",
            ),
            Transition(
                from_state="note_editor",
                to_state="home",
                action="click(save_icon)",
                postcondition="note saved to disk",
            ),
        ],
        strategies=[
            Strategy(
                name="CREATE_NOTE",
                preconditions="markor is launched and showing the file list",
                steps=[
                    "tap floating + button to open new-file dialog",
                    "type filename into the name field",
                    "select 'Markdown (.md)' as the file type",
                    "tap the OK button",
                    "type body content into the editor",
                    "tap save icon to persist",
                ],
                success_signal="new file appears in the home file list",
                fallback="navigate back, retry from home if save toast didn't show",
            ),
        ],
        dead_ends=[
            {
                "state": "note_editor",
                "failed_action": "navigate_back without save",
                "note": "back-navigation discards unsaved edits silently",
            },
        ],
    )
    layer2 = Layer2(
        categories=[
            AbstractCategory(
                name="ADD_ENTRY",
                precondition="a home-like screen is visible; a primary add affordance exists",
                abstract_steps=[
                    "Locate primary add affordance (floating button OR overflow menu OR tab)",
                    "Enter required text fields top to bottom",
                    "Confirm via explicit save control; do NOT rely on back-gesture to save",
                    "Verify the new entry appears in the primary list before declaring success",
                ],
                failure_modes=[
                    "tapping save before all required fields are filled",
                    "confusing the back button with save",
                ],
                verification_checklist=[
                    "new row visible in the main list with the entered values",
                ],
            ),
            AbstractCategory(
                name="DELETE_ENTRY",
                precondition="an entry is visible in a list",
                abstract_steps=[
                    "Long-press the target entry to open the context menu",
                    "Choose Delete from the menu",
                    "Confirm in the resulting dialog",
                ],
                failure_modes=["dismissing the dialog instead of confirming"],
                verification_checklist=["the entry no longer appears in the list"],
            ),
        ],
    )
    return FSM(
        app="markor",
        layer1=layer1,
        layer2=layer2,
        metadata={"built_at": "2026-04-18T10:00:00Z", "n_episodes": 30, "sr": 0.42},
    )


def _empty_fsm() -> FSM:
    """The minimum-viable FSM: required-only fields, all lists empty."""
    return FSM(
        app="dummy_app",
        layer1=Layer1(app="dummy_app", category="Tools"),
        layer2=Layer2(),
    )


# ─────────────────────────────────────────────────────────────────────────
# Schema version
# ─────────────────────────────────────────────────────────────────────────


def test_schema_version_constant_format():
    parts = SCHEMA_VERSION.split(".")
    assert len(parts) == 3
    for p in parts:
        assert p.isdigit(), f"version part {p!r} is not numeric"


def test_default_fsm_version_matches_constant():
    fsm = _empty_fsm()
    assert fsm.version == SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────
# Per-class round-trip
# ─────────────────────────────────────────────────────────────────────────


def test_state_round_trip():
    s = State(id="x", description="d", visual_cues=["a", "b"], resource_hints=["r"])
    assert State.from_json(s.to_json()) == s


def test_state_round_trip_minimal():
    s = State(id="bare")
    assert State.from_json(s.to_json()) == s


def test_transition_round_trip_full():
    t = Transition(
        from_state="a", to_state="b", action="click(x)",
        precondition="p", postcondition="q",
    )
    assert Transition.from_json(t.to_json()) == t


def test_transition_round_trip_minimal():
    t = Transition(from_state="a", to_state="b", action="tap")
    assert Transition.from_json(t.to_json()) == t


def test_strategy_round_trip_full():
    s = Strategy(
        name="DO_X",
        preconditions="screen Y is visible",
        steps=["s1", "s2"],
        success_signal="toast appears",
        fallback="retry from start",
    )
    assert Strategy.from_json(s.to_json()) == s


def test_strategy_round_trip_minimal_steps_empty():
    s = Strategy(name="X", preconditions="p", steps=[], success_signal="ok")
    assert Strategy.from_json(s.to_json()) == s


def test_abstract_category_round_trip_full():
    c = AbstractCategory(
        name="ADD_ENTRY",
        precondition="home visible",
        abstract_steps=["s1", "s2"],
        failure_modes=["fm1"],
        verification_checklist=["vc1"],
    )
    assert AbstractCategory.from_json(c.to_json()) == c


def test_abstract_category_round_trip_minimal():
    c = AbstractCategory(name="X", precondition="p")
    assert AbstractCategory.from_json(c.to_json()) == c


def test_layer1_round_trip():
    layer1 = _full_fsm().layer1
    assert Layer1.from_json(layer1.to_json()) == layer1


def test_layer2_round_trip():
    layer2 = _full_fsm().layer2
    assert Layer2.from_json(layer2.to_json()) == layer2


def test_layer1_round_trip_empty():
    l1 = Layer1(app="x", category="c")
    assert Layer1.from_json(l1.to_json()) == l1


def test_layer2_round_trip_empty():
    l2 = Layer2()
    assert Layer2.from_json(l2.to_json()) == l2


# ─────────────────────────────────────────────────────────────────────────
# FSM round-trip
# ─────────────────────────────────────────────────────────────────────────


def test_fsm_round_trip_full():
    fsm = _full_fsm()
    rebuilt = FSM.from_json(fsm.to_json())
    assert rebuilt == fsm
    # Belt + suspenders: dict-level equality too.
    assert rebuilt.to_json() == fsm.to_json()


def test_fsm_round_trip_empty():
    fsm = _empty_fsm()
    assert FSM.from_json(fsm.to_json()) == fsm


def test_fsm_json_is_real_json_serializable():
    """to_json output must survive ``json.dumps`` + ``json.loads``."""
    fsm = _full_fsm()
    s = json.dumps(fsm.to_json())
    rebuilt = FSM.from_json(json.loads(s))
    assert rebuilt == fsm


# ─────────────────────────────────────────────────────────────────────────
# Unicode + ordering preservation
# ─────────────────────────────────────────────────────────────────────────


def test_round_trip_preserves_unicode_in_strings():
    fsm = FSM(
        app="测试app",
        layer1=Layer1(
            app="测试app", category="Productivity",
            states=[State(id="home", description="主屏 — emoji 🎯 ok")],
        ),
        layer2=Layer2(),
        metadata={"作者": "linqiang"},
    )
    rebuilt = FSM.from_json(json.loads(json.dumps(fsm.to_json(), ensure_ascii=False)))
    assert rebuilt == fsm
    assert "🎯" in rebuilt.layer1.states[0].description


def test_round_trip_preserves_list_order():
    """We must not silently sort or dedupe state/transition/category lists.

    FSM mutation operators rely on positional indices.
    """
    fsm = FSM(
        app="x",
        layer1=Layer1(
            app="x", category="c",
            states=[State(id=f"s{i}") for i in [3, 1, 4, 1, 5, 9, 2, 6]],
            transitions=[Transition("s3", "s1", "go"), Transition("s1", "s4", "go")],
            strategies=[Strategy(name=f"P{i}", preconditions="", steps=[], success_signal="ok")
                        for i in [9, 1, 5]],
            dead_ends=[{"state": f"s{i}", "failed_action": "x"} for i in [7, 0, 2]],
        ),
        layer2=Layer2(categories=[
            AbstractCategory(name=f"C{i}", precondition="p") for i in [4, 1, 3, 2]
        ]),
    )
    rebuilt = FSM.from_json(fsm.to_json())
    assert [s.id for s in rebuilt.layer1.states] == ["s3", "s1", "s4", "s1", "s5", "s9", "s2", "s6"]
    assert [s.name for s in rebuilt.layer1.strategies] == ["P9", "P1", "P5"]
    assert [d["state"] for d in rebuilt.layer1.dead_ends] == ["s7", "s0", "s2"]
    assert [c.name for c in rebuilt.layer2.categories] == ["C4", "C1", "C3", "C2"]


# ─────────────────────────────────────────────────────────────────────────
# Defaults / optional fields
# ─────────────────────────────────────────────────────────────────────────


def test_state_from_json_defaults_when_optional_missing():
    s = State.from_json({"id": "bare"})
    assert s == State(id="bare", description="", visual_cues=[], resource_hints=[])


def test_transition_from_json_defaults_when_optional_missing():
    t = Transition.from_json({"from_state": "a", "to_state": "b", "action": "x"})
    assert t.precondition == "" and t.postcondition == ""


def test_fsm_from_json_metadata_defaults_to_empty():
    minimal_json = {
        "version": SCHEMA_VERSION,
        "app": "x",
        "layer1": {"app": "x", "category": "c"},
        "layer2": {},
    }
    fsm = FSM.from_json(minimal_json)
    assert fsm.metadata == {}


# ─────────────────────────────────────────────────────────────────────────
# Strict validation: missing required field
# ─────────────────────────────────────────────────────────────────────────


def test_state_missing_id_raises():
    try:
        State.from_json({"description": "no id"})
    except ValueError as e:
        assert "id" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_transition_missing_action_raises():
    try:
        Transition.from_json({"from_state": "a", "to_state": "b"})
    except ValueError as e:
        assert "action" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_fsm_missing_layer1_raises():
    try:
        FSM.from_json({
            "version": SCHEMA_VERSION,
            "app": "x",
            "layer2": {},
        })
    except ValueError as e:
        assert "layer1" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_from_json_rejects_non_dict_input():
    try:
        FSM.from_json([1, 2, 3])  # type: ignore[arg-type]
    except ValueError as e:
        assert "dict" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")


# ─────────────────────────────────────────────────────────────────────────
# Schema version policy
# ─────────────────────────────────────────────────────────────────────────


def test_major_version_mismatch_rejected():
    j = _empty_fsm().to_json()
    j["version"] = "99.0.0"
    try:
        FSM.from_json(j)
    except SchemaVersionError as e:
        assert "99" in str(e)
    else:
        raise AssertionError("expected SchemaVersionError")


def test_minor_version_drift_accepted():
    """Same major version, different minor — should still load."""
    j = _empty_fsm().to_json()
    current_major = SCHEMA_VERSION.split(".")[0]
    j["version"] = f"{current_major}.99.0"
    fsm = FSM.from_json(j)
    assert fsm.version == f"{current_major}.99.0"


def test_garbled_version_rejected():
    j = _empty_fsm().to_json()
    j["version"] = "not-a-version"
    try:
        FSM.from_json(j)
    except SchemaVersionError:
        pass
    else:
        raise AssertionError("expected SchemaVersionError")


def test_unknown_optional_fields_ignored():
    """Forward-compat: a future minor version may add fields. Existing
    code should silently drop them, not crash."""
    j = _empty_fsm().to_json()
    j["future_extra_field"] = {"unrecognized": True}
    j["layer1"]["future_field"] = ["whatever"]
    j["layer2"]["future_field"] = 42
    fsm = FSM.from_json(j)  # should not raise
    assert fsm.app == "dummy_app"


# ─────────────────────────────────────────────────────────────────────────
# Equality semantics
# ─────────────────────────────────────────────────────────────────────────


def test_two_identically_built_fsms_compare_equal():
    assert _full_fsm() == _full_fsm()


def test_changing_one_field_breaks_equality():
    a = _full_fsm()
    b = _full_fsm()
    b.metadata["sr"] = 0.99
    assert a != b


def test_changing_state_order_breaks_equality():
    a = _empty_fsm()
    b = _empty_fsm()
    a.layer1.states = [State(id="s1"), State(id="s2")]
    b.layer1.states = [State(id="s2"), State(id="s1")]
    assert a != b


# ─────────────────────────────────────────────────────────────────────────
# to_prompt_text — format invariants
# ─────────────────────────────────────────────────────────────────────────


def test_prompt_text_contains_layer_headers():
    text = _full_fsm().to_prompt_text()
    assert "LAYER 1: APP_SPECIFIC" in text
    assert "LAYER 2: GENERIC" in text
    assert "═══" in text  # the separator line


def test_prompt_text_lists_app_and_category():
    text = _full_fsm().to_prompt_text()
    assert "APP: markor" in text
    assert "CATEGORY: Productivity" in text


def test_prompt_text_renders_state_with_visual_cues():
    text = _full_fsm().to_prompt_text()
    assert "S0: HOME" in text
    assert "floating + button bottom-right" in text
    assert "net.gsantner.markor:id/fab_main" in text


def test_prompt_text_renders_transitions_with_arrow_format():
    text = _full_fsm().to_prompt_text()
    assert "S0 --click(fab_main)--> S1" in text


def test_prompt_text_renders_layer2_categories():
    text = _full_fsm().to_prompt_text()
    assert "CATEGORY: ADD_ENTRY" in text
    assert "abstract_steps:" in text
    assert "Locate primary add affordance" in text


def test_prompt_text_handles_empty_lists_gracefully():
    text = _empty_fsm().to_prompt_text()
    # Both layer headers still printed
    assert "LAYER 1: APP_SPECIFIC" in text
    assert "LAYER 2: GENERIC" in text
    # Empty sections show the "(none)" placeholder, not raise
    assert "STATES:\n  (none)" in text
    assert "TRANSITIONS:\n  (none)" in text
    assert "STRATEGIES:\n  (none)" in text
    assert "DEAD_ENDS:\n  (none)" in text
    assert "(no abstract categories yet)" in text


def test_prompt_text_is_byte_stable():
    """Same FSM in → same string out. Required for KV-cache reuse across
    episodes that share the same FSM."""
    fsm = _full_fsm()
    a = fsm.to_prompt_text()
    b = fsm.to_prompt_text()
    assert a == b


def test_layer2_to_prompt_text_matches_embedded_fsm_render():
    """Layer2.to_prompt_text() with default category='' must produce
    exactly the LAYER-2 block that FSM.to_prompt_text() emits, so the
    refactor introduces no byte drift."""
    fsm = _full_fsm()
    full_text = fsm.to_prompt_text()
    layer2_start = full_text.index(
        "# ═══════════════════════════════════════════════\n"
        "# LAYER 2: GENERIC"
    )
    layer2_block_inside_fsm = full_text[layer2_start:]
    standalone = fsm.layer2.to_prompt_text()
    assert layer2_block_inside_fsm == standalone


def test_layer2_to_prompt_text_injects_category_tag_when_given():
    fsm = _full_fsm()
    without_tag = fsm.layer2.to_prompt_text()
    with_tag = fsm.layer2.to_prompt_text(category="Productivity")
    assert "L_C CATEGORY: Productivity" in with_tag
    assert "L_C CATEGORY:" not in without_tag
    assert with_tag.startswith(
        "# ═══════════════════════════════════════════════\n"
        "# LAYER 2: GENERIC  (transferable, app-agnostic)\n"
        "# ═══════════════════════════════════════════════\n"
        "L_C CATEGORY: Productivity\n"
    )


def test_layer2_to_prompt_text_byte_stable():
    fsm = _full_fsm()
    a = fsm.layer2.to_prompt_text(category="Productivity")
    b = fsm.layer2.to_prompt_text(category="Productivity")
    assert a == b


def test_layer2_to_prompt_text_empty_categories():
    empty = Layer2(categories=[])
    text = empty.to_prompt_text()
    assert "LAYER 2: GENERIC" in text
    assert "(no abstract categories yet)" in text
    tagged = empty.to_prompt_text(category="Finance")
    assert "L_C CATEGORY: Finance" in tagged
    assert "(no abstract categories yet)" in tagged


def test_prompt_text_does_not_mention_app_in_layer2():
    """Layer 2 strings should be app-agnostic; not a hard guarantee in
    schema (linter is Story 2.2.3) but the renderer at least shouldn't
    inject app names into Layer 2 lines."""
    text = _full_fsm().to_prompt_text()
    layer2_start = text.index("LAYER 2: GENERIC")
    layer2_block = text[layer2_start:]
    # Pre-condition for the test: the fixture's Layer 2 was authored
    # without 'markor'. If that ever stops being true, this test is not
    # the right place to enforce it — Story 2.2.3 linter is.
    assert "markor" not in layer2_block.lower()


# ─────────────────────────────────────────────────────────────────────────
# Standalone runner — keeps `python tests/test_fsm_schema.py` working
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
