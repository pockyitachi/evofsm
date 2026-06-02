"""Tests for evofsm_rl.fsm.aggregator + lint_L_C — Story 2.3.

Coverage:

  aggregate_L_C
    * empty list → empty Layer2 (degenerate)
    * single FSM → deep-copy passthrough of its layer2 (no API call)
    * two FSMs → calls Claude, parses merged response, returns Layer2
    * disagreeing category across inputs → ValueError (we never merge
      layer2 from mismatched categories)
    * empty categories in merged response → ValueError

  _build_merge_prompt
    * byte-stable for a given input ordering
    * includes the category name, SOURCE-indexed blocks, and the output
      schema reminder

  lint_L_C
    * passes for a clean merged Layer2
    * flags leak of app-name-A when layer2 mentions app B's name
    * flags leak of any source app's resource hint
    * flags package-name pattern
    * flags app-specific state id from any source
    * de-duplicates identical violations across apps

Run:
    python -m pytest tests/test_fsm_aggregator.py -v
or  python tests/test_fsm_aggregator.py   (no pytest required)
"""

from __future__ import annotations

import json

from evofsm_rl.fsm import (
    AbstractCategory,
    FSM,
    Layer1,
    Layer2,
    State,
    aggregate_L_C,
    lint_L_C,
    load_L_C,
)
from evofsm_rl.fsm.aggregator import _build_merge_prompt


# ─────────────────────────────────────────────────────────────────────────
# Fixture builders
# ─────────────────────────────────────────────────────────────────────────


def _fsm(app: str, category: str, layer2_categories: list[AbstractCategory],
         states: list[State] | None = None) -> FSM:
    return FSM(
        app=app,
        layer1=Layer1(
            app=app,
            category=category,
            states=states or [State(id="home"), State(id="main")],
        ),
        layer2=Layer2(categories=layer2_categories),
    )


def _cat(name: str = "ADD_ENTRY",
         precondition: str = "list-like surface visible",
         abstract_steps: list[str] | None = None,
         failure_modes: list[str] | None = None,
         verification_checklist: list[str] | None = None) -> AbstractCategory:
    return AbstractCategory(
        name=name,
        precondition=precondition,
        abstract_steps=abstract_steps or ["Invoke primary create affordance"],
        failure_modes=failure_modes or ["Confirming before required fields are filled"],
        verification_checklist=verification_checklist or ["new entry visible in list"],
    )


# ─────────────────────────────────────────────────────────────────────────
# aggregate_L_C — passthrough paths
# ─────────────────────────────────────────────────────────────────────────


def test_aggregate_empty_list_returns_empty_layer2():
    result = aggregate_L_C([])
    assert isinstance(result, Layer2)
    assert result.categories == []


def test_aggregate_single_fsm_returns_deep_copy_of_layer2():
    src = _fsm("bluecoins", "Finance",
               layer2_categories=[_cat(name="ADD_TRANSACTION")])
    merged = aggregate_L_C([src])

    # Equal content
    assert merged.to_json() == src.layer2.to_json()

    # Independent identity — mutating the merged result must not
    # mutate the source FSM.
    merged.categories[0].abstract_steps.append("extra step")
    assert "extra step" not in src.layer2.categories[0].abstract_steps


# ─────────────────────────────────────────────────────────────────────────
# aggregate_L_C — multi-app merge path (monkeypatched LLM)
# ─────────────────────────────────────────────────────────────────────────


def _mock_anthropic_returning(payload: dict):
    """Return a function that mimics builder._call_anthropic's signature."""
    def fake(prompt, *, model, max_tokens, temperature):  # noqa: ARG001
        return json.dumps(payload)
    return fake


def test_aggregate_multi_app_calls_claude_and_parses(monkeypatch):
    # Two apps, same category, overlapping and distinct categories.
    a = _fsm("markor", "Productivity", layer2_categories=[
        _cat(name="ADD_ENTRY", abstract_steps=["Tap create affordance"]),
        _cat(name="DELETE_ENTRY", abstract_steps=["Select then tap delete"]),
    ])
    b = _fsm("joplin", "Productivity", layer2_categories=[
        _cat(name="CREATE_ENTRY", abstract_steps=["Invoke primary create control"]),
        _cat(name="QUERY_INFO", abstract_steps=["Locate item", "Read attribute"]),
    ])

    merged_payload = {
        "categories": [
            {
                "name": "ADD_ENTRY",  # merged from ADD_ENTRY+CREATE_ENTRY
                "precondition": "list-like surface visible",
                "abstract_steps": [
                    "Invoke the primary create affordance",
                    "Enter required fields",
                    "Confirm via the explicit save control",
                ],
                "failure_modes": ["Confirming before required fields are filled"],
                "verification_checklist": ["new entry visible in list"],
            },
            {
                "name": "DELETE_ENTRY",
                "precondition": "entry selected",
                "abstract_steps": ["Select then tap delete"],
                "failure_modes": [],
                "verification_checklist": ["entry no longer visible"],
            },
            {
                "name": "QUERY_INFO",
                "precondition": "list-like surface visible",
                "abstract_steps": ["Locate target", "Read attribute"],
                "failure_modes": [],
                "verification_checklist": ["answer matches visible value"],
            },
        ],
    }
    monkeypatch.setattr(
        "evofsm_rl.fsm.builder._call_anthropic",
        _mock_anthropic_returning(merged_payload),
    )

    merged = aggregate_L_C([a, b])
    names = [c.name for c in merged.categories]
    assert names == ["ADD_ENTRY", "DELETE_ENTRY", "QUERY_INFO"]
    # Each merged category is a real AbstractCategory instance
    assert all(isinstance(c, AbstractCategory) for c in merged.categories)


def test_aggregate_accepts_response_wrapped_in_layer2_key(monkeypatch):
    a = _fsm("markor", "Productivity", layer2_categories=[_cat()])
    b = _fsm("joplin", "Productivity", layer2_categories=[_cat()])
    wrapped = {"layer2": {"categories": [
        {"name": "FOO", "precondition": "pre",
         "abstract_steps": [], "failure_modes": [], "verification_checklist": []},
    ]}}
    monkeypatch.setattr(
        "evofsm_rl.fsm.builder._call_anthropic",
        _mock_anthropic_returning(wrapped),
    )
    merged = aggregate_L_C([a, b])
    assert [c.name for c in merged.categories] == ["FOO"]


def test_aggregate_rejects_disagreeing_category():
    a = _fsm("markor", "Productivity", layer2_categories=[_cat()])
    b = _fsm("calculator", "Tools", layer2_categories=[_cat()])
    try:
        aggregate_L_C([a, b])
    except ValueError as e:
        assert "category" in str(e).lower()
    else:
        raise AssertionError("expected ValueError on mismatched categories")


def test_aggregate_rejects_empty_merged_response(monkeypatch):
    a = _fsm("markor", "Productivity", layer2_categories=[_cat()])
    b = _fsm("joplin", "Productivity", layer2_categories=[_cat()])
    monkeypatch.setattr(
        "evofsm_rl.fsm.builder._call_anthropic",
        _mock_anthropic_returning({"categories": []}),
    )
    try:
        aggregate_L_C([a, b])
    except ValueError as e:
        assert "zero categories" in str(e).lower() or "empty" in str(e).lower()
    else:
        raise AssertionError("expected ValueError on empty merged response")


def test_aggregate_rejects_response_missing_categories(monkeypatch):
    a = _fsm("markor", "Productivity", layer2_categories=[_cat()])
    b = _fsm("joplin", "Productivity", layer2_categories=[_cat()])
    monkeypatch.setattr(
        "evofsm_rl.fsm.builder._call_anthropic",
        _mock_anthropic_returning({"not_categories": []}),
    )
    try:
        aggregate_L_C([a, b])
    except ValueError as e:
        assert "categories" in str(e).lower()
    else:
        raise AssertionError("expected ValueError on malformed merged response")


# ─────────────────────────────────────────────────────────────────────────
# _build_merge_prompt
# ─────────────────────────────────────────────────────────────────────────


def test_prompt_includes_category_and_source_indices():
    a = _fsm("markor", "Productivity", layer2_categories=[_cat(name="ADD_ENTRY")])
    b = _fsm("joplin", "Productivity", layer2_categories=[_cat(name="QUERY_INFO")])
    p = _build_merge_prompt([a, b], "Productivity")
    assert "Productivity" in p
    assert "SOURCE 1" in p and "SOURCE 2" in p
    # Does NOT leak concrete app names into the prompt frame
    # (they may still appear inside the JSON blocks — that's fine, the
    # blocks ARE the data the model is merging).
    # The instruction frame should not mention them.
    frame_head = p.split("=== SOURCE 1")[0]
    assert "markor" not in frame_head.lower()
    assert "joplin" not in frame_head.lower()


def test_prompt_is_byte_stable_for_same_input():
    a = _fsm("markor", "Productivity", layer2_categories=[_cat(name="ADD_ENTRY")])
    b = _fsm("joplin", "Productivity", layer2_categories=[_cat(name="QUERY_INFO")])
    p1 = _build_merge_prompt([a, b], "Productivity")
    p2 = _build_merge_prompt([a, b], "Productivity")
    assert p1 == p2


# ─────────────────────────────────────────────────────────────────────────
# lint_L_C
# ─────────────────────────────────────────────────────────────────────────


def _source_fsm_with_rids(app: str, rids: list[str]) -> FSM:
    return FSM(
        app=app,
        layer1=Layer1(app=app, category="Productivity",
                      states=[State(id="main", resource_hints=rids)]),
        layer2=Layer2(categories=[]),  # unused by lint_L_C
    )


def test_lint_L_C_clean_passes():
    merged = Layer2(categories=[_cat()])
    sources = [
        _source_fsm_with_rids("markor", ["net.gsantner.markor:id/document_edit_text"]),
        _source_fsm_with_rids("joplin", ["net.cozic.joplin:id/notes_list"]),
    ]
    passed, errors = lint_L_C(merged, sources)
    assert passed, f"expected clean, got: {errors}"


def test_lint_L_C_catches_any_source_app_name():
    # Leak of app B's name in the merged Layer2
    merged = Layer2(categories=[_cat(
        abstract_steps=["Open the Joplin app and tap create"],
    )])
    sources = [
        _source_fsm_with_rids("markor", []),
        _source_fsm_with_rids("joplin", []),
    ]
    passed, errors = lint_L_C(merged, sources)
    assert not passed
    assert any("app name" in e and "joplin" in e.lower() for e in errors)


def test_lint_L_C_catches_any_source_resource_hint():
    merged = Layer2(categories=[_cat(
        abstract_steps=["Tap com.rammigsoftware.bluecoins:id/fab_main to start"],
    )])
    sources = [
        _source_fsm_with_rids("bluecoins",
                              ["com.rammigsoftware.bluecoins:id/fab_main"]),
        _source_fsm_with_rids("markor", []),
    ]
    passed, errors = lint_L_C(merged, sources)
    assert not passed
    assert any("resource hint" in e and "fab_main" in e for e in errors)


def test_lint_L_C_catches_package_name():
    merged = Layer2(categories=[_cat(
        abstract_steps=["reference net.gsantner.markor for context"],
    )])
    sources = [_source_fsm_with_rids("markor", [])]
    passed, errors = lint_L_C(merged, sources)
    assert not passed
    assert any("package name" in e and "net.gsantner.markor" in e for e in errors)


def test_lint_L_C_catches_app_specific_state_id_from_any_source():
    # app-specific state id only triggers when the state id contains the app name.
    markor = FSM(
        app="markor",
        layer1=Layer1(app="markor", category="Productivity",
                      states=[State(id="markor_editor"),
                              State(id="home")]),
        layer2=Layer2(categories=[]),
    )
    joplin = FSM(
        app="joplin",
        layer1=Layer1(app="joplin", category="Productivity",
                      states=[State(id="note_view")]),
        layer2=Layer2(categories=[]),
    )
    merged = Layer2(categories=[_cat(
        abstract_steps=["Transition to the markor_editor state"],
    )])
    passed, errors = lint_L_C(merged, [markor, joplin])
    assert not passed
    assert any("state id" in e and "markor_editor" in e for e in errors)


def test_lint_L_C_dedups_identical_violations_across_sources():
    # Both source apps have the same hint (contrived); merged layer2 uses it.
    merged = Layer2(categories=[_cat(
        abstract_steps=["Tap the Specific Label Text to continue"],
    )])
    sources = [
        _source_fsm_with_rids("markor", ["Specific Label Text"]),
        _source_fsm_with_rids("joplin", ["Specific Label Text"]),
    ]
    passed, errors = lint_L_C(merged, sources)
    assert not passed
    hint_errs = [e for e in errors if "resource hint" in e
                 and "specific label text" in e.lower()]
    # Must be reported at most once despite being present in both sources
    assert len(hint_errs) == 1, f"expected dedup, got: {hint_errs}"


# ─────────────────────────────────────────────────────────────────────────
# load_L_C — reads the wrapped-Layer2 JSON written by build_L_C.py
# ─────────────────────────────────────────────────────────────────────────


def test_load_L_C_round_trip(tmp_path):
    """Round-trip: Layer2 → write wrapper → load_L_C → same Layer2."""
    import json as _json

    layer2 = Layer2(categories=[_cat(name="ADD_ENTRY")])
    path = tmp_path / "productivity.json"
    path.write_text(_json.dumps(
        {"category": "Productivity", "layer2": layer2.to_json()},
    ))

    category, loaded = load_L_C(path)
    assert category == "Productivity"
    assert loaded.to_json() == layer2.to_json()


def test_load_L_C_rejects_missing_category_key(tmp_path):
    import json as _json

    path = tmp_path / "broken.json"
    path.write_text(_json.dumps({"layer2": {"categories": []}}))
    try:
        load_L_C(path)
    except ValueError as e:
        assert "category" in str(e)
    else:
        raise AssertionError("expected ValueError on missing 'category' key")


def test_load_L_C_rejects_missing_layer2_key(tmp_path):
    import json as _json

    path = tmp_path / "broken.json"
    path.write_text(_json.dumps({"category": "Foo"}))
    try:
        load_L_C(path)
    except ValueError as e:
        assert "layer2" in str(e)
    else:
        raise AssertionError("expected ValueError on missing 'layer2' key")


def test_load_L_C_accepts_string_path(tmp_path):
    """load_L_C must accept str as well as Path."""
    import json as _json

    layer2 = Layer2(categories=[_cat(name="QUERY_INFO")])
    path = tmp_path / "tools.json"
    path.write_text(_json.dumps({"category": "Tools", "layer2": layer2.to_json()}))

    category, loaded = load_L_C(str(path))
    assert category == "Tools"
    assert [c.name for c in loaded.categories] == ["QUERY_INFO"]


# ─────────────────────────────────────────────────────────────────────────
# Standalone runner (pytest-optional)
# ─────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import traceback

    class _MP:
        """Minimal monkeypatch stand-in for pytest-less runs."""
        def __init__(self):
            self._undo = []

        def setattr(self, dotted: str, value):
            module_path, _, attr = dotted.rpartition(".")
            import importlib
            mod = importlib.import_module(module_path)
            old = getattr(mod, attr, None)
            setattr(mod, attr, value)
            self._undo.append((mod, attr, old))

        def undo(self):
            while self._undo:
                mod, attr, old = self._undo.pop()
                setattr(mod, attr, old)

    import inspect
    import tempfile
    from pathlib import Path as _Path
    ns = dict(globals())
    tests = [(name, fn) for name, fn in ns.items()
             if name.startswith("test_") and callable(fn)]
    passed = failed = 0
    for name, fn in tests:
        mp = _MP()
        td = tempfile.TemporaryDirectory()
        try:
            sig = inspect.signature(fn)
            kwargs = {}
            if "monkeypatch" in sig.parameters:
                kwargs["monkeypatch"] = mp
            if "tmp_path" in sig.parameters:
                kwargs["tmp_path"] = _Path(td.name)
            fn(**kwargs)
            passed += 1
            print(f"  ok  {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
        finally:
            mp.undo()
            td.cleanup()
    print(f"\n{passed}/{passed + failed} tests passed")
    raise SystemExit(0 if failed == 0 else 1)
