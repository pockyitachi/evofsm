"""Tests for evofsm_rl.fsm.linter — Story 2.2.3.

Coverage matrix per the four rules:

  Rule 1 — app name (with case + variant handling)
    * literal app name in any Layer 2 text → fail
    * differently-cased app name → fail (Bluecoins / BLUECOINS / bluecoins)
    * snake_case → space variant ("simple_calendar_pro" → "simple calendar pro")
    * known compound variant ("bluecoins" → "blue coins")
    * "_pro"/"_lite" stripped variant ("simple_calendar_pro" → "simple calendar")
    * word-bounded: app "files" must NOT match "filesystem"

  Rule 2 — resource hints
    * exact resource_hint substring in Layer 2 → fail
    * differently-cased resource_hint → fail

  Rule 3 — package-name regex
    * com.foo.bar pattern → fail
    * org.wikipedia, net.gsantner.markor, code.name.monkey.retromusic → fail
    * version strings like "1.2.3" → must NOT trigger
    * naked "com.example" (only 2 segments) → must NOT trigger
    * URLs containing "github.com" but not at the start of a segment → not matched

  Rule 4 — app-specific state ids
    * state id containing app name → fail when referenced in Layer 2
    * generic state id "home" → must NOT trigger even if word "home" is in Layer 2

  General
    * a clean FSM passes
    * empty layer2 (no categories) passes trivially
    * each error line follows the prescribed format and identifies the
      category, field path (e.g. ``abstract_steps[2]``), rule kind, and
      offending substring
    * multiple violations across categories are all reported
    * error ordering is deterministic (depends only on layer2 + sorted variants)

Run:
    python -m pytest tests/test_fsm_linter.py -v
or  python tests/test_fsm_linter.py   (no pytest required)
"""

from __future__ import annotations

from evofsm_rl.fsm import (
    AbstractCategory,
    FSM,
    Layer1,
    Layer2,
    State,
    lint_layer2,
)
from evofsm_rl.fsm.linter import _app_name_variants


# ─────────────────────────────────────────────────────────────────────────
# Fixture builders — small, composable
# ─────────────────────────────────────────────────────────────────────────


def _fsm(app: str, layer2_categories: list[AbstractCategory] | None = None,
         states: list[State] | None = None) -> FSM:
    """Minimal FSM with caller-controlled Layer 2 + Layer 1 states."""
    return FSM(
        app=app,
        layer1=Layer1(
            app=app,
            category="Productivity",
            states=states or [State(id="home"), State(id="editor")],
        ),
        layer2=Layer2(categories=layer2_categories or []),
    )


def _category(
    name: str = "ADD_ENTRY",
    precondition: str = "a home-like screen is visible",
    abstract_steps: list[str] | None = None,
    failure_modes: list[str] | None = None,
    verification_checklist: list[str] | None = None,
) -> AbstractCategory:
    """One AbstractCategory with sensible generic defaults that lint clean."""
    return AbstractCategory(
        name=name,
        precondition=precondition,
        abstract_steps=abstract_steps or [
            "Locate the primary add affordance",
            "Enter required fields top to bottom",
            "Confirm via the explicit save control",
        ],
        failure_modes=failure_modes or [
            "tapping save before all required fields are filled",
        ],
        verification_checklist=verification_checklist or [
            "the new entry appears in the main list",
        ],
    )


# ─────────────────────────────────────────────────────────────────────────
# Sanity: clean FSMs lint clean
# ─────────────────────────────────────────────────────────────────────────


def test_clean_fsm_passes():
    fsm = _fsm("bluecoins", layer2_categories=[_category()])
    passed, errors = lint_layer2(fsm)
    assert passed, f"expected clean lint, got: {errors}"
    assert errors == []


def test_empty_layer2_passes_trivially():
    fsm = _fsm("markor")
    passed, errors = lint_layer2(fsm)
    assert passed and errors == []


# ─────────────────────────────────────────────────────────────────────────
# Rule 1 — app name
# ─────────────────────────────────────────────────────────────────────────


def test_app_name_in_layer2_fails():
    fsm = _fsm("bluecoins", layer2_categories=[_category(
        abstract_steps=[
            "Locate the primary add affordance",
            "Enter the amount",
            "tap the Bluecoins add button",  # ← violates here
        ],
    )])
    passed, errors = lint_layer2(fsm)
    assert not passed
    assert len(errors) == 1
    err = errors[0]
    assert err.startswith("FAIL:")
    assert 'category="ADD_ENTRY"' in err
    assert 'field="abstract_steps[2]"' in err
    assert 'app name "bluecoins"' in err
    assert "Bluecoins add button" in err


def test_app_name_case_insensitive():
    """'Bluecoins' / 'BLUECOINS' / 'bluecoins' all caught."""
    for spelling in ["bluecoins", "Bluecoins", "BLUECOINS", "BlueCoins"]:
        fsm = _fsm("bluecoins", layer2_categories=[_category(
            precondition=f"the {spelling} home screen is visible",
        )])
        passed, errors = lint_layer2(fsm)
        assert not passed, f"failed to catch {spelling!r}"
        assert 'app name' in errors[0]


def test_app_name_compound_variant_blue_coins():
    """'blue coins' (with space) must be caught for app 'bluecoins'."""
    fsm = _fsm("bluecoins", layer2_categories=[_category(
        abstract_steps=["Open the Blue Coins app first"],
    )])
    passed, errors = lint_layer2(fsm)
    assert not passed
    assert 'blue coins' in errors[0].lower()


def test_app_name_snake_to_space_variant():
    """'simple calendar pro' caught for app 'simple_calendar_pro'."""
    fsm = _fsm("simple_calendar_pro", layer2_categories=[_category(
        precondition="the Simple Calendar Pro home view is shown",
    )])
    passed, errors = lint_layer2(fsm)
    assert not passed


def test_app_name_pro_stripped_variant():
    """'simple calendar' (no _pro) caught for app 'simple_calendar_pro'."""
    fsm = _fsm("simple_calendar_pro", layer2_categories=[_category(
        abstract_steps=["Open Simple Calendar from the launcher"],  # 'pro' dropped
    )])
    passed, errors = lint_layer2(fsm)
    assert not passed


def test_app_name_word_bounded_no_substring_false_positive():
    """app 'files' must NOT match the word 'filesystem'."""
    fsm = _fsm("files", layer2_categories=[_category(
        abstract_steps=["Use the filesystem to locate the target"],
    )])
    passed, errors = lint_layer2(fsm)
    # 'files' does not appear as a standalone word — no app-name error
    app_name_errs = [e for e in errors if 'app name' in e]
    assert app_name_errs == [], f"false positive on substring: {app_name_errs}"


def test_app_name_word_bounded_real_match_still_caught():
    """But standalone 'files' IS caught."""
    fsm = _fsm("files", layer2_categories=[_category(
        abstract_steps=["Open the Files app to begin"],
    )])
    passed, errors = lint_layer2(fsm)
    assert not passed
    assert any('files' in e.lower() and 'app name' in e for e in errors)


def test_app_name_one_violation_per_field_max():
    """If a field contains the same app name multiple times, we report
    once — keeps error volume sane for multi-occurrence prose."""
    fsm = _fsm("bluecoins", layer2_categories=[_category(
        abstract_steps=["Open Bluecoins; in Bluecoins, tap the Bluecoins button"],
    )])
    _, errors = lint_layer2(fsm)
    app_name_errs = [e for e in errors if 'app name' in e]
    assert len(app_name_errs) == 1


# ─────────────────────────────────────────────────────────────────────────
# Rule 2 — resource hints
# ─────────────────────────────────────────────────────────────────────────


def test_resource_hint_in_layer2_fails():
    fsm = _fsm(
        "bluecoins",
        states=[
            State(id="home",
                  resource_hints=["com.rammigsoftware.bluecoins:id/fab_main"]),
        ],
        layer2_categories=[_category(
            abstract_steps=[
                "Locate the add button",
                "Tap com.rammigsoftware.bluecoins:id/fab_main to open the form",
            ],
        )],
    )
    passed, errors = lint_layer2(fsm)
    assert not passed
    # Will trip both Rule 2 (resource hint) AND Rule 3 (package pattern)
    # AND Rule 1 (app name "bluecoins" inside the resource id substring) —
    # at least the resource_hint error must be present.
    assert any("resource hint" in e for e in errors)
    assert any('field="abstract_steps[1]"' in e for e in errors)


def test_resource_hint_pure_alpha_dropped_no_false_positive_on_english():
    """A hint like 'filter' / 'reset' / 'folder' must NOT trip on prose.

    These are pure-alphabetic hints that often leak into layer1.resource_hints
    as UI-label text. Dropping them is safe because real app-specific
    resource ids always contain at least one non-alpha character.
    """
    fsm = _fsm(
        "markor",
        states=[
            State(id="home", resource_hints=["filter", "reset", "folder", "back"]),
        ],
        layer2_categories=[_category(
            abstract_steps=[
                "Apply the filter to the list",
                "Reset any partial input before retrying",
                "Traverse the folder hierarchy and return via back navigation",
            ],
        )],
    )
    passed, errors = lint_layer2(fsm)
    hint_errs = [e for e in errors if "resource hint" in e]
    assert hint_errs == [], (
        f"pure-alpha English hints should not produce violations: {hint_errs}"
    )


def test_resource_hint_distinctive_still_caught():
    """A hint with any non-alpha char (digit, punctuation, space) is kept."""
    fsm = _fsm(
        "audio_recorder",
        states=[
            State(id="rec", resource_hints=["Record-1.m4a", "By date"]),
        ],
        layer2_categories=[_category(
            abstract_steps=[
                "Name the entry Record-1.m4a",  # distinctive hint → must fail
            ],
        )],
    )
    _, errors = lint_layer2(fsm)
    assert any("resource hint" in e and "record-1.m4a" in e.lower() for e in errors)


def test_resource_hint_case_insensitive():
    fsm = _fsm(
        "markor",
        states=[
            State(id="editor",
                  resource_hints=["net.gsantner.markor:id/document_edit_text"]),
        ],
        layer2_categories=[_category(
            abstract_steps=[
                "use Net.Gsantner.Markor:Id/Document_edit_text to edit",  # different case
            ],
        )],
    )
    _, errors = lint_layer2(fsm)
    assert any("resource hint" in e for e in errors)


# ─────────────────────────────────────────────────────────────────────────
# Rule 3 — package-name pattern
# ─────────────────────────────────────────────────────────────────────────


def test_package_name_caught():
    fsm = _fsm("calculator", layer2_categories=[_category(
        abstract_steps=["use com.google.android.calculator to launch"],
    )])
    _, errors = lint_layer2(fsm)
    assert any("package name" in e and "com.google.android.calculator" in e for e in errors)


def test_package_name_various_tlds_caught():
    """org.* / net.* / code.* / de.* prefixes all detected."""
    cases = [
        "org.wikipedia",
        "net.gsantner.markor",
        "code.name.monkey.retromusic",
        "de.dennisguse.opentracks",
    ]
    for pkg in cases:
        fsm = _fsm("anything", layer2_categories=[_category(
            abstract_steps=[f"reference {pkg} for context"],
        )])
        _, errors = lint_layer2(fsm)
        assert any("package name" in e and pkg in e for e in errors), \
            f"missed package {pkg!r}: {errors}"


def test_package_name_two_segments_not_flagged():
    """'com.example' alone (only 2 segments) shouldn't match the pattern.

    Real Android packages always have ≥3 segments. 2-segment URLs like
    'github.com' or 'duckduckgo.com' shouldn't false-positive."""
    fsm = _fsm("anything", layer2_categories=[_category(
        abstract_steps=["the project lives at github.com or duckduckgo.com"],
    )])
    _, errors = lint_layer2(fsm)
    pkg_errs = [e for e in errors if "package name" in e]
    assert pkg_errs == [], f"false positive on 2-segment: {pkg_errs}"


def test_version_string_not_flagged_as_package():
    fsm = _fsm("anything", layer2_categories=[_category(
        abstract_steps=["app version 1.2.3 is the current build"],
    )])
    _, errors = lint_layer2(fsm)
    pkg_errs = [e for e in errors if "package name" in e]
    assert pkg_errs == []


# ─────────────────────────────────────────────────────────────────────────
# Rule 4 — app-specific state ids
# ─────────────────────────────────────────────────────────────────────────


def test_app_specific_state_id_caught():
    fsm = _fsm(
        "markor",
        states=[
            State(id="home"),
            State(id="markor_editor"),  # ← contains app name
        ],
        layer2_categories=[_category(
            abstract_steps=["transition to the markor_editor screen"],
        )],
    )
    _, errors = lint_layer2(fsm)
    assert any("app-specific state id" in e and "markor_editor" in e for e in errors)


def test_generic_state_id_not_flagged():
    """Even when 'home' or 'settings' appear in Layer 2, they're fine —
    Rule 4 only flags state ids that themselves contain the app name."""
    fsm = _fsm(
        "calculator",
        states=[State(id="home"), State(id="settings"), State(id="editor")],
        layer2_categories=[_category(
            abstract_steps=[
                "from the home screen",
                "navigate to settings",
                "back to the editor",
            ],
        )],
    )
    _, errors = lint_layer2(fsm)
    state_id_errs = [e for e in errors if "state id" in e]
    assert state_id_errs == [], f"false positive on generic state id: {state_id_errs}"


# ─────────────────────────────────────────────────────────────────────────
# Multi-violation reporting
# ─────────────────────────────────────────────────────────────────────────


def test_multiple_violations_all_reported():
    """A field with both an app name AND a package name must produce both errors."""
    fsm = _fsm("bluecoins", layer2_categories=[_category(
        abstract_steps=[
            "open the Bluecoins app via com.rammigsoftware.bluecoins"
        ],
    )])
    _, errors = lint_layer2(fsm)
    has_app = any("app name" in e for e in errors)
    has_pkg = any("package name" in e for e in errors)
    assert has_app and has_pkg, f"missing one of the rules: {errors}"


def test_violations_across_categories_all_reported():
    fsm = _fsm("markor", layer2_categories=[
        _category(name="ADD_ENTRY", abstract_steps=["open the Markor app"]),
        _category(name="DELETE_ENTRY", precondition="the Markor list is shown"),
    ])
    _, errors = lint_layer2(fsm)
    cats_with_errors = {e.split('"')[1] for e in errors}
    assert cats_with_errors == {"ADD_ENTRY", "DELETE_ENTRY"}


def test_field_path_includes_index_for_list_fields():
    fsm = _fsm("calculator", layer2_categories=[_category(
        abstract_steps=["clean step", "another clean step", "Use the Calculator"],
    )])
    _, errors = lint_layer2(fsm)
    assert any("abstract_steps[2]" in e for e in errors)


def test_passing_lint_returns_empty_list():
    fsm = _fsm("bluecoins", layer2_categories=[
        _category(),  # uses generic defaults
    ])
    passed, errors = lint_layer2(fsm)
    assert passed
    assert errors == []
    assert isinstance(errors, list)


# ─────────────────────────────────────────────────────────────────────────
# Variant-generation helper (private but worth testing)
# ─────────────────────────────────────────────────────────────────────────


def test_variant_generator_includes_literal():
    assert "markor" in _app_name_variants("markor")


def test_variant_generator_snake_case_to_space():
    variants = _app_name_variants("simple_calendar_pro")
    assert "simple_calendar_pro" in variants
    assert "simple calendar pro" in variants


def test_variant_generator_strips_pro_suffix():
    variants = _app_name_variants("simple_calendar_pro")
    assert "simple calendar" in variants
    assert "simple_calendar" in variants


def test_variant_generator_known_compound():
    variants = _app_name_variants("bluecoins")
    assert "bluecoins" in variants
    assert "blue coins" in variants


def test_variant_generator_handles_simple_app():
    """One-word app like 'markor' gets just itself."""
    variants = _app_name_variants("markor")
    assert variants == ["markor"]


# ─────────────────────────────────────────────────────────────────────────
# Standalone runner
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
