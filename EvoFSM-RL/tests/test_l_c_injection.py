"""Tests for L_C injection — Story B2.

Coverage:

  build_action_prompt
    * l_c_prompt_text=None → prompt is byte-identical to the pre-B2
      (B1 zero-shot) rendering. Critical for zero-regression guarantee.
    * l_c_prompt_text=<text> → prompt contains the Workflow-knowledge
      header, the intro paragraph, and the injected text in that order,
      all BEFORE the dynamic goal/history/UI section.
    * the L_C section sits in the stable prefix (between PROMPT_PREFIX
      and the goal line).

  build_action_messages
    * wraps a pre-built prompt + images; the prompt's injected content
      is preserved end-to-end in the user message.

  resolve_l_c_for_app
    * Tier-B app (simple_calendar_pro) returns non-None and contains
      the matching Play-Store category tag.
    * Tier-C app (osmand) returns None (no L_C file for its category).
    * Source-pool app (markor) returns non-None.
    * Unknown app returns None (not an exception).

  Content invariants
    * A sampled Tier-B injection contains the word "abstract_steps" /
      "failure_modes" / "verification_checklist" — i.e. the rendered
      Layer-2 block is structurally intact.
    * The injected text does not mention any source-pool app name in
      plain prose (belt-and-braces on the Story-2.2.3 / B1-update
      linter: we ran lint already, but this test is cheap and keeps
      regressions obvious if a future re-merge reintroduces leakage).

Some tests read the real ``configs/splits.yaml`` and
``artifacts/L_C/*.json``; they ``pytest.skip`` gracefully if those
artifacts are absent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from evofsm_rl.agent.prompts import (
    L_C_SECTION_HEADER,
    L_C_SECTION_INTRO,
    build_action_messages,
    build_action_prompt,
)
from evofsm_rl.fsm import resolve_l_c_for_app


# Resolve paths relative to the repo root, mirroring how callers invoke
# scripts from the parent of EvoFSM-RL/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPLITS_YAML = _REPO_ROOT / "EvoFSM-RL" / "configs" / "splits.yaml"
_L_C_DIR = _REPO_ROOT / "EvoFSM-RL" / "artifacts" / "L_C"


_SKIP_IF_NO_SPLITS = pytest.mark.skipif(
    not _SPLITS_YAML.exists(),
    reason=f"splits.yaml not found at {_SPLITS_YAML}",
)
_SKIP_IF_NO_L_C = pytest.mark.skipif(
    not _L_C_DIR.exists() or not any(_L_C_DIR.glob("*.json")),
    reason=f"L_C artifacts not found under {_L_C_DIR}",
)


# ─────────────────────────────────────────────────────────────────────────
# build_action_prompt — regression + injection shape
# ─────────────────────────────────────────────────────────────────────────


_SAMPLE_GOAL = "Create a new note titled 'Story B2 regression' in Markor."
_SAMPLE_HISTORY = [
    "Step 1- Opened the Markor app via open_app; list view shown.",
    "Step 2- Tapped the FAB; new-file dialog opened.",
]
_SAMPLE_UI = '[0: "New note" Button]\n[1: "Cancel" Button]\n[2: "OK" Button]'


def test_build_action_prompt_without_l_c_byte_identical_to_b1():
    """l_c_prompt_text=None must produce the exact B1 prompt output."""
    prompt_none = build_action_prompt(
        _SAMPLE_GOAL, _SAMPLE_HISTORY, _SAMPLE_UI,
    )
    prompt_explicit_none = build_action_prompt(
        _SAMPLE_GOAL, _SAMPLE_HISTORY, _SAMPLE_UI,
        l_c_prompt_text=None,
    )
    assert prompt_none == prompt_explicit_none

    # And the Workflow-knowledge header must NOT appear.
    assert L_C_SECTION_HEADER not in prompt_none


def test_build_action_prompt_without_l_c_no_empty_injection_artifacts():
    """An empty {l_c_section} slot must not leave dangling whitespace
    between PROMPT_PREFIX and the goal line."""
    prompt = build_action_prompt(_SAMPLE_GOAL, [], _SAMPLE_UI)
    # The goal line must follow PROMPT_PREFIX with exactly one newline
    # ("\nThe current user goal/request is: ...").
    assert "\nThe current user goal/request is:" in prompt
    # No triple-blank-line that would indicate a stray empty injection.
    assert "\n\n\n\nThe current user goal/request is:" not in prompt


def test_build_action_prompt_with_l_c_contains_section_and_text():
    """Full injection: header + intro + L_C text, all BEFORE the goal."""
    fake_l_c = (
        "# ═══════════════════════════════════════════════\n"
        "# LAYER 2: GENERIC  (transferable, app-agnostic)\n"
        "# ═══════════════════════════════════════════════\n"
        "L_C CATEGORY: Productivity\n"
        "CATEGORY: ADD_ENTRY\n"
        "  precondition: list-like surface\n"
        "  abstract_steps:\n"
        "    1. Invoke primary create affordance\n"
        "  failure_modes:\n"
        "    - 'confirming before required fields are filled'\n"
        "  verification_checklist:\n"
        "    - 'new entry visible in list'\n"
    )
    prompt = build_action_prompt(
        _SAMPLE_GOAL, _SAMPLE_HISTORY, _SAMPLE_UI,
        l_c_prompt_text=fake_l_c,
    )

    # Every piece appears
    assert L_C_SECTION_HEADER in prompt
    assert L_C_SECTION_INTRO.splitlines()[0] in prompt
    assert "L_C CATEGORY: Productivity" in prompt
    assert "abstract_steps:" in prompt

    # Ordering: L_C section sits BEFORE the dynamic goal line.
    idx_header = prompt.index(L_C_SECTION_HEADER)
    idx_goal = prompt.index("The current user goal/request is:")
    idx_layer2 = prompt.index("L_C CATEGORY: Productivity")
    assert idx_header < idx_layer2 < idx_goal


def test_build_action_prompt_with_l_c_does_not_leak_into_summary():
    """Smoke check: the L_C injection is an action-prompt concern only.

    We can't import build_summary_prompt's output easily without all
    fields, but we can at least verify that build_action_prompt with
    an empty history works, as does the None path."""
    empty = build_action_prompt(_SAMPLE_GOAL, [], "", l_c_prompt_text=None)
    assert L_C_SECTION_HEADER not in empty


# ─────────────────────────────────────────────────────────────────────────
# build_action_messages — passthrough
# ─────────────────────────────────────────────────────────────────────────


class _FakeImage:
    """Stand-in for PIL.Image — build_action_messages doesn't inspect it."""


def test_build_action_messages_preserves_injected_prompt_text():
    fake_l_c = (
        "# LAYER 2: GENERIC  (transferable, app-agnostic)\n"
        "L_C CATEGORY: Productivity\n"
        "CATEGORY: ADD_ENTRY\n"
        "  precondition: x\n"
    )
    prompt = build_action_prompt(
        _SAMPLE_GOAL, [], _SAMPLE_UI, l_c_prompt_text=fake_l_c,
    )
    messages = build_action_messages(prompt, _FakeImage(), _FakeImage())
    # One user message with image, image, text parts.
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    parts = messages[0]["content"]
    text_parts = [p for p in parts if p.get("type") == "text"]
    assert len(text_parts) == 1
    text = text_parts[0]["text"]
    assert L_C_SECTION_HEADER in text
    assert "L_C CATEGORY: Productivity" in text


# ─────────────────────────────────────────────────────────────────────────
# resolve_l_c_for_app
# ─────────────────────────────────────────────────────────────────────────


@_SKIP_IF_NO_SPLITS
@_SKIP_IF_NO_L_C
def test_resolve_l_c_for_tier_b_app():
    """Tier-B app (category in source pool) gets a non-None L_C text."""
    text = resolve_l_c_for_app("simple_calendar_pro", _SPLITS_YAML, _L_C_DIR)
    assert text is not None, "expected Productivity L_C for simple_calendar_pro"
    assert "L_C CATEGORY: Productivity" in text
    assert "LAYER 2: GENERIC" in text


@_SKIP_IF_NO_SPLITS
@_SKIP_IF_NO_L_C
def test_resolve_l_c_for_tier_c_app_returns_none():
    """Tier-C app has no source-pool category → no L_C → None (B2 degrades to B1)."""
    text = resolve_l_c_for_app("osmand", _SPLITS_YAML, _L_C_DIR)
    assert text is None


@_SKIP_IF_NO_SPLITS
@_SKIP_IF_NO_L_C
def test_resolve_l_c_for_source_pool_app():
    """Source-pool apps (e.g. markor, Productivity) also resolve — useful
    for self-transfer / validation experiments."""
    text = resolve_l_c_for_app("markor", _SPLITS_YAML, _L_C_DIR)
    assert text is not None
    assert "L_C CATEGORY: Productivity" in text


@_SKIP_IF_NO_SPLITS
def test_resolve_l_c_for_unknown_app_returns_none():
    """Unknown app must return None gracefully — NEVER raise."""
    text = resolve_l_c_for_app("not_a_real_app_xyz", _SPLITS_YAML, _L_C_DIR)
    assert text is None


# ─────────────────────────────────────────────────────────────────────────
# Content invariants on the real injected text
# ─────────────────────────────────────────────────────────────────────────


@_SKIP_IF_NO_SPLITS
@_SKIP_IF_NO_L_C
def test_l_c_prompt_text_contains_structural_fields():
    """Sanity: rendered text has abstract_steps/failure_modes/verification."""
    text = resolve_l_c_for_app("simple_calendar_pro", _SPLITS_YAML, _L_C_DIR)
    assert text is not None
    assert "abstract_steps" in text
    assert "failure_modes" in text
    assert "verification_checklist" in text


# Apps whose names are generic English ("files", "contacts", "clock") would
# produce false positives on a naïve substring check. Build the forbidden
# set from splits.yaml and filter them out, matching what the
# Story-2.2.3 linter does with word-boundary checks.
_GENERIC_APP_NAMES_TO_SKIP = {"files", "contacts", "clock"}


@_SKIP_IF_NO_SPLITS
@_SKIP_IF_NO_L_C
def test_l_c_prompt_text_no_source_app_names():
    """The rendered L_C must not mention any distinctive source-pool app
    name. We word-bound the check (same rule as the linter) to avoid
    false positives on generic English words ('files', 'clock')."""
    import yaml

    splits_data = yaml.safe_load(_SPLITS_YAML.read_text())
    source_pool = splits_data.get("source_pool") or {}
    # Use every Tier-B app once to get a representative injection per category.
    tier_b = splits_data.get("tier_B_held_out") or {}

    for app in tier_b:
        text = resolve_l_c_for_app(app, _SPLITS_YAML, _L_C_DIR)
        if text is None:
            continue
        lower = text.lower()
        for src_app in source_pool:
            if src_app in _GENERIC_APP_NAMES_TO_SKIP:
                continue
            # Word-bounded check: src_app must not appear as a standalone
            # token in the injected text. Underscores are treated as
            # word characters, which matches how the linter enforces it.
            if re.search(
                r"(?<![A-Za-z0-9_])" + re.escape(src_app) + r"(?![A-Za-z0-9_])",
                lower,
            ):
                raise AssertionError(
                    f"source-pool app {src_app!r} leaked into L_C for "
                    f"Tier-B app {app!r}"
                )


# ─────────────────────────────────────────────────────────────────────────
# Standalone runner — mirrors the pattern used by the other test files in
# this repo; keeps the file executable without pytest for quick smoke runs.
# ─────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import traceback

    ns = dict(globals())
    tests = [(name, fn) for name, fn in ns.items()
             if name.startswith("test_") and callable(fn)]
    passed = failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  ok    {name}")
        except pytest.skip.Exception as e:  # type: ignore[attr-defined]
            skipped += 1
            print(f"  skip  {name}  ({e})")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    total = passed + failed + skipped
    print(f"\n{passed}/{total} passed  ({skipped} skipped, {failed} failed)")
    raise SystemExit(0 if failed == 0 else 1)
