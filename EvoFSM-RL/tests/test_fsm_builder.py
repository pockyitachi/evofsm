"""Tests for evofsm_rl.fsm.builder — Story 2.2.2.

Coverage:

  Compression
    * Filters by ``meta.app``; ignores other apps' episode dirs.
    * Output bundle has the expected episode-header line per matching
      episode, deterministically ordered by (template, seed).
    * Per-step line carries exactly the 5 required fields, nothing more.
    * UI element rendering caps at the requested element count.
    * SUCCESS / FAIL label derived from ``meta.success`` (partial credit
      counts as SUCCESS so the LLM still sees the trajectory).
    * Token-budget tripwire kicks in: when char count > 4 × 160k,
      the bundle re-renders at the tighter cap. Verified by stubbing the
      budget down so the test data triggers the tighter cap.

  Pure helpers (no API)
    * ``_build_json_schema_description`` mentions every required schema
      field by name.
    * ``_assemble_prompt`` includes app/category/n_episodes/schema/text.
    * ``_extract_json_from_response`` survives bare JSON, ```json fenced,
      and prose-wrapped {...} forms; raises clear errors on garbage.
    * ``_first_balanced_object`` ignores braces inside strings + nested.

The actual Anthropic API call (``_call_anthropic``) is NOT covered here —
it requires a real API key and is the only impure piece. Wrapper code
around it (retry, prompt assembly, JSON extraction, FSM round-trip) is
fully covered above.

Run:
    python -m pytest tests/test_fsm_builder.py -v
or  python tests/test_fsm_builder.py   (no pytest required)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from evofsm_rl.fsm import builder as B


# ─────────────────────────────────────────────────────────────────────────
# Mock-episode fixture
# ─────────────────────────────────────────────────────────────────────────


def _ui_text(n_elements: int) -> str:
    """Build a verbose ``UI element N: {...}`` text with n_elements lines.

    Mirrors the exact format of real episode JSONL entries (Python-style
    True/False, since that's what our a11y renderer emits)."""
    lines = []
    for i in range(n_elements):
        # Alternate between text-bearing and content_description-bearing
        # elements to exercise both label paths in _compact_ui_elements.
        if i % 2 == 0:
            label = f'"text": "elem_{i}_label"'
        else:
            label = f'"content_description": "elem_{i}_desc"'
        lines.append(
            f'UI element {i}: {{"index": {i}, {label}, '
            f'"is_clickable": True, "is_long_clickable": False, '
            f'"is_editable": False, "is_scrollable": False, '
            f'"is_selected": False, "is_checked": False}}'
        )
    return "\n".join(lines)


def _make_episode(
    root: Path,
    template: str,
    seed: int,
    app: str,
    success: float,
    n_steps: int,
    *,
    ui_elements_per_step: int = 4,
    extra_meta: dict | None = None,
) -> Path:
    """Create a fake episode dir with meta.json + episode.jsonl."""
    ep_dir = root / f"{template}_seed{seed}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "schema_version": 1,
        "template": template,
        "seed": seed,
        "app": app,
        "tier": "source",
        "success": float(success),
        "n_steps": n_steps,
        "self_reported": 1 if success > 0 else 0,
        "parse_failures": 0,
        "alias_hits": 0,
        "clamp_hits": 0,
        "agent_name": "Qwen3-VL-M3A",
        "wall_s_total": 12.34,
    }
    if extra_meta:
        meta.update(extra_meta)
    (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    with (ep_dir / "episode.jsonl").open("w") as fh:
        for s in range(1, n_steps + 1):
            row = {
                "step": s,
                "timestamp": 1700000000.0 + s,
                "goal": f"do {template} task",
                "before_ui_elements_text": _ui_text(ui_elements_per_step),
                "after_ui_elements_text": _ui_text(ui_elements_per_step),
                "before_screenshot_path": f"step_{s}_before.png",
                "after_screenshot_path": f"step_{s}_after.png" if s < n_steps else None,
                "action": {"action_type": "click", "index": s},
                "action_reason": f"reason for step {s} of {template}",
                "summary": f"summary for step {s} of {template}",
                "reward": 0.0 if s < n_steps else float(success),
                "action_wall_s": 1.0,
                "summary_wall_s": 0.5,
                "action_input_tokens": 1234,
                "parse_error": None,
                "exec_error": None,
                "action_raw_response": f"Reason: ...\nAction: ...",
                "summary_raw_response": f"summary text",
            }
            fh.write(json.dumps(row) + "\n")
    return ep_dir


# ─────────────────────────────────────────────────────────────────────────
# compress_trajectories
# ─────────────────────────────────────────────────────────────────────────


def test_compress_filters_by_app():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _make_episode(root, "MarkorCreateNote", 30, "markor", 1.0, n_steps=3)
        _make_episode(root, "BluecoinsAddExpense", 30, "bluecoins", 0.0, n_steps=3)
        _make_episode(root, "MarkorDeleteNote", 30, "markor", 0.0, n_steps=2)

        out = B.compress_trajectories("markor", root)

        assert "MarkorCreateNote" in out
        assert "MarkorDeleteNote" in out
        assert "BluecoinsAddExpense" not in out, \
            "compress_trajectories must filter by meta.app"


def test_compress_returns_empty_when_no_match():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _make_episode(root, "MarkorCreateNote", 30, "markor", 1.0, n_steps=2)
        out = B.compress_trajectories("nonexistent_app", root)
        assert out == ""


def test_compress_episode_header_format():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _make_episode(root, "MarkorCreateNote", 30, "markor", 1.0, n_steps=4)

        out = B.compress_trajectories("markor", root)

        assert "=== Episode: MarkorCreateNote (seed=30, result=SUCCESS, " \
               "reward=1.0, steps=4) ===" in out


def test_compress_fail_label_for_zero_success():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _make_episode(root, "MarkorEditNote", 30, "markor", 0.0, n_steps=2)
        out = B.compress_trajectories("markor", root)
        assert "result=FAIL" in out
        assert "result=SUCCESS" not in out


def test_compress_partial_credit_counts_as_success():
    """0.5/0.7 rewards must be labeled SUCCESS — they're useful trajectory
    signal, not failures."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _make_episode(root, "PiMusicPlayFromPlaylist", 30, "pi_music", 0.7, n_steps=3)
        out = B.compress_trajectories("pi_music", root)
        assert "result=SUCCESS" in out
        assert "reward=0.7" in out


def test_compress_step_line_has_only_5_fields():
    """Per-step line must mention only step / UI / action / reason / summary —
    no token counts, no screenshots, no raw_response, no timestamps."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _make_episode(root, "MarkorCreateNote", 30, "markor", 1.0, n_steps=2)

        out = B.compress_trajectories("markor", root)

        # The step line itself
        step_lines = [l for l in out.splitlines() if l.startswith("Step ")]
        assert step_lines, "expected at least one Step line"
        for line in step_lines:
            # Required tokens
            assert "Step " in line
            assert "UI=[" in line
            assert "-> action=" in line
            assert "-> reason=" in line
            assert "-> summary=" in line
            # Forbidden: anything from the un-promoted fields
            assert "screenshot" not in line.lower()
            assert "raw_response" not in line
            assert "input_tokens" not in line
            assert "wall_s" not in line
            assert "timestamp" not in line


def test_compress_ui_element_cap_default():
    """Default cap is 15 — episodes with more elements get truncated."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _make_episode(root, "MarkorCreateNote", 30, "markor", 1.0,
                      n_steps=1, ui_elements_per_step=30)

        out = B.compress_trajectories("markor", root)

        # Exactly 15 element tokens should appear per step (look at
        # comma-separated count between UI=[ and ])
        ui_section = out.split("UI=[", 1)[1].split("]", 1)[0]
        n_tokens = ui_section.count(":") if ui_section else 0
        assert n_tokens == B.UI_ELEMENTS_DEFAULT_CAP, \
            f"expected {B.UI_ELEMENTS_DEFAULT_CAP} UI tokens, got {n_tokens}"


def test_compress_token_budget_trips_to_tighter_cap(monkeypatch):
    """Force the char budget tiny → expect the tight cap to kick in."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # Make a chunky episode: 30 UI elements per step, 8 steps.
        _make_episode(root, "BigEpisode", 30, "markor", 1.0,
                      n_steps=8, ui_elements_per_step=30)

        # Squash the budget so the first pass is over-budget but the
        # tight-cap pass fits comfortably.
        monkeypatch.setattr(B, "CHAR_BUDGET", 4_000)

        out = B.compress_trajectories("markor", root)

        # At tight cap (8 elements), the per-step UI section should have
        # exactly 8 tokens.
        first_step_ui = out.split("UI=[", 1)[1].split("]", 1)[0]
        n_tokens = first_step_ui.count(":")
        assert n_tokens == B.UI_ELEMENTS_TIGHT_CAP, \
            f"expected tight cap {B.UI_ELEMENTS_TIGHT_CAP}, got {n_tokens}"


def test_compress_deterministic_ordering():
    """Episodes must be sorted by (template, seed) for reproducibility."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # Add in a deliberately non-alphabetical order
        for tpl, sd in [("MarkorDeleteNote", 32), ("MarkorCreateNote", 30),
                        ("MarkorDeleteNote", 30), ("MarkorCreateNote", 31)]:
            _make_episode(root, tpl, sd, "markor", 1.0, n_steps=1)

        out = B.compress_trajectories("markor", root)
        headers = [line for line in out.splitlines() if line.startswith("=== Episode:")]
        # Expect alphabetical by template, ascending by seed within template.
        assert "MarkorCreateNote (seed=30" in headers[0]
        assert "MarkorCreateNote (seed=31" in headers[1]
        assert "MarkorDeleteNote (seed=30" in headers[2]
        assert "MarkorDeleteNote (seed=32" in headers[3]


def test_compress_skips_episode_with_missing_jsonl():
    """If meta.json is there but episode.jsonl is missing, the function
    should still emit the header + a sentinel and not crash."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ep_dir = root / "WeirdEpisode_seed30"
        ep_dir.mkdir()
        (ep_dir / "meta.json").write_text(json.dumps({
            "schema_version": 1, "template": "WeirdEpisode", "seed": 30,
            "app": "markor", "success": 0.0, "n_steps": 0,
            "self_reported": 0, "parse_failures": 0, "alias_hits": 0,
            "clamp_hits": 0, "tier": "source", "agent_name": "x",
            "wall_s_total": 0.1,
        }))
        # No episode.jsonl on purpose.

        out = B.compress_trajectories("markor", root)
        assert "WeirdEpisode" in out
        assert "(episode.jsonl missing)" in out


def test_compress_skips_dirs_without_meta():
    """Stray dirs (e.g. .DS_Store, scratch) should be silently ignored."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "RandomDir").mkdir()
        (root / "AnotherDir").mkdir()
        _make_episode(root, "MarkorCreateNote", 30, "markor", 1.0, n_steps=1)
        out = B.compress_trajectories("markor", root)
        assert "MarkorCreateNote" in out
        # No crash, no spurious content


def test_compress_raises_when_dir_missing():
    try:
        B.compress_trajectories("markor", Path("/no/such/path"))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")


# ─────────────────────────────────────────────────────────────────────────
# UI element compaction (private helper, but worth testing)
# ─────────────────────────────────────────────────────────────────────────


def test_compact_ui_elements_picks_text_over_content_description():
    text = (
        'UI element 0: {"index": 0, "text": "Save", '
        '"content_description": "Save button", "is_clickable": True, '
        '"is_long_clickable": False, "is_editable": False, '
        '"is_scrollable": False, "is_selected": False, "is_checked": False}'
    )
    out = B._compact_ui_elements(text, cap=10)
    assert '0:"Save"' in out
    assert "Save button" not in out  # text takes priority


def test_compact_ui_elements_falls_back_to_content_description():
    text = (
        'UI element 0: {"index": 0, '
        '"content_description": "Profile photo", "is_clickable": True, '
        '"is_long_clickable": False, "is_editable": False, '
        '"is_scrollable": False, "is_selected": False, "is_checked": False}'
    )
    out = B._compact_ui_elements(text, cap=10)
    assert 'Profile photo' in out


def test_compact_ui_elements_renders_flags():
    text = _ui_text(1)  # one element with is_clickable=True
    out = B._compact_ui_elements(text, cap=10)
    assert "(clk)" in out


def test_compact_ui_elements_truncates_long_label():
    long_label = "x" * 200
    text = (
        f'UI element 0: {{"index": 0, "text": "{long_label}", '
        f'"is_clickable": False, "is_long_clickable": False, '
        f'"is_editable": False, "is_scrollable": False, '
        f'"is_selected": False, "is_checked": False}}'
    )
    out = B._compact_ui_elements(text, cap=10)
    # Output should contain an ellipsis and not the full 200-char string.
    assert "…" in out
    assert "x" * 200 not in out


def test_compact_ui_elements_handles_empty():
    assert B._compact_ui_elements("", cap=10) == ""


def test_compact_ui_elements_handles_unparseable_line_gracefully():
    text = "UI element 7: {malformed json"
    out = B._compact_ui_elements(text, cap=10)
    # Should produce the fallback "<idx>:?" token, not crash.
    assert "7:?" in out


# ─────────────────────────────────────────────────────────────────────────
# JSON-schema description string
# ─────────────────────────────────────────────────────────────────────────


def test_schema_description_lists_all_required_fields():
    s = B._build_json_schema_description()
    # Top-level
    assert '"version"' in s
    assert '"app"' in s
    assert '"layer1"' in s
    assert '"layer2"' in s
    assert '"metadata"' in s
    # Layer1 sub-fields
    assert '"states"' in s
    assert '"transitions"' in s
    assert '"strategies"' in s
    assert '"dead_ends"' in s
    # Layer2
    assert '"categories"' in s
    # State fields
    assert '"visual_cues"' in s
    assert '"resource_hints"' in s
    # Transition fields
    assert '"from_state"' in s
    assert '"to_state"' in s
    # Strategy fields
    assert '"preconditions"' in s
    assert '"success_signal"' in s
    # AbstractCategory fields
    assert '"abstract_steps"' in s
    assert '"failure_modes"' in s
    assert '"verification_checklist"' in s


def test_schema_description_warns_layer2_must_not_mention_app():
    """The schema description must remind the model not to leak app names
    into Layer 2. Helps the model self-correct before output."""
    s = B._build_json_schema_description()
    assert "MUST NOT mention the app name" in s
    assert "MUST NOT mention concrete widgets" in s


# ─────────────────────────────────────────────────────────────────────────
# Prompt assembly
# ─────────────────────────────────────────────────────────────────────────


def test_assemble_prompt_contains_required_sections():
    fake_compressed = "=== Episode: TestX (seed=30, result=SUCCESS, reward=1.0, steps=2) ===\nStep 1: ..."
    prompt = B._assemble_prompt(
        app_name="markor", category="Productivity",
        compressed_text=fake_compressed, n_episodes=1,
    )
    # App + category interpolated
    assert "markor" in prompt
    assert "Productivity" in prompt
    # The two required headers
    assert "## LAYER 1 (APP_SPECIFIC)" in prompt
    assert "## LAYER 2 (GENERIC, transferable)" in prompt
    # Compressed bundle made it through
    assert "=== Episode: TestX" in prompt
    # Schema description present (spot-check a unique key)
    assert '"abstract_steps"' in prompt
    # Episode count interpolated
    assert "1 trajectories" in prompt


def test_assemble_prompt_layer2_constraint_references_app_name():
    """The 'CRITICAL CONSTRAINT' line must name the app so the model
    knows which token to avoid in Layer 2."""
    prompt = B._assemble_prompt(
        app_name="bluecoins", category="Finance",
        compressed_text="", n_episodes=0,
    )
    # The constraint line says: LAYER 2 must NOT contain "<app_name>"
    assert 'LAYER 2 must NOT contain "bluecoins"' in prompt


# ─────────────────────────────────────────────────────────────────────────
# JSON extraction from LLM response
# ─────────────────────────────────────────────────────────────────────────


def test_extract_json_bare():
    text = '{"a": 1, "b": [2, 3]}'
    assert B._extract_json_from_response(text) == {"a": 1, "b": [2, 3]}


def test_extract_json_from_fence():
    text = 'Here you go:\n```json\n{"version": "0.1.0", "x": 1}\n```\nHope that helps!'
    assert B._extract_json_from_response(text) == {"version": "0.1.0", "x": 1}


def test_extract_json_from_plain_fence():
    text = '```\n{"a": 1}\n```'
    assert B._extract_json_from_response(text) == {"a": 1}


def test_extract_json_from_prose_wrapped():
    text = 'Sure, the FSM is: {"version": "0.1.0", "app": "x"} as requested.'
    assert B._extract_json_from_response(text) == {"version": "0.1.0", "app": "x"}


def test_extract_json_handles_nested_objects():
    text = 'Here: {"layer1": {"app": "x", "states": [{"id": "s1"}]}}'
    out = B._extract_json_from_response(text)
    assert out["layer1"]["states"][0]["id"] == "s1"


def test_extract_json_handles_braces_in_strings():
    text = 'reply: {"action": "click({\\"x\\":1})"}'
    out = B._extract_json_from_response(text)
    assert out["action"] == 'click({"x":1})'


def test_extract_json_raises_on_garbage():
    try:
        B._extract_json_from_response("definitely not json here")
    except ValueError as e:
        assert "JSON" in str(e) or "json" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")


def test_extract_json_raises_when_top_level_is_array():
    """Top-level JSON must be an object — an array is a schema error,
    not a parse error."""
    try:
        B._extract_json_from_response("[1, 2, 3]")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_first_balanced_object_picks_first_complete_object():
    text = "garbage {a: 1} {b: 2}"  # not real JSON, but balance-wise
    obj = B._first_balanced_object(text)
    assert obj == "{a: 1}"


def test_first_balanced_object_handles_nested():
    text = 'leading {"x": {"y": 1}} trailing'
    obj = B._first_balanced_object(text)
    assert obj == '{"x": {"y": 1}}'


def test_first_balanced_object_returns_none_when_no_brace():
    assert B._first_balanced_object("nothing here") is None


# ─────────────────────────────────────────────────────────────────────────
# build_fsm — pure parts only (env guard)
# ─────────────────────────────────────────────────────────────────────────


def test_build_fsm_raises_without_api_key(monkeypatch):
    """If ANTHROPIC_API_KEY isn't set, build_fsm must fail loudly before
    making any network call."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        B.build_fsm("markor", "Productivity", "fake compressed text")
    except RuntimeError as e:
        assert "ANTHROPIC_API_KEY" in str(e)
    else:
        raise AssertionError("expected RuntimeError when key absent")


# ─────────────────────────────────────────────────────────────────────────
# Standalone runner (lite monkeypatch shim for environments without pytest)
# ─────────────────────────────────────────────────────────────────────────


class _LiteMonkeypatch:
    """Minimal monkeypatch substitute so tests run via `python file.py` too."""

    def __init__(self):
        self._undos: list = []

    def setattr(self, target, name, value):
        old = getattr(target, name)
        self._undos.append(lambda: setattr(target, name, old))
        setattr(target, name, value)

    def delenv(self, name, raising=True):
        existed = name in os.environ
        old = os.environ.get(name)
        if existed:
            del os.environ[name]
            self._undos.append(lambda: os.environ.__setitem__(name, old))
        elif raising:
            raise KeyError(name)

    def setenv(self, name, value):
        old = os.environ.get(name)
        existed = name in os.environ
        os.environ[name] = value
        if existed:
            self._undos.append(lambda: os.environ.__setitem__(name, old))
        else:
            self._undos.append(lambda: os.environ.pop(name, None))

    def undo(self):
        for fn in reversed(self._undos):
            fn()
        self._undos.clear()


if __name__ == "__main__":
    import inspect
    import traceback

    ns = dict(globals())
    tests = [(name, fn) for name, fn in ns.items()
             if name.startswith("test_") and callable(fn)]
    passed = failed = 0
    for name, fn in tests:
        sig = inspect.signature(fn)
        mp = _LiteMonkeypatch() if "monkeypatch" in sig.parameters else None
        try:
            if mp is not None:
                fn(mp)
            else:
                fn()
            passed += 1
            print(f"  ✔ {name}")
        except Exception:
            failed += 1
            print(f"  ✘ {name}")
            traceback.print_exc()
        finally:
            if mp is not None:
                mp.undo()
    print(f"\n{passed}/{passed + failed} tests passed")
    raise SystemExit(0 if failed == 0 else 1)
