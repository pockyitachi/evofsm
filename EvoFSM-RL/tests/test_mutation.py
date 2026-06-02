"""Tests for evofsm_rl.fsm.mutation — Story 3.3.

Covers the 12 acceptance items from the story brief, grouped:

  Trajectory compression (1-2):
    1. compress_trajectory_for_reflection reads meta + jsonl, extracts
       step / action / reason / summary / ui_elements (cap at 10) / status.
    2. format_trajectories_for_prompt produces text containing task_name,
       reward, step numbers, action, reason.

  Prompt building (3-6):
    3. build_reflection_prompt returns a single-user-turn list containing
       the app name and the trajectory text.
    4. With task_category, LAYER 2 section appears.
    5. Without task_category, LAYER 2 section is absent.
    6. build_diff_prompt contains FSM text, reflection text, schema, example.

  Diff parsing integration (7-8):
    7. A realistic Claude-style response round-trips through
       parse_diff_from_llm_response.
    8. A mixed L1+L2 op diff parses all ops.

  mutate_fsm integration (9-12, all API mocked):
    9.  Happy path: mocked valid reflection + valid diff → new FSM
        reflects diff changes.
    10. Invalid JSON on first diff attempt, valid on second → succeeds
        on retry.
    11. Always-failing mock → raises MutationError.
    12. Diff with only non-applicable ops → MutationError after
        apply_diff produces empty applied list.

All API interactions are monkeypatched at
``evofsm_rl.fsm.mutation._call_claude`` — no real network calls.

Run::
    python -m pytest tests/test_mutation.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evofsm_rl.fsm import mutation as mut
from evofsm_rl.fsm.mutation import (
    CompressedStep,
    CompressedTrajectory,
    MutationError,
    build_diff_prompt,
    build_reflection_prompt,
    compress_trajectory_for_reflection,
    format_trajectories_for_prompt,
    mutate_fsm,
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
# Fixture helpers
# ─────────────────────────────────────────────────────────────────────


def _sample_fsm() -> FSM:
    return FSM(
        app="markor",
        layer1=Layer1(
            app="markor",
            category="Productivity",
            states=[State(id="home"), State(id="note_editor")],
            transitions=[
                Transition(from_state="home", to_state="note_editor",
                           action="click(fab)"),
            ],
            strategies=[
                Strategy(name="CREATE_NOTE", preconditions="home visible",
                         steps=["click fab"], success_signal="note in list"),
            ],
        ),
        layer2=Layer2(categories=[
            AbstractCategory(
                name="ADD_ENTRY", precondition="list visible",
                abstract_steps=["locate primary affordance"],
                failure_modes=["confirming too early"],
                verification_checklist=["new entry visible"],
            ),
        ]),
    )


def _write_episode(tmp_path: Path, *, reward: float = 1.0) -> Path:
    """Create a minimal Story-2.0 episode dir and return its Path."""
    ep_dir = tmp_path / "SomeTemplate_seed30"
    ep_dir.mkdir()
    meta = {
        "app": "test_app",
        "template": "SomeTemplate",
        "seed": 30,
        "reward": reward,
        "goal": "Do something in the test app",
    }
    (ep_dir / "meta.json").write_text(json.dumps(meta))
    steps = [
        {
            "step": 1,
            "action": {"action_type": "click", "x": 540, "y": 1200},
            "action_reason": "Tap the add button",
            "summary": "Opened add form",
            "before_ui_elements_text": [
                "Add button", "Title bar", "Menu icon", "Search", "Settings",
                "Home tab", "Recent tab", "Favorites", "Profile",
                "Notification", "Extra1", "Extra2",
            ],
            "status": "ok",
        },
        {
            "step": 2,
            "action": {"action_type": "input_text", "text": "hello"},
            "action_reason": "Type into field",
            "summary": "Text entered",
            "before_ui_elements_text": ["Text field", "Keyboard"],
            "status": "ok",
        },
    ]
    with (ep_dir / "episode.jsonl").open("w") as f:
        for s in steps:
            f.write(json.dumps(s) + "\n")
    return ep_dir


# ═════════════════════════════════════════════════════════════════════
# Trajectory compression
# ═════════════════════════════════════════════════════════════════════


def test_compress_trajectory_reads_meta_and_steps(tmp_path: Path):
    """Test 1: episode dir → CompressedTrajectory with all key fields set."""
    ep_dir = _write_episode(tmp_path, reward=1.0)
    traj = compress_trajectory_for_reflection(ep_dir)

    assert traj.task_name == "SomeTemplate"
    assert traj.seed == 30
    assert traj.reward == 1.0
    assert traj.task_goal == "Do something in the test app"
    assert traj.n_steps == 2
    assert len(traj.steps) == 2

    s1 = traj.steps[0]
    assert s1.step == 1
    assert s1.action == {"action_type": "click", "x": 540, "y": 1200}
    assert s1.action_reason == "Tap the add button"
    assert s1.summary == "Opened add form"
    # UI list capped at 10 entries.
    assert len(s1.ui_elements) == 10
    assert s1.ui_elements[0] == "Add button"
    assert "Extra1" not in s1.ui_elements  # past the cap
    assert s1.status == "ok"


def test_format_trajectories_for_prompt_contains_key_fields():
    """Test 2: prompt-formatted text has task_name, reward, steps, actions."""
    traj = CompressedTrajectory(
        task_name="MarkorCreateNote",
        task_goal="Create a note titled Foo",
        seed=42,
        reward=0.0,
        n_steps=1,
        steps=[
            CompressedStep(
                step=1,
                action={"action_type": "click", "index": 7},
                action_reason="Tap FAB to create new note",
                summary="FAB tapped; dialog appeared",
                ui_elements=["FAB (+)", "Empty note list", "App title"],
                status="ok",
            ),
        ],
    )
    text = format_trajectories_for_prompt([traj])
    assert "MarkorCreateNote" in text
    assert "seed=42" in text
    assert "reward=0.00" in text
    assert "steps=1" in text
    assert "Step 1:" in text
    assert "action_type" in text and "click" in text
    assert "Tap FAB to create new note" in text
    assert "FAB tapped" in text


# ═════════════════════════════════════════════════════════════════════
# Prompt building
# ═════════════════════════════════════════════════════════════════════


def _one_user_message(msgs: list[dict]) -> str:
    """Assert shape: exactly one user turn; return its content."""
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    return msgs[0]["content"]


def test_build_reflection_prompt_shape_and_content():
    """Test 3: single user turn, contains app name + trajectory text."""
    traj = CompressedTrajectory(
        task_name="SomeTask", task_goal="g", seed=30, reward=1.0,
        n_steps=1, steps=[
            CompressedStep(step=1, action={"action_type": "click"},
                           action_reason="r", summary="s",
                           ui_elements=["btn"], status="ok"),
        ],
    )
    msgs = build_reflection_prompt(_sample_fsm(), [traj], task_category="")
    content = _one_user_message(msgs)

    # Contains the app name.
    assert "markor" in content
    # Contains the trajectory header.
    assert "SomeTask" in content
    # Contains the FSM prompt text (look for a LAYER header that
    # FSM.to_prompt_text always emits).
    assert "LAYER 1: APP_SPECIFIC" in content
    # Has the L1 insights block.
    assert "LAYER 1 INSIGHTS" in content


def test_build_reflection_prompt_with_category_includes_layer2_block():
    """Test 4: task_category non-empty ⇒ LAYER 2 section appears."""
    msgs = build_reflection_prompt(_sample_fsm(), [],
                                    task_category="Productivity")
    content = _one_user_message(msgs)
    assert "LAYER 2 INSIGHTS" in content
    assert "Productivity" in content


def test_build_reflection_prompt_without_category_omits_layer2_block():
    """Test 5: empty task_category ⇒ no LAYER 2 section."""
    msgs = build_reflection_prompt(_sample_fsm(), [], task_category="")
    content = _one_user_message(msgs)
    assert "LAYER 2 INSIGHTS" not in content


def test_build_diff_prompt_contains_fsm_reflection_schema_example():
    """Test 6: diff prompt has FSM text, reflection text, schema, example."""
    reflection = "Step 1 clicked FAB, but FSM lacks a settings state."
    msgs = build_diff_prompt(_sample_fsm(), reflection)
    content = _one_user_message(msgs)

    # FSM present (same banner check as above).
    assert "LAYER 1: APP_SPECIFIC" in content
    # Reflection text verbatim.
    assert reflection in content
    # Schema keys.
    assert '"properties"' in content
    assert '"ops"' in content
    # Example values.
    assert "search_results" in content


# ═════════════════════════════════════════════════════════════════════
# Diff parsing integration (no mock API — just realistic text → parse)
# ═════════════════════════════════════════════════════════════════════


_REALISTIC_CLAUDE_RESPONSE = """Based on my analysis, here are the changes:

```json
{
  "ops": [
    {
      "layer": "layer1",
      "target": "state",
      "op": "add",
      "key": "search_results",
      "value": {
        "id": "search_results",
        "description": "Search results after query",
        "visual_cues": ["result list", "search bar at top"],
        "resource_hints": []
      }
    },
    {
      "layer": "layer1",
      "target": "transition",
      "op": "add",
      "key": "home->search_results",
      "value": {
        "from_state": "home",
        "to_state": "search_results",
        "action": "tap(search_icon) then input_text(query)"
      }
    },
    {
      "layer": "layer2",
      "target": "category",
      "op": "modify",
      "key": "ADD_ENTRY",
      "value": {
        "failure_modes": ["confirming too early", "wrong target selected"]
      }
    }
  ],
  "reflection_summary": "Added search state + transition; refined ADD_ENTRY failure modes",
  "layer_tag": "both"
}
```
"""


def test_realistic_claude_response_parses():
    """Test 7: end-to-end parse of a fenced-block style LLM response."""
    from evofsm_rl.fsm.diff import parse_diff_from_llm_response
    diff = parse_diff_from_llm_response(_REALISTIC_CLAUDE_RESPONSE)
    assert len(diff.ops) == 3
    assert diff.reflection_summary.startswith("Added search state")
    assert diff.layer_tag == "both"


def test_mixed_l1_l2_ops_all_parsed():
    """Test 8: ops spanning both layers survive parsing."""
    from evofsm_rl.fsm.diff import parse_diff_from_llm_response
    diff = parse_diff_from_llm_response(_REALISTIC_CLAUDE_RESPONSE)
    layers = {op.layer for op in diff.ops}
    assert layers == {"layer1", "layer2"}
    # Per-layer counts match the example.
    l1 = [op for op in diff.ops if op.layer == "layer1"]
    l2 = [op for op in diff.ops if op.layer == "layer2"]
    assert len(l1) == 2
    assert len(l2) == 1


# ═════════════════════════════════════════════════════════════════════
# mutate_fsm — integration with mocked API
# ═════════════════════════════════════════════════════════════════════


_VALID_REFLECTION_RESPONSE = (
    "=== LAYER 1 INSIGHTS (APP_SPECIFIC for \"markor\") ===\n"
    "- Step 1 the agent tapped a non-existent floating button; the FSM "
    "is missing a search_results state.\n"
)

_VALID_DIFF_RESPONSE = """```json
{
  "ops": [
    {
      "layer": "layer1",
      "target": "state",
      "op": "add",
      "key": "search_results",
      "value": {
        "id": "search_results",
        "description": "results after search",
        "visual_cues": ["results list"],
        "resource_hints": []
      }
    }
  ],
  "reflection_summary": "Added search_results state",
  "layer_tag": "layer1"
}
```"""


class _Fake:
    """Stateful fake for monkeypatching _call_claude.

    Each invocation pops the next response from ``responses``. A
    response may be a string (returned directly) or an Exception
    instance (raised). Once exhausted, StopIteration is raised.
    """

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, messages, *, model, max_tokens, temperature,
                 max_api_retries=3):
        self.calls.append({
            "messages": messages,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        if not self.responses:
            raise AssertionError("fake _call_claude: ran out of responses")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def test_mutate_fsm_happy_path_applies_diff(monkeypatch):
    """Test 9: mocked reflection + valid diff → mutated FSM reflects change."""
    fake = _Fake([_VALID_REFLECTION_RESPONSE, _VALID_DIFF_RESPONSE])
    monkeypatch.setattr(mut, "_call_claude", fake)

    fsm = _sample_fsm()
    new_fsm, diff, reflection_text = mutate_fsm(fsm, [], task_category="")

    # Exactly two API calls (reflection + one diff attempt).
    assert len(fake.calls) == 2
    # Reflection came back verbatim.
    assert "search_results" in reflection_text
    # Diff had our single op.
    assert len(diff.ops) == 1
    # Mutated FSM now has the new state; original is unchanged.
    new_ids = {s.id for s in new_fsm.layer1.states}
    assert "search_results" in new_ids
    orig_ids = {s.id for s in fsm.layer1.states}
    assert "search_results" not in orig_ids


def test_mutate_fsm_retries_on_unparseable_diff(monkeypatch):
    """Test 10: first diff call returns garbage, second returns valid → OK."""
    # Reflection call returns valid text once. The diff call fails first
    # (no JSON present ⇒ parse_diff_from_llm_response raises ValueError)
    # then succeeds on the retry.
    fake = _Fake([
        _VALID_REFLECTION_RESPONSE,
        "I'm sorry, I cannot help with that.",  # 1st diff attempt, no JSON
        _VALID_DIFF_RESPONSE,                    # 2nd diff attempt, valid
    ])
    monkeypatch.setattr(mut, "_call_claude", fake)
    # Shorten retry sleep so the test isn't slow.
    monkeypatch.setattr(mut, "RETRY_BACKOFFS_S", (0.0, 0.0, 0.0))

    new_fsm, diff, _reflection = mutate_fsm(
        _sample_fsm(), [], task_category="",
    )
    assert len(fake.calls) == 3
    assert len(diff.ops) == 1
    assert any(s.id == "search_results" for s in new_fsm.layer1.states)


def test_mutate_fsm_raises_when_all_retries_fail(monkeypatch):
    """Test 11: every diff attempt returns garbage → MutationError."""
    # Reflection still succeeds; every diff call fails to parse.
    fake = _Fake([
        _VALID_REFLECTION_RESPONSE,
        "no JSON here",
        "still no JSON",
        "nope",
    ])
    monkeypatch.setattr(mut, "_call_claude", fake)
    monkeypatch.setattr(mut, "RETRY_BACKOFFS_S", (0.0, 0.0, 0.0))

    with pytest.raises(MutationError) as excinfo:
        mutate_fsm(_sample_fsm(), [], task_category="",
                    max_retries=3)
    assert "diff generation" in str(excinfo.value).lower()
    # All diff attempts were made (1 reflection + 3 diff tries).
    assert len(fake.calls) == 4


def test_build_reflection_prompt_layer2_only_mentions_category_level(
    monkeypatch,
):
    """Test (L2-only): reflection prompt calls out CATEGORY-LEVEL scope
    and does NOT ask for LAYER 1 insights."""
    msgs = build_reflection_prompt(
        _sample_fsm(), [], task_category="Productivity",
        layer2_only=True,
    )
    content = _one_user_message(msgs)
    assert "CATEGORY-LEVEL" in content
    assert "ABSTRACT STRATEGY ONLY" in content
    # No LAYER 1 insights header in L2-only mode.
    assert "LAYER 1 INSIGHTS" not in content
    # Full FSM banner is not surfaced in L2-only mode either (we only
    # show the Layer-2 block).
    assert "LAYER 1: APP_SPECIFIC" not in content
    assert "LAYER 2: GENERIC" in content  # from layer2.to_prompt_text


def test_build_diff_prompt_layer2_only_requires_layer2_ops(monkeypatch):
    """Test (L2-only): diff prompt constrains ops to layer2 / category."""
    msgs = build_diff_prompt(
        _sample_fsm(), "reflection text here",
        layer2_only=True, task_category="Productivity",
    )
    content = _one_user_message(msgs)
    # The key constraint phrasing.
    assert 'ALL operations MUST have "layer": "layer2"' in content
    assert "NO layer1 operations are allowed" in content
    # The L2-only example is surfaced (not the mixed example).
    assert "OPEN_DETAIL_VIEW" in content
    # Full FSM banner suppressed, layer-2 block shown with category tag.
    assert "LAYER 1: APP_SPECIFIC" not in content
    assert "L_C CATEGORY: Productivity" in content


def test_mutate_fsm_layer2_only_filters_layer1_ops_from_response(
    monkeypatch,
):
    """Test (L2-only): if the model emits a mixed L1+L2 diff despite the
    prompt constraint, mutate_fsm drops the L1 ops and logs a warning."""
    mixed_diff = json.dumps({
        "ops": [
            {"layer": "layer1", "target": "state", "op": "add",
             "key": "sneaky_state",
             "value": {"id": "sneaky_state",
                       "description": "should not land"}},
            {"layer": "layer2", "target": "category", "op": "modify",
             "key": "ADD_ENTRY",
             "value": {"failure_modes": ["refined mode 1", "refined mode 2"]}},
        ],
        "reflection_summary": "mixed — mutate_fsm should strip L1",
        "layer_tag": "both",
    })
    fake = _Fake([_VALID_REFLECTION_RESPONSE, mixed_diff])
    monkeypatch.setattr(mut, "_call_claude", fake)
    monkeypatch.setattr(mut, "RETRY_BACKOFFS_S", (0.0, 0.0, 0.0))

    new_fsm, diff, _reflection = mutate_fsm(
        _sample_fsm(), [],
        task_category="Productivity", layer2_only=True,
    )

    # The applied diff has only the layer2 op.
    assert len(diff.ops) == 1
    assert diff.ops[0].layer == "layer2"
    assert diff.layer_tag == "layer2"
    # Verify the filter worked end-to-end: the new FSM's layer1 is
    # unchanged (no sneaky_state), layer2 reflects the edit.
    new_state_ids = {s.id for s in new_fsm.layer1.states}
    assert "sneaky_state" not in new_state_ids
    add_entry = next(c for c in new_fsm.layer2.categories
                      if c.name == "ADD_ENTRY")
    assert add_entry.failure_modes == ["refined mode 1", "refined mode 2"]


def test_mutate_fsm_raises_when_no_ops_apply(monkeypatch):
    """Test 12: diff parses but every op gets skipped → MutationError."""
    # Build a diff whose only op is "remove a state that doesn't exist" —
    # apply_diff will put it in `skipped`, leaving `applied` empty.
    bad_diff = json.dumps({
        "ops": [
            {"layer": "layer1", "target": "state", "op": "remove",
             "key": "not_in_the_fsm"},
        ],
        "reflection_summary": "tried to remove a ghost",
        "layer_tag": "layer1",
    })
    fake = _Fake([_VALID_REFLECTION_RESPONSE, bad_diff])
    monkeypatch.setattr(mut, "_call_claude", fake)
    monkeypatch.setattr(mut, "RETRY_BACKOFFS_S", (0.0, 0.0, 0.0))

    with pytest.raises(MutationError) as excinfo:
        mutate_fsm(_sample_fsm(), [], task_category="")
    msg = str(excinfo.value)
    assert "skipped" in msg.lower()


# ─────────────────────────────────────────────────────────────────────
# Bootstrap (Tier-C option b) prompt tests
# ─────────────────────────────────────────────────────────────────────


def _empty_layer2_fsm():
    """FSM with non-empty Layer-1 but EMPTY Layer-2 categories list.

    Mirrors the state of a Tier-C target app at bootstrap time:
    population is initialized with ``Layer2(categories=[])``.
    """
    from evofsm_rl.fsm.schema import FSM, Layer1, Layer2

    return FSM(
        app="opentracks",
        layer1=Layer1(
            app="opentracks",
            category="Sports Tracking",
            states=[],
            transitions=[],
            strategies=[],
            dead_ends=[],
        ),
        layer2=Layer2(categories=[]),
        metadata={"source": "test"},
    )


def test_build_reflection_prompt_bootstrap_frames_cold_start():
    """Bootstrap reflection prompt explicitly tells the model that L_C
    starts empty and must be created from scratch."""
    msgs = build_reflection_prompt(
        _empty_layer2_fsm(), [], task_category="Sports Tracking",
        layer2_only=True, bootstrap=True,
    )
    content = _one_user_message(msgs)
    # Bootstrap framing must surface.
    assert "BOOTSTRAP" in content or "COLD START" in content
    assert "do NOT have any existing abstract action library" in content
    # Asks for the three core L2 fields.
    assert "abstract_steps" in content
    assert "failure_modes" in content
    assert "verification_checklist" in content
    # App-agnostic enforcement still present.
    assert "opentracks" in content  # app surfaced
    assert "app-specific" in content.lower() or "OTHER apps" in content


def test_build_reflection_prompt_bootstrap_differs_from_l2_only():
    """The bootstrap variant must differ from the default L2-only variant
    in its framing (so we don't accidentally route to the wrong prompt)."""
    fsm = _empty_layer2_fsm()
    boot = _one_user_message(build_reflection_prompt(
        fsm, [], task_category="Sports Tracking",
        layer2_only=True, bootstrap=True,
    ))
    normal = _one_user_message(build_reflection_prompt(
        fsm, [], task_category="Sports Tracking",
        layer2_only=True, bootstrap=False,
    ))
    assert boot != normal
    # Sentinel that's only in bootstrap.
    assert "BOOTSTRAP" in boot
    assert "BOOTSTRAP" not in normal


def test_build_diff_prompt_bootstrap_emits_add_op_guidance():
    """Bootstrap diff prompt should steer the model toward 'add' ops on
    the 'category' target (since LAYER 2 starts empty)."""
    msgs = build_diff_prompt(
        _empty_layer2_fsm(), "reflection text here",
        layer2_only=True, task_category="Sports Tracking",
        bootstrap=True,
    )
    content = _one_user_message(msgs)
    # Frames the situation as bootstrap.
    assert "BOOTSTRAP" in content
    # Tells model the L_C is empty.
    assert "EMPTY" in content or "empty" in content
    # Still enforces layer2-only + app-agnostic.
    assert 'ALL operations MUST have "layer": "layer2"' in content
    assert "app-agnostic" in content
    # Steers toward "add" ops.
    assert '"add"' in content


def test_build_diff_prompt_bootstrap_with_existing_l2_still_works():
    """Bootstrap diff prompt should not crash if called with a non-empty
    Layer-2 (degenerate case — caller misconfigured)."""
    from evofsm_rl.fsm.schema import AbstractCategory, FSM, Layer1, Layer2

    fsm = FSM(
        app="x",
        layer1=Layer1(app="x", category="Y", states=[], transitions=[],
                      strategies=[], dead_ends=[]),
        layer2=Layer2(categories=[
            AbstractCategory(
                name="Y",
                precondition="",
                abstract_steps=["step1"],
                failure_modes=["fm1"],
                verification_checklist=["v1"],
            ),
        ]),
        metadata={"source": "test"},
    )
    msgs = build_diff_prompt(
        fsm, "reflection", layer2_only=True, task_category="Y",
        bootstrap=True,
    )
    content = _one_user_message(msgs)
    # Should still work, no crash. The empty-state notice is omitted.
    assert "BOOTSTRAP" in content


# ─────────────────────────────────────────────────────────────────────
# Pytest-less standalone runner (monkeypatch fallback)
# ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import inspect
    import tempfile
    import traceback
    from pathlib import Path as _Path

    class _MP:
        def __init__(self):
            self._undo = []

        def setattr(self, target, value):
            import importlib
            if isinstance(target, str):
                mod_name, _, attr = target.rpartition(".")
                mod = importlib.import_module(mod_name)
            else:
                mod = target
                attr = value  # not supported in the standalone runner
                raise NotImplementedError
            old = getattr(mod, attr, None)
            setattr(mod, attr, value)
            self._undo.append((mod, attr, old))

        def undo(self):
            while self._undo:
                mod, attr, old = self._undo.pop()
                setattr(mod, attr, old)

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
                class _MPShim:
                    def __init__(self, mp): self.mp = mp
                    def setattr(self, mod, attr, value=None):
                        # pytest's monkeypatch.setattr has two shapes; we
                        # only ever use the module-attr-string form here.
                        if value is None and isinstance(attr, str):
                            value = attr  # unused
                        if isinstance(mod, str):
                            self.mp.setattr(mod, attr)
                        else:
                            # Module-object + attr string form.
                            old = getattr(mod, attr, None)
                            setattr(mod, attr, value)
                            self.mp._undo.append((mod, attr, old))
                kwargs["monkeypatch"] = _MPShim(mp)
            if "tmp_path" in sig.parameters:
                kwargs["tmp_path"] = _Path(td.name)
            fn(**kwargs)
            passed += 1
            print(f"  ok    {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
        finally:
            mp.undo()
            td.cleanup()
    total = passed + failed
    print(f"\n{passed}/{total} passed")
    raise SystemExit(0 if failed == 0 else 1)
