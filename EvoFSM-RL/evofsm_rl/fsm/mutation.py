"""Mutation self-reflection + diff generation — Story 3.3 (Epic 3 / B3).

This is the "brain" of the evolution loop. Given the current best FSM and
a handful of recent rollout trajectories produced under that FSM, the
``mutate_fsm`` orchestrator runs two sequential Claude API calls:

  1. **Reflection** — a free-form analysis of where the agent went right
     or wrong, split into a LAYER-1 (app-specific) section and an
     optional LAYER-2 (category-generic) section (§3.8 layering).
  2. **Diff** — a structured JSON :class:`FSMDiff` grounded in the
     reflection text, which we then feed through
     :func:`evofsm_rl.fsm.diff.apply_diff` to produce the new FSM.

The module has no emulator or filesystem side-effects beyond reading
episode directories for trajectory compression; all I/O to Claude goes
through a single private helper ``_call_claude`` that tests monkeypatch.

Model defaults
--------------
The default model is ``claude-opus-4-7`` (current Opus 4.7 per the
project environment list). Mutation quality directly determines
evolution quality — a weak reflection or malformed diff wastes a
whole tournament round, and the aggregate API spend across an Epic-3
sweep is small relative to the emulator/GPU time it takes to evaluate
each child FSM. Callers that want to trade quality for cost can pass
``model="claude-sonnet-4-6"`` explicitly.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from evofsm_rl.fsm.diff import (
    DIFF_JSON_SCHEMA,
    FSMDiff,
    apply_diff,
    parse_diff_from_llm_response,
    validate_fsm_integrity,
)
from evofsm_rl.fsm.schema import FSM


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "claude-opus-4-7"  # Opus 4.7 — mutation quality > API cost
DEFAULT_MAX_RETRIES = 3
# Opus 4.7 deprecated the ``temperature`` request parameter; passing it yields
# 400 invalid_request_error. Default to ``None`` (use the model's built-in
# sampling) and let callers pinning older Sonnet/Haiku models override.
DEFAULT_TEMP_REFLECTION: float | None = None
DEFAULT_TEMP_DIFF: float | None = None
DEFAULT_MAX_TOKENS_REFLECTION = 4096
# Opus 4.7 is verbose on multi-op diffs with full list replacements
# (markor-scale FSMs can easily push past 2K); 8K gives plenty of
# headroom while still being cheap on output-token cost.
DEFAULT_MAX_TOKENS_DIFF = 8192
RETRY_BACKOFFS_S: tuple[float, ...] = (2.0, 8.0, 32.0)

# Cap UI-element lines to avoid giant prompt tails.
_UI_ELEMENTS_PER_STEP = 10
_UI_ELEMENT_LINE_MAX = 160
_REASON_MAX = 400
_SUMMARY_MAX = 400


# ─────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────


class MutationError(Exception):
    """Raised when the mutation pipeline fails irrecoverably.

    Cases:
      * Reflection API call fails after all retries.
      * Diff API call + parse fails after all retries.
      * Diff parsed successfully but apply_diff applied zero ops while
        the diff had at least one op (i.e. everything was skipped —
        the resulting FSM would be identical to the input, which
        defeats the point of the round).
    """


# ─────────────────────────────────────────────────────────────────────
# Part 1: Trajectory compression for reflection
# ─────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class CompressedStep:
    """One step of a trajectory, compressed for reflection."""

    step: int
    action: dict[str, Any]
    action_reason: str
    summary: str
    ui_elements: list[str]  # at most ``_UI_ELEMENTS_PER_STEP`` entries
    status: str             # "ok" | "failed_parse" | "failed_exec"


@dataclasses.dataclass(frozen=True)
class CompressedTrajectory:
    """One episode trajectory, compressed for reflection."""

    task_name: str
    task_goal: str
    seed: int
    reward: float
    n_steps: int
    steps: list[CompressedStep]


def compress_trajectory_for_reflection(
    episode_dir: Path | str,
) -> CompressedTrajectory:
    """Compress one persisted episode directory for reflection.

    Expects the Story-2.0 on-disk schema:
      * ``{episode_dir}/meta.json`` with ``app``, ``template``, ``seed``,
        ``success`` (or ``reward``), optional ``goal``.
      * ``{episode_dir}/episode.jsonl`` one JSON per step, with fields
        ``step``, ``action`` (dict, may be ``action_json`` on older
        trajectories), ``action_reason``, ``summary``,
        ``before_ui_elements_text`` (list of strings), optional ``status``.

    UI-element lists are truncated to the top ``_UI_ELEMENTS_PER_STEP``
    entries with each line clipped at ``_UI_ELEMENT_LINE_MAX`` chars so
    a pathologically long a11y string can't blow up the prompt.
    """
    episode_dir = Path(episode_dir)
    meta_path = episode_dir / "meta.json"
    jsonl_path = episode_dir / "episode.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found at {meta_path}")
    if not jsonl_path.exists():
        raise FileNotFoundError(f"episode.jsonl not found at {jsonl_path}")

    meta = json.loads(meta_path.read_text())
    task_name = str(meta.get("template", episode_dir.name))
    # "reward" or "success" — whichever is present.
    if "reward" in meta:
        reward = float(meta.get("reward", 0.0))
    else:
        reward = float(meta.get("success", 0.0))
    seed = int(meta.get("seed", 0))
    task_goal = str(meta.get("goal", "") or "")

    steps: list[CompressedStep] = []
    with jsonl_path.open() as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                s = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.warning(
                    "compress_trajectory_for_reflection: skipping bad JSONL "
                    "line %d of %s: %s", lineno, jsonl_path, e,
                )
                continue
            steps.append(_compress_step(s, lineno))

    if task_goal == "" and steps:
        # Many of our episodes store the goal on the first step rather
        # than in meta.json. Fall back to that.
        first_goal = ""
        with jsonl_path.open() as fh:
            for raw in fh:
                try:
                    first = json.loads(raw)
                    first_goal = str(first.get("goal", "") or "")
                    break
                except json.JSONDecodeError:
                    continue
        task_goal = first_goal

    return CompressedTrajectory(
        task_name=task_name,
        task_goal=task_goal,
        seed=seed,
        reward=reward,
        n_steps=len(steps),
        steps=steps,
    )


def _compress_step(raw: dict[str, Any], lineno: int) -> CompressedStep:
    """Build a ``CompressedStep`` from one episode.jsonl row."""
    step_num = int(raw.get("step", lineno))

    action = raw.get("action")
    if isinstance(action, str):
        # Some older agents serialized action as JSON string; try to parse.
        try:
            action = json.loads(action)
        except (json.JSONDecodeError, TypeError):
            action = {"raw": action}
    if not isinstance(action, dict):
        action = raw.get("action_json") if isinstance(
            raw.get("action_json"), dict,
        ) else {}

    action_reason = _clip(str(raw.get("action_reason", "") or ""), _REASON_MAX)
    summary = _clip(str(raw.get("summary", "") or ""), _SUMMARY_MAX)

    elements_raw = raw.get("before_ui_elements_text") or []
    if not isinstance(elements_raw, list):
        elements_raw = []
    ui_elements: list[str] = []
    for elem in elements_raw[:_UI_ELEMENTS_PER_STEP]:
        ui_elements.append(_clip(str(elem), _UI_ELEMENT_LINE_MAX))

    status = _infer_step_status(raw)

    return CompressedStep(
        step=step_num,
        action=dict(action),
        action_reason=action_reason,
        summary=summary,
        ui_elements=ui_elements,
        status=status,
    )


def _infer_step_status(raw: dict[str, Any]) -> str:
    """Heuristic: ok / failed_parse / failed_exec.

    - Explicit ``"status"`` field wins if present.
    - Otherwise flag parse failures first, then execution failures,
      otherwise "ok".
    """
    explicit = raw.get("status")
    if isinstance(explicit, str) and explicit:
        return explicit
    if raw.get("parse_failure") or raw.get("parse_error"):
        return "failed_parse"
    if raw.get("exec_error") or raw.get("exec_failure") or raw.get("error"):
        return "failed_exec"
    return "ok"


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


def format_trajectories_for_prompt(
    trajectories: list[CompressedTrajectory],
) -> str:
    """Render compressed trajectories as text for the reflection prompt.

    Output layout, repeated per trajectory::

        === Trajectory: <task_name> (seed=<seed>, reward=<reward>, steps=<N>) ===
        goal: <task_goal>
        Step 1: UI=[elem1, elem2, ...] → action=<action> → reason="<reason>" → summary="<summary>" [<status>]
        Step 2: ...

    Successes and failures both get their full step sequence — failure
    steps in particular are the most information-dense input for the
    mutation reflection.
    """
    if not trajectories:
        return "(no trajectories available)"

    blocks: list[str] = []
    for traj in trajectories:
        header = (
            f"=== Trajectory: {traj.task_name} "
            f"(seed={traj.seed}, reward={traj.reward:.2f}, "
            f"steps={traj.n_steps}) ==="
        )
        lines: list[str] = [header]
        if traj.task_goal:
            lines.append(f"goal: {traj.task_goal}")
        for s in traj.steps:
            ui_str = ", ".join(s.ui_elements) if s.ui_elements else "(empty)"
            action_json = _compact_json(s.action)
            status_tag = f" [{s.status}]" if s.status and s.status != "ok" else ""
            lines.append(
                f"Step {s.step}: "
                f"UI=[{ui_str}] "
                f"→ action={action_json} "
                f'→ reason="{s.action_reason}" '
                f'→ summary="{s.summary}"'
                f"{status_tag}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _compact_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(obj)


# ─────────────────────────────────────────────────────────────────────
# Part 2: Reflection prompt (§3.8 layered)
# ─────────────────────────────────────────────────────────────────────


def build_reflection_prompt(
    fsm: FSM,
    trajectories: list[CompressedTrajectory],
    task_category: str = "",
    *,
    layer2_only: bool = False,
    bootstrap: bool = False,
) -> list[dict[str, Any]]:
    """Construct the single-turn reflection prompt for Claude.

    Modes:
      * ``layer2_only=False`` (default, B3 full-FSM or ablation path):
        shows the complete two-layer FSM and asks for LAYER-1 + LAYER-2
        insights in separate blocks. If ``task_category`` is empty,
        only the LAYER-1 block is requested (Tier-C fallback).
      * ``layer2_only=True`` (B3 canonical L_C-evolution path): shows
        only the LAYER-2 block and asks exclusively for category-level
        insights. Used by :func:`run_l_c_evolution` when the "FSM"
        being evolved is just a Layer-2 wrapped in an empty FSM shell.
        Requires a non-empty ``task_category``.
    """
    app = fsm.layer1.app
    trajs_text = format_trajectories_for_prompt(trajectories)

    if layer2_only:
        if bootstrap:
            return _build_bootstrap_l2_reflection_prompt(
                fsm, app, task_category, trajs_text,
            )
        return _build_l2_only_reflection_prompt(
            fsm, app, task_category, trajs_text,
        )

    fsm_text = fsm.to_prompt_text()
    sections: list[str] = [
        f'You are analyzing an Android GUI agent\'s performance on app "{app}".',
        "The agent used the following two-layer FSM to guide its actions:",
        "",
        fsm_text,
        "",
        "Here are the agent's trajectories on this app:",
        "",
        trajs_text,
        "",
    ]

    if task_category:
        sections.append(
            "Analyze these trajectories and produce TWO SEPARATE LISTS of insights:"
        )
        sections.append("")
        sections.extend(_layer1_block(app))
        sections.append("")
        sections.extend(_layer2_block(app, task_category))
    else:
        sections.append(
            "Analyze these trajectories and produce a single list of insights:"
        )
        sections.append("")
        sections.extend(_layer1_block(app))

    sections.append("")
    sections.append(
        "Be concrete and specific. Reference step numbers from the trajectories. "
        "Each insight should be actionable (something that could become a diff "
        "operation)."
    )

    return [{"role": "user", "content": "\n".join(sections)}]


def _build_bootstrap_l2_reflection_prompt(
    fsm: FSM, app: str, task_category: str, trajs_text: str,
) -> list[dict[str, Any]]:
    """Bootstrap LAYER-2 reflection: target-app trajectories are the ONLY
    source of category knowledge (no pre-built L_C exists).

    Used in the Tier-C "bootstrap" variant (CLAUDE.md 2026-05-13 option b),
    where the target app's Play Store category has no source-pool match,
    so there's no learned ``L_C`` to start from. The reflection must
    synthesize category-level abstractions from scratch using only the
    target-app trajectories — a strictly harder cold-start problem than
    the standard B3/B4 path, which iteratively edits an existing L_C.

    The prompt deliberately:
      * Acknowledges the empty starting state so the model doesn't try to
        "preserve" non-existent prior content,
      * Asks for the full set of abstract patterns (workflows, failure
        modes, verification signals) rather than incremental insights,
      * Still enforces app-agnosticism — even with one app's data, the
        proposed Layer 2 must generalize to other apps in the same
        category.
    """
    sections = [
        f'You are analyzing an Android GUI agent\'s performance on app "{app}" '
        f'to BOOTSTRAP a category-level abstract action library for category '
        f'"{task_category}".',
        "",
        "IMPORTANT context — this is a COLD START:",
        f'- We do NOT have any existing abstract action library for "{task_category}".',
        f'- "{app}" is the only data point we have for this category right now.',
        "- Your job is to extract category-level patterns from the trajectories below "
        "and propose an INITIAL abstract library. The library must be designed so it "
        "generalizes to OTHER apps in the same category we may encounter later.",
        "",
        f'Trajectories on "{app}":',
        "",
        trajs_text,
        "",
        "Synthesize the following CATEGORY-LEVEL abstractions:",
        "",
        f'1. **abstract_steps**: an ordered list of high-level steps that a generic '
        f'app in category "{task_category}" would need to perform to complete tasks of '
        f'this kind. (Think: workflow stages, not button clicks.)',
        "",
        f'2. **failure_modes**: high-level failure patterns visible in the '
        f'trajectories that would plausibly recur in OTHER apps of this category '
        f'(e.g., "agent confuses calendar week-view with month-view" generalizes; '
        f'"agent failed to find the +Event button in Simple Calendar Pro" does not).',
        "",
        f'3. **verification_checklist**: signals that, if observed, indicate the task '
        f'is genuinely complete versus apparently complete. Again, app-agnostic.',
        "",
        "STRICT REQUIREMENTS:",
        f'- Do NOT mention "{app}" by name, package id, button labels, or any other '
        f'app-specific identifier. Every entry must apply to OTHER apps in category '
        f'"{task_category}".',
        "- Reference step numbers from the trajectories as evidence.",
        "- It is OK (expected, even) to produce 6-12 abstract_steps; this is the "
        "initial library and should be reasonably complete, not minimal.",
        "- If the trajectories are too sparse to support a confident pattern, say so "
        "explicitly rather than fabricating one.",
    ]
    return [{"role": "user", "content": "\n".join(sections)}]


def _build_l2_only_reflection_prompt(
    fsm: FSM, app: str, task_category: str, trajs_text: str,
) -> list[dict[str, Any]]:
    """LAYER-2-only reflection: we're evolving category knowledge, not
    app-specific structure."""
    # Surface just the Layer-2 block with its category tag so the model
    # knows which abstract library it is editing.
    l2_text = fsm.layer2.to_prompt_text(category=task_category)

    sections = [
        "You are analyzing an Android GUI agent's performance to improve "
        "CATEGORY-LEVEL abstract strategies (not app-specific knowledge).",
        "",
        f'The agent is using the following abstract action library for category '
        f'"{task_category}":',
        "",
        l2_text,
        "",
        f'Here are the agent\'s trajectories on app "{app}":',
        "",
        trajs_text,
        "",
        "Analyze these trajectories and identify improvements to the ABSTRACT "
        "STRATEGY ONLY:",
        "- Are the abstract_steps missing a critical step that caused failure?",
        "- Are there new failure_modes discovered from these episodes?",
        "- Does the verification_checklist need additional signals?",
        "- Are any abstract_steps misleading or too vague?",
        "",
        "IMPORTANT:",
        f"- Do NOT propose app-specific changes (screen names, button names, "
        f'resource IDs). "{app}" is one instance of category "{task_category}"; '
        f"your edits must apply to OTHER apps in the same category too.",
        "- Every insight must be phrased in app-agnostic terms.",
        "- Reference step numbers from the trajectories as evidence.",
    ]
    return [{"role": "user", "content": "\n".join(sections)}]


def _layer1_block(app: str) -> list[str]:
    return [
        f'=== LAYER 1 INSIGHTS (APP_SPECIFIC for "{app}") ===',
        f'Things that are true about "{app}" specifically:',
        "- Missing states: screens the agent visited that are not in the FSM",
        "- Wrong visual cues or resource hints",
        "- Incorrect or missing transitions between states",
        "- App-specific UI quirks discovered during the episode",
        "- Error recovery paths that the agent needed but the FSM lacks",
    ]


def _layer2_block(app: str, category: str) -> list[str]:
    return [
        f'=== LAYER 2 INSIGHTS (GENERIC, category = "{category}") ===',
        f'Things that are true about task category "{category}" across apps:',
        "- Abstract workflow patterns that generalize beyond this specific app",
        "- Failure modes that would apply to OTHER apps doing the same type of task",
        "- Verification signals that would generalize",
        (
            f'IMPORTANT: Do NOT mention "{app}" or any app-specific widget names '
            "in Layer 2. If you cannot describe it without naming the app, put it "
            "in Layer 1."
        ),
    ]


# ─────────────────────────────────────────────────────────────────────
# Part 3: Diff generation prompt
# ─────────────────────────────────────────────────────────────────────


# Example spelled out once as a string literal so the prompt stays
# byte-stable regardless of dict ordering quirks in other json dumps.
_DIFF_EXAMPLE_L2_ONLY_JSON = """{
  "ops": [
    {
      "layer": "layer2",
      "target": "category",
      "op": "modify",
      "key": "QUERY_INFO",
      "value": {
        "failure_modes": [
          "reading truncated text from an overview instead of opening the detail view",
          "answering before scrolling the full list",
          "confusing relative date labels with the requested absolute date"
        ],
        "verification_checklist": [
          "answer exactly matches a value visible in the detail view",
          "a terminal status action is emitted exactly once"
        ]
      }
    },
    {
      "layer": "layer2",
      "target": "category",
      "op": "add",
      "key": "OPEN_DETAIL_VIEW",
      "value": {
        "name": "OPEN_DETAIL_VIEW",
        "precondition": "a list view shows entries whose relevant attribute may be truncated",
        "abstract_steps": [
          "Identify the target row using a visual cue that is guaranteed to be complete at this zoom level",
          "Tap the row to open its dedicated detail view",
          "Read the full attribute value from the detail view, not from the list cell"
        ],
        "failure_modes": [
          "Attempting to read the attribute directly from the truncated list cell"
        ],
        "verification_checklist": [
          "the detail view header shows the expected entry title in full"
        ]
      }
    }
  ],
  "reflection_summary": "Refined QUERY_INFO failure modes and added OPEN_DETAIL_VIEW category to handle truncated list views",
  "layer_tag": "layer2"
}"""


_DIFF_EXAMPLE_JSON = """{
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
      "key": "QUERY_INFO",
      "value": {
        "failure_modes": [
          "reading truncated text instead of opening detail view",
          "searching by wrong field"
        ]
      }
    }
  ],
  "reflection_summary": "Added search_results state and transition; updated QUERY failure modes",
  "layer_tag": "both"
}"""


def build_diff_prompt(
    fsm: FSM,
    reflection_text: str,
    *,
    layer2_only: bool = False,
    task_category: str = "",
    bootstrap: bool = False,
) -> list[dict[str, Any]]:
    """Construct the diff-generation prompt for Claude.

    Modes:
      * ``layer2_only=False``: full two-layer prompt with general rules.
        Targets can be any of state/transition/strategy/dead_end/category.
      * ``layer2_only=True``: the FSM context shrinks to just its
        Layer 2, and the rules require every op to target LAYER-2
        categories. Produces one worked example (see
        :data:`_DIFF_EXAMPLE_L2_ONLY_JSON`) instead of the mixed-layer
        one. ``task_category`` is surfaced in the header so the model
        knows which L_C it is editing.
    """
    schema_json = json.dumps(DIFF_JSON_SCHEMA, indent=2)

    if layer2_only:
        if bootstrap:
            return _build_bootstrap_l2_diff_prompt(
                fsm, reflection_text, task_category, schema_json,
            )
        return _build_l2_only_diff_prompt(
            fsm, reflection_text, task_category, schema_json,
        )

    sections = [
        "You are an FSM editor for an Android GUI agent.",
        "",
        "Here is the current FSM:",
        "",
        fsm.to_prompt_text(),
        "",
        "Here is the analysis of the agent's recent performance:",
        "",
        reflection_text,
        "",
        "Based on this analysis, propose edits to the FSM as a JSON diff.",
        "",
        "Rules:",
        "1. Each operation must be tagged with \"layer\": \"layer1\" or \"layer2\".",
        "2. Only include operations that are directly supported by evidence in "
        "the analysis. Do NOT invent speculative changes.",
        "3. For \"modify\" operations on list fields (visual_cues, steps, "
        "failure_modes, etc.), include the COMPLETE updated list, not just "
        "the additions.",
        "4. Keep changes focused — prefer 3-8 targeted operations over wholesale "
        "rewrites.",
        "5. State IDs should be lowercase_with_underscores (e.g. \"search_results\", "
        "not \"SearchResults\").",
        "6. Transition keys use \"from_state->to_state\" format.",
        "7. LAYER2 operations must NOT contain app-specific names or resource IDs.",
        "",
        "Output ONLY the JSON object, nothing else. Schema:",
        "",
        schema_json,
        "",
        "Example of a valid diff:",
        _DIFF_EXAMPLE_JSON,
    ]
    return [{"role": "user", "content": "\n".join(sections)}]


def _build_bootstrap_l2_diff_prompt(
    fsm: FSM, reflection_text: str, task_category: str, schema_json: str,
) -> list[dict[str, Any]]:
    """Bootstrap LAYER-2 diff prompt: emit INSERT operations to populate
    an initially-empty L_C from reflection content.

    Differs from ``_build_l2_only_diff_prompt`` in that:
      * The "current Layer 2" shown to the model is explicitly framed as
        empty/stub, so the model knows to ADD rather than MODIFY.
      * Rule list emphasizes ``add`` ops on the ``category`` target.
      * Worked example shows ``add`` ops populating a category from
        scratch.
    """
    is_empty = not fsm.layer2.categories
    if is_empty:
        l2_text = (
            f'(EMPTY — no abstract action library exists yet for category '
            f'"{task_category}". Your edits will create the initial entries.)'
        )
    else:
        l2_text = fsm.layer2.to_prompt_text(category=task_category)

    cat_clause = (
        f' for category "{task_category}"' if task_category else ""
    )
    sections = [
        f"You are an abstract-strategy editor{cat_clause}. This is the "
        f"BOOTSTRAP phase: the abstract action library starts EMPTY and "
        f"you must populate it from scratch.",
        "",
        "Here is the current LAYER 2:",
        "",
        l2_text,
        "",
        "Here is the analysis of the agent's recent performance:",
        "",
        reflection_text,
        "",
        "Propose edits to the LAYER 2 as a JSON diff.",
        "",
        "Rules:",
        "1. ALL operations MUST have \"layer\": \"layer2\". NO layer1 operations.",
        "2. Only target \"category\" is valid (no state/transition/strategy/"
        "dead_end targets in this mode).",
        f'3. Since LAYER 2 starts empty, your operations should primarily be '
        f'`"op": "add"` to introduce a new category entry under '
        f'"{task_category}" with its abstract_steps, failure_modes, and '
        f'verification_checklist populated. A single comprehensive `add` op '
        f'is usually better than many small ones in the bootstrap phase.',
        "4. Only include entries that are directly supported by evidence in "
        "the analysis. Do NOT invent speculative content.",
        "5. Every entry MUST remain app-agnostic: no app names, no package "
        "identifiers, no resource IDs, no concrete widget names.",
        "6. Aim for a complete-but-tight initial library: 6-12 abstract_steps "
        "is healthy; 3-6 failure_modes; 3-6 verification signals.",
        "",
        "Output ONLY the JSON object, nothing else. Schema:",
        "",
        schema_json,
        "",
        "Example of a valid bootstrap diff (showing one `add` op that "
        "populates a category from scratch):",
        json.dumps({
            "reflection_summary": "Bootstrap initial L_C for the category "
                                  "from N target-app trajectories.",
            "layer_tag": "layer2",
            "ops": [
                {
                    "op": "add",
                    "layer": "layer2",
                    "target": "category",
                    "value": {
                        "name": "<task_category>",
                        "abstract_steps": ["...", "..."],
                        "failure_modes": ["..."],
                        "verification_checklist": ["..."],
                    },
                },
            ],
        }, indent=2),
    ]
    return [{"role": "user", "content": "\n".join(sections)}]


def _build_l2_only_diff_prompt(
    fsm: FSM, reflection_text: str, task_category: str, schema_json: str,
) -> list[dict[str, Any]]:
    """LAYER-2-only diff prompt used by L_C evolution."""
    l2_text = fsm.layer2.to_prompt_text(category=task_category)
    cat_clause = (
        f' for category "{task_category}"' if task_category else ""
    )
    sections = [
        f"You are an abstract-strategy editor{cat_clause}.",
        "",
        "Here is the current LAYER 2 (category-level abstract action library):",
        "",
        l2_text,
        "",
        "Here is the analysis of the agent's recent performance:",
        "",
        reflection_text,
        "",
        "Propose edits to the LAYER 2 as a JSON diff.",
        "",
        "Rules:",
        "1. ALL operations MUST have \"layer\": \"layer2\". NO layer1 operations "
        "are allowed — this is category-level evolution only.",
        "2. Only target \"category\" is valid (no state/transition/strategy/"
        "dead_end targets in this mode).",
        "3. Only include operations that are directly supported by evidence "
        "in the analysis. Do NOT invent speculative changes.",
        "4. For \"modify\" operations on list fields (abstract_steps, "
        "failure_modes, verification_checklist), include the COMPLETE "
        "updated list, not just the additions.",
        "5. Keep changes focused — prefer 2-6 targeted operations over "
        "wholesale rewrites.",
        "6. Every edit MUST remain app-agnostic: no app names, no package "
        "identifiers, no resource IDs, no concrete widget names. If a "
        "proposed change cannot be described without naming the app, it "
        "does not belong in LAYER 2.",
        '7. Set "layer_tag": "layer2".',
        "",
        "Output ONLY the JSON object, nothing else. Schema:",
        "",
        schema_json,
        "",
        "Example of a valid LAYER-2-only diff:",
        _DIFF_EXAMPLE_L2_ONLY_JSON,
    ]
    return [{"role": "user", "content": "\n".join(sections)}]


# ─────────────────────────────────────────────────────────────────────
# Part 4: Mutation orchestrator
# ─────────────────────────────────────────────────────────────────────


def mutate_fsm(
    fsm: FSM,
    trajectories: list[CompressedTrajectory],
    *,
    task_category: str = "",
    layer2_only: bool = False,
    bootstrap: bool = False,
    model: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    temperature_reflection: float | None = DEFAULT_TEMP_REFLECTION,
    temperature_diff: float | None = DEFAULT_TEMP_DIFF,
    max_tokens_reflection: int = DEFAULT_MAX_TOKENS_REFLECTION,
    max_tokens_diff: int = DEFAULT_MAX_TOKENS_DIFF,
) -> tuple[FSM, FSMDiff, str]:
    """Run the full mutation pipeline: reflect → diff → apply.

    Pipeline:
      1. Build the layered reflection prompt and call Claude → reflection text.
      2. Build the diff prompt (with reflection embedded) and call Claude →
         raw diff JSON. Retry on API errors *and* on JSON parse errors up to
         ``max_retries`` attempts.
      3. Feed the raw JSON through
         :func:`evofsm_rl.fsm.diff.parse_diff_from_llm_response`.
      4. Apply the parsed diff via :func:`apply_diff`.
      5. Validate integrity via :func:`validate_fsm_integrity` (warnings
         are logged but don't raise).
      6. If the diff had ops but none applied, raise
         :class:`MutationError`.

    Returns a 3-tuple of:
      * the mutated :class:`~evofsm_rl.fsm.schema.FSM`,
      * the :class:`FSMDiff` that was applied (may contain skipped ops),
      * the verbatim reflection text (for logging / audit / visualization).
    """
    # Reflection. Internal API retries only (parse of reflection text is
    # not structured — we just pass the whole string along).
    try:
        reflection_text = _call_claude(
            build_reflection_prompt(
                fsm, trajectories, task_category,
                layer2_only=layer2_only,
                bootstrap=bootstrap,
            ),
            model=model,
            max_tokens=max_tokens_reflection,
            temperature=temperature_reflection,
            max_api_retries=max_retries,
        )
    except Exception as e:
        raise MutationError(
            f"Reflection call failed after {max_retries} attempts: {e}"
        ) from e

    reflection_text = reflection_text.strip() or "(empty reflection response)"

    # Diff. Retry on both API errors and parse errors up to max_retries.
    diff: FSMDiff | None = None
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw_diff = _call_claude(
                build_diff_prompt(
                    fsm, reflection_text,
                    layer2_only=layer2_only,
                    task_category=task_category,
                    bootstrap=bootstrap,
                ),
                model=model,
                max_tokens=max_tokens_diff,
                temperature=temperature_diff,
                max_api_retries=1,  # outer loop owns the retry budget
            )
            diff = parse_diff_from_llm_response(raw_diff)
            break
        except Exception as e:
            last_exc = e
            logger.warning(
                "Diff generation attempt %d/%d failed: %s",
                attempt, max_retries, e,
            )
            if attempt >= max_retries:
                break
            sleep_for = RETRY_BACKOFFS_S[
                min(attempt - 1, len(RETRY_BACKOFFS_S) - 1)
            ]
            time.sleep(sleep_for)

    if diff is None:
        raise MutationError(
            f"Diff generation failed after {max_retries} attempts: {last_exc}"
        ) from last_exc

    # In layer2-only mode, post-filter any layer1 ops the model may have
    # slipped in despite the rules. We build a fresh FSMDiff so the
    # applier only sees well-scoped ops and the returned diff reflects
    # what was actually applied.
    if layer2_only:
        l1_ops = [op for op in diff.ops if op.layer == "layer1"]
        if l1_ops:
            logger.warning(
                "mutate_fsm(layer2_only=True): dropping %d layer1 op(s) "
                "the model proposed despite the prompt constraint; "
                "keys=%s",
                len(l1_ops), [op.key for op in l1_ops],
            )
            diff = FSMDiff(
                ops=[op for op in diff.ops if op.layer == "layer2"],
                reflection_summary=diff.reflection_summary,
                layer_tag="layer2",
            )

    # Apply.
    result = apply_diff(fsm, diff)

    # Enforce: a non-empty diff must produce at least one applied op.
    if diff.ops and not result.applied:
        reasons = [r for (_, r) in result.skipped]
        raise MutationError(
            f"Diff had {len(diff.ops)} op(s) but all were skipped. "
            f"Reasons: {reasons}"
        )

    # Advisory post-apply integrity check; log only.
    warnings = validate_fsm_integrity(result.fsm)
    if warnings:
        logger.warning(
            "Mutated FSM has %d integrity warning(s): %s",
            len(warnings), warnings,
        )

    logger.info(
        "mutate_fsm: %d ops applied, %d skipped, %d warnings",
        len(result.applied), len(result.skipped), len(result.warnings),
    )

    return result.fsm, diff, reflection_text


# ─────────────────────────────────────────────────────────────────────
# Private: Claude API call with retry (monkeypatchable single impure point)
# ─────────────────────────────────────────────────────────────────────


def _call_claude(
    messages: list[dict[str, Any]],
    *,
    model: str,
    max_tokens: int,
    temperature: float | None,
    max_api_retries: int = 3,
) -> str:
    """Invoke the Anthropic SDK and return the joined text response.

    Retries on transient ``anthropic.APIError`` / ``APIConnectionError``
    /``APIStatusError`` up to ``max_api_retries`` attempts with
    exponential backoff (2s, 8s, 32s). Does NOT retry on parse errors —
    that's the caller's concern (``mutate_fsm`` wraps this function in
    its own retry loop for the diff phase).

    Separated out as a top-level function so tests can monkeypatch it
    at ``evofsm_rl.fsm.mutation._call_claude`` and never touch the real
    SDK.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it in your shell before "
            "calling mutate_fsm()."
        )

    # Lazy import so test paths that monkeypatch us never load anthropic.
    import anthropic

    client = anthropic.Anthropic()
    last_exc: Exception | None = None
    # Some newer models (Opus 4.7) deprecated the ``temperature`` request
    # parameter. Pass it only when the caller explicitly set a value.
    api_kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if temperature is not None:
        api_kwargs["temperature"] = temperature

    for attempt in range(1, max_api_retries + 1):
        try:
            response = client.messages.create(**api_kwargs)
            parts = [
                block.text
                for block in response.content
                if getattr(block, "type", None) == "text"
            ]
            return "".join(parts)
        except (
            anthropic.APIError,
            anthropic.APIStatusError,
            anthropic.APIConnectionError,
        ) as e:
            last_exc = e
            if attempt >= max_api_retries:
                break
            sleep_for = RETRY_BACKOFFS_S[
                min(attempt - 1, len(RETRY_BACKOFFS_S) - 1)
            ]
            logger.warning(
                "Anthropic API call failed (attempt %d/%d): %s — "
                "retrying in %.1fs",
                attempt, max_api_retries, e, sleep_for,
            )
            time.sleep(sleep_for)

    assert last_exc is not None
    raise last_exc


__all__ = [
    "CompressedStep",
    "CompressedTrajectory",
    "DEFAULT_MODEL",
    "MutationError",
    "build_diff_prompt",
    "build_reflection_prompt",
    "compress_trajectory_for_reflection",
    "format_trajectories_for_prompt",
    "mutate_fsm",
]
