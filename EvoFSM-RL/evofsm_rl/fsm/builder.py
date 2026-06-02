"""FSM builder — Story 2.2.2.

Two pieces:

  1. ``compress_trajectories(app, traj_dir) -> str`` — reads all
     ``{template}_seed{N}/`` episode dirs whose ``meta.json`` matches
     the requested ``app``, and produces a single compact text bundle
     suitable for an LLM prompt. Per-step the bundle keeps only the
     5 fields needed for FSM synthesis:

         step, action, action_reason, summary, before_ui_elements_text

     The verbose UI-elements text is rendered down to one short
     ``[idx: "label" (flags)]`` token per element and capped at 15
     elements per step. Token budget enforced at ~160K input tokens
     (≈ 640K chars at 4 chars/token); over-budget bundles are
     re-rendered with the cap dropped to 8 elements and a warning logged.

  2. ``build_fsm(app, category, compressed_text) -> FSM`` — sends the
     bundle to Claude with a strict two-layer schema instruction,
     parses the JSON response, and returns an `FSM` dataclass instance.

The Anthropic API call is the only impure bit; everything else is pure
and unit-tested. The builder retries on transient API errors with
exponential backoff (2s, 4s, 8s) and never silently swallows a JSON
parse failure.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from evofsm_rl.fsm.schema import FSM

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────────────────

# Approx 4 chars per token for English. We aim well under Claude Opus 4.7's
# 200K input window so the prompt scaffold + LLM headroom still fit.
TOKEN_BUDGET = 160_000
CHARS_PER_TOKEN_ESTIMATE = 4
CHAR_BUDGET = TOKEN_BUDGET * CHARS_PER_TOKEN_ESTIMATE

UI_ELEMENTS_DEFAULT_CAP = 15  # first pass
UI_ELEMENTS_TIGHT_CAP = 8     # fallback when over budget

# Per-element label cap so a long content_description doesn't blow up the
# bundle. 60 chars is plenty for "Save changes and return to home" etc.
UI_LABEL_MAX_LEN = 60

# Long action_reason / summary are useful but at some point they're noise.
# Cap defensively.
TEXT_FIELD_MAX_LEN = 600


# Anthropic call config (per Linqiang's spec)
DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.3
RETRY_BACKOFFS_S = (2.0, 4.0, 8.0)


# ─────────────────────────────────────────────────────────────────────────
# Compress trajectories
# ─────────────────────────────────────────────────────────────────────────


def compress_trajectories(app_name: str, trajectory_dir: Path | str) -> str:
    """Read all episodes for ``app_name`` and emit one compressed bundle.

    Args:
        app_name: ``meta.app`` value to filter on (e.g. ``"markor"``).
        trajectory_dir: Path containing ``{template}_seed{N}/`` subdirs
            (each with a ``meta.json`` and ``episode.jsonl``).

    Returns:
        A single string. Episodes are deterministically ordered
        (by template name, then seed). Empty string if no episodes
        match ``app_name``.
    """
    trajectory_dir = Path(trajectory_dir)
    if not trajectory_dir.is_dir():
        raise FileNotFoundError(f"trajectory_dir does not exist: {trajectory_dir}")

    # Collect matching episodes (by reading meta.json on every dir).
    matching: list[tuple[Path, dict[str, Any]]] = []
    for ep_dir in sorted(trajectory_dir.iterdir()):
        meta_path = ep_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("skipping unreadable meta: %s", meta_path)
            continue
        if meta.get("app") == app_name:
            matching.append((ep_dir, meta))
    matching.sort(key=lambda t: (t[1].get("template", ""), t[1].get("seed", 0)))

    if not matching:
        logger.warning("compress_trajectories: no episodes matched app=%r in %s",
                       app_name, trajectory_dir)
        return ""

    # First pass: render at the default UI cap.
    text = _render_bundle(matching, ui_cap=UI_ELEMENTS_DEFAULT_CAP)
    est_tokens = len(text) // CHARS_PER_TOKEN_ESTIMATE
    if len(text) <= CHAR_BUDGET:
        logger.info("compress_trajectories(app=%s): %d episodes, ~%d tokens",
                    app_name, len(matching), est_tokens)
        return text

    # Over-budget: re-render with the tighter cap and warn.
    logger.warning(
        "compress_trajectories(app=%s): first pass ~%d tokens (cap=%d UI elems) "
        "EXCEEDS %d-token budget; re-rendering with cap=%d.",
        app_name, est_tokens, UI_ELEMENTS_DEFAULT_CAP, TOKEN_BUDGET, UI_ELEMENTS_TIGHT_CAP,
    )
    text = _render_bundle(matching, ui_cap=UI_ELEMENTS_TIGHT_CAP)
    est_tokens = len(text) // CHARS_PER_TOKEN_ESTIMATE
    if len(text) > CHAR_BUDGET:
        logger.warning(
            "compress_trajectories(app=%s): STILL ~%d tokens after tight UI cap. "
            "The LLM call may overflow context — consider sampling fewer episodes "
            "or shortening reason/summary fields.",
            app_name, est_tokens,
        )
    else:
        logger.info("compress_trajectories(app=%s): re-rendered to ~%d tokens",
                    app_name, est_tokens)
    return text


def _render_bundle(matching: list[tuple[Path, dict[str, Any]]], *, ui_cap: int) -> str:
    """Format a list of (episode_dir, meta) into the compressed bundle.

    Format (deterministic):

        === Episode: {template} (seed={seed}, result={SUCCESS/FAIL}, reward={reward}, steps={n}) ===
        Step 1: UI=[...] -> action={...} -> reason="..." -> summary="..."
        Step 2: ...
    """
    chunks: list[str] = []
    for ep_dir, meta in matching:
        chunks.append(_render_episode(ep_dir, meta, ui_cap=ui_cap))
    return "\n\n".join(chunks)


def _render_episode(ep_dir: Path, meta: dict[str, Any], *, ui_cap: int) -> str:
    """Render a single episode block."""
    template = meta.get("template", ep_dir.name)
    seed = meta.get("seed", "?")
    success = float(meta.get("success", 0.0))
    n_steps = int(meta.get("n_steps", 0))
    # SUCCESS / FAIL based on success > 0 (partial credit counts as a partial-success
    # signal worth keeping). 0.0 → FAIL; 0.5/0.7/1.0 → SUCCESS.
    label = "SUCCESS" if success > 0 else "FAIL"

    header = (
        f"=== Episode: {template} (seed={seed}, result={label}, "
        f"reward={success}, steps={n_steps}) ==="
    )
    lines = [header]

    jsonl_path = ep_dir / "episode.jsonl"
    if not jsonl_path.exists():
        lines.append("  (episode.jsonl missing)")
        return "\n".join(lines)

    for raw in jsonl_path.read_text().splitlines():
        if not raw.strip():
            continue
        try:
            sd = json.loads(raw)
        except json.JSONDecodeError:
            lines.append(f"  Step ?: <unparseable jsonl line>")
            continue
        lines.append(_render_step(sd, ui_cap=ui_cap))
    return "\n".join(lines)


def _render_step(sd: dict[str, Any], *, ui_cap: int) -> str:
    """Render one step as a single line with the 5 required fields."""
    step = sd.get("step", "?")
    ui = _compact_ui_elements(sd.get("before_ui_elements_text") or "", cap=ui_cap)
    # action is already a dict (or None on parse fail) — render compactly
    action = sd.get("action")
    action_str = json.dumps(action, ensure_ascii=False) if action is not None else "null"
    reason = _truncate(sd.get("action_reason") or "", TEXT_FIELD_MAX_LEN)
    summary = _truncate(sd.get("summary") or "", TEXT_FIELD_MAX_LEN)
    return (
        f"Step {step}: UI=[{ui}] -> action={action_str} "
        f"-> reason={reason!r} -> summary={summary!r}"
    )


# Permissive: only requires "UI element N:" header. The body — if any —
# is parsed as JSON; on failure we still emit a fallback "<idx>:?" token
# so callers can see something was there.
_UI_ELEMENT_LINE_RE = re.compile(r"^UI element (\d+):\s*(.*)$")


def _compact_ui_elements(text: str, *, cap: int) -> str:
    """Convert verbose ``UI element N: {...}`` lines into compact tokens.

    Output is a comma-separated string of ``<idx>:"<label>"<flags>`` tokens
    for the FIRST ``cap`` elements. Drops elements past the cap. Falls
    back gracefully if a line doesn't parse.
    """
    if not text:
        return ""
    tokens: list[str] = []
    for line in text.splitlines():
        if len(tokens) >= cap:
            break
        m = _UI_ELEMENT_LINE_RE.match(line.strip())
        if not m:
            continue
        idx = m.group(1)
        body = m.group(2).strip()
        try:
            obj = json.loads(body.replace("True", "true").replace("False", "false"))
        except json.JSONDecodeError:
            tokens.append(f'{idx}:?')
            continue
        # Pick the most informative label.
        label = (
            obj.get("text")
            or obj.get("content_description")
            or obj.get("hint_text")
            or ""
        )
        if isinstance(label, str) and len(label) > UI_LABEL_MAX_LEN:
            label = label[:UI_LABEL_MAX_LEN - 1] + "…"
        flags: list[str] = []
        if obj.get("is_clickable"):
            flags.append("clk")
        if obj.get("is_editable"):
            flags.append("edt")
        if obj.get("is_scrollable"):
            flags.append("scr")
        flag_str = f"({','.join(flags)})" if flags else ""
        tokens.append(f'{idx}:"{label}"{flag_str}')
    return ", ".join(tokens)


def _truncate(s: str, max_len: int) -> str:
    """Truncate string to ``max_len`` chars with an ellipsis."""
    if len(s) <= max_len:
        return s
    return s[:max_len - 1] + "…"


# ─────────────────────────────────────────────────────────────────────────
# JSON schema description (fed into the LLM prompt so it knows the shape)
# ─────────────────────────────────────────────────────────────────────────


def _build_json_schema_description() -> str:
    """Render a human-readable description of the FSM JSON schema.

    Keeps in sync with ``evofsm_rl/fsm/schema.py`` by walking the
    dataclass fields. Format is plain text — easier for an LLM to follow
    than a formal JSON Schema document, and small enough to fit comfortably
    in the prompt.
    """
    return (
        '{\n'
        '  "version": "0.1.0",                              # SCHEMA_VERSION literal\n'
        '  "app": "<app_name>",                             # str, same as parameter\n'
        '  "layer1": {\n'
        '    "app": "<app_name>",\n'
        '    "category": "<play_store_category>",          # str, same as parameter\n'
        '    "states": [\n'
        '      {\n'
        '        "id": "<short_snake_case_id>",             # str, REQUIRED. e.g. "home", "note_editor"\n'
        '        "description": "<one-line human label>",   # str, may be empty\n'
        '        "visual_cues": ["<UI feature 1>", ...],    # list[str]; e.g. "floating + button bottom-right"\n'
        '        "resource_hints": ["<id 1>", ...]          # list[str]; resource-id substrings if visible\n'
        '      }\n'
        '    ],\n'
        '    "transitions": [\n'
        '      {\n'
        '        "from_state": "<state.id>",                # str, REQUIRED, must match a states[].id\n'
        '        "to_state": "<state.id>",                  # str, REQUIRED, must match a states[].id\n'
        '        "action": "<short action label>",          # str, REQUIRED. e.g. "click(fab_main)", "scroll_down"\n'
        '        "precondition": "<optional guard>",        # str, may be empty\n'
        '        "postcondition": "<optional postcondition>"\n'
        '      }\n'
        '    ],\n'
        '    "strategies": [\n'
        '      {\n'
        '        "name": "<UPPER_SNAKE_CASE_NAME>",         # str, REQUIRED. e.g. "CREATE_NOTE"\n'
        '        "preconditions": "<starting state>",       # str, REQUIRED\n'
        '        "steps": ["<step 1>", "<step 2>", ...],    # list[str], ordered\n'
        '        "success_signal": "<how to know it worked>", # str, REQUIRED\n'
        '        "fallback": "<recovery if it fails>"       # str, may be empty\n'
        '      }\n'
        '    ],\n'
        '    "dead_ends": [\n'
        '      {\n'
        '        "state": "<state.id>",                     # where the agent got stuck\n'
        '        "failed_action": "<what was tried>",       # action that did not work\n'
        '        "note": "<why it failed / what to avoid>"\n'
        '      }\n'
        '    ]\n'
        '  },\n'
        '  "layer2": {\n'
        '    "categories": [\n'
        '      {\n'
        '        "name": "<UPPER_SNAKE_CASE>",              # e.g. "ADD_ENTRY", "DELETE_ENTRY", "QUERY_INFO"\n'
        '        "precondition": "<abstract starting state>", # MUST NOT mention the app name\n'
        '        "abstract_steps": ["<step 1>", ...],       # MUST NOT mention concrete widgets/ids\n'
        '        "failure_modes": ["<common pitfall 1>", ...],\n'
        '        "verification_checklist": ["<check 1>", ...]\n'
        '      }\n'
        '    ]\n'
        '  },\n'
        '  "metadata": {                                    # free-form provenance dict\n'
        '    "built_at": "<ISO timestamp>",\n'
        '    "n_episodes": <int>,\n'
        '    "n_successful": <int>\n'
        '  }\n'
        '}'
    )


# ─────────────────────────────────────────────────────────────────────────
# Prompt assembly
# ─────────────────────────────────────────────────────────────────────────


def _assemble_prompt(app_name: str, category: str, compressed_text: str,
                     n_episodes: int) -> str:
    """Build the full user message sent to the LLM."""
    return _PROMPT_TEMPLATE.format(
        app_name=app_name,
        category=category,
        n_episodes=n_episodes,
        json_schema=_build_json_schema_description(),
        compressed_text=compressed_text,
    )


_PROMPT_TEMPLATE = """\
You are an expert at extracting finite state machines (FSMs) from Android GUI agent trajectories.

Below are {n_episodes} trajectories of an agent operating on the app "{app_name}" (Play Store category: {category}). The trajectories include both successful and failed attempts.

Analyze ALL trajectories and produce a two-layer FSM:

## LAYER 1 (APP_SPECIFIC)
- Identify all distinct app screens/states observed (e.g. HomeScreen, EditorView, SettingsPage)
- For each state: list visual_cues (UI features visible on screen) and resource_hints (resource IDs if apparent from UI element text)
- List all observed state transitions (which action triggers the transition)
- Extract strategies from successful trajectories (step-by-step sequences that completed tasks)
- Extract dead_ends from failed trajectories (where the agent got stuck and why)

## LAYER 2 (GENERIC, transferable)
- Generalize the observed tasks into abstract categories (e.g. ADD_ENTRY, DELETE_ENTRY, QUERY_INFO, EDIT_ENTRY, NAVIGATE_TO)
- For each category: provide abstract preconditions, abstract steps, common failure modes, and verification checklist
- CRITICAL CONSTRAINT: LAYER 2 must NOT contain "{app_name}" or any app-specific widget names, resource IDs, or state names. If you cannot describe something without naming the app, it belongs in LAYER 1.

Output strictly valid JSON matching this schema (no other text):
{json_schema}

## Trajectories

{compressed_text}
"""


# ─────────────────────────────────────────────────────────────────────────
# JSON extraction from LLM response (tolerant)
# ─────────────────────────────────────────────────────────────────────────


def _extract_json_from_response(text: str) -> dict[str, Any]:
    """Pull a JSON object out of the model's response text.

    Tries in order:
      1. Whole-text json.loads (works when model returned bare JSON)
      2. Strip ```json ... ``` fence and parse the body
      3. Find the first balanced ``{...}`` and parse that

    Raises:
        ValueError if nothing parses, with the original text head in the
        message so callers can debug.
    """
    text = text.strip()

    # Attempt 1: raw
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        raise ValueError("top-level JSON is not an object")
    except json.JSONDecodeError:
        pass

    # Attempt 2: ```json … ``` (or plain ```) fence
    fence_match = re.search(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Attempt 3: first balanced top-level {...}
    extracted = _first_balanced_object(text)
    if extracted is not None:
        try:
            obj = json.loads(extracted)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Found a {{...}} block but it failed to parse as JSON: {e}. "
                f"Response head: {text[:200]!r}"
            ) from e

    raise ValueError(
        f"Could not extract any JSON object from response. "
        f"Response head: {text[:200]!r}"
    )


def _first_balanced_object(text: str) -> str | None:
    """Return the first top-level balanced ``{...}`` substring of ``text``."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


# ─────────────────────────────────────────────────────────────────────────
# Anthropic call (with retries) — the only impure piece
# ─────────────────────────────────────────────────────────────────────────


def _call_anthropic(prompt: str, *, model: str, max_tokens: int,
                    temperature: float) -> str:
    """Call Claude via the Anthropic SDK and return the response text.

    Retries 3 times on transient errors with exponential backoff
    (2s → 4s → 8s). Raises the last exception if all retries fail.
    Reads ``ANTHROPIC_API_KEY`` from env (the SDK does this implicitly).
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it in your shell before "
            "calling build_fsm()."
        )

    # Lazy import so unit tests that don't hit the API don't pay the cost.
    import anthropic

    client = anthropic.Anthropic()
    last_exc: Exception | None = None
    attempts = len(RETRY_BACKOFFS_S) + 1
    for attempt in range(1, attempts + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            # Concatenate all text blocks (response.content is a list).
            parts = [block.text for block in response.content
                     if getattr(block, "type", None) == "text"]
            return "".join(parts)
        except (anthropic.APIError, anthropic.APIStatusError,
                anthropic.APIConnectionError) as e:
            last_exc = e
            if attempt >= attempts:
                break
            sleep_for = RETRY_BACKOFFS_S[attempt - 1]
            logger.warning(
                "Anthropic API call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt, attempts, e, sleep_for,
            )
            time.sleep(sleep_for)
    assert last_exc is not None
    raise last_exc


# ─────────────────────────────────────────────────────────────────────────
# Public: build_fsm
# ─────────────────────────────────────────────────────────────────────────


def build_fsm(
    app_name: str,
    category: str,
    compressed_text: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> FSM:
    """Synthesize an ``F^0_a`` FSM for one app from its compressed trajectories.

    Args:
        app_name: app key (e.g. ``"markor"``). Used in prompt + Layer 1.
        category: Play Store category (e.g. ``"Productivity"``). Used in
            prompt + Layer 1. Caller looks up from
            ``configs/task_categories.csv``.
        compressed_text: output of ``compress_trajectories(app_name, ...)``.
        model: Claude model id. Default: opus 4.7 (200K context).
        max_tokens: cap on the response. 8192 is comfortable for a
            two-layer FSM with ~10 states + ~10 transitions + a few
            categories.
        temperature: sampling temp. 0.3 is the spec default — low enough
            that two runs on the same input agree on structure, high
            enough that the model commits to concrete choices.

    Returns:
        Validated ``FSM`` instance with version + metadata populated.

    Raises:
        ValueError if the response cannot be parsed as JSON or doesn't
        match the FSM schema (after retry exhaustion). Never silently
        returns a partial / synthesized fallback.
    """
    # n_episodes for the prompt — count from the compressed bundle headers.
    n_episodes = compressed_text.count("=== Episode: ")
    prompt = _assemble_prompt(app_name, category, compressed_text, n_episodes)

    response_text = _call_anthropic(
        prompt, model=model, max_tokens=max_tokens, temperature=temperature,
    )
    data = _extract_json_from_response(response_text)

    # Force the parameters we control onto the LLM output (defensive — the
    # model usually echoes correctly but we don't want to depend on it).
    data["app"] = app_name
    if "layer1" not in data:
        raise ValueError(
            f"LLM response missing required 'layer1' field. Response head: "
            f"{response_text[:300]!r}"
        )
    data["layer1"]["app"] = app_name
    data["layer1"]["category"] = category

    fsm = FSM.from_json(data)

    # Stamp provenance metadata if the model didn't.
    fsm.metadata.setdefault("n_episodes_input", n_episodes)
    fsm.metadata.setdefault("synthesizer_model", model)
    fsm.metadata.setdefault("synthesizer_temperature", temperature)
    return fsm


__all__ = [
    "TOKEN_BUDGET",
    "UI_ELEMENTS_DEFAULT_CAP",
    "UI_ELEMENTS_TIGHT_CAP",
    "build_fsm",
    "compress_trajectories",
]
