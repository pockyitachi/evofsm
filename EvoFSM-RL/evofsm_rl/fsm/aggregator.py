"""LAYER-2 aggregator — Story 2.3.

Merges the ``layer2`` blocks from multiple per-app FSMs that share a
Play Store category into a single ``L_C`` (``Layer2``) for use in
Tier-B / Tier-C transfer.

Contract:
    ``aggregate_L_C(fsms) -> Layer2``
    * Empty list  → empty Layer2 (degenerate case; caller typically
      avoids this).
    * Single FSM  → deep-copy of ``fsms[0].layer2`` (no LLM call; the
      single-app category is already the L_C).
    * Multiple FSMs → Claude-mediated semantic merge. Categories with
      the same or semantically-similar names collapse into one;
      abstract_steps / failure_modes / verification_checklist are
      union-merged with duplicate-semantic pruning. Categories unique
      to one app are preserved verbatim.

The merge step uses ``claude-opus-4-7`` because the 1M-context window
comfortably fits all per-app Layer2 blocks for the largest category
(Productivity: 3 apps × ~11 categories × ~15 items each ≈ a few KB).

We reuse ``_call_anthropic`` and ``_extract_json_from_response`` from
``evofsm_rl.fsm.builder`` for consistent retry / parse behavior.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

from evofsm_rl.fsm.schema import AbstractCategory, FSM, Layer2


logger = logging.getLogger(__name__)


# Same model + sampling params as the per-app builder. 0.3 is conservative
# enough that two runs on the same input agree on structure.
DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_MAX_TOKENS = 16384  # generous — merged L_C for Productivity can be long
DEFAULT_TEMPERATURE = 0.3


# ─────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────


def aggregate_L_C(
    fsms: list[FSM],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Layer2:
    """Merge LAYER2 blocks from same-category FSMs into a unified L_C.

    Args:
        fsms: FSMs sharing a Play Store category. Order is preserved
            when feeding them to the LLM (prompt is byte-stable w.r.t.
            the caller's ordering).
        model / max_tokens / temperature: passed through to Claude on
            the multi-app merge path. Ignored on single-app passthrough.

    Returns:
        A new ``Layer2`` — never a shared reference to any input's
        layer2 (always a deep copy for the single-app path, fresh
        ``AbstractCategory`` instances for the merged path).

    Raises:
        ValueError if the LLM response cannot be parsed as a Layer2.
        RuntimeError if ``ANTHROPIC_API_KEY`` is not set (only on the
        multi-app path).
    """
    if not fsms:
        return Layer2(categories=[])

    if len(fsms) == 1:
        # Single-app category (e.g. Finance=bluecoins alone). L_C = layer2.
        return copy.deepcopy(fsms[0].layer2)

    # Multi-app: semantic merge via Claude.
    category = fsms[0].layer1.category
    if not all(f.layer1.category == category for f in fsms):
        mismatches = sorted({f.layer1.category for f in fsms})
        raise ValueError(
            f"aggregate_L_C: FSMs disagree on layer1.category: {mismatches}. "
            f"All inputs must share the same Play Store category."
        )

    return _merge_layer2_via_claude(
        fsms, category, model=model,
        max_tokens=max_tokens, temperature=temperature,
    )


# ─────────────────────────────────────────────────────────────────────────
# Multi-app merge (LLM-mediated)
# ─────────────────────────────────────────────────────────────────────────


def _merge_layer2_via_claude(
    fsms: list[FSM], category: str, *,
    model: str, max_tokens: int, temperature: float,
) -> Layer2:
    """Build the merge prompt, call Claude, parse the response as Layer2.

    The source apps are labeled ``SOURCE 1`` / ``SOURCE 2`` / ... rather
    than by app name, so the model isn't nudged toward mentioning them
    in the merged output.
    """
    # Lazy import so unit tests that monkeypatch _call_anthropic never
    # import anthropic at all.
    from evofsm_rl.fsm.builder import (
        _call_anthropic,
        _extract_json_from_response,
    )

    prompt = _build_merge_prompt(fsms, category)
    n_apps = len(fsms)
    logger.info(
        "aggregate_L_C: merging %d apps for category %r via %s (prompt %.1fKB)",
        n_apps, category, model, len(prompt) / 1024,
    )

    response_text = _call_anthropic(
        prompt, model=model, max_tokens=max_tokens, temperature=temperature,
    )
    data = _extract_json_from_response(response_text)

    # Response may be either a bare Layer2 ({"categories": [...]}) or a
    # wrapped object like {"layer2": {"categories": [...]}}. Accept both.
    if "categories" not in data and "layer2" in data:
        data = data["layer2"]
    if "categories" not in data:
        raise ValueError(
            f"Merged L_C response missing 'categories' key. "
            f"Response head: {response_text[:300]!r}"
        )

    merged = Layer2.from_json(data)
    if not merged.categories:
        raise ValueError(
            f"Merged L_C has zero categories — refusing to write empty "
            f"output. Response head: {response_text[:300]!r}"
        )
    return merged


def _build_merge_prompt(fsms: list[FSM], category: str) -> str:
    """Assemble the merge prompt. Byte-stable for a given input ordering."""
    n = len(fsms)
    header = (
        f'You are merging abstract workflow patterns from {n} apps in the '
        f'same Play Store category "{category}".\n\n'
        "Below are the LAYER2 blocks from each source app. Each LAYER2 "
        "contains a list of abstract categories, where each category has:\n"
        '  - "name" (e.g. ADD_ENTRY, QUERY_INFO)\n'
        '  - "precondition" (prose)\n'
        '  - "abstract_steps" (list of app-agnostic step strings)\n'
        '  - "failure_modes" (list of prose strings)\n'
        '  - "verification_checklist" (list of prose strings)\n\n'
        "Your task: merge them into a single unified LAYER2.\n\n"
        "Merge rules:\n"
        "  1. Categories with the same or semantically-similar names "
        "(e.g. ADD_ENTRY / CREATE_ENTRY, QUERY_INFO / LOOKUP_FIELD) "
        "must be merged into ONE. Choose the clearest name — prefer "
        "the shorter/more-generic form.\n"
        "  2. For a merged category:\n"
        '     - "abstract_steps" — keep the most comprehensive ordered '
        "sequence; where sources disagree on ordering, prefer the "
        "sequence that covers more sources; absorb unique steps from "
        "each source as additional items in the right position.\n"
        '     - "failure_modes" — UNION. Deduplicate only when two '
        "entries describe literally the same failure; keep all unique "
        "failures across sources.\n"
        '     - "verification_checklist" — UNION with the same dedup rule.\n'
        '     - "precondition" — pick the most complete wording (or '
        "lightly combine if both add information).\n"
        "  3. Categories that exist in only one source are kept as-is.\n"
        "  4. The merged LAYER2 must remain fully app-agnostic:\n"
        "     - NO concrete app names, NO resource IDs, NO package names, "
        "NO dotted Android identifiers.\n"
        "     - Use abstraction vocabulary only (e.g. 'primary create "
        "affordance', 'confirmation control', 'list view').\n"
        "  5. Do NOT invent new failure modes or checklist items. "
        "Only consolidate and reorganize what the sources already say.\n\n"
        "Output format: a single JSON object matching this exact schema, "
        "and nothing else (no prose, no markdown fences):\n"
        '{"categories": [\n'
        '  {"name": str, "precondition": str, "abstract_steps": [str, ...], '
        '"failure_modes": [str, ...], "verification_checklist": [str, ...]},\n'
        '  ...\n'
        ']}\n\n'
    )

    # Per-source blocks (labeled by index, not app name).
    source_blocks = []
    for i, fsm in enumerate(fsms, start=1):
        source_blocks.append(
            f"=== SOURCE {i} (LAYER2) ===\n"
            + json.dumps(fsm.layer2.to_json(), ensure_ascii=False, indent=2)
        )
    sources = "\n\n".join(source_blocks)

    footer = (
        "\n\n"
        "Produce the merged LAYER2 JSON now. Remember: output JSON only, "
        "no surrounding prose."
    )
    return header + sources + footer


def category_to_slug(category: str) -> str:
    """Filename-safe slug for a Play Store category name.

    Canonical mapping used by ``scripts/build_L_C.py`` and any consumer
    that needs to resolve a category → L_C file path:

        "Music & Audio" → "music_audio"
        "Tools"         → "tools"
        "Productivity"  → "productivity"

    Rule: lowercase, then replace each run of non-alphanumeric characters
    with a single underscore, strip leading/trailing underscores. The
    function is deliberately *not* the naïve ``.replace(" ", "")`` — it
    keeps the output human-readable for multi-word categories.
    """
    out: list[str] = []
    prev_under = False
    for ch in category.lower():
        if ch.isalnum():
            out.append(ch)
            prev_under = False
        elif not prev_under:
            out.append("_")
            prev_under = True
    return "".join(out).strip("_")


def load_L_C(path: str | Path) -> tuple[str, Layer2]:
    """Load an ``artifacts/L_C/{slug}.json`` file.

    The on-disk format is ``{"category": <name>, "layer2": <Layer2 JSON>}``
    (a wrapped Layer 2 — not a full FSM). This helper validates the wrapper
    and returns the category label alongside the parsed ``Layer2`` so
    downstream hand-off code (Tier-B / Tier-C init, prompt rendering) can
    consume them directly.

    Args:
        path: path to an L_C JSON file written by ``scripts/build_L_C.py``.

    Returns:
        ``(category, layer2)`` tuple.

    Raises:
        ValueError if the file lacks the required wrapper keys.
    """
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"L_C file {path} is not a JSON object")
    for key in ("category", "layer2"):
        if key not in payload:
            raise ValueError(
                f"L_C file {path} missing required key {key!r} "
                f"(got keys: {sorted(payload.keys())})"
            )
    return payload["category"], Layer2.from_json(payload["layer2"])


__all__ = [
    "aggregate_L_C",
    "category_to_slug",
    "load_L_C",
    "DEFAULT_MODEL",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
]
