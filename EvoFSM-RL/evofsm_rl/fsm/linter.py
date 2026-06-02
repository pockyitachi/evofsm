"""LAYER 2 linter — Story 2.2.3.

Verifies that ``fsm.layer2`` contains nothing app-specific. Layer 2 is
the transferable, category-keyed knowledge that gets carried across apps
via L_C; if any concrete app token leaks in (the app name, a resource
id, a package, or an app-specific state id), the abstraction is
contaminated and downstream Tier-B / Tier-C transfer experiments are
silently invalid.

Rules (each violation produces one ``FAIL:`` line in the returned list):

    1. **App name** — the literal ``layer1.app`` value, case-insensitive,
       plus snake_case→space variants and a small hand-curated alias map
       for compound words (e.g. ``bluecoins`` → ``blue coins``). Match
       is word-bounded to avoid spurious hits on ``files`` inside
       ``filesystem``, etc.
    2. **Resource hints** — any string that appears verbatim in any
       ``layer1.states[*].resource_hints`` list. Resource ids are by
       definition app-specific (they're prefixed with the app's package
       name) and have no business in Layer 2.
    3. **Package-name pattern** — anything matching the dotted-identifier
       form used by Android packages (``com.foo.bar``, ``org.wikipedia``,
       ``net.gsantner.markor``, ``code.name.monkey.retromusic``, ...).
       The regex covers the TLD prefixes used by all 12 source-pool apps.
    4. **App-specific state ids** — only state ids that themselves
       contain the app name are flagged (``markor_editor`` → fail,
       ``home`` / ``settings`` → fine). This avoids false positives on
       generic state names like ``home`` that legitimately appear in
       Layer 2 text.

A single text field can produce multiple errors (one per violated rule).
The linter never raises; callers get back a (passed, [error_lines]) tuple.
"""

from __future__ import annotations

import re

from evofsm_rl.fsm.schema import FSM, AbstractCategory, Layer2


# Hand-curated decompositions for compound words the agent commonly
# spells with a space. Add entries as new apps surface false negatives
# in production. Keys are lowercase ``app`` values.
KNOWN_COMPOUND_VARIANTS: dict[str, list[str]] = {
    "bluecoins": ["blue coins"],
    "snapseed": ["snap seed"],
    "tasks_org": ["tasks.org"],
    "pi_music": ["pi music player"],   # e.g. "Pi Music Player" full app name
}


# Package-name regex. Covers TLD prefixes seen in source pool:
#   com.rammigsoftware.bluecoins      ← 3+ segments
#   net.gsantner.markor               ← 3 segments
#   org.wikipedia                     ← 2 segments (real Android package)
#   code.name.monkey.retromusic       ← 4 segments
#   de.dennisguse.opentracks          ← 3 segments
# Anchored at a known TLD-like prefix at the start; needs at least one
# `.identifier` after it. Random URLs like "github.com" / "duckduckgo.com"
# don't match because their leftmost segment is not in the prefix list.
# Version strings like "1.2.3" don't match because they don't start with
# a TLD prefix word.
_PACKAGE_RE = re.compile(
    r"\b(?:com|org|net|io|de|edu|gov|name|code)\.[\w]+(?:\.[\w]+)*\b",
    re.IGNORECASE,
)


def lint_layer2(fsm: FSM) -> tuple[bool, list[str]]:
    """Check that LAYER 2 contains no app-specific content.

    Args:
        fsm: the FSM whose ``layer2`` block to verify. Layer 1 is read
            for context (app name, resource hints, state ids) but not
            checked.

    Returns:
        ``(passed, errors)``. ``passed`` is True iff ``errors`` is empty.
        Each error is a single string starting with ``FAIL:`` and
        identifying the category, field path, and offending substring.
    """
    errors: list[str] = []

    app_variants = _app_name_variants(fsm.layer1.app)
    resource_hints = _collect_resource_hints(fsm)
    app_specific_state_ids = _collect_app_specific_state_ids(fsm)

    for cat in fsm.layer2.categories:
        for field_path, text in _walk_category_text(cat):
            errors.extend(_check_field(
                cat_name=cat.name,
                field_path=field_path,
                text=text,
                app_variants=app_variants,
                resource_hints=resource_hints,
                app_specific_state_ids=app_specific_state_ids,
            ))

    return (len(errors) == 0, errors)


# ─────────────────────────────────────────────────────────────────────────
# Helpers (private)
# ─────────────────────────────────────────────────────────────────────────


def _app_name_variants(app: str) -> list[str]:
    """Return all spellings to forbid for one app.

    Always includes the literal ``app`` value (lowercased). Adds:
      * snake_case → space form (``simple_calendar_pro`` → ``simple calendar pro``)
      * snake_case stripped of trailing ``_pro`` / ``_lite`` qualifier
        (``simple calendar`` from ``simple_calendar_pro``) — Layer 2
        prose tends to drop these qualifiers
      * any entries from ``KNOWN_COMPOUND_VARIANTS``

    All returned strings are lowercase.
    """
    app_lower = app.lower()
    variants: set[str] = {app_lower}

    if "_" in app_lower:
        spaced = app_lower.replace("_", " ")
        variants.add(spaced)
        # Strip _pro / _lite qualifier variants
        for suffix in ("_pro", "_lite"):
            if app_lower.endswith(suffix):
                stripped = app_lower[: -len(suffix)]
                variants.add(stripped)
                variants.add(stripped.replace("_", " "))

    if app_lower in KNOWN_COMPOUND_VARIANTS:
        variants.update(KNOWN_COMPOUND_VARIANTS[app_lower])

    # Sort for deterministic error ordering across runs.
    return sorted(variants)


def _collect_resource_hints(fsm: FSM) -> set[str]:
    """All non-empty resource_hints from layer1, lowercased, *distinctive*.

    Two filters applied:

    1. Length ≤ 2 hints dropped (``"ok"``, ``"pi"``, ``"no"``) — they
       produced substring false-positives in prose even with
       word-boundary matching.
    2. Pure-alphabetic hints dropped (``"filter"``, ``"reset"``,
       ``"folder"``, ``"search"``, ``"back"``, ``"home"``, ``"delete"``,
       ``"configuration"``). Hints in the source FSMs often capture
       UI-label text, which overlaps heavily with common English
       vocabulary used in LAYER-2 prose. Real app-specific resource
       identifiers always contain at least one non-alpha character
       (``:``, ``/``, ``.``, ``_``, a digit, or a space / multi-word
       phrase like "By date"), so this filter keeps the signal
       (``com.foo:id/bar``, ``Record-1.m4a``, ``By date``) and drops
       the noise (bare English words). App-name leaks are still
       caught by Rule 1 independently.
    """
    def _is_distinctive(s: str) -> bool:
        return any(not ch.isalpha() for ch in s)

    return {
        h.strip().lower()
        for s in fsm.layer1.states
        for h in s.resource_hints
        if h and h.strip()
        and len(h.strip()) > 2
        and _is_distinctive(h.strip())
    }


def _collect_app_specific_state_ids(fsm: FSM) -> set[str]:
    """State ids that contain the app name as a substring (lowercase).

    Generic ids like ``home``, ``settings``, ``editor`` are intentionally
    NOT included — they're legitimate Layer 2 vocabulary.
    """
    app_lower = fsm.layer1.app.lower()
    return {
        s.id.lower()
        for s in fsm.layer1.states
        if s.id and app_lower in s.id.lower()
    }


def _walk_category_text(cat: AbstractCategory):
    """Yield (field_path_for_error_msg, text) for every text in a category."""
    yield "name", cat.name
    yield "precondition", cat.precondition
    for i, step in enumerate(cat.abstract_steps):
        yield f"abstract_steps[{i}]", step
    for i, fm in enumerate(cat.failure_modes):
        yield f"failure_modes[{i}]", fm
    for i, vc in enumerate(cat.verification_checklist):
        yield f"verification_checklist[{i}]", vc


def _check_field(
    *,
    cat_name: str,
    field_path: str,
    text: str,
    app_variants: list[str],
    resource_hints: set[str],
    app_specific_state_ids: set[str],
) -> list[str]:
    """Run all rules against one text field. Returns 0+ error lines."""
    if not text:
        return []
    errors: list[str] = []

    # Rule 1: app name (with variants)
    for variant in app_variants:
        if _word_contains(text, variant):
            errors.append(_format_error(
                cat_name, field_path, "app name", variant, text,
            ))
            break  # one app-name violation per field is enough; don't spam

    # Rule 2: literal resource_hints — word-bounded, case-insensitive.
    # Category names are exempt (they are Layer-2 vocabulary keys like
    # ``DELETE_ENTRY`` / ``REMOVE_ENTRY`` that can legitimately contain
    # short English words overlapping with app resource-id fragments).
    if field_path != "name":
        for hint in resource_hints:
            if re.search(r"\b" + re.escape(hint) + r"\b", text, re.IGNORECASE):
                errors.append(_format_error(
                    cat_name, field_path, "resource hint", hint, text,
                ))

    # Rule 3: package-name pattern
    for m in _PACKAGE_RE.finditer(text):
        errors.append(_format_error(
            cat_name, field_path, "package name", m.group(0), text,
        ))

    # Rule 4: app-specific state ids
    for sid in app_specific_state_ids:
        if _word_contains(text, sid):
            errors.append(_format_error(
                cat_name, field_path, "app-specific state id", sid, text,
            ))

    return errors


def _word_contains(text: str, target: str) -> bool:
    """Word-bounded case-insensitive substring check.

    Accepts targets that contain spaces (multi-word variants like
    ``"simple calendar pro"``); the boundary is enforced at the start
    and end of the whole target, not at each space.
    """
    if not target:
        return False
    # \b doesn't fire cleanly between '_' and a word char (both are word chars
    # to the regex engine), so for word-bounded match we use lookarounds that
    # treat ANY non-word char as a boundary.
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(target) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, text, re.IGNORECASE) is not None


def _format_error(cat_name: str, field_path: str, rule_kind: str,
                  found: str, in_text: str) -> str:
    """Render one error line in the format Linqiang asked for."""
    snippet = in_text if len(in_text) <= 100 else in_text[:97] + "..."
    return (
        f'FAIL: category="{cat_name}", field="{field_path}", '
        f'found {rule_kind} "{found}" in "{snippet}"'
    )


def lint_L_C(
    merged_layer2: Layer2, source_fsms: list[FSM],
) -> tuple[bool, list[str]]:
    """Lint a merged L_C against the specifics of every source app.

    Story 2.3. After ``aggregate_L_C`` produces a unified Layer 2 for a
    Play Store category, this check guarantees that *none* of the
    contributing apps' specifics (names, resource ids, app-specific
    state ids) have survived the merge. A single-app leak is enough to
    fail — we return the union of all per-app violations.

    Args:
        merged_layer2: the aggregated L_C block.
        source_fsms: every FSM that contributed to the merge. The
            linter collects app-name variants, resource hints, and
            app-specific state ids from each, and checks ``merged_layer2``
            against the union.

    Returns:
        ``(passed, errors)`` with the same error-line format as
        :func:`lint_layer2`. Errors are deduplicated across apps so a
        leak visible to multiple source apps is reported once.
    """
    errors: list[str] = []

    # Per-app app-name variants
    variants_per_app: list[tuple[str, list[str]]] = [
        (fsm.layer1.app, _app_name_variants(fsm.layer1.app))
        for fsm in source_fsms
    ]
    # Union of resource hints (length filter applied in _collect_resource_hints)
    all_hints: set[str] = set()
    for fsm in source_fsms:
        all_hints |= _collect_resource_hints(fsm)
    # Per-app app-specific state ids (each app's state ids that contain its own name)
    state_ids_per_app: list[set[str]] = [
        _collect_app_specific_state_ids(fsm) for fsm in source_fsms
    ]

    for cat in merged_layer2.categories:
        for field_path, text in _walk_category_text(cat):
            if not text:
                continue

            # Rule 1: app name — report one violation per field per app
            # (don't spam the same text with every variant of one app)
            for app_name, variants in variants_per_app:
                for variant in variants:
                    if _word_contains(text, variant):
                        errors.append(_format_error(
                            cat.name, field_path, "app name", variant, text,
                        ))
                        break

            # Rule 2: resource hints — word-bounded, skip category.name field
            if field_path != "name":
                for hint in all_hints:
                    if re.search(
                        r"\b" + re.escape(hint) + r"\b", text, re.IGNORECASE,
                    ):
                        errors.append(_format_error(
                            cat.name, field_path, "resource hint", hint, text,
                        ))

            # Rule 3: package-name pattern
            for m in _PACKAGE_RE.finditer(text):
                errors.append(_format_error(
                    cat.name, field_path, "package name", m.group(0), text,
                ))

            # Rule 4: app-specific state ids (each app's set)
            for sids in state_ids_per_app:
                for sid in sids:
                    if _word_contains(text, sid):
                        errors.append(_format_error(
                            cat.name, field_path, "app-specific state id", sid, text,
                        ))

    # Dedup while preserving first-occurrence order.
    seen: set[str] = set()
    deduped: list[str] = []
    for e in errors:
        if e not in seen:
            deduped.append(e)
            seen.add(e)

    return (len(deduped) == 0, deduped)


__all__ = [
    "KNOWN_COMPOUND_VARIANTS",
    "lint_layer2",
    "lint_L_C",
]
