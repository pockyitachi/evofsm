"""FSM diff format + parser + applier — Story 3.2 (Epic 3 / B3).

The evolution loop (Story 3.5) asks Claude to mutate an FSM by emitting a
structured diff rather than rewriting the FSM from scratch. A diff is a
small list of atomic edit operations against the two-layer FSM schema,
plus a short natural-language reflection summary explaining why.

This module owns three concerns, in order:

  1. :class:`DiffOp` / :class:`FSMDiff` — the schema and JSON round-trip
     (with tolerant parsing for common LLM output quirks).
  2. :func:`apply_diff` — deterministic application onto a deep-copied
     FSM, returning an :class:`ApplyResult` that lists the ops that
     landed, the ops that were skipped (with reasons), and non-fatal
     warnings. The original FSM is never mutated.
  3. :func:`validate_fsm_integrity` — post-apply structural check that
     flags dangling transitions, duplicate ids, empty abstract steps,
     etc. Returns a list of human-readable warnings (empty = clean).

Plus two helpers:
  - :func:`parse_diff_from_llm_response` — extract + parse JSON from a
    raw Claude response (tries bare JSON, fenced ``` ```json``` block,
    then the first balanced top-level ``{...}``).
  - :data:`DIFF_JSON_SCHEMA` — the JSON-Schema description surfaced to
    the mutation prompt so Claude knows what shape to emit.

Design principles
-----------------
- **Immutability for the caller.** ``apply_diff`` deep-copies the input
  FSM and mutates the copy. A single ``ApplyResult`` object reports
  everything the caller needs.
- **Lenient parsing, strict application.** The ``FSMDiff.from_json``
  path accepts reasonable LLM quirks (case-insensitive enums, plural
  target names, extra keys). The applier, in contrast, refuses to
  invent data: a malformed op is *skipped*, not silently treated as
  something else — the sole exception is "modify non-existent" which
  we upgrade to "add" because that's the most common LLM mistake and
  it's still semantically safe.
- **No hidden I/O.** No file access, no network calls, no global state.
  Pure transformation of `FSM + FSMDiff -> FSM`.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import re
from typing import Any, Literal

from evofsm_rl.fsm.schema import (
    AbstractCategory,
    FSM,
    Layer1,
    Layer2,
    State,
    Strategy,
    Transition,
)


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Types + constants
# ─────────────────────────────────────────────────────────────────────

Layer = Literal["layer1", "layer2"]
L1Target = Literal["state", "transition", "strategy", "dead_end"]
L2Target = Literal["category"]
OpKind = Literal["add", "modify", "remove"]


_VALID_LAYERS: set[str] = {"layer1", "layer2"}
_VALID_OPS: set[str] = {"add", "modify", "remove"}

# Canonical targets, plus tolerated singular/plural/casing synonyms.
_TARGET_SYNONYMS: dict[str, str] = {
    "state": "state",
    "states": "state",
    "transition": "transition",
    "transitions": "transition",
    "strategy": "strategy",
    "strategies": "strategy",
    "dead_end": "dead_end",
    "dead_ends": "dead_end",
    "deadend": "dead_end",
    "deadends": "dead_end",
    "category": "category",
    "categories": "category",
}

# Which targets belong to which layer (caught at parse time).
_LAYER_TARGETS: dict[str, set[str]] = {
    "layer1": {"state", "transition", "strategy", "dead_end"},
    "layer2": {"category"},
}


DIFF_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ops": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "layer": {
                        "type": "string",
                        "enum": ["layer1", "layer2"],
                    },
                    "target": {
                        "type": "string",
                        "enum": [
                            "state", "transition", "strategy",
                            "dead_end", "category",
                        ],
                    },
                    "op": {
                        "type": "string",
                        "enum": ["add", "modify", "remove"],
                    },
                    "key": {"type": "string"},
                    "value": {"type": "object"},
                },
                "required": ["layer", "target", "op", "key"],
            },
        },
        "reflection_summary": {"type": "string"},
        "layer_tag": {
            "type": "string",
            "enum": ["layer1", "layer2", "both"],
        },
    },
    "required": ["ops"],
}


# ─────────────────────────────────────────────────────────────────────
# DiffOp / FSMDiff — schema + JSON
# ─────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class DiffOp:
    """One atomic edit operation on an FSM.

    Attributes:
        layer: ``"layer1"`` or ``"layer2"``.
        target: Element kind within the layer. Layer 1 accepts
            ``state`` / ``transition`` / ``strategy`` / ``dead_end``;
            Layer 2 accepts ``category``.
        op: ``"add"``, ``"modify"``, or ``"remove"``.
        key: Identifier for the element the op targets. Conventions:
            ``state`` — state id (``"home"``); ``transition`` —
            ``"from_state->to_state"`` (``"home->editor"``);
            ``strategy`` — strategy name; ``dead_end`` — caller-chosen
            identifier (typically a short label); ``category`` —
            category name (``"ADD_ENTRY"``).
        value: Payload for ``add`` / ``modify`` ops. ``None`` for
            ``remove``. For ``modify`` with patch semantics, only the
            keys present are updated; list-valued fields
            (``visual_cues``, ``steps``, ``failure_modes``, ...) are
            *replaced wholesale*, not merged.
    """

    layer: Layer
    target: L1Target | L2Target
    op: OpKind
    key: str
    value: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "layer": self.layer,
            "target": self.target,
            "op": self.op,
            "key": self.key,
        }
        if self.value is not None:
            out["value"] = _deep_jsonable(self.value)
        return out


@dataclasses.dataclass(frozen=True)
class FSMDiff:
    """A batch of :class:`DiffOp` to apply atomically onto an FSM.

    Attributes:
        ops: Ordered list of atomic edits. Applied in order by
            :func:`apply_diff`; an op that fails (e.g. missing value)
            is reported in ``ApplyResult.skipped`` but does not abort
            subsequent ops.
        reflection_summary: Short natural-language rationale, usually
            provided by the mutation LLM. Stored verbatim; not used by
            the applier.
        layer_tag: Hint for the caller's bookkeeping: which layer(s)
            this diff touches. Not enforced against ``ops``.
    """

    ops: list[DiffOp]
    reflection_summary: str = ""
    layer_tag: Literal["layer1", "layer2", "both"] = "both"

    def to_json(self) -> dict[str, Any]:
        return {
            "ops": [op.to_json() for op in self.ops],
            "reflection_summary": self.reflection_summary,
            "layer_tag": self.layer_tag,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "FSMDiff":
        """Parse a diff from JSON, tolerating common LLM quirks.

        Tolerated quirks:
          * Case-insensitive ``op`` / ``layer`` / ``target`` values.
          * Pluralized targets (``"states"`` → ``"state"``,
            ``"categories"`` → ``"category"``, ...).
          * Missing ``reflection_summary`` and/or ``layer_tag``.
          * Unknown top-level keys are ignored silently.

        Skipped (never raised):
          * Ops with unknown / mistyped ``layer`` / ``target`` / ``op``.
          * ``add`` / ``modify`` ops lacking a ``value``.
          * Targets that don't belong to the stated layer
            (e.g. ``layer2 / state``).

        Every skipped op is logged at WARNING level so the caller can
        surface it if they want to.
        """
        if not isinstance(data, dict):
            raise ValueError(
                f"FSMDiff.from_json: expected a dict, got {type(data).__name__}"
            )

        raw_ops = data.get("ops", [])
        if not isinstance(raw_ops, list):
            raise ValueError(
                f"FSMDiff.from_json: 'ops' must be a list, "
                f"got {type(raw_ops).__name__}"
            )

        ops: list[DiffOp] = []
        for i, raw in enumerate(raw_ops):
            parsed = _parse_one_op(raw, index=i)
            if parsed is not None:
                ops.append(parsed)

        reflection_summary = str(data.get("reflection_summary", "") or "")
        layer_tag_raw = str(data.get("layer_tag", "both") or "both").lower()
        if layer_tag_raw not in ("layer1", "layer2", "both"):
            logger.warning(
                "FSMDiff.from_json: unknown layer_tag %r, defaulting to 'both'",
                layer_tag_raw,
            )
            layer_tag_raw = "both"

        return cls(
            ops=ops,
            reflection_summary=reflection_summary,
            layer_tag=layer_tag_raw,  # type: ignore[arg-type]
        )


def _parse_one_op(raw: Any, *, index: int) -> DiffOp | None:
    """Best-effort parse of one op dict. Returns None on any skip path."""
    if not isinstance(raw, dict):
        logger.warning(
            "FSMDiff.from_json: op #%d is not an object, skipping: %r",
            index, raw,
        )
        return None

    layer = str(raw.get("layer", "")).strip().lower()
    target_raw = str(raw.get("target", "")).strip().lower()
    op = str(raw.get("op", "")).strip().lower()
    key = raw.get("key", None)

    if layer not in _VALID_LAYERS:
        logger.warning(
            "FSMDiff.from_json: op #%d has unknown layer %r, skipping",
            index, layer,
        )
        return None

    target = _TARGET_SYNONYMS.get(target_raw)
    if target is None:
        logger.warning(
            "FSMDiff.from_json: op #%d has unknown target %r, skipping",
            index, target_raw,
        )
        return None

    if target not in _LAYER_TARGETS[layer]:
        logger.warning(
            "FSMDiff.from_json: op #%d target %r not valid for %s, skipping",
            index, target, layer,
        )
        return None

    if op not in _VALID_OPS:
        logger.warning(
            "FSMDiff.from_json: op #%d has unknown op %r, skipping",
            index, op,
        )
        return None

    if not isinstance(key, str) or not key:
        logger.warning(
            "FSMDiff.from_json: op #%d has missing/non-string key, skipping",
            index,
        )
        return None

    value = raw.get("value", None)
    if value is not None and not isinstance(value, dict):
        logger.warning(
            "FSMDiff.from_json: op #%d has non-dict value, skipping",
            index,
        )
        return None

    if op in ("add", "modify") and value is None:
        logger.warning(
            "FSMDiff.from_json: op #%d (%s/%s/%s) missing 'value', "
            "skipping at parse time",
            index, layer, target, op,
        )
        return None

    return DiffOp(
        layer=layer,  # type: ignore[arg-type]
        target=target,  # type: ignore[arg-type]
        op=op,  # type: ignore[arg-type]
        key=key,
        value=value,
    )


def _deep_jsonable(value: Any) -> Any:
    """Deep-copy and drop any non-JSON-able content gracefully."""
    if isinstance(value, dict):
        return {k: _deep_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_jsonable(v) for v in value]
    return value


# ─────────────────────────────────────────────────────────────────────
# apply_diff
# ─────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class ApplyResult:
    """Outcome of :func:`apply_diff`.

    Attributes:
        fsm: The mutated FSM (always a fresh deep copy — the caller's
            input is untouched).
        applied: DiffOps that landed successfully, in application order.
        skipped: ``(op, reason)`` pairs for ops the applier couldn't
            or wouldn't apply (e.g. remove of a non-existent element).
        warnings: Non-fatal messages about ops that applied with caveats
            (e.g. "modify on a non-existent state — treated as add").
    """

    fsm: FSM
    applied: list[DiffOp] = dataclasses.field(default_factory=list)
    skipped: list[tuple[DiffOp, str]] = dataclasses.field(default_factory=list)
    warnings: list[str] = dataclasses.field(default_factory=list)


def apply_diff(fsm: FSM, diff: FSMDiff) -> ApplyResult:
    """Apply ``diff`` to ``fsm`` and return a fresh FSM + a trace.

    The input FSM is deep-copied up-front; the copy is mutated and
    returned in ``ApplyResult.fsm``. No mutation of ``fsm``.

    Each op is dispatched to a per-target handler:

    ============  ====================================================
    target        handler semantics
    ============  ====================================================
    state         add/modify/remove by ``key`` = state id. ``remove``
                  also deletes transitions that reference the state.
    transition    add/modify/remove by ``key`` = ``"from->to"``. Value
                  wins over key when both disagree.
    strategy      add/modify/remove by ``key`` = strategy name.
    dead_end      add/modify/remove by ``key`` = caller label; match
                  on (``state``, ``failed_action``) fields for
                  modify/remove.
    category      add/modify/remove on Layer 2 by ``key`` = category
                  name.
    ============  ====================================================

    For ``modify`` ops the value is treated as a *patch*: only fields
    present in the value are written onto the target element; list
    fields are replaced wholesale (no per-item merge).
    """
    new_fsm = copy.deepcopy(fsm)
    result = ApplyResult(fsm=new_fsm)

    for op in diff.ops:
        try:
            _dispatch_op(new_fsm, op, result)
        except _SkipOp as exc:
            result.skipped.append((op, str(exc)))
            logger.warning(
                "apply_diff skipped %s/%s/%s key=%r: %s",
                op.layer, op.target, op.op, op.key, exc,
            )

    return result


class _SkipOp(Exception):
    """Raised by handlers to abort a single op cleanly."""


def _dispatch_op(new_fsm: FSM, op: DiffOp, result: ApplyResult) -> None:
    """Route one op to the appropriate handler."""
    if op.layer == "layer1":
        if op.target == "state":
            _apply_state_op(new_fsm.layer1, op, result)
        elif op.target == "transition":
            _apply_transition_op(new_fsm.layer1, op, result)
        elif op.target == "strategy":
            _apply_strategy_op(new_fsm.layer1, op, result)
        elif op.target == "dead_end":
            _apply_dead_end_op(new_fsm.layer1, op, result)
        else:
            raise _SkipOp(f"unknown layer1 target {op.target!r}")
    elif op.layer == "layer2":
        if op.target == "category":
            _apply_category_op(new_fsm.layer2, op, result)
        else:
            raise _SkipOp(f"unknown layer2 target {op.target!r}")
    else:
        raise _SkipOp(f"unknown layer {op.layer!r}")


# ── Per-target handlers ──────────────────────────────────────────────


def _apply_state_op(layer1: Layer1, op: DiffOp, result: ApplyResult) -> None:
    idx = _find_index(layer1.states, lambda s: s.id == op.key)

    if op.op == "add":
        if op.value is None:
            raise _SkipOp("add/state requires 'value'")
        if idx is not None:
            result.warnings.append(
                f"state {op.key!r} already exists; add treated as modify"
            )
            _patch_state(layer1.states[idx], op.value)
        else:
            layer1.states.append(_state_from_value(op.key, op.value))
        result.applied.append(op)
        return

    if op.op == "modify":
        if op.value is None:
            raise _SkipOp("modify/state requires 'value'")
        if idx is None:
            result.warnings.append(
                f"state {op.key!r} not found for modify; treated as add"
            )
            layer1.states.append(_state_from_value(op.key, op.value))
        else:
            _patch_state(layer1.states[idx], op.value)
        result.applied.append(op)
        return

    if op.op == "remove":
        if idx is None:
            raise _SkipOp(f"state {op.key!r} not found for remove")
        removed_id = layer1.states[idx].id
        del layer1.states[idx]
        # Cascade: drop transitions that reference the removed state.
        before = len(layer1.transitions)
        layer1.transitions = [
            t for t in layer1.transitions
            if t.from_state != removed_id and t.to_state != removed_id
        ]
        dropped = before - len(layer1.transitions)
        if dropped:
            result.warnings.append(
                f"removed {dropped} transition(s) referencing "
                f"deleted state {removed_id!r}"
            )
        result.applied.append(op)
        return

    raise _SkipOp(f"unknown op {op.op!r}")


def _state_from_value(key: str, value: dict[str, Any]) -> State:
    """Materialize a :class:`State` from a diff value dict.

    The state id falls back to ``key`` if the value dict doesn't carry
    one — this is the common case since the op's ``key`` already is the
    state id.
    """
    return State(
        id=str(value.get("id", key)),
        description=str(value.get("description", "")),
        visual_cues=list(value.get("visual_cues", [])),
        resource_hints=list(value.get("resource_hints", [])),
    )


def _patch_state(state: State, value: dict[str, Any]) -> None:
    if "id" in value:
        state.id = str(value["id"])
    if "description" in value:
        state.description = str(value["description"])
    if "visual_cues" in value:
        state.visual_cues = list(value["visual_cues"])
    if "resource_hints" in value:
        state.resource_hints = list(value["resource_hints"])


# Transitions ----------------------------------------------------------

_TRANSITION_KEY_RE = re.compile(r"^(?P<from>[^>]+?)\s*->\s*(?P<to>[^>]+?)$")


def _split_transition_key(key: str) -> tuple[str, str] | None:
    """Parse ``"from->to"`` style keys. Returns ``None`` on failure."""
    m = _TRANSITION_KEY_RE.match(key)
    if not m:
        return None
    return m.group("from").strip(), m.group("to").strip()


def _find_transition_index(
    transitions: list[Transition],
    from_state: str,
    to_state: str,
) -> int | None:
    for i, t in enumerate(transitions):
        if t.from_state == from_state and t.to_state == to_state:
            return i
    return None


def _apply_transition_op(
    layer1: Layer1, op: DiffOp, result: ApplyResult,
) -> None:
    value = op.value or {}

    # Resolve identity from value if provided, else fall back to key.
    from_state = value.get("from_state") or None
    to_state = value.get("to_state") or None
    if from_state is None or to_state is None:
        parsed = _split_transition_key(op.key)
        if parsed is None:
            raise _SkipOp(
                f"transition key {op.key!r} is not in 'from->to' form and "
                f"value lacks from_state/to_state"
            )
        from_state = from_state or parsed[0]
        to_state = to_state or parsed[1]

    idx = _find_transition_index(layer1.transitions, from_state, to_state)

    if op.op == "add":
        if op.value is None or "action" not in op.value:
            raise _SkipOp("add/transition requires value with 'action'")
        if idx is not None:
            result.warnings.append(
                f"transition {from_state!r}->{to_state!r} exists; add "
                f"treated as modify"
            )
            _patch_transition(layer1.transitions[idx], op.value)
        else:
            state_ids = {s.id for s in layer1.states}
            if from_state not in state_ids or to_state not in state_ids:
                result.warnings.append(
                    f"transition {from_state!r}->{to_state!r} references "
                    f"unknown state(s); added anyway"
                )
            layer1.transitions.append(
                _transition_from_value(from_state, to_state, op.value)
            )
        result.applied.append(op)
        return

    if op.op == "modify":
        if op.value is None:
            raise _SkipOp("modify/transition requires 'value'")
        if idx is None:
            if "action" not in op.value:
                raise _SkipOp(
                    "modify/transition target missing and value has no "
                    "'action' to seed an add"
                )
            result.warnings.append(
                f"transition {from_state!r}->{to_state!r} not found for "
                f"modify; treated as add"
            )
            layer1.transitions.append(
                _transition_from_value(from_state, to_state, op.value)
            )
        else:
            _patch_transition(layer1.transitions[idx], op.value)
        result.applied.append(op)
        return

    if op.op == "remove":
        if idx is None:
            raise _SkipOp(
                f"transition {from_state!r}->{to_state!r} not found for remove"
            )
        del layer1.transitions[idx]
        result.applied.append(op)
        return

    raise _SkipOp(f"unknown op {op.op!r}")


def _transition_from_value(
    from_state: str, to_state: str, value: dict[str, Any],
) -> Transition:
    return Transition(
        from_state=str(from_state),
        to_state=str(to_state),
        action=str(value.get("action", "")),
        precondition=str(value.get("precondition", "")),
        postcondition=str(value.get("postcondition", "")),
    )


def _patch_transition(t: Transition, value: dict[str, Any]) -> None:
    if "from_state" in value:
        t.from_state = str(value["from_state"])
    if "to_state" in value:
        t.to_state = str(value["to_state"])
    if "action" in value:
        t.action = str(value["action"])
    if "precondition" in value:
        t.precondition = str(value["precondition"])
    if "postcondition" in value:
        t.postcondition = str(value["postcondition"])


# Strategies -----------------------------------------------------------


def _apply_strategy_op(
    layer1: Layer1, op: DiffOp, result: ApplyResult,
) -> None:
    idx = _find_index(layer1.strategies, lambda s: s.name == op.key)

    if op.op == "add":
        if op.value is None:
            raise _SkipOp("add/strategy requires 'value'")
        if idx is not None:
            result.warnings.append(
                f"strategy {op.key!r} exists; add treated as modify"
            )
            _patch_strategy(layer1.strategies[idx], op.value)
        else:
            layer1.strategies.append(_strategy_from_value(op.key, op.value))
        result.applied.append(op)
        return

    if op.op == "modify":
        if op.value is None:
            raise _SkipOp("modify/strategy requires 'value'")
        if idx is None:
            result.warnings.append(
                f"strategy {op.key!r} not found for modify; treated as add"
            )
            layer1.strategies.append(_strategy_from_value(op.key, op.value))
        else:
            _patch_strategy(layer1.strategies[idx], op.value)
        result.applied.append(op)
        return

    if op.op == "remove":
        if idx is None:
            raise _SkipOp(f"strategy {op.key!r} not found for remove")
        del layer1.strategies[idx]
        result.applied.append(op)
        return

    raise _SkipOp(f"unknown op {op.op!r}")


def _strategy_from_value(key: str, value: dict[str, Any]) -> Strategy:
    return Strategy(
        name=str(value.get("name", key)),
        preconditions=str(value.get("preconditions", "")),
        steps=list(value.get("steps", [])),
        success_signal=str(value.get("success_signal", "")),
        fallback=str(value.get("fallback", "")),
    )


def _patch_strategy(s: Strategy, value: dict[str, Any]) -> None:
    if "name" in value:
        s.name = str(value["name"])
    if "preconditions" in value:
        s.preconditions = str(value["preconditions"])
    if "steps" in value:
        s.steps = list(value["steps"])
    if "success_signal" in value:
        s.success_signal = str(value["success_signal"])
    if "fallback" in value:
        s.fallback = str(value["fallback"])


# Dead ends ------------------------------------------------------------


def _match_dead_end(
    dead_ends: list[dict[str, Any]], value: dict[str, Any] | None,
    key: str,
) -> int | None:
    """Find a dead_end by matching (state, failed_action) from value,
    falling back to ``key`` as the state token if value is empty."""
    if value:
        state = value.get("state", "")
        action = value.get("failed_action", "")
        for i, de in enumerate(dead_ends):
            if de.get("state", "") == state and de.get("failed_action", "") == action:
                return i
    # Fallback: match either field loosely against the key.
    for i, de in enumerate(dead_ends):
        if key in (de.get("state", ""), de.get("failed_action", "")):
            return i
    return None


def _apply_dead_end_op(
    layer1: Layer1, op: DiffOp, result: ApplyResult,
) -> None:
    if op.op == "add":
        if op.value is None:
            raise _SkipOp("add/dead_end requires 'value'")
        layer1.dead_ends.append(dict(op.value))
        result.applied.append(op)
        return

    if op.op == "modify":
        if op.value is None:
            raise _SkipOp("modify/dead_end requires 'value'")
        idx = _match_dead_end(layer1.dead_ends, op.value, op.key)
        if idx is None:
            result.warnings.append(
                f"dead_end {op.key!r} not found for modify; treated as add"
            )
            layer1.dead_ends.append(dict(op.value))
        else:
            layer1.dead_ends[idx].update(op.value)
        result.applied.append(op)
        return

    if op.op == "remove":
        idx = _match_dead_end(layer1.dead_ends, op.value, op.key)
        if idx is None:
            raise _SkipOp(f"dead_end {op.key!r} not found for remove")
        del layer1.dead_ends[idx]
        result.applied.append(op)
        return

    raise _SkipOp(f"unknown op {op.op!r}")


# Layer 2 categories ---------------------------------------------------


def _apply_category_op(
    layer2: Layer2, op: DiffOp, result: ApplyResult,
) -> None:
    idx = _find_index(layer2.categories, lambda c: c.name == op.key)

    if op.op == "add":
        if op.value is None:
            raise _SkipOp("add/category requires 'value'")
        if idx is not None:
            result.warnings.append(
                f"category {op.key!r} exists; add treated as modify"
            )
            _patch_category(layer2.categories[idx], op.value)
        else:
            layer2.categories.append(_category_from_value(op.key, op.value))
        result.applied.append(op)
        return

    if op.op == "modify":
        if op.value is None:
            raise _SkipOp("modify/category requires 'value'")
        if idx is None:
            result.warnings.append(
                f"category {op.key!r} not found for modify; treated as add"
            )
            layer2.categories.append(_category_from_value(op.key, op.value))
        else:
            _patch_category(layer2.categories[idx], op.value)
        result.applied.append(op)
        return

    if op.op == "remove":
        if idx is None:
            raise _SkipOp(f"category {op.key!r} not found for remove")
        del layer2.categories[idx]
        result.applied.append(op)
        return

    raise _SkipOp(f"unknown op {op.op!r}")


def _category_from_value(key: str, value: dict[str, Any]) -> AbstractCategory:
    return AbstractCategory(
        name=str(value.get("name", key)),
        precondition=str(value.get("precondition", "")),
        abstract_steps=list(value.get("abstract_steps", [])),
        failure_modes=list(value.get("failure_modes", [])),
        verification_checklist=list(value.get("verification_checklist", [])),
    )


def _patch_category(c: AbstractCategory, value: dict[str, Any]) -> None:
    if "name" in value:
        c.name = str(value["name"])
    if "precondition" in value:
        c.precondition = str(value["precondition"])
    if "abstract_steps" in value:
        c.abstract_steps = list(value["abstract_steps"])
    if "failure_modes" in value:
        c.failure_modes = list(value["failure_modes"])
    if "verification_checklist" in value:
        c.verification_checklist = list(value["verification_checklist"])


# ── Misc ─────────────────────────────────────────────────────────────


def _find_index(items: list[Any], predicate) -> int | None:
    for i, item in enumerate(items):
        if predicate(item):
            return i
    return None


# ─────────────────────────────────────────────────────────────────────
# Integrity checker
# ─────────────────────────────────────────────────────────────────────


def validate_fsm_integrity(fsm: FSM) -> list[str]:
    """Structural post-apply sanity checks.

    Returns a list of human-readable warnings. An empty list means the
    FSM is clean.

    Checks:
      1. Every transition's ``from_state`` / ``to_state`` refers to a
         declared state.
      2. Strategy names are non-empty and unique.
      3. Every Layer-2 category has at least one ``abstract_steps`` entry.
      4. State ids are unique.
      5. ``(from_state, to_state)`` pairs on transitions are unique.

    This function is advisory; the applier will happily produce an FSM
    that violates these checks (e.g. a forward-reference transition
    added before the target state). Call after a mutation round when
    you want to surface issues to the caller / user.
    """
    warnings: list[str] = []

    # 1 + 4: states
    state_ids = [s.id for s in fsm.layer1.states]
    seen: dict[str, int] = {}
    for sid in state_ids:
        seen[sid] = seen.get(sid, 0) + 1
    for sid, count in seen.items():
        if count > 1:
            warnings.append(f"duplicate state id: {sid!r} ({count} occurrences)")
    state_set = set(state_ids)

    # 1 + 5: transitions
    pair_counts: dict[tuple[str, str], int] = {}
    for t in fsm.layer1.transitions:
        if t.from_state not in state_set:
            warnings.append(
                f"transition from_state {t.from_state!r} "
                f"(-> {t.to_state!r}) does not exist in states"
            )
        if t.to_state not in state_set:
            warnings.append(
                f"transition to_state {t.to_state!r} "
                f"({t.from_state!r} ->) does not exist in states"
            )
        pair = (t.from_state, t.to_state)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    for pair, count in pair_counts.items():
        if count > 1:
            warnings.append(
                f"duplicate transition pair: {pair[0]!r} -> {pair[1]!r} "
                f"({count} occurrences)"
            )

    # 2: strategies
    strategy_names: list[str] = []
    for i, s in enumerate(fsm.layer1.strategies):
        if not s.name:
            warnings.append(f"strategy at index {i} has empty name")
            continue
        strategy_names.append(s.name)
    name_counts: dict[str, int] = {}
    for n in strategy_names:
        name_counts[n] = name_counts.get(n, 0) + 1
    for name, count in name_counts.items():
        if count > 1:
            warnings.append(
                f"duplicate strategy name: {name!r} ({count} occurrences)"
            )

    # 3: layer 2
    for c in fsm.layer2.categories:
        if not c.abstract_steps:
            warnings.append(
                f"layer2 category {c.name!r} has empty abstract_steps"
            )

    return warnings


# ─────────────────────────────────────────────────────────────────────
# LLM response extraction
# ─────────────────────────────────────────────────────────────────────


def parse_diff_from_llm_response(raw_text: str) -> FSMDiff:
    """Extract + parse an :class:`FSMDiff` from a raw LLM response.

    Tries, in order:
      1. The full text as bare JSON.
      2. The first ```json ... ``` fenced block.
      3. The first balanced ``{...}`` substring.

    Raises:
        ValueError if no valid JSON object can be found, with the head
        of the raw text included for debugging.
    """
    if not isinstance(raw_text, str):
        raise ValueError(
            f"parse_diff_from_llm_response: expected str, "
            f"got {type(raw_text).__name__}"
        )

    text = raw_text.strip()

    # Attempt 1: bare JSON.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return FSMDiff.from_json(obj)
    except json.JSONDecodeError:
        pass

    # Attempt 2: fenced ```json or plain ``` block.
    fence = re.search(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return FSMDiff.from_json(obj)
        except json.JSONDecodeError:
            pass

    # Attempt 3: first balanced top-level {...}.
    extracted = _first_balanced_object(text)
    if extracted is not None:
        try:
            obj = json.loads(extracted)
            if isinstance(obj, dict):
                return FSMDiff.from_json(obj)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"parse_diff_from_llm_response: found a {{...}} block but "
                f"it failed to parse as JSON: {e}. Response head: "
                f"{text[:200]!r}"
            ) from e

    raise ValueError(
        f"parse_diff_from_llm_response: could not find any JSON object "
        f"in response. Response head: {text[:200]!r}"
    )


def _first_balanced_object(text: str) -> str | None:
    """Return the first top-level balanced ``{...}`` substring, or None."""
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


__all__ = [
    "ApplyResult",
    "DiffOp",
    "FSMDiff",
    "DIFF_JSON_SCHEMA",
    "apply_diff",
    "parse_diff_from_llm_response",
    "validate_fsm_integrity",
]
