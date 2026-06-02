"""Two-layer FSM schema + JSON serializer + prompt-text formatter.

Story 2.2.1. The schema mirrors the spec in `plan/algorithm_design.md` §3.7
plus the small structural additions Linqiang requested:

  * `State.description`        — human-readable label alongside the id
  * `Transition.precondition / postcondition` — optional guards
  * `Layer1.strategies`        — named multi-step playbooks distilled
                                  from successful trajectories
  * `Layer1.dead_ends`         — recorded failure-state/action pairs
                                  from failed trajectories

Three serialization formats are supported:

  1. **Python dataclass instances** — what mutation operators / RL
     code edits in memory.
  2. **JSON dict** (`FSM.to_json()` / `FSM.from_json()`) — what gets
     written to disk in Story 1.3's snapshot directories. Round-trips
     exactly: ``FSM.from_json(fsm.to_json()) == fsm``.
  3. **Prompt text** (`FSM.to_prompt_text()`) — the structured
     human-readable block injected into the agent's system prompt.
     Format is byte-stable so two prompts built from the same FSM
     produce identical KV-cache prefixes.

Versioning: the top-level FSM carries a `version` field set to
``SCHEMA_VERSION`` ("0.1.0"). `from_json` rejects unknown major versions
loudly so we don't silently drift after a future schema bump.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


# Bump the MAJOR portion when the wire format changes incompatibly;
# bump MINOR for additive changes (new optional fields).
SCHEMA_VERSION = "0.1.0"


class SchemaVersionError(ValueError):
    """Raised when ``from_json`` sees a version it doesn't know how to read."""


# ─────────────────────────────────────────────────────────────────────────
# LAYER 1 — APP_SPECIFIC (non-transferable)
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class State:
    """A concrete UI state of one app (e.g. "the note editor screen of Markor").

    Attributes:
        id: Stable identifier used by transitions (e.g. ``"note_editor"``).
        description: Human-readable one-liner.
        visual_cues: Strings describing what the screen looks like — used
            by the agent to recognize the state from a screenshot.
        resource_hints: Optional Android `resource-id` substrings that, if
            present in the a11y tree, strongly indicate this state.
            Optional because not every app exposes stable resource ids.
    """

    id: str
    description: str = ""
    visual_cues: list[str] = field(default_factory=list)
    resource_hints: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "visual_cues": list(self.visual_cues),
            "resource_hints": list(self.resource_hints),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "State":
        _require_keys(data, {"id"}, "State")
        return cls(
            id=data["id"],
            description=data.get("description", ""),
            visual_cues=list(data.get("visual_cues", [])),
            resource_hints=list(data.get("resource_hints", [])),
        )


@dataclass
class Transition:
    """A directed edge ``from_state --action--> to_state`` in LAYER 1."""

    from_state: str
    to_state: str
    action: str
    precondition: str = ""
    postcondition: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "from_state": self.from_state,
            "to_state": self.to_state,
            "action": self.action,
            "precondition": self.precondition,
            "postcondition": self.postcondition,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Transition":
        _require_keys(data, {"from_state", "to_state", "action"}, "Transition")
        return cls(
            from_state=data["from_state"],
            to_state=data["to_state"],
            action=data["action"],
            precondition=data.get("precondition", ""),
            postcondition=data.get("postcondition", ""),
        )


@dataclass
class Strategy:
    """A named multi-step playbook for accomplishing one user-level goal.

    Distilled from successful trajectories during FSM synthesis. Reused
    by the agent at runtime when the goal text matches the strategy's
    intent.
    """

    name: str
    preconditions: str
    steps: list[str]
    success_signal: str
    fallback: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "preconditions": self.preconditions,
            "steps": list(self.steps),
            "success_signal": self.success_signal,
            "fallback": self.fallback,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Strategy":
        _require_keys(data, {"name", "preconditions", "success_signal"}, "Strategy")
        return cls(
            name=data["name"],
            preconditions=data["preconditions"],
            steps=list(data.get("steps", [])),
            success_signal=data["success_signal"],
            fallback=data.get("fallback", ""),
        )


@dataclass
class Layer1:
    """Everything tied to one app's visual/structural identity.

    Discarded on hand-off; not transferable.
    """

    app: str
    category: str  # Play Store category (e.g. "Productivity"), used to look up Layer 2 row
    states: list[State] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    strategies: list[Strategy] = field(default_factory=list)
    dead_ends: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "app": self.app,
            "category": self.category,
            "states": [s.to_json() for s in self.states],
            "transitions": [t.to_json() for t in self.transitions],
            "strategies": [s.to_json() for s in self.strategies],
            "dead_ends": [dict(d) for d in self.dead_ends],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Layer1":
        _require_keys(data, {"app", "category"}, "Layer1")
        return cls(
            app=data["app"],
            category=data["category"],
            states=[State.from_json(s) for s in data.get("states", [])],
            transitions=[Transition.from_json(t) for t in data.get("transitions", [])],
            strategies=[Strategy.from_json(s) for s in data.get("strategies", [])],
            dead_ends=[dict(d) for d in data.get("dead_ends", [])],
        )


# ─────────────────────────────────────────────────────────────────────────
# LAYER 2 — GENERIC (transferable, app-agnostic)
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class AbstractCategory:
    """One transferable workflow archetype keyed by play_category.

    The strings in here MUST NOT mention concrete app names, resource
    ids, or visual specifics. Linter (Story 2.2.3) enforces this.
    """

    name: str
    precondition: str
    abstract_steps: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    verification_checklist: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "precondition": self.precondition,
            "abstract_steps": list(self.abstract_steps),
            "failure_modes": list(self.failure_modes),
            "verification_checklist": list(self.verification_checklist),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "AbstractCategory":
        _require_keys(data, {"name", "precondition"}, "AbstractCategory")
        return cls(
            name=data["name"],
            precondition=data["precondition"],
            abstract_steps=list(data.get("abstract_steps", [])),
            failure_modes=list(data.get("failure_modes", [])),
            verification_checklist=list(data.get("verification_checklist", [])),
        )


@dataclass
class Layer2:
    """Container for the transferable abstract categories of one FSM."""

    categories: list[AbstractCategory] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"categories": [c.to_json() for c in self.categories]}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Layer2":
        return cls(
            categories=[AbstractCategory.from_json(c) for c in data.get("categories", [])],
        )

    # ── Prompt text ───────────────────────────────────────────────

    def to_prompt_text(self, category: str = "") -> str:
        """Render the LAYER-2 block as prompt-ready text.

        Used both from :meth:`FSM.to_prompt_text` (embedded inside the
        full FSM render) and standalone when injecting a per-category
        ``L_C`` into the agent prompt on Tier-B / Tier-C hand-off.

        Args:
            category: optional Play Store category label. When non-empty
                (L_C hand-off path), an extra ``L_C CATEGORY: <name>``
                line is emitted right after the LAYER-2 header so the
                agent knows which category's abstractions are attached.
                Default ``""`` keeps the output byte-identical to the
                previous inline rendering in ``FSM.to_prompt_text``.

        Output is deterministic: byte-stable for identical input
        (required for KV-cache reuse).
        """
        out: list[str] = []
        out.append("# ═══════════════════════════════════════════════")
        out.append("# LAYER 2: GENERIC  (transferable, app-agnostic)")
        out.append("# ═══════════════════════════════════════════════")
        if category:
            out.append(f"L_C CATEGORY: {category}")
        if not self.categories:
            out.append("(no abstract categories yet)")
        else:
            for c_i, c in enumerate(self.categories):
                if c_i > 0:
                    out.append("")
                out.append(f"CATEGORY: {c.name}")
                out.append(f"  precondition: {c.precondition}")
                if c.abstract_steps:
                    out.append("  abstract_steps:")
                    for j, step in enumerate(c.abstract_steps, start=1):
                        out.append(f"    {j}. {step}")
                if c.failure_modes:
                    out.append("  failure_modes:")
                    for fm in c.failure_modes:
                        out.append(f"    - {fm!r}")
                if c.verification_checklist:
                    out.append("  verification_checklist:")
                    for vc in c.verification_checklist:
                        out.append(f"    - {vc!r}")
        return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────
# Top-level FSM
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class FSM:
    """One per-app FSM ``F^0_a`` (and later F_a, F_a*).

    `metadata` is a free-form dict for provenance / metrics — e.g.
    ``{"built_at": "2026-04-18T...", "n_episodes": 30, "sr": 0.42}``.
    Stored verbatim, not validated.
    """

    app: str
    layer1: Layer1
    layer2: Layer2
    version: str = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── JSON ──────────────────────────────────────────────────────

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "app": self.app,
            "layer1": self.layer1.to_json(),
            "layer2": self.layer2.to_json(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "FSM":
        _require_keys(data, {"version", "app", "layer1", "layer2"}, "FSM")
        version = data["version"]
        _check_version_compatible(version)
        return cls(
            version=version,
            app=data["app"],
            layer1=Layer1.from_json(data["layer1"]),
            layer2=Layer2.from_json(data["layer2"]),
            metadata=dict(data.get("metadata", {})),
        )

    # ── Prompt text (matches §3.7 layout) ─────────────────────────

    def to_prompt_text(self) -> str:
        """Render the FSM as the structured text block defined in §3.7.

        Output is deterministic: byte-stable for identical input. Suitable
        for direct concatenation into the agent's system prompt; safe for
        KV-cache reuse across episodes that share the same FSM.
        """
        out: list[str] = []

        # ── LAYER 1 header ──────────────────────────────────────
        out.append("# ═══════════════════════════════════════════════")
        out.append("# LAYER 1: APP_SPECIFIC  (non-transferable)")
        out.append("# ═══════════════════════════════════════════════")
        out.append(f"APP: {self.layer1.app}")
        out.append(f"CATEGORY: {self.layer1.category}")

        # STATES
        out.append("STATES:")
        if not self.layer1.states:
            out.append("  (none)")
        else:
            for i, s in enumerate(self.layer1.states):
                header = f"  S{i}: {s.id.upper()}"
                if s.description:
                    header += f" — {s.description}"
                out.append(header)
                if s.visual_cues:
                    out.append(f"    visual_cues: {_render_str_list(s.visual_cues)}")
                if s.resource_hints:
                    out.append(f"    resource_hints: {_render_str_list(s.resource_hints)}")

        # TRANSITIONS
        out.append("TRANSITIONS:")
        if not self.layer1.transitions:
            out.append("  (none)")
        else:
            id_to_idx = {s.id: i for i, s in enumerate(self.layer1.states)}
            for t in self.layer1.transitions:
                src = f"S{id_to_idx[t.from_state]}" if t.from_state in id_to_idx else t.from_state
                dst = f"S{id_to_idx[t.to_state]}" if t.to_state in id_to_idx else t.to_state
                line = f"  {src} --{t.action}--> {dst}"
                if t.precondition:
                    line += f"  [pre: {t.precondition}]"
                if t.postcondition:
                    line += f"  [post: {t.postcondition}]"
                out.append(line)

        # STRATEGIES (extension over §3.7's example)
        out.append("STRATEGIES:")
        if not self.layer1.strategies:
            out.append("  (none)")
        else:
            for st in self.layer1.strategies:
                out.append(f"  {st.name}:")
                if st.preconditions:
                    out.append(f"    preconditions: {st.preconditions}")
                if st.steps:
                    out.append("    steps:")
                    for j, step in enumerate(st.steps, start=1):
                        out.append(f"      {j}. {step}")
                out.append(f"    success_signal: {st.success_signal}")
                if st.fallback:
                    out.append(f"    fallback: {st.fallback}")

        # DEAD_ENDS (extension)
        out.append("DEAD_ENDS:")
        if not self.layer1.dead_ends:
            out.append("  (none)")
        else:
            for de in self.layer1.dead_ends:
                state = de.get("state", "?")
                action = de.get("failed_action", "?")
                note = de.get("note", "")
                tail = f"  [{note}]" if note else ""
                out.append(f"  at {state}: {action} fails{tail}")

        out.append("")

        # ── LAYER 2 — delegate to Layer2.to_prompt_text (no category
        # tag: we don't need the L_C-style header inside a full FSM
        # render, and the default keeps byte-output stable vs pre-refactor).
        out.append(self.layer2.to_prompt_text())

        return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────
# Helpers (module-private)
# ─────────────────────────────────────────────────────────────────────────


def _require_keys(data: dict[str, Any], required: set[str], cls_name: str) -> None:
    """Raise ``ValueError`` listing missing required keys."""
    if not isinstance(data, dict):
        raise ValueError(f"{cls_name}.from_json expected a dict, got {type(data).__name__}")
    missing = required - set(data.keys())
    if missing:
        raise ValueError(
            f"{cls_name}.from_json missing required field(s): "
            f"{sorted(missing)}"
        )


def _check_version_compatible(version: str) -> None:
    """Reject FSM JSON whose major version differs from ``SCHEMA_VERSION``.

    Minor version drift is accepted (additive-only changes); the unknown
    fields just get ignored by the per-class ``from_json``s above (they
    use ``data.get(...)`` everywhere).
    """
    if not isinstance(version, str):
        raise SchemaVersionError(
            f"FSM version must be a string, got {type(version).__name__}"
        )
    parts = version.split(".")
    if len(parts) < 2:
        raise SchemaVersionError(
            f"FSM version must be 'MAJOR.MINOR[.PATCH]', got {version!r}"
        )
    try:
        loaded_major = int(parts[0])
    except ValueError as e:
        raise SchemaVersionError(
            f"FSM version major component must be an int, got {parts[0]!r}"
        ) from e
    current_major = int(SCHEMA_VERSION.split(".")[0])
    if loaded_major != current_major:
        raise SchemaVersionError(
            f"FSM JSON is version {version!r} but this code only reads major "
            f"version {current_major} (current SCHEMA_VERSION={SCHEMA_VERSION!r}). "
            "Migrate the JSON or upgrade the schema module."
        )


def _render_str_list(xs: list[str]) -> str:
    """Render a list of strings as a JSON-ish array, deterministically."""
    return "[" + ", ".join(repr(x) for x in xs) + "]"
