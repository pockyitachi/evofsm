"""App-name → L_C prompt-text resolver — Story B2.

When the B2 baseline driver iterates over the T_eval template list of a
held-out app, it needs to decide two things per app:

  1. Does this app's Play Store category appear in the source pool?
     (Tier-B ⇒ yes; Tier-C ⇒ no.)
  2. If yes, where is the corresponding ``artifacts/L_C/{slug}.json``?

Both questions are answered by :func:`resolve_l_c_for_app`, which
centralizes the lookup so the runner and the tests agree on the logic.
Source-pool apps are also accepted (useful for validation / debugging /
future self-transfer experiments); held-out apps whose category has no
L_C file (Tier-C) return ``None``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from evofsm_rl.fsm.aggregator import category_to_slug, load_L_C
from evofsm_rl.fsm.schema import FSM


def _find_category(splits_data: dict[str, Any], app_name: str) -> str | None:
    """Scan source_pool / tier_B_held_out / tier_C_held_out for ``app_name``.

    Returns the Play Store category string or ``None`` if the app is not
    registered in any of the three pools.
    """
    for pool_key in ("source_pool", "tier_B_held_out", "tier_C_held_out"):
        pool = splits_data.get(pool_key) or {}
        if app_name in pool:
            return pool[app_name].get("category")
    return None


def resolve_l_c_for_app(
    app_name: str,
    splits_yaml_path: str | Path,
    l_c_dir: str | Path,
) -> str | None:
    """Return the L_C prompt-text for one app, or ``None`` if unavailable.

    Args:
        app_name: canonical snake_case app key (must match the keys in
            ``configs/splits.yaml``).
        splits_yaml_path: path to ``configs/splits.yaml``.
        l_c_dir: directory holding ``{slug}.json`` files written by
            ``scripts/build_L_C.py``.

    Returns:
        - Non-empty prompt text (ready to splice into the agent's action
          prompt via ``agent.set_l_c_prompt_text``) when the app's
          category has a corresponding L_C file.
        - ``None`` when either (a) the app is unknown to splits.yaml,
          or (b) the app's category has no L_C file on disk (Tier-C
          fallthrough — B2 degrades to B1 for these).

    The returned string includes the ``L_C CATEGORY: <name>`` tag so
    the agent knows which category the transferred knowledge belongs to.
    """
    splits_path = Path(splits_yaml_path)
    with splits_path.open() as fh:
        splits_data = yaml.safe_load(fh)

    category = _find_category(splits_data, app_name)
    if category is None:
        return None

    slug = category_to_slug(category)
    lc_path = Path(l_c_dir) / f"{slug}.json"
    if not lc_path.exists():
        # Tier-C categories (no source-pool coverage) won't have a file
        # in artifacts/L_C/. That's the designed B2 degradation path.
        return None

    _, layer2 = load_L_C(lc_path)
    return layer2.to_prompt_text(category=category)


def resolve_app_guidance(
    app_name: str,
    splits_yaml_path: str | Path,
    l_c_dir: str | Path,
    fsm_dir: str | Path | None = None,
) -> tuple[str | None, str]:
    """Three-tier resolver for the symbolic guidance to inject for one app.

    Tier order (most → least specific):
      1. ``app``  — a per-app static FSM exists at ``{fsm_dir}/{app}.json``
         ⇒ inject the FULL app FSM (Layer-1 states/transitions/strategies/
         dead_ends + the app's own Layer-2). Most specific knowledge.
      2. ``category`` — no app FSM, but the app's Play-category has an L_C
         file at ``{l_c_dir}/{slug}.json`` ⇒ inject category Layer-2.
      3. ``bootstrap`` — neither exists ⇒ return ``(None, "bootstrap")``;
         the caller bootstraps from the target app's own trajectories.

    ``fsm_dir=None`` disables tier-1 (collapses to the original
    category→bootstrap behaviour of :func:`resolve_l_c_for_app`).

    Returns ``(prompt_text_or_None, tier)`` where ``tier`` is one of
    ``"app" | "category" | "bootstrap"`` — the tier label lets the caller
    log / analyse which knowledge source fired.
    """
    # ── Tier 1: app-level static FSM ────────────────────────────────
    if fsm_dir is not None:
        fsm_path = Path(fsm_dir) / f"{app_name}.json"
        if fsm_path.exists():
            fsm = FSM.from_json(json.loads(fsm_path.read_text()))
            return fsm.to_prompt_text(), "app"

    # ── Tier 2: category-level L_C ──────────────────────────────────
    text = resolve_l_c_for_app(app_name, splits_yaml_path, l_c_dir)
    if text is not None:
        return text, "category"

    # ── Tier 3: bootstrap (caller handles) ──────────────────────────
    return None, "bootstrap"


__all__ = ["resolve_l_c_for_app", "resolve_app_guidance"]
